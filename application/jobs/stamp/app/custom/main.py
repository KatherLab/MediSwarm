import logging
import sys

from torch import tensor

import modeling.lightning_model
from modeling.registry import MODEL_REGISTRY

print("Training script is using:", modeling.lightning_model.__file__)

logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s"))
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)

import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import lightning
import lightning.pytorch
import lightning.pytorch.accelerators
import lightning.pytorch.accelerators.accelerator
import torch
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
import logging
import nvflare.client.lightning as flare
import nvflare.client as flare_util

from data import (
    BagDataset,
    PatientData,
    dataloader_from_patient_data,
    filter_complete_patient_data_,
    patient_to_ground_truth_from_clini_table_,
    slide_to_patient_from_slide_table_,
)
from modeling.lightning_model import (
    Bags,
    BagSizes,
    EncodedTargets,
    LitVisionTransformer,
)
from transforms import VaryPrecisionTransform
from modeling.types import Category, CoordinatesBatch, GroundTruth, PandasLabel, PatientId


def train_categorical_model_(
        *,
        clini_table: Path,
        slide_table: Path,
        feature_dir: Path,
        output_dir: Path,
        patient_label: PandasLabel,
        ground_truth_label: PandasLabel,
        filename_label: PandasLabel,
        categories: Sequence[Category] | None,
        # Dataset and -loader parameters
        bag_size: int,
        num_workers: int,
        # Training paramenters
        batch_size: int,
        max_epochs: int,
        patience: int,
        accelerator: str | Accelerator,
        # Experimental features
        use_vary_precision_transform: bool,
        use_alibi: bool,
) -> None:
    """Trains a model.

    Args:
        clini_table:
            An excel or csv file to read the clinical information from.
            Must at least have the columns specified in the arguments

            `patient_label` (containing a unique patient ID)
            and `ground_truth_label` (containing the ground truth to train for).
        slide_table:
            An excel or csv file to read the patient-slide associations from.
            Must at least have the columns specified in the arguments
            `patient_label` (containing the patient ID)
            and `filename_label`
            (containing a filename relative to `feature_dir`
            in which some of the patient's features are stored).
        feature_dir:
            See `slide_table`.
        output_dir:
            Path into which to output the artifacts (trained model etc.)
            generated during training.
        patient_label:
            See `clini_table`, `slide_table`.
        ground_truth_label:
            See `clini_table`.
        filename_label:
            See `slide_table`.
        categories:
            Categories of the ground truth.
            Set to `None` to automatically infer.
    """
    # Read and parse data from out clini and slide table
    patient_to_ground_truth = patient_to_ground_truth_from_clini_table_(
        clini_table_path=clini_table,
        ground_truth_label=ground_truth_label,
        patient_label=patient_label,
    )
    slide_to_patient = slide_to_patient_from_slide_table_(
        slide_table_path=slide_table,
        feature_dir=feature_dir,
        patient_label=patient_label,
        filename_label=filename_label,
    )

    # Clean data (remove slides without ground truth, missing features, etc.)
    patient_to_data = filter_complete_patient_data_(
        patient_to_ground_truth=patient_to_ground_truth,
        slide_to_patient=slide_to_patient,
        drop_patients_with_missing_ground_truth=True,
    )

    # Train the model
    model, train_dl, valid_dl = setup_model_for_training(
        patient_to_data=patient_to_data,
        categories=categories,
        bag_size=bag_size,
        batch_size=batch_size,
        num_workers=num_workers,
        ground_truth_label=ground_truth_label,
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,
        train_transform=(
            VaryPrecisionTransform(min_fraction_bits=1)
            if use_vary_precision_transform
            else None
        ),
        use_alibi=use_alibi,
    )
    train_model_(
        output_dir=output_dir,
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator,
    )


def train_model_(
        *,
        output_dir: Path,
        model: lightning.LightningModule,
        train_dl: DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
        valid_dl: DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
        max_epochs: int,
        patience: int,
        accelerator: str | Accelerator,
) -> lightning.LightningModule:
    """Trains a model.

    Returns:
        The model with the best validation loss during training.
    """
    torch.set_float32_matmul_precision("high")

    model_checkpoint = ModelCheckpoint(
        monitor="validation_loss",
        mode="min",
        filename="checkpoint-{epoch:02d}-{validation_loss:0.3f}",
    )
    trainer = lightning.Trainer(
        default_root_dir=output_dir,
        callbacks=[
            # EarlyStopping(monitor="validation_loss", mode="min", patience=patience),
            model_checkpoint,
        ],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.5,
        logger=CSVLogger(save_dir=output_dir),
        log_every_n_steps=len(train_dl),
    )
    flare_util.init()
    SITE_NAME = flare.get_site_name()
    flare.patch(trainer)  # Patch trainer to enable swarm learning
    logger.info(f"Site name: {SITE_NAME}")

    logger.info(" About to enter flare.is_running loop")

    while flare.is_running():
        logger.info("[DEBUG] waiting to receive swarm model (10s timeout)")
        input_model = flare.receive()
        # logger.info('ModelClass state dict keys()', model.state_dict().keys())
        '''
        if input_model is not None:
            logger.info("==== Swarm model received ====")
            logger.info(
                f"input_model.params.keys() = {list(input_model.params.keys())[:10]} ... total = {len(input_model.params)}")
            # log input_model.params
            logger.info(f"input_model.params.keys() = {input_model.params.keys()}")
            logger.info(
                f"model.state_dict().keys() = {list(model.state_dict().keys())[:10]} ... total = {len(model.state_dict())}")
            try:
                model.load_state_dict(input_model.params)
            except Exception as e:
                logger.error("load_state_dict failed:", exc_info=True)
                raise

        logger.info(f"[DEBUG] Got input model: {input_model}")
        if input_model is None:
            logger.info("[DEBUG] no swarm_start received in 10s")
        else:
            model.load_state_dict(input_model.params)

            logger.info("[DEBUG] received swarm_start:", input_model)
        logger.info(f"Current round: {input_model.current_round}")
        '''

        logger.info(f"input_model.params.keys() = {input_model.params.keys()}")
        logger.info(
            f"model.state_dict().keys() = {list(model.state_dict().keys())[:10]} ... total = {len(model.state_dict())}")
        # log type(model)
        logger.info(f"model class: {type(model)}")

        #model = input_model.model

        trainer.fit(
            model=model,
            train_dataloaders=train_dl,
            val_dataloaders=valid_dl,
        )
        shutil.copy(model_checkpoint.best_model_path, output_dir / "model.ckpt")

    return LitVisionTransformer.load_from_checkpoint(model_checkpoint.best_model_path)


def setup_model_for_training(
        *,
        patient_to_data: Mapping[PatientId, PatientData[GroundTruth]],
        categories: Sequence[Category] | None,
        bag_size: int,
        batch_size: int,
        num_workers: int,
        train_transform: Callable[[torch.Tensor], torch.Tensor] | None,
        use_alibi: bool,
        # Metadata, has no effect on model training
        ground_truth_label: PandasLabel,
        clini_table: Path,
        slide_table: Path,
        feature_dir: Path,
) -> tuple[
    lightning.LightningModule,
    DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
]:
    """Creates a model and dataloaders for training"""

    # Do a stratified train-validation split
    ground_truths = [
        patient_data.ground_truth
        for patient_data in patient_to_data.values()
        if patient_data.ground_truth is not None
    ]

    if len(ground_truths) != len(patient_to_data):
        raise ValueError(
            "patient_to_data must have a ground truth defined for all targets!"
        )

    train_patients, valid_patients = cast(
        tuple[Sequence[PatientId], Sequence[PatientId]],
        train_test_split(
            list(patient_to_data), stratify=ground_truths, shuffle=True, random_state=0
        ),
    )

    train_dl, train_categories = dataloader_from_patient_data(
        patient_data=[patient_to_data[patient] for patient in train_patients],
        categories=categories,
        bag_size=bag_size,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=train_transform,
    )
    del categories  # Let's not accidentally reuse the original categories
    valid_dl, _ = dataloader_from_patient_data(
        patient_data=[patient_to_data[patient] for patient in valid_patients],
        bag_size=None,  # Use all the patient data for validation
        categories=train_categories,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        transform=None,
    )
    if overlap := set(train_patients) & set(valid_patients):
        raise RuntimeError(
            f"unreachable: unexpected overlap between training and validation set: {overlap}"
        )

    # Sample one bag to infer the input dimensions of the model
    bags, coords, bag_sizes, targets = cast(
        tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets], next(iter(train_dl))
    )
    _, _, dim_feats = bags.shape

    # Weigh classes inversely to their occurrence
    category_counts = cast(BagDataset, train_dl.dataset).ground_truths.sum(dim=0)
    cat_ratio_reciprocal = category_counts.sum() / category_counts
    category_weights = cat_ratio_reciprocal / cat_ratio_reciprocal.sum()

    if len(train_categories) <= 1:
        raise ValueError(f"not enough categories to train on: {train_categories}")
    elif any(category_counts < 16):
        underpopulated_categories = {
            category: count
            for category, count in zip(train_categories, category_counts, strict=True)
            if count < 16
        }
        _logger.warning(
            f"Some categories do not have enough samples to meaningfully train a model: {underpopulated_categories}. "
            "You may want to consider removing these categories; the model will likely overfit on the few samples available."
        )

    # Train the model
    model_info = MODEL_REGISTRY['vit']

    ModelClass = model_info["model_class"]
    '''
            categories = ['WT', 'MUT']
        dim_input = 1536
        category_weights =[0.8676, 0.1324]
            dim_model = 512
            dim_feedforward=512
            n_heads=8
            n_layers=2
            dropout=0.25
use_alibi=False
            ground_truth_label='isMSIH'
            train_patients=['TCGA-AH-6644', 'TCGA-AA-3664', 'TCGA-AZ-4614', 'TCGA-D5-5537', 'TCGA-G4-6628', 'TCGA-EI-6882', 'TCGA-AF-2689', 'TCGA-F4-6805', 'TCGA-AZ-4681', 'TCGA-AG-3882', 'TCGA-A6-6648', 'TCGA-AA-A017', 'TCGA-G4-6303', 'TCGA-EI-6508', 'TCGA-AF-6136', 'TCGA-A6-6654', 'TCGA-AD-6895', 'TCGA-CM-5860', 'TCGA-A6-4105', 'TCGA-F5-6810', 'TCGA-QG-A5YV', 'TCGA-DC-6155', 'TCGA-F5-6812', 'TCGA-QL-A97D', 'TCGA-AZ-4615', 'TCGA-CK-6751', 'TCGA-AH-6547', 'TCGA-AA-3980', 'TCGA-CL-5918', 'TCGA-DM-A282', 'TCGA-AA-3514', 'TCGA-EI-6509', 'TCGA-AF-3911', 'TCGA-CL-5917', 'TCGA-A6-2686', 'TCGA-AA-3950', 'TCGA-AA-3549', 'TCGA-AA-3973', 'TCGA-D5-6922', 'TCGA-AG-3887', 'TCGA-CA-6717', 'TCGA-CM-6171', 'TCGA-AD-6964', 'TCGA-A6-6142', 'TCGA-AD-6890', 'TCGA-A6-6138', 'TCGA-AD-A5EK', 'TCGA-D5-6929', 'TCGA-G4-6588', 'TCGA-AZ-4616', 'TCGA-AA-3561', 'TCGA-QG-A5Z2', 'TCGA-DM-A1HA', 'TCGA-CL-4957', 'TCGA-CA-5255', 'TCGA-DC-6158', 'TCGA-G4-6295', 'TCGA-AY-A69D', 'TCGA-DM-A1D6', 'TCGA-AG-3896', 'TCGA-A6-6652', 'TCGA-CM-6680', 'TCGA-AA-3821', 'TCGA-D5-5540', 'TCGA-DC-5869', 'TCGA-AG-4008', 'TCGA-CK-4952', 'TCGA-AA-3875', 'TCGA-AA-A02W', 'TCGA-EI-6514', 'TCGA-DM-A288', 'TCGA-AG-3890', 'TCGA-AA-3979', 'TCGA-NH-A5IV', 'TCGA-G4-6297', 'TCGA-D5-6927', 'TCGA-AA-3680', 'TCGA-NH-A8F8', 'TCGA-A6-5662', 'TCGA-A6-2674', 'TCGA-AG-A01N', 'TCGA-DM-A1D8', 'TCGA-AA-3837', 'TCGA-AA-A01I', 'TCGA-DM-A0XF', 'TCGA-F4-6855', 'TCGA-CA-6715', 'TCGA-AA-3975', 'TCGA-AA-3989', 'TCGA-AG-A016', 'TCGA-A6-5666', 'TCGA-CK-6746', 'TCGA-CM-4752', 'TCGA-CM-6676', 'TCGA-D5-6541', 'TCGA-A6-5667', 'TCGA-AF-2691', 'TCGA-AD-A5EJ', 'TCGA-F5-6864', 'TCGA-AG-3726', 'TCGA-A6-6649', 'TCGA-AA-A03J', 'TCGA-AF-2687', 'TCGA-AA-3858', 'TCGA-G4-6321', 'TCGA-G5-6641', 'TCGA-G4-6309', 'TCGA-D5-5538', 'TCGA-DM-A28K', 'TCGA-AG-3878', 'TCGA-AZ-6598', 'TCGA-AA-3688', 'TCGA-AG-A01W', 'TCGA-AA-3517', 'TCGA-D5-6898', 'TCGA-CM-5341', 'TCGA-CM-6167', 'TCGA-D5-6531', 'TCGA-DC-6683', 'TCGA-AF-2690', 'TCGA-CM-6161', 'TCGA-A6-5660', 'TCGA-NH-A8F7', 'TCGA-CM-6163', 'TCGA-AA-3968', 'TCGA-CM-5868', 'TCGA-AA-3976', 'TCGA-A6-A565', 'TCGA-AM-5821', 'TCGA-AD-6965', 'TCGA-AY-A71X', 'TCGA-AZ-4682', 'TCGA-AG-3892', 'TCGA-A6-2675', 'TCGA-F4-6569', 'TCGA-AG-3893', 'TCGA-CM-5862', 'TCGA-AZ-6606', 'TCGA-CM-6678', 'TCGA-D5-6931', 'TCGA-D5-6923', 'TCGA-AG-3575', 'TCGA-AA-A00Z', 'TCGA-D5-6540', 'TCGA-A6-3808', 'TCGA-CM-4743', 'TCGA-DC-4749', 'TCGA-T9-A92H', 'TCGA-A6-3810', 'TCGA-AD-6889', 'TCGA-G4-6320', 'TCGA-AA-3971', 'TCGA-AZ-4308', 'TCGA-D5-7000', 'TCGA-DM-A28A', 'TCGA-AG-A01L', 'TCGA-EI-6512', 'TCGA-A6-5661', 'TCGA-AG-3599', 'TCGA-AG-3902', 'TCGA-AD-6548', 'TCGA-AA-A022', 'TCGA-AG-3580', 'TCGA-A6-A567', 'TCGA-AA-3679', 'TCGA-CA-5256', 'TCGA-F4-6808', 'TCGA-D5-6536', 'TCGA-DM-A1D9', 'TCGA-AG-3885', 'TCGA-DM-A1D0', 'TCGA-CM-6674', 'TCGA-EI-6507', 'TCGA-F4-6459', 'TCGA-A6-6137', 'TCGA-AD-6899', 'TCGA-EI-6885', 'TCGA-WS-AB45', 'TCGA-D5-6930', 'TCGA-AU-6004', 'TCGA-AY-6196', 'TCGA-AA-A010', 'TCGA-AG-A00C', 'TCGA-4T-AA8H', 'TCGA-G4-6302', 'TCGA-AA-3966', 'TCGA-AF-2693', 'TCGA-D5-6926', 'TCGA-DM-A0XD', 'TCGA-AA-3854', 'TCGA-D5-6932', 'TCGA-EI-7004', 'TCGA-AG-3594', 'TCGA-EI-6510', 'TCGA-AA-3715', 'TCGA-AA-A01X', 'TCGA-F4-6570', 'TCGA-EI-6511', 'TCGA-CM-5861', 'TCGA-CA-5254', 'TCGA-G4-6317', 'TCGA-DM-A28M', 'TCGA-EI-6506', 'TCGA-AG-A020', 'TCGA-AG-3583', 'TCGA-G4-6294', 'TCGA-CM-6165', 'TCGA-D5-6535', 'TCGA-D5-5541', 'TCGA-DM-A1DB', 'TCGA-AG-A01J', 'TCGA-AG-4001', 'TCGA-AG-A00Y', 'TCGA-AA-3949', 'TCGA-AA-3842', 'TCGA-CA-6716', 'TCGA-CM-5348', 'TCGA-AA-3696', 'TCGA-AA-3833', 'TCGA-NH-A6GC', 'TCGA-CK-4947', 'TCGA-AA-3846', 'TCGA-A6-2677', 'TCGA-A6-2684', 'TCGA-NH-A6GA', 'TCGA-F5-6811', 'TCGA-A6-4107', 'TCGA-F5-6571', 'TCGA-CK-6747', 'TCGA-AA-3666', 'TCGA-CK-5914', 'TCGA-G4-6627', 'TCGA-DC-6681', 'TCGA-AZ-4315', 'TCGA-DT-5265', 'TCGA-CM-4751', 'TCGA-CM-6168', 'TCGA-AY-5543', 'TCGA-AZ-6607', 'TCGA-AG-3901', 'TCGA-AA-3695', 'TCGA-AD-6963', 'TCGA-CK-5913', 'TCGA-AF-2692', 'TCGA-G4-6307', 'TCGA-AA-A01T', 'TCGA-DY-A1DG', 'TCGA-AA-A01C', 'TCGA-AZ-6603', 'TCGA-AG-3727', 'TCGA-AG-3909', 'TCGA-D5-6539', 'TCGA-AZ-5407', 'TCGA-NH-A6GB', 'TCGA-AA-3851', 'TCGA-AF-5654', 'TCGA-AA-3530', 'TCGA-F5-6813', 'TCGA-AA-3845', 'TCGA-AG-3584', 'TCGA-AM-5820', 'TCGA-AA-3982', 'TCGA-D5-6924', 'TCGA-AA-3984', 'TCGA-5M-AATE', 'TCGA-AG-3894', 'TCGA-AU-3779', 'TCGA-A6-2671', 'TCGA-DC-6682', 'TCGA-A6-6653', 'TCGA-DM-A28E', 'TCGA-AZ-5403', 'TCGA-CK-4948', 'TCGA-AF-6655', 'TCGA-AA-A01P', 'TCGA-AA-A024', 'TCGA-F4-6809', 'TCGA-AA-3548', 'TCGA-QG-A5YW', 'TCGA-AA-3818', 'TCGA-EI-6513', 'TCGA-AY-4071', 'TCGA-D5-6532', 'TCGA-AZ-6600', 'TCGA-CM-6172', 'TCGA-AG-A01Y', 'TCGA-AA-3675', 'TCGA-AD-6888', 'TCGA-AH-6897', 'TCGA-AA-3852', 'TCGA-CM-4747', 'TCGA-AG-A02N', 'TCGA-AA-A02F', 'TCGA-DM-A1D7', 'TCGA-AA-3678', 'TCGA-CM-6677', 'TCGA-AA-A004', 'TCGA-D5-5539', 'TCGA-F5-6861', 'TCGA-AA-3544', 'TCGA-F4-6463', 'TCGA-AG-4015', 'TCGA-AF-3400', 'TCGA-CM-6169', 'TCGA-F4-6461', 'TCGA-AD-6901', 'TCGA-DM-A280', 'TCGA-AA-3819', 'TCGA-DM-A0X9', 'TCGA-G4-6315', 'TCGA-AA-3850', 'TCGA-CA-5797', 'TCGA-AA-3856', 'TCGA-AY-6386', 'TCGA-AG-3598', 'TCGA-CM-5863', 'TCGA-AG-3605', 'TCGA-AZ-6599', 'TCGA-CM-5344', 'TCGA-AA-3693', 'TCGA-G4-6306', 'TCGA-F4-6854', 'TCGA-A6-5656', 'TCGA-DY-A0XA', 'TCGA-A6-2685', 'TCGA-AY-6197', 'TCGA-4N-A93T', 'TCGA-NH-A50U', 'TCGA-A6-2683', 'TCGA-AA-3952', 'TCGA-DC-6160', 'TCGA-A6-3807', 'TCGA-AA-3522', 'TCGA-G4-6304', 'TCGA-AA-3526', 'TCGA-DC-4745', 'TCGA-AA-3877', 'TCGA-CM-6162', 'TCGA-AA-3692', 'TCGA-AA-3524', 'TCGA-AG-3881', 'TCGA-AA-3681', 'TCGA-CK-4951', 'TCGA-AA-3534', 'TCGA-A6-A56B', 'TCGA-DM-A1D4', 'TCGA-5M-AAT5', 'TCGA-AG-4022', 'TCGA-AA-3812', 'TCGA-D5-6529', 'TCGA-AA-3673', 'TCGA-EI-6917', 'TCGA-EI-6884', 'TCGA-AG-A011', 'TCGA-AY-A8YK', 'TCGA-EI-6881', 'TCGA-A6-2678', 'TCGA-AA-3947', 'TCGA-A6-5659', 'TCGA-AA-3994', 'TCGA-D5-6534', 'TCGA-AG-3574', 'TCGA-F5-6465', 'TCGA-CA-6719', 'TCGA-AA-3532', 'TCGA-G4-6322', 'TCGA-AF-A56K', 'TCGA-AA-3864', 'TCGA-AA-3685', 'TCGA-AA-3986']
            valid_patients=['TCGA-AG-4021', 'TCGA-AG-A02X', 'TCGA-AA-A02H', 'TCGA-F4-6807', 'TCGA-G4-6311', 'TCGA-AF-A56L', 'TCGA-CM-4748', 'TCGA-AZ-6605', 'TCGA-G4-6299', 'TCGA-AA-3667', 'TCGA-5M-AAT4', 'TCGA-5M-AAT6', 'TCGA-F4-6704', 'TCGA-AH-6544', 'TCGA-3L-AA1B', 'TCGA-CK-4950', 'TCGA-AA-3956', 'TCGA-G4-6323', 'TCGA-AA-3831', 'TCGA-AH-6903', 'TCGA-CA-5796', 'TCGA-AA-3684', 'TCGA-EI-7002', 'TCGA-F4-6460', 'TCGA-AA-3844', 'TCGA-AA-3848', 'TCGA-D5-6928', 'TCGA-A6-5665', 'TCGA-AA-A01S', 'TCGA-DM-A28H', 'TCGA-DC-6157', 'TCGA-DY-A1DD', 'TCGA-AA-3519', 'TCGA-AA-3855', 'TCGA-F4-6806', 'TCGA-AA-3520', 'TCGA-CM-6675', 'TCGA-AG-3898', 'TCGA-CK-6748', 'TCGA-G4-6298', 'TCGA-G4-6626', 'TCGA-QG-A5Z1', 'TCGA-DM-A28G', 'TCGA-A6-2681', 'TCGA-DM-A28F', 'TCGA-G4-6586', 'TCGA-CK-5916', 'TCGA-AG-A002', 'TCGA-AG-A026', 'TCGA-AA-A02E', 'TCGA-F5-6464', 'TCGA-EI-6883', 'TCGA-F4-6856', 'TCGA-AG-A015', 'TCGA-CM-5349', 'TCGA-DM-A1DA', 'TCGA-D5-6530', 'TCGA-A6-A566', 'TCGA-NH-A50V', 'TCGA-F5-6814', 'TCGA-CM-6166', 'TCGA-CI-6622', 'TCGA-AA-3977', 'TCGA-CA-6718', 'TCGA-AA-3841', 'TCGA-AA-3521', 'TCGA-NH-A50T', 'TCGA-AA-A01R', 'TCGA-AG-A008', 'TCGA-CM-4746', 'TCGA-AA-A02R', 'TCGA-DC-6154', 'TCGA-AA-3531', 'TCGA-F5-6863', 'TCGA-F4-6703', 'TCGA-AY-A54L', 'TCGA-AA-3811', 'TCGA-AA-3814', 'TCGA-CK-5912', 'TCGA-A6-3809', 'TCGA-A6-5657', 'TCGA-EF-5830', 'TCGA-CM-6164', 'TCGA-SS-A7HO', 'TCGA-CM-5864', 'TCGA-AF-4110', 'TCGA-AF-6672', 'TCGA-AA-3866', 'TCGA-AA-A01V', 'TCGA-AA-A00N', 'TCGA-D5-6538', 'TCGA-CM-6170', 'TCGA-D5-6537', 'TCGA-AA-A02Y', 'TCGA-DY-A1DC', 'TCGA-AG-3883', 'TCGA-AD-5900', 'TCGA-AA-3955', 'TCGA-AG-3581', 'TCGA-A6-5664', 'TCGA-DY-A1DF', 'TCGA-AA-A02O', 'TCGA-AA-3527', 'TCGA-F5-6702', 'TCGA-AH-6643', 'TCGA-G4-6293', 'TCGA-DC-5337', 'TCGA-AA-A03F', 'TCGA-CI-6624', 'TCGA-AG-3602', 'TCGA-AA-3538', 'TCGA-AA-A01Z', 'TCGA-AA-3558', 'TCGA-AA-3560', 'TCGA-AA-3867', 'TCGA-A6-6650', 'TCGA-CK-5915', 'TCGA-A6-6651', 'TCGA-AY-4070', 'TCGA-AF-A56N', 'TCGA-D5-6533', 'TCGA-AA-3529', 'TCGA-AG-A032', 'TCGA-QG-A5YX']

    '''
    model = LitVisionTransformer(categories=['WT', 'MUT'],
                                 dim_input=1536,
                                 category_weights=tensor([0.8676, 0.1324]),
                                 dim_model=512,
                                 dim_feedforward=512,
                                 n_heads=8,
                                 n_layers=2,
                                 dropout=0.25,
                                 use_alibi=False,
                                 ground_truth_label='isMSIH',
                                 train_patients=['TCGA-AH-6644', 'TCGA-AA-3664', 'TCGA-AZ-4614', 'TCGA-D5-5537',
                                                 'TCGA-G4-6628', 'TCGA-EI-6882', 'TCGA-AF-2689', 'TCGA-F4-6805',
                                                 'TCGA-AZ-4681', 'TCGA-AG-3882', 'TCGA-A6-6648', 'TCGA-AA-A017',
                                                 'TCGA-G4-6303', 'TCGA-EI-6508', 'TCGA-AF-6136', 'TCGA-A6-6654',
                                                 'TCGA-AD-6895', 'TCGA-CM-5860', 'TCGA-A6-4105', 'TCGA-F5-6810',
                                                 'TCGA-QG-A5YV', 'TCGA-DC-6155', 'TCGA-F5-6812', 'TCGA-QL-A97D',
                                                 'TCGA-AZ-4615', 'TCGA-CK-6751', 'TCGA-AH-6547', 'TCGA-AA-3980',
                                                 'TCGA-CL-5918', 'TCGA-DM-A282', 'TCGA-AA-3514', 'TCGA-EI-6509',
                                                 'TCGA-AF-3911', 'TCGA-CL-5917', 'TCGA-A6-2686', 'TCGA-AA-3950',
                                                 'TCGA-AA-3549', 'TCGA-AA-3973', 'TCGA-D5-6922', 'TCGA-AG-3887',
                                                 'TCGA-CA-6717', 'TCGA-CM-6171', 'TCGA-AD-6964', 'TCGA-A6-6142',
                                                 'TCGA-AD-6890', 'TCGA-A6-6138', 'TCGA-AD-A5EK', 'TCGA-D5-6929',
                                                 'TCGA-G4-6588', 'TCGA-AZ-4616', 'TCGA-AA-3561', 'TCGA-QG-A5Z2',
                                                 'TCGA-DM-A1HA', 'TCGA-CL-4957', 'TCGA-CA-5255', 'TCGA-DC-6158',
                                                 'TCGA-G4-6295', 'TCGA-AY-A69D', 'TCGA-DM-A1D6', 'TCGA-AG-3896',
                                                 'TCGA-A6-6652', 'TCGA-CM-6680', 'TCGA-AA-3821', 'TCGA-D5-5540',
                                                 'TCGA-DC-5869', 'TCGA-AG-4008', 'TCGA-CK-4952', 'TCGA-AA-3875',
                                                 'TCGA-AA-A02W', 'TCGA-EI-6514', 'TCGA-DM-A288', 'TCGA-AG-3890',
                                                 'TCGA-AA-3979', 'TCGA-NH-A5IV', 'TCGA-G4-6297', 'TCGA-D5-6927',
                                                 'TCGA-AA-3680', 'TCGA-NH-A8F8', 'TCGA-A6-5662', 'TCGA-A6-2674',
                                                 'TCGA-AG-A01N', 'TCGA-DM-A1D8', 'TCGA-AA-3837', 'TCGA-AA-A01I',
                                                 'TCGA-DM-A0XF', 'TCGA-F4-6855', 'TCGA-CA-6715', 'TCGA-AA-3975',
                                                 'TCGA-AA-3989', 'TCGA-AG-A016', 'TCGA-A6-5666', 'TCGA-CK-6746',
                                                 'TCGA-CM-4752', 'TCGA-CM-6676', 'TCGA-D5-6541', 'TCGA-A6-5667',
                                                 'TCGA-AF-2691', 'TCGA-AD-A5EJ', 'TCGA-F5-6864', 'TCGA-AG-3726',
                                                 'TCGA-A6-6649', 'TCGA-AA-A03J', 'TCGA-AF-2687', 'TCGA-AA-3858',
                                                 'TCGA-G4-6321', 'TCGA-G5-6641', 'TCGA-G4-6309', 'TCGA-D5-5538',
                                                 'TCGA-DM-A28K', 'TCGA-AG-3878', 'TCGA-AZ-6598', 'TCGA-AA-3688',
                                                 'TCGA-AG-A01W', 'TCGA-AA-3517', 'TCGA-D5-6898', 'TCGA-CM-5341',
                                                 'TCGA-CM-6167', 'TCGA-D5-6531', 'TCGA-DC-6683', 'TCGA-AF-2690',
                                                 'TCGA-CM-6161', 'TCGA-A6-5660', 'TCGA-NH-A8F7', 'TCGA-CM-6163',
                                                 'TCGA-AA-3968', 'TCGA-CM-5868', 'TCGA-AA-3976', 'TCGA-A6-A565',
                                                 'TCGA-AM-5821', 'TCGA-AD-6965', 'TCGA-AY-A71X', 'TCGA-AZ-4682',
                                                 'TCGA-AG-3892', 'TCGA-A6-2675', 'TCGA-F4-6569', 'TCGA-AG-3893',
                                                 'TCGA-CM-5862', 'TCGA-AZ-6606', 'TCGA-CM-6678', 'TCGA-D5-6931',
                                                 'TCGA-D5-6923', 'TCGA-AG-3575', 'TCGA-AA-A00Z', 'TCGA-D5-6540',
                                                 'TCGA-A6-3808', 'TCGA-CM-4743', 'TCGA-DC-4749', 'TCGA-T9-A92H',
                                                 'TCGA-A6-3810', 'TCGA-AD-6889', 'TCGA-G4-6320', 'TCGA-AA-3971',
                                                 'TCGA-AZ-4308', 'TCGA-D5-7000', 'TCGA-DM-A28A', 'TCGA-AG-A01L',
                                                 'TCGA-EI-6512', 'TCGA-A6-5661', 'TCGA-AG-3599', 'TCGA-AG-3902',
                                                 'TCGA-AD-6548', 'TCGA-AA-A022', 'TCGA-AG-3580', 'TCGA-A6-A567',
                                                 'TCGA-AA-3679', 'TCGA-CA-5256', 'TCGA-F4-6808', 'TCGA-D5-6536',
                                                 'TCGA-DM-A1D9', 'TCGA-AG-3885', 'TCGA-DM-A1D0', 'TCGA-CM-6674',
                                                 'TCGA-EI-6507', 'TCGA-F4-6459', 'TCGA-A6-6137', 'TCGA-AD-6899',
                                                 'TCGA-EI-6885', 'TCGA-WS-AB45', 'TCGA-D5-6930', 'TCGA-AU-6004',
                                                 'TCGA-AY-6196', 'TCGA-AA-A010', 'TCGA-AG-A00C', 'TCGA-4T-AA8H',
                                                 'TCGA-G4-6302', 'TCGA-AA-3966', 'TCGA-AF-2693', 'TCGA-D5-6926',
                                                 'TCGA-DM-A0XD', 'TCGA-AA-3854', 'TCGA-D5-6932', 'TCGA-EI-7004',
                                                 'TCGA-AG-3594', 'TCGA-EI-6510', 'TCGA-AA-3715', 'TCGA-AA-A01X',
                                                 'TCGA-F4-6570', 'TCGA-EI-6511', 'TCGA-CM-5861', 'TCGA-CA-5254',
                                                 'TCGA-G4-6317', 'TCGA-DM-A28M', 'TCGA-EI-6506', 'TCGA-AG-A020',
                                                 'TCGA-AG-3583', 'TCGA-G4-6294', 'TCGA-CM-6165', 'TCGA-D5-6535',
                                                 'TCGA-D5-5541', 'TCGA-DM-A1DB', 'TCGA-AG-A01J', 'TCGA-AG-4001',
                                                 'TCGA-AG-A00Y', 'TCGA-AA-3949', 'TCGA-AA-3842', 'TCGA-CA-6716',
                                                 'TCGA-CM-5348', 'TCGA-AA-3696', 'TCGA-AA-3833', 'TCGA-NH-A6GC',
                                                 'TCGA-CK-4947', 'TCGA-AA-3846', 'TCGA-A6-2677', 'TCGA-A6-2684',
                                                 'TCGA-NH-A6GA', 'TCGA-F5-6811', 'TCGA-A6-4107', 'TCGA-F5-6571',
                                                 'TCGA-CK-6747', 'TCGA-AA-3666', 'TCGA-CK-5914', 'TCGA-G4-6627',
                                                 'TCGA-DC-6681', 'TCGA-AZ-4315', 'TCGA-DT-5265', 'TCGA-CM-4751',
                                                 'TCGA-CM-6168', 'TCGA-AY-5543', 'TCGA-AZ-6607', 'TCGA-AG-3901',
                                                 'TCGA-AA-3695', 'TCGA-AD-6963', 'TCGA-CK-5913', 'TCGA-AF-2692',
                                                 'TCGA-G4-6307', 'TCGA-AA-A01T', 'TCGA-DY-A1DG', 'TCGA-AA-A01C',
                                                 'TCGA-AZ-6603', 'TCGA-AG-3727', 'TCGA-AG-3909', 'TCGA-D5-6539',
                                                 'TCGA-AZ-5407', 'TCGA-NH-A6GB', 'TCGA-AA-3851', 'TCGA-AF-5654',
                                                 'TCGA-AA-3530', 'TCGA-F5-6813', 'TCGA-AA-3845', 'TCGA-AG-3584',
                                                 'TCGA-AM-5820', 'TCGA-AA-3982', 'TCGA-D5-6924', 'TCGA-AA-3984',
                                                 'TCGA-5M-AATE', 'TCGA-AG-3894', 'TCGA-AU-3779', 'TCGA-A6-2671',
                                                 'TCGA-DC-6682', 'TCGA-A6-6653', 'TCGA-DM-A28E', 'TCGA-AZ-5403',
                                                 'TCGA-CK-4948', 'TCGA-AF-6655', 'TCGA-AA-A01P', 'TCGA-AA-A024',
                                                 'TCGA-F4-6809', 'TCGA-AA-3548', 'TCGA-QG-A5YW', 'TCGA-AA-3818',
                                                 'TCGA-EI-6513', 'TCGA-AY-4071', 'TCGA-D5-6532', 'TCGA-AZ-6600',
                                                 'TCGA-CM-6172', 'TCGA-AG-A01Y', 'TCGA-AA-3675', 'TCGA-AD-6888',
                                                 'TCGA-AH-6897', 'TCGA-AA-3852', 'TCGA-CM-4747', 'TCGA-AG-A02N',
                                                 'TCGA-AA-A02F', 'TCGA-DM-A1D7', 'TCGA-AA-3678', 'TCGA-CM-6677',
                                                 'TCGA-AA-A004', 'TCGA-D5-5539', 'TCGA-F5-6861', 'TCGA-AA-3544',
                                                 'TCGA-F4-6463', 'TCGA-AG-4015', 'TCGA-AF-3400', 'TCGA-CM-6169',
                                                 'TCGA-F4-6461', 'TCGA-AD-6901', 'TCGA-DM-A280', 'TCGA-AA-3819',
                                                 'TCGA-DM-A0X9', 'TCGA-G4-6315', 'TCGA-AA-3850', 'TCGA-CA-5797',
                                                 'TCGA-AA-3856', 'TCGA-AY-6386', 'TCGA-AG-3598', 'TCGA-CM-5863',
                                                 'TCGA-AG-3605', 'TCGA-AZ-6599', 'TCGA-CM-5344', 'TCGA-AA-3693',
                                                 'TCGA-G4-6306', 'TCGA-F4-6854', 'TCGA-A6-5656', 'TCGA-DY-A0XA',
                                                 'TCGA-A6-2685', 'TCGA-AY-6197', 'TCGA-4N-A93T', 'TCGA-NH-A50U',
                                                 'TCGA-A6-2683', 'TCGA-AA-3952', 'TCGA-DC-6160', 'TCGA-A6-3807',
                                                 'TCGA-AA-3522', 'TCGA-G4-6304', 'TCGA-AA-3526', 'TCGA-DC-4745',
                                                 'TCGA-AA-3877', 'TCGA-CM-6162', 'TCGA-AA-3692', 'TCGA-AA-3524',
                                                 'TCGA-AG-3881', 'TCGA-AA-3681', 'TCGA-CK-4951', 'TCGA-AA-3534',
                                                 'TCGA-A6-A56B', 'TCGA-DM-A1D4', 'TCGA-5M-AAT5', 'TCGA-AG-4022',
                                                 'TCGA-AA-3812', 'TCGA-D5-6529', 'TCGA-AA-3673', 'TCGA-EI-6917',
                                                 'TCGA-EI-6884', 'TCGA-AG-A011', 'TCGA-AY-A8YK', 'TCGA-EI-6881',
                                                 'TCGA-A6-2678', 'TCGA-AA-3947', 'TCGA-A6-5659', 'TCGA-AA-3994',
                                                 'TCGA-D5-6534', 'TCGA-AG-3574', 'TCGA-F5-6465', 'TCGA-CA-6719',
                                                 'TCGA-AA-3532', 'TCGA-G4-6322', 'TCGA-AF-A56K', 'TCGA-AA-3864',
                                                 'TCGA-AA-3685', 'TCGA-AA-3986'],
                                 valid_patients=['TCGA-AG-4021', 'TCGA-AG-A02X', 'TCGA-AA-A02H', 'TCGA-F4-6807',
                                                 'TCGA-G4-6311', 'TCGA-AF-A56L', 'TCGA-CM-4748', 'TCGA-AZ-6605',
                                                 'TCGA-G4-6299', 'TCGA-AA-3667', 'TCGA-5M-AAT4', 'TCGA-5M-AAT6',
                                                 'TCGA-F4-6704', 'TCGA-AH-6544', 'TCGA-3L-AA1B', 'TCGA-CK-4950',
                                                 'TCGA-AA-3956', 'TCGA-G4-6323', 'TCGA-AA-3831', 'TCGA-AH-6903',
                                                 'TCGA-CA-5796', 'TCGA-AA-3684', 'TCGA-EI-7002', 'TCGA-F4-6460',
                                                 'TCGA-AA-3844', 'TCGA-AA-3848', 'TCGA-D5-6928', 'TCGA-A6-5665',
                                                 'TCGA-AA-A01S', 'TCGA-DM-A28H', 'TCGA-DC-6157', 'TCGA-DY-A1DD',
                                                 'TCGA-AA-3519', 'TCGA-AA-3855', 'TCGA-F4-6806', 'TCGA-AA-3520',
                                                 'TCGA-CM-6675', 'TCGA-AG-3898', 'TCGA-CK-6748', 'TCGA-G4-6298',
                                                 'TCGA-G4-6626', 'TCGA-QG-A5Z1', 'TCGA-DM-A28G', 'TCGA-A6-2681',
                                                 'TCGA-DM-A28F', 'TCGA-G4-6586', 'TCGA-CK-5916', 'TCGA-AG-A002',
                                                 'TCGA-AG-A026', 'TCGA-AA-A02E', 'TCGA-F5-6464', 'TCGA-EI-6883',
                                                 'TCGA-F4-6856', 'TCGA-AG-A015', 'TCGA-CM-5349', 'TCGA-DM-A1DA',
                                                 'TCGA-D5-6530', 'TCGA-A6-A566', 'TCGA-NH-A50V', 'TCGA-F5-6814',
                                                 'TCGA-CM-6166', 'TCGA-CI-6622', 'TCGA-AA-3977', 'TCGA-CA-6718',
                                                 'TCGA-AA-3841', 'TCGA-AA-3521', 'TCGA-NH-A50T', 'TCGA-AA-A01R',
                                                 'TCGA-AG-A008', 'TCGA-CM-4746', 'TCGA-AA-A02R', 'TCGA-DC-6154',
                                                 'TCGA-AA-3531', 'TCGA-F5-6863', 'TCGA-F4-6703', 'TCGA-AY-A54L',
                                                 'TCGA-AA-3811', 'TCGA-AA-3814', 'TCGA-CK-5912', 'TCGA-A6-3809',
                                                 'TCGA-A6-5657', 'TCGA-EF-5830', 'TCGA-CM-6164', 'TCGA-SS-A7HO',
                                                 'TCGA-CM-5864', 'TCGA-AF-4110', 'TCGA-AF-6672', 'TCGA-AA-3866',
                                                 'TCGA-AA-A01V', 'TCGA-AA-A00N', 'TCGA-D5-6538', 'TCGA-CM-6170',
                                                 'TCGA-D5-6537', 'TCGA-AA-A02Y', 'TCGA-DY-A1DC', 'TCGA-AG-3883',
                                                 'TCGA-AD-5900', 'TCGA-AA-3955', 'TCGA-AG-3581', 'TCGA-A6-5664',
                                                 'TCGA-DY-A1DF', 'TCGA-AA-A02O', 'TCGA-AA-3527', 'TCGA-F5-6702',
                                                 'TCGA-AH-6643', 'TCGA-G4-6293', 'TCGA-DC-5337', 'TCGA-AA-A03F',
                                                 'TCGA-CI-6624', 'TCGA-AG-3602', 'TCGA-AA-3538', 'TCGA-AA-A01Z',
                                                 'TCGA-AA-3558', 'TCGA-AA-3560', 'TCGA-AA-3867', 'TCGA-A6-6650',
                                                 'TCGA-CK-5915', 'TCGA-A6-6651', 'TCGA-AY-4070', 'TCGA-AF-A56N',
                                                 'TCGA-D5-6533', 'TCGA-AA-3529', 'TCGA-AG-A032', 'TCGA-QG-A5YX'],
                                 )

    '''
    model = ModelClass(
        categories=[np.str_('MUT'), np.str_('WT')],
    dim_input = 1536,
    category_weights = tensor([0.8676, 0.1324]),
    dim_model=512,
     dim_feedforward=512,
     n_heads=8,
     n_layers=2,
     dropout=0.25,

        use_alibi=use_alibi,
        ground_truth_label=ground_truth_label,
        train_patients=train_patients,
        valid_patients=valid_patients,
        clini_table=clini_table,
        slide_table=slide_table,
        feature_dir=feature_dir,)
    '''
    logger.info(
        f"Model 'instantiated with {len(model.state_dict())} parameters."
    )
    return model, train_dl, valid_dl


def _add_file_handle_(logger: logging.Logger, *, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(output_dir / "logfile.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def main():
    ''''
    output_dir = os.getenv("TRAINING_OUTPUT_DIR")
    print('output_dir:', output_dir)
    #_add_file_handle_(_logger, output_dir=Path(output_dir))
    #_logger.info("Using training configuration from environment variables.")

    train_categorical_model_(
        output_dir=Path(output_dir),
        clini_table=Path(os.getenv("TRAINING_CLINI_TABLE")),
        slide_table=Path(os.getenv("TRAINING_SLIDE_TABLE")),
        feature_dir=Path(os.getenv("TRAINING_FEATURE_DIR")),
        patient_label=os.getenv("TRAINING_PATIENT_LABEL"),
        ground_truth_label=os.getenv("TRAINING_GROUND_TRUTH_LABEL"),
        filename_label=os.getenv("TRAINING_FILENAME_LABEL"),
        categories=os.getenv("TRAINING_CATEGORIES").split(","),
        # Dataset and loader parameters
        bag_size=int(os.getenv("TRAINING_BAG_SIZE")),
        num_workers=int(os.getenv("TRAINING_NUM_WORKERS")),
        # Training parameters
        batch_size=int(os.getenv("TRAINING_BATCH_SIZE")),
        max_epochs=int(os.getenv("TRAINING_MAX_EPOCHS")),
        patience=int(os.getenv("TRAINING_PATIENCE")),
        accelerator=os.getenv("TRAINING_ACCELERATOR"),
        # Experimental features
        use_vary_precision_transform=os.getenv("USE_VARY_PRECISION_TRANSFORM", "False").lower() == "true",
        use_alibi=os.getenv("USE_ALIBI", "False").lower() == "true",
    )
'''
    output_dir = "/mnt/swarm_alpha/ECDP2025/pathology_data/TCGA/TCGA-CRC/STAMP_crossval_new"
    #logger.info('output_dir:', output_dir)
    # _add_file_handle_(_logger, output_dir=Path(output_dir))
    #_logger.info("Using training configuration from environment variables.")

    train_categorical_model_(
        output_dir=Path(output_dir),
        clini_table=Path("/mnt/swarm_alpha/ECDP2025/pathology_data/TCGA/TCGA-CRC/TCGA-CRC-DX_clini.xlsx"),
        slide_table=Path("/mnt/swarm_alpha/ECDP2025/pathology_data/TCGA/TCGA-CRC/TCGA-CRC-DX_slide_h5.csv"),
        feature_dir=Path("/mnt/swarm_alpha/ECDP2025/pathology_data/TCGA/TCGA-CRC/uni2-02627079"),
        patient_label="PATIENT",
        ground_truth_label="isMSIH",
        filename_label="FILENAME",
        categories=None,
        # Dataset and loader parameters
        bag_size=512,
        num_workers=15,
        # Training parameters
        batch_size=64,
        max_epochs=1,  # 64
        patience=16,
        accelerator="gpu",
        # Experimental features
        use_vary_precision_transform='true',
        use_alibi='false',
    )
if __name__ == "__main__":
    main()
