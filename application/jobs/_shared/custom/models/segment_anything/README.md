# segment_anything (SAM-Med2D, trimmed)

Model code for the SAM-Med2D image encoder used by the `MST_SAMMed2D` backbone
(`models/mst.py`, `backbone_type="sammed2d"`). Contributed by Zhongying Deng (University of
Cambridge) on branch `custom_cam_ZD`; ported onto `main` in September 2026.

Origin: [OpenGVLab/SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D), itself derived from
[facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).
Both are Apache-2.0; the copyright headers are kept as received.

What is here: `build_sam.py` (model registry and checkpoint loading, with the SAM-Med2D
adapter layers and 256-pixel position embeddings) and `modeling/` (ViT image encoder with
`Adapter_Layer`, prompt encoder, mask decoder, two-way transformer). Only the image encoder is
used for classification; the prompt encoder and mask decoder are built so that the published
checkpoint loads without key surgery, then discarded.

What was left out: `predictor.py`, `predictor_sammed.py`, `automatic_mask_generator.py` and
`utils/` (they import cv2, pycocotools and albumentations, none of which is in the ODELIA
image, and none is needed to run the encoder).

Weights: `sam-med2d_b.pth` from the SAM-Med2D release (Google Drive / Baidu links in their
README). It is not committed. Put it at `/MediSwarm/pretrained_weights/sam-med2d_b.pth` in the
image (the build's `_cacheAndCopyPretrainedModelWeights.sh` copies it from the build cache if
present) or point `SAM_MED2D_CHECKPOINT` at it. Without it the encoder starts from random
weights and logs a warning — fine for a preflight, useless for training.
