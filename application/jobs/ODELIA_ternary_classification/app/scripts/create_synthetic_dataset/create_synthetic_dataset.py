#!/usr/bin/env python3
"""Write a small synthetic ODELIA data set (three sites) for tests and CI.

usage: create_synthetic_dataset.py <output folder> [--bd]

--bd additionally writes each site's crops as BreastDivider would lay them out:
data_unilateral_BD/<UID>/Sub_1.nii.gz with a voxel spacing that varies per case, a
processing.log naming BD_unilateralize.py, diagnostic PNGs and the removals CSV next to the
crops, and metadata_unilateral_BD/ as a copy of metadata_unilateral/. That is what
CONFIG=unilateral_BD reads, and what the crop-format guard (data_layout.py) recognises as
BreastDivider output.
"""

import argparse
import csv
from itertools import product
import numpy as np
import os
import pathlib
import shutil
import SimpleITK as sitk
from tqdm import tqdm

np.random.seed(1)

unproblematic_sites = ('client_A', 'client_B')
problematic_sites = ('client_P',)
sites = unproblematic_sites + problematic_sites  # this must match the swarm project definition

size = (32, 256, 256)
num_images_per_site = 15
metadata_folder = 'metadata_unilateral'
data_folder = 'data_unilateral'
other_unused_folders = ('data_raw', 'data')
folders = other_unused_folders + (metadata_folder, data_folder)
some_age = 42 * 365
num_folds = 5


def create_folder_structure(output_folder) -> None:
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)
    for i, site in enumerate(sites):
        os.mkdir(output_folder / site)
        for folder in folders:
            os.mkdir(output_folder / site / folder)


def get_image(i: int, j: int, lesion_class: int):
    def _get_partially_random_array():
        rng = np.random.Generator(np.random.SFC64())
        small_size = (size[0], size[1]//4, size[2]//4)
        random_array = rng.integers(low=-10, high=10, size=small_size, dtype=np.int16)
        array = np.pad(random_array, ((0, 0), ((size[1]-small_size[1])//2, (size[1]-small_size[1])//2), ((size[2]-small_size[2])//2, (size[2]-small_size[2])//2)) )
        return array

    def _set_to_class(image, i: int, j: int):
        if lesion_class == 0:
            array[:, i, j] = -50
        elif lesion_class == 1:
            array[:, i, j] = 200
        else:
            array[:size[2] // 2, i, j] = 200
            array[size[2] // 2:, i, j] = 50
        return array

    array = _get_partially_random_array()
    array = _set_to_class(array, i, j)
    image = sitk.GetImageFromArray(array)
    return image


def save_table(output_folder, site: str, table_data: dict) -> None:
    def write_split_csv(output_folder, site: str, table_data: dict) -> None:
        with open(output_folder / site / metadata_folder / 'split.csv', 'w') as output_csv:
            split_fields = ('UID', 'Fold', 'Split')
            writer = csv.DictWriter(output_csv, fieldnames=split_fields)
            writer.writeheader()
            for linedata in table_data:
                writer.writerow({sf: linedata[sf] for sf in split_fields})

    def _get_annotation_data(table_data: dict, annotation_fields: tuple) -> list:
        annotation_data = [{af: linedata[af] for af in annotation_fields} for linedata in table_data]
        entries = list({tuple(d.items()) for d in annotation_data})
        entries.sort()
        annotation_data = [dict(t) for t in entries]
        return annotation_data

    def write_annotation_csv(output_folder, site: str, table_data: dict) -> None:
        with open(output_folder / site / metadata_folder / 'annotation.csv', 'w') as output_csv:
            annotation_fields = ('UID', 'PatientID', 'Age', 'Lesion')
            writer = csv.DictWriter(output_csv, fieldnames=annotation_fields)
            writer.writeheader()

            annotation_data = _get_annotation_data(table_data, annotation_fields)
            for linedata in annotation_data:
                writer.writerow(linedata)

    write_split_csv(output_folder, site, table_data)
    write_annotation_csv(output_folder, site, table_data)


def get_split(fold: int, num: int) -> str:
    # mimic 60/20/20 split that slightly differs between folds
    index = ((fold + num) % num_images_per_site) / num_images_per_site
    if index < 0.6:
        return 'train'
    elif index < 0.8:
        return 'val'
    else:
        return 'test'


bd_data_folder = 'data_unilateral_BD'
bd_metadata_folder = 'metadata_unilateral_BD'
# Per-axis spacing ranges (mm) of the 912 BreastDivider crops of the Duke benchmark
# (docs/PREPROCESSING_BENCHMARK_BREASTDIVIDER.md); the standard crops are all 0.7 x 0.7 x 3.
bd_spacing_range = ((0.51, 0.85), (0.60, 0.99), (3.75, 8.25))
bd_removals_header = 'case_id,seg_path,reason,label,n_components,voxels_before,voxels_after,voxels_removed\n'


def bd_processing_log(site_root: pathlib.Path) -> str:
    """A processing.log in the shape BD_main.py 0.2.2 writes (command, version, commit, worker
    command, settings, segmentation report), marked as synthetic in its first line."""
    out = site_root / bd_data_folder
    stamp = '2026-10-01 12:00:00'
    return '\n'.join([
        f'{stamp} [INFO] # synthetic stub written by create_synthetic_dataset.py --bd (MediSwarm tests), '
        f'not real BreastDivider output',
        f'{stamp} [INFO] Command: python BD_main.py {site_root} --single-site --mode=normal --num_workers=4 '
        f'--subs 1 --force',
        f'{stamp} [INFO] Package version: 0.2.2',
        f'{stamp} [INFO] Package version source: _version.py',
        f'{stamp} [INFO] Git commit: a084145',
        f'{stamp} [INFO] Worker command: python BD_unilateralize.py --mode normal --data_root {site_root / "data"} '
        f'--seg_root {site_root / "BD_Output"} --output {out} --log {out / "processing.log"} --num_workers 4 '
        f'--xy_resolution 256 --force --subs 1',
        f'{stamp} [INFO] root:1782 mode            : normal',
        f'{stamp} [INFO] root:1788 xy_resolution   : 256 (output grid only; box-finding above is unaffected)',
        f'{stamp} [PROGRESS] root:939 Segmentation report: 0 untrusted-fallback case(s), 0 component-removal(s) '
        f'across 0 case(s) -> {out / "segmentation_cleanup_removals.csv"}',
        f'{stamp} [INFO] root:1691 Outputs: {out}',
        '',
    ])


def write_breastdivider_variant(output_folder, site: str) -> None:
    """Mirror a site's standard crops into a BreastDivider-style layout.

    Same voxels and the same UIDs (BreastDivider names its crops <CaseID>_left/_right from the
    same bilateral input), so the labels fit; only the header spacing varies per case, as it
    does when BreastDivider fits its box to each breast. The extra files BreastDivider leaves
    next to the crops are written too, so tests see that the loader ignores them.
    """
    site_root = output_folder / site
    source, target = site_root / data_folder, site_root / bd_data_folder
    os.mkdir(target)
    exams = set()
    for case_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        image = sitk.ReadImage(str(case_dir / 'Sub_1.nii.gz'))
        image.SetSpacing(tuple(round(float(np.random.uniform(low, high)), 4) for low, high in bd_spacing_range))
        os.mkdir(target / case_dir.name)
        sitk.WriteImage(image, str(target / case_dir.name / 'Sub_1.nii.gz'))
        exams.add(case_dir.name.rsplit('_', 1)[0])
    for exam in sorted(exams)[:3]:
        sitk.WriteImage(sitk.Image(8, 8, sitk.sitkUInt8), str(target / f'{exam}_diagnostic.png'))
    (target / 'segmentation_cleanup_removals.csv').write_text(bd_removals_header)
    (target / 'processing.log').write_text(bd_processing_log(site_root))
    shutil.copytree(site_root / metadata_folder, site_root / bd_metadata_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output_folder')
    parser.add_argument('--bd', action='store_true',
                        help='also write data_unilateral_BD/ and metadata_unilateral_BD/ (CONFIG=unilateral_BD)')
    args = parser.parse_args()

    output_folder = pathlib.Path(args.output_folder)
    create_folder_structure(output_folder)

    for i, site in enumerate(sites):
        table_data = []
        for j in tqdm(range(num_images_per_site), f'Generating synthetic images for {site}'):
            lesion_class = j % 3
            patientid = f'ID_{j:03d}'
            for side in ('left', 'right'):
                uid = f'{patientid}_{side}'
                side_folder = output_folder / site / data_folder / uid
                os.mkdir(side_folder)
                image = get_image(i, j, lesion_class)
                sitk.WriteImage(image, side_folder / 'Sub_1.nii.gz')
                if ((site not in problematic_sites) or (j < num_images_per_site - 1)):  # one image per side without table entry for problematic sites
                    for f in range(num_folds):
                        table_data.append(
                            {'UID': uid, 'PatientID': patientid, 'Lesion': lesion_class, 'Age': some_age + i + j, 'Fold': f,
                             'Split': get_split(j, f)})

        if site in problematic_sites:
            for patientid in ('SomeUID_both', 'ID_016_right', 'ID_016_left', 'ID_998_right', 'ID_999_left'):
                folder = output_folder/site/data_folder/patientid
                os.mkdir(folder)
                shutil.copyfile(output_folder/site/data_folder/'ID_000_left'/'Sub_1.nii.gz', folder/'Sub_1.nii.gz')

            # one table entry per fold without image that is a duplicate in two parts of the split
            for f in range(num_folds):
                j = num_images_per_site + 1
                for side in ('left', 'right'):
                    patientid = f'ID_{j:03d}'
                    uid = f'{patientid}_{side}'
                    table_data.append({'UID': uid, 'PatientID': patientid, 'Lesion': 0, 'Age': 0, 'Fold': f, 'Split': 'train'})
                    table_data.append({'UID': uid, 'PatientID': patientid, 'Lesion': 0, 'Age': 0, 'Fold': f, 'Split': 'val'})
                    table_data.append({'UID': uid, 'PatientID': patientid, 'Lesion': 0, 'Age': 0, 'Fold': f, 'Split': 'test'})
                    patientid = f'ID_{j+1:03d}'
                    uid = f'{patientid}_{side}'
                    table_data.append({'UID': uid, 'PatientID': patientid, 'Lesion': 0, 'Age': 0, 'Fold': f, 'Split': 'train'})

                table_data.append({'UID': 'SomeUID_both', 'PatientID': 'SomeUID', 'Lesion': 0, 'Age': 0, 'Fold': f, 'Split': 'train'}) # one entry not ending in _left or _right
                table_data.append({'UID': 'ID_998_right', 'PatientID': 'SomeUID', 'Lesion': 0, 'Age': 0, 'Fold': f, 'Split': 'train'}) # one entry with _left only
                table_data.append({'UID': 'ID_999_left', 'PatientID': 'SomeUID', 'Lesion': 0, 'Age': 0, 'Fold': f, 'Split': 'train'}) # one entry with _left only

        save_table(output_folder, site, table_data)
        if args.bd:
            write_breastdivider_variant(output_folder, site)
