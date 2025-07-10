#!/usr/bin/env python3

import csv
from itertools import product
import numpy as np
import os
import pathlib
import shutil
import sys
import SimpleITK as sitk
from tqdm import tqdm

np.random.seed(1)

size = (32, 256, 256)
num_images_per_site = 15
sites = ('client_A', 'client_B')  # this must match the swarm project definition
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
    # create three different types of images depending on the class
    array = np.random.randint(-10, 10, size=size, dtype=np.int16)
    if lesion_class == 0:
        array[:, i, j] = -50
    elif lesion_class == 1:
        array[:, i, j] = 200
    else:
        array[:size[2] // 2, i, j] = 200
        array[size[2] // 2:, i, j] = 50
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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: create_synthetic_dataset.py <output folder>')
        exit(1)

    output_folder = pathlib.Path(sys.argv[1])
    create_folder_structure(output_folder)

    for i, site in enumerate(sites):
        table_data = []
        for j in tqdm(range(num_images_per_site), f'Generating synthetic images for {site}'):
            lesion_class = j % 3
            image = get_image(i, j, lesion_class)
            for side in ('left', 'right'):
                patientid = f'ID_{j:03d}'
                uid = f'{patientid}_{side}'
                side_folder = output_folder / site / data_folder / uid
                os.mkdir(side_folder)
                # sitk.WriteImage(image, side_folder/'Pre.nii.gz')
                sitk.WriteImage(image, side_folder / 'Sub_1.nii.gz')
                # sitk.WriteImage(image, side_folder/'T2.nii.gz')
                for f in range(num_folds):
                    table_data.append(
                        {'UID': uid, 'PatientID': patientid, 'Lesion': lesion_class, 'Age': some_age + i + j, 'Fold': f,
                         'Split': get_split(j, f)})

        save_table(output_folder, site, table_data)
