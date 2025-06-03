#!/usr/bin/env python3

import csv
import numpy as np
import os
import pathlib
import shutil
import sys
import SimpleITK as sitk

size = (512,512,32)
num_images_per_site = 15
sites = ('client_A', 'client_B')  # this must match the swarm project definition
metadata_folder = 'metadata_unilateral'
data_folder = 'data_unilateral'
other_unused_folders = ('data_raw', 'data')
folders = other_unused_folders + (metadata_folder, data_folder)


def create_folder_structure(output_folder) -> None:
    shutil.rmtree(output_folder, ignore_errors=True)
    os.mkdir(output_folder)
    for i, site in enumerate(sites):
        os.mkdir(output_folder/site)
        for folder in folders:
            os.mkdir(output_folder/site/folder)


def get_image(i: int, j:int, annotation_class: int):
    # create three different types of images depending on the class
    array = np.full(size, 100, dtype=np.uint16)
    if annotation_class == 0:
        array[i,j,:] = 50
    elif annotation_class == 1:
        array[i,j,:] = 200
    else:
        array[i,j,:size[2]//2] = 200
        array[i,j,size[2]//2:] = 50
    image = sitk.GetImageFromArray(array)
    return image


def save_table(output_folder, site: str, table_data: dict) -> None:
    def get_split(fold: int, num: int) -> str:
        # mimic 60/20/20 split that slightly differs between folds
        index = ( (fold + num) % num_images_per_site ) / num_images_per_site
        if index < 0.6:
            return 'train'
        elif index < 0.8:
            return 'val'
        else:
            return 'test'

    with open(output_folder/site/metadata_folder/'split.csv', 'w') as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=('PatientID','UID','Class','Fold','Split'))
        writer.writeheader()
        for i in range(5):
            for j, linedata in enumerate(table_data):
                linedata['Fold'] = i
                linedata['Split'] = get_split(i, j)
                writer.writerow(linedata)

    with open(output_folder/site/metadata_folder/'annotation.csv', 'w') as output_csv:
        writer = csv.DictWriter(output_csv, fieldnames=('PatientID','UID','Class','Fold','Split'))
        writer.writeheader()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: create_synthetic_dataset.py <output folder>')
        exit(1)

    output_folder = pathlib.Path(sys.argv[1])
    create_folder_structure(output_folder)

    for i, site in enumerate(sites):
        table_data = []
        for j in range (num_images_per_site):
            annotation_class = j % 3

            image = get_image(i, j, annotation_class)

            for side in ('left', 'right'):
                id__ = f'ID_{j:03d}'
                id_ = f'{id__}_{side}'
                side_folder = output_folder/site/data_folder/id_
                os.mkdir(side_folder)
                sitk.WriteImage(image, side_folder/'Sub.nii.gz')
                table_data.append({'PatientID': id__,'UID': id_, 'Class': annotation_class})

        save_table(output_folder, site, table_data)
