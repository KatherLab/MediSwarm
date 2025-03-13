from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch  
# 					TODO maybe adapt to ImageOrSubjectToTensor (both functions in one) for all centers.
from ..augmentation.augmentations_3d import ImageToTensor, ZNormalization, CropOrPad, UKA_CropOrPad

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ODELIA_Dataset3D(data.Dataset):
    ALL_INSTITUTIONS = ['CAM', 'RSH', 'RUMC', 'UKA', 'UMCU']

    def __init__(
            self,
            path_root=None,
            institutions="UKA",
            fold = 0,
            split= None,
            fraction=None,
            transform = None,
            flip = False,
            random_rotate=False,
            random_inverse=False,
            noise=False, 
            to_tensor = True,
        ):
        self.path_root = Path(self.PATH_ROOT if path_root is None else path_root)
        self.split = split 
        
        if (institutions is None) or (institutions == "ODELIA"):
            institutions = self.ALL_INSTITUTIONS
        elif isinstance(institutions, str):
            institutions = [institutions]
        self.institutions = institutions
        
        logger.info(f"-------------------- {path_root} -----------------------")
        for path_file in self.path_root.iterdir():
            logger.info(f"-------------------- {path_file} -----------------------")
        

        if transform is None: 
            self.transform = tio.Compose([
                tio.Flip((1,0)), # Just for viewing, otherwise upside down
                UKA_CropOrPad((224, 224, 32), random_center=random_rotate), 

                ZNormalization(per_channel=True, masking_method=lambda x:(x>x.min()) & (x<x.max()), percentiles=(0.5, 99.5)), 

                # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x), # WARNING: Padding mask 
                tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if random_inverse else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(lambda x: x),

                ImageToTensor() if to_tensor else tio.Lambda(lambda x: x)             
            ])
        else:
            self.transform = transform
        
        path_csv = self.path_root/'metadata/split.csv'
        df = ds.load_split(path_csv, fold=fold, split=split, fraction=fraction)
            
        # Verify files exist
        uids = self.run_item_crawler(self.path_root/'data_unilateral')
        df = df[df['UID'].isin(uids)]

        self.df = df # pd.concat(df).reset_index()
        self.item_pointers = self.df.index.tolist()

        


    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        target = item['Class']
        uid = item['UID']

        # Use only binary label (Cancer yes/no)
        if self.institutions != "DUKE":
            target = int(target == 2)
    
        img = self.load_img(self.path_root/'data_unilateral'/uid/'Sub.nii.gz')
        img = self.transform(img)

        return {'uid':uid, 'source': img, 'target':target}
    
    def get_label(self):
        return self.df['Class']
    
    @classmethod
    def load_split(cls, filepath_or_buffer=None, fold=0, split=None, fraction=None):
        df = pd.read_csv(filepath_or_buffer)
        df = df[df['Fold'] == fold]
        if split is not None:
            df = df[df['Split'] == split]   
        if fraction is not None:
            df = df.sample(frac=fraction, random_state=0).reset_index()
        return df
    
    @classmethod
    def run_item_crawler(cls, path_root, **kwargs):
        return [path.relative_to(path_root).name for path in Path(path_root).iterdir() if path.is_dir() ]
