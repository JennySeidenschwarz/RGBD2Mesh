import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path
import glob
import imageio
import random
import trimesh

from src.utils.train_util import instantiate_from_config


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):
        # if stage in ['fit', TrainerFn.VALIDATING]:
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        # else:
        #     raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='objaverse/',
        meta_fname='valid_paths.json',
        image_dir='rendering_zero123plus',
        validation=False,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = image_dir

        self.paths = glob.glob(f"{os.path.join(self.root_dir, self.image_dir)}/*/*")
        random.seed(10)
        random.shuffle(self.paths)
        
        total_objects = len(self.paths)
        if validation:
            # self.paths = self.paths[-16:] # used last 16 as validation
            self.paths = self.paths[:100] # used last 16 as validation
            self.visualize = random.choices(list(range(100)))
        else:
            # self.paths = self.paths[:-16]
            self.paths = self.paths[:100] # used last 16 as validation
            self.visualize = list()
        print('============= length of dataset %d =============' % len(self.paths))

        clean_paths = list()
        for path in self.paths:
            if not os.path.isfile(os.path.join(path, 'mesh.ply')):
                continue
            mesh = trimesh.load(os.path.join(path, 'mesh.ply'))
            if isinstance(mesh, trimesh.base.Trimesh) and len(mesh.vertices) > 2:
                clean_paths.append(path)
        self.paths = clean_paths
        print('============= length of dataset after cleaning %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        if path[-3:] == 'exr':
            img = imageio.imread(path, format='EXR-FI')
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.open(path)
        pil_img = pil_img.resize((self.input_image_size, self.input_image_size), resample=Image.BICUBIC)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        if image.shape[-1] == 4:
            alpha = image[:, :, 3:]
            image = image[:, :, :3] * alpha + color * (1 - alpha)
        else:
            alpha = np.ones_like(image[:, :, :1])

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def __getitem__(self, index, with_depth=True):
        if with_depth:
            return self.__getitem_with_depth__(index)
        else:
            return self.__getitem_without_depth__(index)

    def __getitem_without_depth__(self, index):
        while True:
            image_path = self.paths[index]

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            depth_list = []
            try:
                for idx in range(7):
                    img, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bkg_color)
                    depth = imageio.imread(os.path.join(image_path, 'depth_%03d.exr' % idx), format='EXR-FI')[:, :, 0]
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    img_list.append(img)
                    depth_list.append(depth)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        imgs = torch.stack(img_list, dim=0).float()

        data = {
            'cond_imgs': imgs[0],           # (3, H, W)
            'target_imgs': imgs[1:],        # (6, 3, H, W)
        }

        return data

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def __getitem_with_depth__(self, index):
        bg_white = [1., 1., 1.]
        bg_black = [0., 0., 0.]

        while True:
            image_list = []
            alpha_list = []
            depth_list = []
            normal_list = []
            pose_list = []
            intrinsics_list = []
            image_path = self.paths[index]
            try:
                for idx in range(7):
                    image, alpha = self.load_im(os.path.join(image_path, '%03d.png' % idx), bg_white)
                    normal = imageio.imread(os.path.join(image_path, 'normal_%03d.exr' % idx), format='EXR-FI')
                    normal = torch.from_numpy(normal).unsqueeze(0)
                    depth = imageio.imread(os.path.join(image_path, 'depth_%03d.exr' % idx), format='EXR-FI')
                    depth = torch.from_numpy(depth).unsqueeze(0)
                    pose = np.vstack([np.load(os.path.join(image_path, 'extrinsics_%03d.npy' % idx)), np.array([[0, 0, 0, 1]])])
                    intrinsics = np.load(os.path.join(image_path, 'intrinsics.npy'))

                    image_list.append(image)
                    alpha_list.append(alpha)
                    depth_list.append(depth)
                    normal_list.append(normal)
                    pose_list.append(pose)
                    intrinsics_list.append(intrinsics)

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
            
        images = torch.stack(image_list, dim=0).float()                 # (6+V, 3, H, W)
        alphas = torch.stack(alpha_list, dim=0).float()                 # (6+V, 1, H, W)
        depths = torch.stack(depth_list, dim=0).float()                 # (6+V, 1, H, W)
        normals = torch.stack(normal_list, dim=0).float()               # (6+V, 3, H, W)
        w2cs = torch.from_numpy(np.stack(pose_list, axis=0)).float()    # (6+V, 4, 4)
        c2ws = torch.linalg.inv(w2cs).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()    # (6+V, 4, 4)

        data = {
            'cond_imgs': images[0],           # (6, 3, H, W)
            'cond_alphas': alphas[0],           # (6, 1, H, W) 
            'cond_depths': depths[0],           # (6, 1, H, W)
            'cond_normals': normals[0],         # (6, 3, H, W)
            'cond_c2ws': c2ws[0],               # (6, 4, 4)
            'cond_Ks': intrinsics[0],                   # (6, 3, 3)

            # lrm generator input and supervision
            'target_images': images[1:],          # (V, 3, H, W)
            'target_alphas': alphas[1:],          # (V, 1, H, W)
            'target_depths': depths[1:],          # (V, 1, H, W)
            'target_normals': normals[1:],        # (V, 3, H, W)
            'target_c2ws': c2ws[1:],              # (V, 4, 4)
            'target_Ks': intrinsics[1:],                  # (V, 3, 3)
            'mesh_path': os.path.join(image_path, 'mesh.ply'),
            'visualize': True if index in self.visualize else False
        }

        

        return data