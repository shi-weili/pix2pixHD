### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class PlanetMarsDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.fnames = sorted(make_dataset(os.path.join(self.dir, 'elevation'), fnameOnly=True))

    def __getitem__(self, index):
        fname = self.fnames[index]

        ### input A (topo data)
        A_path = os.path.join(self.dir, 'elevation', fname)
        elevation = Image.open(A_path)
        elevation = elevation.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        elevation = transforms.ToTensor()(elevation)
        elevation = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(elevation)

        land_mask = Image.open(os.path.join(self.dir, 'land_mask', fname))
        land_mask = land_mask.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        land_mask = transforms.ToTensor()(land_mask)
        land_mask = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(land_mask)
    
        latitude = Image.open(os.path.join(self.dir, 'latitude', fname))
        latitude = latitude.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        latitude = transforms.ToTensor()(latitude).type(torch.FloatTensor) / 64800
        latitude = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(latitude)

        existing_image = Image.open(os.path.join(self.dir, 'existing_image', fname))
        existing_image = existing_image.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        existing_image = transforms.ToTensor()(existing_image)
        existing_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(existing_image)

        existing_image_mask = Image.open(os.path.join(self.dir, 'existing_image_mask', fname))
        existing_image_mask = existing_image_mask.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        existing_image_mask = transforms.ToTensor()(existing_image_mask).type(torch.FloatTensor) / 64800
        existing_image_mask = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(existing_image_mask)

        A_tensor = torch.cat((elevation, land_mask, latitude, existing_image, existing_image_mask), dim=0)

        B_tensor = inst_tensor = feat_tensor = 0

        ### input B (real satelite images)
        if self.opt.isTrain:
            generated_image = Image.open(os.path.join(self.dir, 'generated_image', fname))
            generated_image = generated_image.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
            generated_image = transforms.ToTensor()(generated_image)
            generated_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(generated_image)
            B_tensor = generated_image                          

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.fnames) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'PlanetMarsDataset'