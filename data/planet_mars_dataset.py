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
        self.fnames = sorted(make_dataset(os.path.join(self.dir, 'topo'), fnameOnly=True))

    def __getitem__(self, index):
        fname = self.fnames[index]

        ### input A (topo data)
        A_path = os.path.join(self.dir, 'topo', fname)
        topo = Image.open(A_path)
        topo = topo.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        topo = transforms.ToTensor()(topo)
        topo = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(topo)

        land = Image.open(os.path.join(self.dir, 'land', fname))
        land = land.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        land = transforms.ToTensor()(land)
        land = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(land)

        longi = Image.open(os.path.join(self.dir, 'longi', fname))
        longi = longi.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        longi = transforms.ToTensor()(longi).type(torch.FloatTensor) / 64800
        longi = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(longi)
    
        lati = Image.open(os.path.join(self.dir, 'lati', fname))
        lati = lati.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        lati = transforms.ToTensor()(lati).type(torch.FloatTensor) / 64800
        lati = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lati)

        A_tensor = torch.cat((topo, land, longi, lati), dim=0)

        B_tensor = inst_tensor = feat_tensor = 0

        ### input B (real satelite images)
        if self.opt.isTrain:
            bm = Image.open(os.path.join(self.dir, 'bm', fname))
            bm = bm.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
            bm = transforms.ToTensor()(bm)
            bm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(bm)
            B_tensor = bm                          

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor, 
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.fnames) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'PlanetMarsDataset'