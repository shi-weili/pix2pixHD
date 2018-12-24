import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from PIL import ImageFilter

class SimpleGridDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        if (opt.multi_phase_dataset):
            self.dir = os.path.join(opt.dataroot, opt.phase)
        else:
            self.dir = opt.dataroot

        self.fnames = sorted(make_dataset(os.path.join(self.dir, 'topo'), fnameOnly=True))

    def __getitem__(self, index):
        fname = self.fnames[index]

        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        layers = []

        topo = Image.open(os.path.join(self.dir, 'topo', fname))
        topo = topo.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

        if self.opt.topo_filter == 'detail':
            topo = topo.filter(ImageFilter.DETAIL)
        elif self.opt.topo_filter == 'edge_enhance':
            topo = topo.filter(ImageFilter.EDGE_ENHANCE)
        elif self.opt.topo_filter == 'edge_enhance_more':
            topo = topo.filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif self.opt.topo_filter == 'sharpen':
            topo = topo.filter(ImageFilter.SHARPEN)

        if topo.mode == 'L':
            topo = transforms.ToTensor()(topo)
        else:
            topo = transforms.ToTensor()(topo).type(torch.FloatTensor) / 65535

        topo = topo[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        topo = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(topo)
        layers.append(topo)

        if self.opt.land_ocean == 'land_mask' or self.opt.land_ocean == 'both':
            land = Image.open(os.path.join(self.dir, 'land', fname))
            land = land.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

            if land.mode == 'L':
                land = transforms.ToTensor()(land)
            else:
                land = transforms.ToTensor()(land).type(torch.FloatTensor) / 65535

            land = land[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            land = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(land)              
            layers.append(land)
        
        if self.opt.land_ocean == 'distance_to_ocean' or self.opt.land_ocean == 'both':
            docean = Image.open(os.path.join(self.dir, 'docean', fname))
            docean = docean.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

            if docean.mode == 'L':
                docean = transforms.ToTensor()(docean)
            else:
                docean = transforms.ToTensor()(docean).type(torch.FloatTensor) / 65535

            docean = docean[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            docean = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(docean)
            layers.append(docean)
    
        if self.opt.longi == 'monotone':
            longi = Image.open(os.path.join(self.dir, 'longi', fname))
            longi = longi.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

            if longi.mode == 'L':
                longi = transforms.ToTensor()(longi)
            else:
                longi = transforms.ToTensor()(longi).type(torch.FloatTensor) / 65535

            longi = longi[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            longi = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(longi)
            layers.append(longi)

        elif self.opt.longi == 'circular':
            longi = Image.open(os.path.join(self.dir, 'clongi', fname))
            longi = longi.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

            if longi.mode == 'L':
                longi = transforms.ToTensor()(longi)
            else:
                longi = transforms.ToTensor()(longi).type(torch.FloatTensor) / 65535

            longi = longi[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            longi = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(longi)
            layers.append(longi)

        if self.opt.lati == 'monotone':
            lati = Image.open(os.path.join(self.dir, 'lati', fname))
            lati = lati.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

            if lati.mode == 'L':
                lati = transforms.ToTensor()(lati)
            else:
                lati = transforms.ToTensor()(lati).type(torch.FloatTensor) / 65535

            lati = lati[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            lati = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lati)
            layers.append(lati)

        elif self.opt.lati == 'symmetric':
            lati = Image.open(os.path.join(self.dir, 'slati', fname))
            lati = lati.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

            if lati.mode == 'L':
                lati = transforms.ToTensor()(lati)
            else:
                lati = transforms.ToTensor()(lati).type(torch.FloatTensor) / 65535

            lati = lati[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            lati = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lati)
            layers.append(lati)

        A = torch.cat(layers, dim=0)

        inst_tensor = feat_tensor = 0

        if self.opt.isTrain:
            if self.opt.bm_version == 'bm':
                bm = Image.open(os.path.join(self.dir, 'bm', fname))
                bm = bm.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            elif self.opt.bm_version == 'bm_land_ocean_ice':
                bm = Image.open(os.path.join(self.dir, 'bm_land_ocean_ice', fname))
                bm = bm.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)

            if bm.mode == 'RGB' or bm.mode == 'RGBA':
                bm = transforms.ToTensor()(bm)
            else:
                bm = transforms.ToTensor()(bm).type(torch.FloatTensor) / 65535

            bm = bm[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize, :3]
            bm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(bm)
            B = bm       

        else:
            B = np.zeros_like(topo)  

        input_dict = {'label': A, 'inst': inst_tensor, 'image': B, 
                      'feat': feat_tensor, 'path': fname}

        return input_dict

    def __len__(self):
        return len(self.fnames) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'SimpleGridDataset'
