import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize, get_trans_txt_mc,get_trans_txt_dc
from data.image_folder import make_dataset
from data.text_folder import make_dataset_txt
from PIL import Image
import numpy as np
import copy
import torchvision.transforms as transforms
import random

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        dir_A = '_A'
        self.dir_A = os.path.join(opt.dataroot, (opt.phase + dir_A))
        self.A_paths = sorted(make_dataset(self.dir_A))
        
        ### input txt (label txts)
        dir_txt = 'raw_txt/'
        self.dir_txt = os.path.join(opt.dataroot, (dir_txt + opt.phase))
        self.txt_paths = sorted(make_dataset_txt(self.dir_txt))

        ### input B (real images)
        dir_B = '_B'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        ### input C (condition images mask)
        self.dir_C = os.path.join(opt.dataroot,'cond',opt.phase)
        self.C_paths = sorted(make_dataset(self.dir_C))

        self.dataset_size = len(self.A_paths)

        type_path = self.txt_paths[0]

        txt_total = np.loadtxt(type_path, dtype=str, encoding="utf-8").tolist()
        self.type = float(txt_total[-1])

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index] #逐一获取路径
        A = Image.open(A_path).convert("RGB")

        transform_A = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        A_tensor = transform_A(A)

        ### input B (real images)
        B_path = self.B_paths[index]
        B = Image.open(B_path).convert('RGB')
        transform_B = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        B_tensor = transform_B(B)

        ### input C (condition mask)
        C_path = self.C_paths[index]
        C = Image.open(C_path).convert('RGB')
        transform_C = transforms.ToTensor()
        C_tensor = transform_C(C)

        ### input txt (real txts)
        txt_path =self.txt_paths[index]

        txt_lines = np.loadtxt(txt_path, dtype=str, encoding="utf-8").tolist()

        txt1, txt2, txt3, txt4, txt5 = [], [], [], [], []

        real_txt = [txt1, txt2, txt3, txt4, txt5]
        for i, line in enumerate(txt_lines):
            real_txt[i].extend(map(float, line.split(",")))
        txt_tensor = 0
        if txt5[0]>2:
            txt_tensor = get_trans_txt_dc(self.opt,real_txt,C_tensor)
        else:
            txt_tensor = get_trans_txt_mc(self.opt, real_txt, C_tensor)

        ### fake txt (fake txts)
        if txt5[0] > 2:
            while True:
                txt6 = [round(random.uniform(2.3, 2.95), 2)]
                if abs(txt6[0]-txt5[0])>0.1:
                    break
        else:
            while True:
                txt6 = [round(random.uniform(1.1, 1.4), 2)]
                if abs(txt6[0] - txt5[0]) > 0.1:
                    break

        fake_txt = [txt1, txt2, txt3, txt4, txt6]
        if txt5[0]>2:
            fake_txt_tensor = get_trans_txt_dc(self.opt,fake_txt,C_tensor)
        else:
            fake_txt_tensor = get_trans_txt_mc(self.opt, fake_txt, C_tensor)

        input_dict = {'label': A_tensor,'label_txt': txt_tensor,'fake_txt': fake_txt_tensor ,'image': B_tensor,'mask':C_tensor,'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'