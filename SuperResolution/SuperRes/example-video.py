from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import os
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input images path')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)

#data = os.listdir(opt.input_image)
data = cv2.VideoCapture(opt.input_image)

model= torch.load(opt.model)
img_to_tensor = ToTensor()

outvid = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(640, 380))

temp = 0
#for i in tqdm(data):
while(data.isOpened()):
    try:
        #img = Image.open(opt.input_image+"/"+i).convert('YCbCr')
        _, frame = data.read()
        print(temp)
        temp+=1
        img = Image.fromarray(frame)
        y, cb, cr = img.split()

        input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])

        if opt.cuda:
            model = model.cuda()
            input = input.cuda()

        out = model(input)
        out = out.cpu()
        out_img_y = out[0].detach().numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb,
                                    out_img_cr]).convert('RGB')
    except:
        break
outvid.write(np.array(out_img))
data.release()
outvid.release()

    #    out_img.save(f"{opt.output_filename}_{str(i)}_.jpg")

