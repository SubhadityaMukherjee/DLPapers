import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


# check if image

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png",".jpg",
                                                              ".jpeg"])

# open a single image and convert it into YCbCr
"""
(The difference between YCbCr and RGB is that RGB represents colors as
 combinations of red, green and blue signals, while YCbCr represents colors as
combinations of a brightness signal and two chroma signals.)
"""

def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

# Custom dataloader to get data from folder

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform = None, target_transform =
                 None):
        super(DatasetFromFolder, self).__init__()

        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir)
                               if is_image_file(x)]
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = input.copy()
        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        
        return input, target
    
    def __len__(self):
        return len(self.image_filenames)

# Get valid crop size
def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)

# Input batch transforms
def input_transform(crop_size, upscale_factor):
    return Compose(
        [
            CenterCrop(crop_size),
            Resize(crop_size//upscale_factor),
            ToTensor(),
        ]
    )

# Output batch transforms
def target_transform(crop_size):
    return Compose(
        [
            CenterCrop(crop_size),
            ToTensor(),
        ]
    )

def get_training_set(args, upscale_factor):
    root_dir = args.data_path
    train_dir = join(root_dir, "train")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    return DatasetFromFolder(train_dir, input_transform =
                             input_transform(crop_size, upscale_factor),
                             target_transform = target_transform(crop_size))

def get_test_set(args, upscale_factor):
    root_dir = args.data_path
    test_dir = join(root_dir, "test")
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    return DatasetFromFolder(test_dir, input_transform =
                             input_transform(crop_size, upscale_factor),
                             target_transform = target_transform(crop_size))



