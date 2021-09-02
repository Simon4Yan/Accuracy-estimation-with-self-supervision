import os
import random

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm import trange

# ===================================================== #
# -----------     load original dataset     ----------- #
# ===================================================== #
'''An example to load original datasets (based on Pytorch's dataloader)'''
NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])

teset = torchvision.datasets.CIFAR10(root='./dataset/',
                                     train=False, download=True, transform=te_transforms)
teset_raw = teset.data
print('Loaded original set')

# ===================================================== #
# -----------     Image Transformations     ----------- #
# ===================================================== #
'''
In our paper, the transformations are: 
{Sharpness, translateX/Y, Color, Autocontrast, Brightness, Rotation}.
The users can customize the transformation list based on the their own data.
The users can use more transformations for the selection.
We refer the readers to https://imgaug.readthedocs.io/ for more details of transformations.
'''
list = [
    iaa.pillike.EnhanceColor(),  # remove a random fraction of color from input images
    iaa.Sharpen(alpha=(0.0, 1.0)),  # apply a sharpening filter kernel to images
    iaa.ChangeColorTemperature((1100, 10000 // 2)),  # change the temperature to a provided Kelvin value.
    iaa.pillike.Equalize(),  # equalize the image histogram
    iaa.Solarize(0.5, threshold=(32, 128)),  # invert the colors
    iaa.Multiply((0.8, 1.2), per_channel=0.2),  # make some images brighter and some darker
    iaa.pillike.Autocontrast(),  # adjust contrast by cutting off p% of lowest/highest histogram values
    iaa.Grayscale(alpha=(0.0, 0.5)),  # remove colors with varying strength
]

# example 2
# list = [
#     iaa.Grayscale(alpha=(0.0, 0.5)),  # remove colors with varying strengths
#     iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=(0.25,0.5)),  # move pixels locally around with random strengths
#     iaa.PiecewiseAffine(scale=(0.01, 0.05)),  # distort local areas with varying strength
#     iaa.Invert(0.05, per_channel=True),  # invert color channels
#     iaa.pillike.FilterBlur(),  # apply a blur filter kernel to images
#     iaa.pillike.EnhanceBrightness(),  # change the brightness of images
#     iaa.Fog(),  # add fog to images
#     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.025*255), per_channel=0.5)  # Add gaussian noise to some images
# ]

# ===================================================== #
# -----------  Generate Synthetic datasets  ----------- #
# ===================================================== #
'''
Generate 800 and 200 synthetic datasets for training and validation, respectively
'''
tesize = 10000
num_sets = 500
# Change to your path
try:
    os.makedirs('/mnt/home/dwj/AutoEval/CIFAR-10_Setup/dataset_GroupC')
except:
    print('Alread has this path')

for num in trange(num_sets):
    num_sel = random.randint(1, 4)  # use more transformation to introduce dataset diversity
    list_sel = random.sample(list, int(num_sel))
    random.shuffle(list_sel)
    seq = iaa.Sequential(list_sel)

    new_data = np.zeros(teset_raw.shape).astype(np.uint8)
    for i in range(tesize):
        data = teset_raw[i]
        ia.seed(i + num * tesize)  # add random for each dataset
        new_data[i] = seq(image=data)

    np.save('/mnt/home/dwj/AutoEval/CIFAR-10_Setup/dataset_GroupC/new_data_' + str(num).zfill(3), new_data)

print('Finished, thanks!')

# ===================================================== #
# -----------  Load Synthetic datasets  ----------- #
# ===================================================== #
# An example to load synthetic datasets (based on Pytorch's dataloader)
'''
for i in range(500):
    teset_raw = np.load('/mnt/home/dwj/AutoEval/CIFAR-10_Setup/dataset_GroupA/new_data_' + str(i).zfill(3)+ '.npy') # your path
    teset = torchvision.datasets.CIFAR10(root=YourPATH,
                                         train=False, download=True, transform=te_transforms)
    teset.data = teset_raw
    teloader = torch.utils.data.DataLoader(teset, batch_size=64,
                                           shuffle=False, num_workers=2)
'''
