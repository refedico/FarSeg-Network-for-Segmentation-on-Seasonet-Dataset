import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import SeasoNet
from torch.utils.data import DataLoader, Subset
import os
import random
from torchvision import transforms
from utils import create_subsets
from solver import Solver
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def transform_sample(sample):
    transform = transforms.Compose([transforms.Resize((224, 224))])
    sample["image"] = transform(sample["image"])
    return sample

dataset_path = "/hpc/scratch/federico.putamorsi/deeplearning_project/seasons"
season = "Spring"

test_set = SeasoNet(
    root=dataset_path,
    split="test",
    transforms=transform_sample,
    seasons=[season],
    grids=[1],
    download=False
)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=8)

model_path = "D:\Desktop\DLAGM_project\model_weights.pth"
model = torch.load(model_path)
model.eval()

# Random sample
random_index = random.randint(0, len(test_set) - 1)
sample = test_set[random_index]
transformed_sample = transform_sample(sample)

with torch.no_grad():
    input_tensor = transformed_sample["image"].unsqueeze(0) # Add cause of the batch size
    predictions = model(input_tensor)

predictions_np = predictions.squeeze(0).numpy()

# Convert predictions to PIL Image
predictions_image = Image.fromarray((predictions_np[0] * 255).astype(np.uint8))

# Convert input image to PIL Image
input_image = transforms.ToPILImage()(transformed_sample["image"])

# Overlay predictions on input image
overlay = Image.blend(input_image.convert("RGBA"), predictions_image.convert("RGBA"), alpha=0.5)

# Display images using PIL
input_image.show(title='Input Image')
predictions_image.show(title='Prediction Image')
overlay.show(title='Overlay')