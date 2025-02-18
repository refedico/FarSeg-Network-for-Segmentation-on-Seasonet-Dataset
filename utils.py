import os
import torch
import torch.nn as nn
import random
from torch.utils.data import Subset, ConcatDataset
from sklearn.metrics import jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torchvision import transforms

def calculate_metrics(pred, target, num_classes=33):
    iou = jaccard_score(target.reshape(-1), pred.reshape(-1), average='weighted', labels=range(num_classes)) # reshape(-1) = view(-1) because we need a 1D tensor for sklearn
    precision = precision_score(target.reshape(-1), pred.reshape(-1), average='weighted', labels=range(num_classes), zero_division=0)
    recall = recall_score(target.reshape(-1), pred.reshape(-1), average='weighted', labels=range(num_classes), zero_division=0)
    accuracy = (pred == target).sum() / (pred == target).size 
    return iou, precision, recall, accuracy

def load_or_generate_indices(file_path, dataset_size, subset_size):
    if os.path.exists(file_path):
        print(f"Loading indexes from {file_path}...")
        with open(file_path, "r") as f:
            return [int(line.strip()) for line in f]
    else:
        print(f"Generte new random indexes in {file_path}...")
        indices = random.sample(range(dataset_size), subset_size)
        with open(file_path, "w") as f:
            f.writelines(f"{idx}\n" for idx in indices)
        return indices

def create_subsets(train_set, test_set, dataset_path, season, train_percentage, test_percentage, mixed_test_set, sum_test_set = None, spr_test_set = None, fall_test_set = None):
    if season not in {"Summer", "Spring", "Fall"}:
        print("Error: Invalid season")
        return None, None
    
    # Standard creation of names for these configuration files
    subset_indices_file = os.path.join(dataset_path, season + "_subset_indices_"+ str(int(train_percentage*100)) + "per.txt")
    train_size = len(train_set)
    subset_size_train = max(1, int(train_percentage * train_size))
    train_indices = load_or_generate_indices(subset_indices_file, train_size, subset_size_train)

    # Normal flow with train and test of the same season
    if not mixed_test_set:
        test_indices_file = os.path.join(dataset_path, season + "_test_indices_" + str(int(test_percentage*100)) + "per.txt") 
        test_size = len(test_set)
        subset_size_test = max(1, int(test_percentage * test_size))
        test_indices = load_or_generate_indices(test_indices_file, test_size, subset_size_test)
        return Subset(train_set, train_indices), Subset(test_set, test_indices)
    else:
        # Load the test sets for each season
        N = 500 # Number of sample for each season
        fall_test_indices_file = os.path.join(dataset_path, "/seasons/mixed_test_set/fall_test_indices.txt")
        sum_test_indices_file = os.path.join(dataset_path, "/seasons/mixed_test_set/sum_test_indices.txt")
        spr_test_indices_file = os.path.join(dataset_path, "/seasons/mixed_test_set/spring_test_indices.txt")

        fall_test_indices  = load_or_generate_indices(fall_test_indices_file, len(fall_test_set), N)
        sum_test_indices = load_or_generate_indices(sum_test_indices_file, len(sum_test_set), N)
        spr_test_indices = load_or_generate_indices(spr_test_indices_file, len(spr_test_set), N)

        fall_test_subset = Subset(fall_test_set, fall_test_indices)
        sum_test_subset = Subset(sum_test_set, sum_test_indices)
        spr_test_subset = Subset(spr_test_set, spr_test_indices)

        combined_test_subset = ConcatDataset([fall_test_subset, sum_test_subset, spr_test_subset])

        return Subset(train_set, train_indices), combined_test_subset

def param_flop_counts(model, input = torch.randn(1, 12, 224, 224)):
    flops = FlopCountAnalysis(model ,input)
    params = parameter_count_table(model)

    return flops, params


def plot_model_stats(epochs, epoch_losses, train_iou_scores, train_accuracy):
    # Old function with matplotlib because I did everything on hpc
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses, label="Loss", marker="o")
    plt.plot(range(1, epochs+1), train_iou_scores, label="IoU", marker="o")
    plt.plot(range(1, epochs+1), train_accuracy, label="Accuracy", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("/hpc/scratch/federico.putamorsi/deeplearning_project/plot1.png")
    plt.close()

class DepthwiseSeparableConv(nn.Module):
    # Definition of the layer to make separable convolutions for mobilenet
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        # Each input channel is treated separately with its own convolution with groups=in_channel
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
