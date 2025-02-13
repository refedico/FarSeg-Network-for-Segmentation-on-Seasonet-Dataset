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

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_name', type=str, default="run_1", help='name of current run')
    parser.add_argument('--model_name', type=str, default="first_train", help='name of the model to be saved/loaded')

    parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--validation', type=bool, default=False, help='True if you want validation, False otherwise') # New
    parser.add_argument('--pretrained', type=bool, default=True, help='True if you want pretrained weights, False otherwise') # New
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss", choices=['CrossEntropyLoss', 'DiceLoss', "DiceLoss"],  help='type of loss')
    parser.add_argument('--batch_size', type=int, default=32, help='number of elements in batch size')
    parser.add_argument('--patience', type=int, default=5, help='number of epochs to wait before early stopping') # New
    parser.add_argument('--resolution', type=int, default=224, help='resolution of input image in pixel(for example 320, 224, 96, 32)') # New
    parser.add_argument('--workers', type=int, default=4, help='number of workers in data loader')
    parser.add_argument('--augumentation', type=bool, default=False, help='True for augument, False otherwhise') # New

    parser.add_argument('--backbone', type=str, default="resnet18", help="backbone used")

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'], help='optimizer used for training')

    parser.add_argument('--dataset_path', type=str, default= "/hpc/scratch/federico.putamorsi/deeplearning_project/seasons", help='path where the dataset is located')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='path where to save the trained model')
    parser.add_argument('--pruning', type=float, default=0, help='percentage of weights to be reset, False otherwhise') # New

    parser.add_argument('--resume_train', action='store_true', help='load the model from checkpoint before training')
    parser.add_argument('--train_percentage', type=float, default=0.001, help='percentage of the train dataset to use for training') # New
    parser.add_argument('--test_percentage', type=float, default=0.0001, help='percentage of the test dataset to use for testing') # New
    parser.add_argument('--season', type=str, default='Summer', help='season to use from the SeasoNet dataset') # New
    parser.add_argument('--flop', type=bool, default=False, help='True to display Multi Adds and parameters number') # New
    parser.add_argument('--mobilenets', type=bool, default=False, help='True to use the mobilenets architecture, False otherwise') # New
    parser.add_argument('--mixed_testset', type=bool, default=False, help='To choose mixed test sets on three seasons') # New

    return parser.parse_args()

def main(args):
    writer = SummaryWriter('./runs/' + args.run_name)

    def transform_sample(sample):
       if args.augumentation:
           transform = transforms.Compose([
                    transforms.Resize((args.resolution, args.resolution)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=30),
                ])
       else:
           transform = transforms.Compose([transforms.Resize((args.resolution, args.resolution))])

       sample["image"] = transform(sample["image"])
       return sample


    # Load SeasoNet dataset: train, test and val 
    train_set = SeasoNet(
        root=args.dataset_path,
        split="train",
        transforms=transform_sample,
        seasons=[args.season],
        grids=[1],
        download=False
    )

    test_set = SeasoNet(
        root=args.dataset_path,
        split="test",
        transforms=transform_sample,
        seasons=[args.season],
        grids=[1],
        download=False
    )
    
    val_set = SeasoNet(
        root=args.dataset_path,
        split="val",
        transforms=transform_sample,
        seasons=[args.season],
        grids=[1],
        download=False
    )
    
    # Load all test sets if you want testing for task2 with mixed_test
    if args.mixed_testset:
        sum_test_set = SeasoNet(root=args.dataset_path, split="test", transforms=transform_sample, seasons=["Summer"], grids=[1], download=False)
        spr_test_set = SeasoNet(root=args.dataset_path, split="test", transforms=transform_sample, seasons=["Spring"], grids=[1], download=False)
        fall_test_set = SeasoNet(root=args.dataset_path, split="test", transforms=transform_sample, seasons=["Fall"], grids=[1], download=False)

        # Create subsets extracting indexes from the original dataset
        # the indices were implemented to ensure consistency between experiments, so as not to let the random division of runs impact the results
        train_subset, test_subset = create_subsets(train_set, test_set,args.dataset_path, args.season,args.train_percentage, 
                                                args.test_percentage, mixed_test_set=args.mixed_testset, 
                                                sum_test_set=sum_test_set, spr_test_set=spr_test_set, fall_test_set=fall_test_set)
    else:
        train_subset, test_subset = create_subsets(train_set, test_set,args.dataset_path, args.season,args.train_percentage, 
                                                args.test_percentage, mixed_test_set=args.mixed_testset)


    # DataLoaders creation
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    print(f"Dimension of training set: {len(train_loader.dataset)} sample")
    print(f"Dimension of test set: {len(test_loader.dataset)} sample")
    print(f"Dimension of validation set: {len(val_loader.dataset)} sample")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Solver initialization
    solver = Solver(train_loader=train_loader,
                    test_loader=test_loader,
                    val_loader=val_loader,
                    device=device,
                    writer=writer,
                    args=args)

    # Visualization of a (random) sample
    solver.visualize_sample()

    # Train and test model
    solver.train()
    solver.test()


    # Prune model
    if args.pruning > 0:
        solver.pruning_model(args.pruning)

if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
