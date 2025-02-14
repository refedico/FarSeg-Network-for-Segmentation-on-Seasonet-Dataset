import torch
import torch.optim as optim
import torch.nn as nn
import os
import torch.nn.functional as F
import numpy as np
import random
from torchvision import transforms
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from PIL import Image
from sklearn.metrics import jaccard_score, precision_score, recall_score
from utils import calculate_metrics, plot_model_stats, param_flop_counts
import torch.nn.utils.prune as prune

from model import FarSegNetwork

class Solver(object):
    def __init__(self, train_loader, test_loader, val_loader, device, writer, args):
        self.args = args
        self.model_name = f'model_{self.args.model_name}.pth'
        num_classes = 33

        # Define the model
        self.net = FarSegNetwork(num_classes=num_classes, 
                         backbone=self.args.backbone, 
                         pretrained=self.args.pretrained,
                         mobilenets=self.args.mobilenets).to(device)

        # Load a pretrained model if resume flag is set
        if self.args.resume_train:
            self.load_model()

        # Choose loss function
        if self.args.loss == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.args.loss == "DiceLoss":
            self.criterion = smp.losses.DiceLoss(mode='multiclass')
        elif self.args.loss == "FocalLoss":
            self.criterion = smp.losses.FocalLoss(mode='multiclass')
        
        # Choose optimizer
        if self.args.optimizer == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.optimizer == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

        # Show number of params/flops
        if self.args.flop:
            flops, params = param_flop_counts(self.net)
            print(f"FLOPs: {flops.total()}", params)


        self.epochs = self.args.epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.writer = writer


    def save_model(self):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        torch.save(self.net.state_dict(), check_path)
        print("Model saved!")


    def load_model(self):
        check_path = os.path.join(self.args.checkpoint_path, self.model_name)
        self.net.load_state_dict(torch.load(check_path))
        print("Model loaded!")

    def prune_model(self, amount=0.3):
        for name, module in self.net.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')

        self.model_name = "pruned " + self.model_name 
        self.save_model()
        print("Pruning completed!")

    def visualize_and_save_sample(self, index=None, overlay=False):
        save_path = "visualizations"
        os.makedirs(save_path, exist_ok=True)

        if index is None:
           index = random.randint(0, len(self.test_loader.dataset) - 1)

        batch_size = self.test_loader.batch_size
        batch_index = index // batch_size  # Batch index in the DataLoader
        sample_index = index % batch_size  # Sample index within the batch

        for i, batch in enumerate(self.test_loader):
            if i == batch_index:
                sample = {
                "image": batch["image"][sample_index],
                "mask": batch["mask"][sample_index]
            } # as expected by the plot method
                input_image = batch["image"][sample_index].unsqueeze(0).to(self.device)
                break

        file_path = os.path.join(save_path, f"sample_{index}.png")

        if overlay:
            self.net.eval()
            with torch.no_grad():
                output = self.net(input_image)
                pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
        
            sample["prediction"] = pred_mask
        
        fig = self.test_loader.dataset.dataset.plot(sample, show_titles=True, s$
        fig.savefig(file_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Sample saved at: {file_path}")



    
    def train(self):
        self.net.train()

        epoch_losses = []
        train_iou_scores = []
        train_accuracy = []
        val_losses = []

        # Early Styopping
        best_val_loss = float('inf')
        patience_counter = 0

        print("Start of training...")
        for epoch in range(self.epochs):
            running_loss = 0
            iou_scores = []
            accuracy_scores = []
            for i, batch in enumerate(self.train_loader, 0):
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(images)
                loss = self.criterion(outputs, masks)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy() # preds.shape = (32, 33, 120, 120) -> (32, 120, 120)
                targets = masks.cpu().numpy() # targets.shape = (32, 120, 120)
        
                for pred, target in zip(preds, targets): # loop to calculate metrics for each batch
                    iou, _, _, accuracy = calculate_metrics(pred, target)
                    iou_scores.append(iou)
                    accuracy_scores.append(accuracy)

            # Lists were used for convenience with matplotlib
            epoch_losses.append(running_loss / len(self.train_loader))
            train_iou_scores.append(np.mean(iou_scores))
            train_accuracy.append(np.mean(accuracy_scores))

            if self.args.validation: # validation mode if flag is set
                val_loss = self.validate()
                val_losses.append(val_loss)

                self.writer.add_scalars('Training/Metrics', { # Adding validation loss to tensorboard
                'Loss': epoch_losses[-1],
                'IoU': train_iou_scores[-1],
                'Accuracy': train_accuracy[-1],
                'Val Loss': val_losses[-1]}, epoch)

                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

                # Early stopping basic implementation
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model() # save only if it's the best
                else:
                    patience_counter += 1

                if patience_counter >= self.args.patience:
                    print("Interrupted by Early stopping call.")
                    break
            else: # no validation mode, normal flow
                self.writer.add_scalars('Training/Metrics', {
                'Loss': epoch_losses[-1],
                'IoU': train_iou_scores[-1],
                'Accuracy': train_accuracy[-1]}, epoch)
                print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {epoch_losses[-1]:.4f}")

        if not self.args.validation:
            self.save_model()

        #plot_model_stats(self.epochs, epoch_losses, train_iou_scores, train_accuracy) old method with matplotlib
        self.writer.flush()
        self.writer.close()
        print('Finished Training')   
    

    def validate(self):
        self.net.eval()
        running_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)

                outputs = self.net(images)
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()

                # To calculate the metrics here too, does it make sense? I already did it in the test set
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()

        val_loss = running_loss / len(self.val_loader)
        return val_loss
    
    def test(self): 
        self.net.eval()
        iou_scores, dice_scores, precision_scores, recall_scores, accuracy_scores = [], [], [], [], []

        with torch.no_grad():
            for batch in self.test_loader:
                images, masks = batch['image'].to(self.device), batch['mask'].to(self.device)
                outputs = self.net(images)

                dice_loss = smp.losses.DiceLoss(mode='multiclass')
                dice = 1 - dice_loss(outputs, masks)  # I use dice loss to calculate the coefficient -> dice_loss = 1 - dice
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                targets = masks.cpu().numpy()
                
                # Calculate metrics for each batch
                for pred, target in zip(preds, targets):
                    iou, precision, recall, accuracy = calculate_metrics(pred, target)
                    iou_scores.append(iou)
                    dice_scores.append(dice)
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    accuracy_scores.append(accuracy)

        print(f"Mean IoU: {np.mean(iou_scores):.4f}")
        print(f"Mean Dice Coefficient: {(sum(dice_scores) / len(dice_scores)):.4f}")
        print(f"Mean Precision: {np.mean(precision_scores):.4f}")
        print(f"Mean Recall: {np.mean(recall_scores):.4f}")
        print(f"Pixel Accuracy: {np.mean(accuracy_scores):.4f}")

        self.writer.add_scalar("IoU/Test", np.mean(iou_scores))
        self.writer.add_scalar("Dice/Test", (sum(dice_scores) / len(dice_scores)))
        self.writer.add_scalar("Precision/Test", np.mean(precision_scores))
        self.writer.add_scalar("Recall/Test", np.mean(recall_scores))
        self.writer.add_scalar("Accuracy/Test", np.mean(accuracy_scores))

        self.net.train()


