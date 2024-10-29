from monai.transforms.compose import Transform, MapTransform
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt


class MinMax(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] -= np.min(d[key])
            d[key] /= np.max(d[key])
        return d
    
def binary_output(output, num_class, keep_dim=True):
    shape = output.shape
    argmax_idx = torch.argmax(output, axis=1, keepdim=True)
    argmax_oh = F.one_hot(argmax_idx, num_classes=num_class)
    if keep_dim:
        argmax_oh = torch.squeeze(argmax_oh, dim=1)
    if len(shape) == 5:
        argmax_oh = argmax_oh.permute(0,4,1,2,3)
    elif len(shape) == 4:
        argmax_oh = argmax_oh.permute(0,3,1,2)
    return argmax_oh


def visualize_test(pred_dict, image_size):
    ncols, nrows = 10, 3*3
    interval = int(image_size[-1]//(ncols*nrows/3))
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols,nrows))
    cnt1, cnt2, cnt3 = 0, 0, 0
    for i in range(nrows):
        for j in range(ncols):
            if i%3 == 0:
                axes[i,j].imshow(pred_dict['input'][0][0,0,:,:,cnt1])
                cnt1+=interval
            elif i%3 == 1:
                axes[i,j].imshow(pred_dict['target'][0][0,0,:,:,cnt2])
                cnt2+=interval
            else:
                axes[i,j].imshow(pred_dict['output'][0][0,:,:,cnt3])
                cnt3+=interval
            axes[i,j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig("./plot/visual.png")
    plt.show()  

def plot_loss_dice_score(losses, dice_scores):
    epochs = [i for i in range(len(losses['train']))]
    train_loss = losses['train']
    val_loss = losses['val']
    train_dice = dice_scores['train']
    val_dice = dice_scores['val']
    
    fig , ax = plt.subplots(1,2)
    fig.set_size_inches(12,6)
    
    ax[0].plot(epochs , train_loss , 'b-*' , label = 'Training Loss')
    ax[0].plot(epochs , val_loss , 'g-*' , label = 'Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    
    ax[1].plot(epochs , train_dice , 'go-' , label = 'Training Dice score')
    ax[1].plot(epochs , val_dice , 'ro-' , label = 'Validation Dice score')
    ax[1].set_title('Training & Validation Dice score')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Dice score")
    plt.savefig("./plot/loss.png")
    plt.show()