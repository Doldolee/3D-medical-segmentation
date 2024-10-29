import torch
import torch.nn as nn
import monai

from data import get_dataset
from net import UNETR, VNet
from handler import ModelHandler
import argparse

from util import plot_loss_dice_score, visualize_test


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    data_type, net_type  = args.data_type, args.net_type
    parallel, image_size = args.parallel, args.image_size
    epoch, model_save_path = args.epoch, args.model_save_path
    batch_size, num_class = args.batch_size, args.num_class
    # data_type = decathron_spleen
    train_loader, val_loader = get_dataset(data_type, image_size, batch_size)

    if net_type == 'unetr':
        net = UNETR(img_shape=image_size, input_dim=1, output_dim=2, 
                embed_dim=768, patch_size=16, num_heads=8, dropout=0., light_r=4)
    
    elif net_type == 'vnet':
        net = VNet(in_ch=1, num_class=num_class)

    
    if parallel == True:
        net = nn.DataParallel(net)


    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = monai.losses.DiceLoss(include_background=False, to_onehot_y=False, softmax=True)
    metric_dice = monai.metrics.DiceMetric(include_background=False, reduction='mean')

    model_handler = ModelHandler(net, train_loader, val_loader, optimizer, criterion, metric_dice)
    losses, dice_scores = model_handler.epoch_train(epoch, num_class, model_save_path, model_save = True)

    # recommend using jupyter notebook
    plot_loss_dice_score(losses, dice_scores)
    weight_path = model_save_path + "/92_0.41817347208658856.pth"
    pred_dict = model_handler.inference(val_loader, load_weight = False, weight_path = weight_path)
    visualize_test(pred_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script Arguments")
    parser.add_argument('--data_type', type=str, default='decathron_spleen', help='Type of dataset')
    parser.add_argument('--net_type', type=str, default='unetr', choices=['vnet', 'unetr'], help='Type of network')
    parser.add_argument('--parallel', type=bool, default=True, help='Use parallel training or not')
    parser.add_argument('--image_size', type=tuple, default=(128, 128, 128), help='Size of input images')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--model_save_path', type=str, default='./pth', help='Path to save the model')
    parser.add_argument('--num_class', type=int, default=2, help='num class')

    args = parser.parse_args()

    main(args)