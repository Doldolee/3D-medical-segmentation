import torch
from tqdm import tqdm
from monai.losses.dice import DiceLoss, one_hot
import os
from util import binary_output

class ModelHandler:
    def __init__(self, net, train_loader, val_loader, optimizer, criterion, metric):
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.dataloaders_dict = {'train':train_loader, 'val' : val_loader}
        self.device = 'cuda'
        self.net = net.to(self.device)
        
        torch.backends.cudnn.benchmark = True

    def epoch_train(self, epochs, num_class, model_save_path, model_save = True):
        losses = {'train':[], 'val':[]}
        dice_scores = {'train':[], 'val':[]}
        best_metric, best_epoch = 999, -1
        
        
        for epoch in range(1, epochs+1):
            train_loss, train_dice_score, val_loss, val_dice_score = self.step_train_val(epoch, epochs, num_class)
            
            losses['train'].append(train_loss)
            losses['val'].append(val_loss)
            dice_scores['train'].append(train_dice_score)
            dice_scores['val'].append(val_dice_score)
        
            if losses['val'][-1] < best_metric:
                best_metric = losses['val'][-1]
                best_epoch = epoch
                print(f'Best record! [{epoch}] Test Loss: {val_loss:.6f}, Dice score: {val_dice_score:.6f}')
                if model_save:
                    net_name = f'{best_epoch}_{best_metric}.pth'
                    torch.save(self.net.state_dict(), os.path.join(model_save_path, net_name))
                    print('saved model')
                    
        return losses, dice_scores
    
    def step_train_val(self, epoch, epochs, num_class):
        train_loss, train_dice_score = 0, 0
        val_loss, val_dice_score = 0, 0
        for phase in ['train', 'val']:
            if phase == 'train':
                self.net.train()
            else:
                self.net.eval()

            epoch_loss, epoch_dice_score = 0, 0
            epoch_iterator = tqdm(self.dataloaders_dict[phase], desc="Training (X / X EPOCHS) (loss=X.X) (dice score=%.5f)", dynamic_ncols=True)
        
            for step, batch in enumerate(epoch_iterator):
                img, mask = (batch['img'].to(self.device), batch['mask'].to(self.device))
                mask = torch.squeeze(mask, dim=1)
                mask = one_hot(mask[:, None, ...], num_classes=num_class)
                # print(mask.shape)
                
                self.optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    output = self.net(img)
                    step_loss = self.criterion(output, mask)

                    if phase == 'train':
                        step_loss.backward()
                        self.optimizer.step()
        
                epoch_loss += step_loss.item()
        
                bi_output = binary_output(output, num_class=num_class)
                self.metric(bi_output, mask)
                step_dice_score = self.metric.aggregate().item()
                epoch_dice_score += step_dice_score
        
                epoch_iterator.set_description("Training (%d / %d EPOCHS) (loss=%2.5f) (dice score=%.5f)" % (epoch, epochs, step_loss, step_dice_score))
        
            epoch_loss /= len(epoch_iterator)
            epoch_dice_score /= len(epoch_iterator)

            if phase == 'train':
                train_loss, train_dice_score = epoch_loss, epoch_dice_score
            else:
                val_loss, val_dice_score = epoch_loss, epoch_dice_score
            
            self.metric.reset() # reset the status for next round
        
        return train_loss, train_dice_score, val_loss, val_dice_score


    def inference(self, loader, load_weight = True, weight_path = None):
        pred_dict = {'input':[], 'target':[], 'output':[]}
        
        if load_weight:
            self.net.load_state_dict(torch.load(os.path.join(weight_path)))
        
        self.net.to(self.device)
        self.net.eval()

        with torch.no_grad():
            for i, data in enumerate(loader):
                img, mask = data["img"].cpu(), data["mask"].cpu()
        
                output = self.net(img).detach().cpu()
                output = torch.argmax(output, dim=1)
                
                pred_dict['input'].append(img)
                pred_dict['target'].append(mask)
                pred_dict['output'].append(output)
                
        return pred_dict