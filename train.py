import argparse
import os
import random
import time
import utils
from dataset.CD_dataset import CDDataset
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
from models.ConvFormer import ConvFormer
import losses
from metric_tool import ConfuseMatrixMeter
from utils import str2bool
import warnings

warnings.filterwarnings("ignore")
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

# torch.cuda.set_device(0)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="Test",
                        help='project_name')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

    # model
    parser.add_argument('--dim', metavar='dim', default=48, type=int)
    parser.add_argument('--num_classes', default=2, type=int,  help='number of classes')
    parser.add_argument('--image_size', default=224, type=int, help='image size')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)

    # dataset
    parser.add_argument('--dataset', default='LEVIR', help='dataset name')
    parser.add_argument('--img_dir', default="/data/lmt/Dataset/LEVIR_Dataset/LEVIR_224_overlap",
                        help='dataset name')
    # optimizer
    parser.add_argument('--optimizer', default='AdamW', choices=['Adam','AdamW', 'SGD'], help='')
    parser.add_argument('--lr', default=0.0005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,  help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=0.0001, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    config = parser.parse_args()

    return config


# args = parser.parse_args()
def train(config, running_metric, train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_metric.clear()
    train_loss_list = []
    train_acc_list = []
    x = random.randint(0, len(train_loader))
    pbar = tqdm(train_loader, desc=f"Train epoch:[{epoch}/{config['epochs']}]")
    for i_batch, sampled_batch in enumerate(pbar):
        image_batch1, image_batch2, label_batch = sampled_batch['A'], sampled_batch['B'], sampled_batch["L"]
        image_batch1, image_batch2, label_batch = image_batch1.cuda(), image_batch2.cuda(), label_batch.squeeze().cuda()
        outputs = model(image_batch1, image_batch2)
        loss = criterion(outputs, label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_acc = running_metric.update_cm(pr=torch.argmax(outputs, dim=1).cpu().numpy(),
                                               gt=label_batch.cpu().numpy())
        train_loss_list.append(loss)
        train_acc_list.append(running_acc)
        if i_batch == x and epoch % 10 == 0:
            utils.draw(sampled_batch, outputs, f"./vis/{config['name']}", epoch, x, 'train')
    train_scores = running_metric.get_scores()
    train_scores['loss'] = sum(train_loss_list) / len(train_loss_list)
    pbar.close()
    return train_scores


def validate(config, val_loader, running_metric, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    val_loss_list = []
    val_acc_list = []
    running_metric.clear()
    x = random.randint(0, len(val_loader))
    # x = 1
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Val epoch:[{epoch}/{config['epochs']}]")
        for i_batch, sampled_batch in enumerate(pbar):
            image_batch1, image_batch2, label_batch = sampled_batch['A'], sampled_batch['B'], sampled_batch["L"]
            image_batch1, image_batch2, label_batch = image_batch1.cuda(), image_batch2.cuda(), label_batch.squeeze().cuda()
            outputs = model(image_batch1, image_batch2)
            loss = criterion(outputs, label_batch)
            val_loss_list.append(loss.item())
            running_acc = running_metric.update_cm(pr=torch.argmax(outputs, dim=1).cpu().numpy(),
                                                   gt=label_batch.cpu().numpy())
            val_acc_list.append(running_acc)
            if i_batch == x and epoch % 10 == 0:
                utils.draw(sampled_batch, outputs, f"./vis/{config['name']}", epoch, x, 'val')
        train_scores = running_metric.get_scores()
        train_scores['loss'] = sum(val_loss_list) / len(val_loss_list)
        pbar.close()

    return train_scores


def save_model(epoch, best_f1_1, best_epoch_id, model, optimizer, scheduler, save_dir, model_name):
    torch.save({
        'epoch_id': epoch,
        'best_val_F1': best_f1_1,
        'best_epoch_id': best_epoch_id,
        'model_G_state_dict': model.state_dict(),
        'optimizer_G_state_dict': optimizer.state_dict(),
        'exp_lr_scheduler_G_state_dict': scheduler.state_dict(),
    }, os.path.join(save_dir, model_name))
    print(f"************ saved {model_name}*************")


def main():
    config = vars(parse_args())
    # create model
    model = ConvFormer(embed_dim=config['dim'])
    time_str = time.strftime("%m-%d-%H-%M")
    # config['name'] = "Test"
    config['name'] = f"{type(model).__name__}_{config['dim']}_{config['dataset']}_{time_str}"
    save_dir = f'outputs/{config["name"]}'
    os.makedirs(save_dir, exist_ok=True)
    print('-' * 20)
    with open(f'outputs/{config["name"]}/parameters.txt', 'w', encoding='utf-8') as f:
        for key in config:
            print('%s: %s' % (key, config[key]))
            f.write('%s: %s\n' % (key, config[key]))
    print('-' * 20)

    if config['loss'] == 'BCELoss':
        criterion = losses.cross_entropy
    elif config['loss'] == 'WBCEDiceLoss':
        criterion = losses.WBCEDiceLoss()
    else:
        criterion = losses.BCEDiceLoss()

    cudnn.benchmark = True

    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model = model.cuda()
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        pass
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        pass
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    train_dataset = CDDataset(root_dir=config['img_dir'], split='train', img_size=config['image_size'],
                              is_train=True, label_transform='norm')
    val_dataset = CDDataset(root_dir=config['img_dir'], split='val', img_size=config['image_size'],
                            is_train=False, label_transform='norm')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"])
    val_best_f1_1 = 0
    best_epoch_id = 0
    writer = SummaryWriter(f"./runs/{config['name']}")
    running_metric = ConfuseMatrixMeter(n_class=config['num_classes'])
    for epoch in range(1, config['epochs'] + 1):
        train_scores = train(config, running_metric, train_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('train/train_loss', train_scores["loss"], epoch)
        writer.add_scalar('train/train_mf1', train_scores["mf1"], epoch)
        writer.add_scalar('train/train_miou', train_scores["miou"], epoch)
        writer.add_scalar('train/train_F1_1', train_scores["F1_1"], epoch)
        writer.add_scalar('train/train_acc', train_scores["acc"], epoch)
        # evaluate on validation set
        val_scores = validate(config, val_loader, running_metric, model, criterion, epoch)
        writer.add_scalar('val/val_loss', val_scores["loss"], epoch)
        writer.add_scalar('val/val_mf1', val_scores["mf1"], epoch)
        writer.add_scalar('val/val_miou', val_scores["miou"], epoch)
        writer.add_scalar('val/val_F1_1', val_scores["F1_1"], epoch)
        writer.add_scalar('val/val_acc', val_scores["acc"], epoch)
        print(f'train_score:F1:{train_scores["F1_1"]},val_score:F1:{val_scores["F1_1"]}')
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        writer.add_scalar('train/train_lr', lr, epoch)
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        if val_scores["F1_1"] > val_best_f1_1:
            val_best_f1_1 = val_scores["F1_1"]
            best_epoch_id = epoch
            with open(f'outputs/{config["name"]}/val_scores.txt', 'w', encoding='utf-8') as f:
                for key in val_scores:
                    f.write('%s: %s\n' % (key, val_scores[key]))
            save_model(epoch, val_best_f1_1, best_epoch_id, model, optimizer, scheduler, save_dir, "last_model2.pth")
        save_model(epoch, val_best_f1_1, best_epoch_id, model, optimizer, scheduler, save_dir, "best_model.pth")
        print(f"********** the best val F1 at peoch{best_epoch_id}, "
              f"best val F1:{round(val_best_f1_1, 4)}***********")
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
