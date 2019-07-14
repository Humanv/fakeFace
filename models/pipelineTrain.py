import shutil
import torch
import sys
import os
import pandas as pd
import numpy as np
from itertools import chain
from glob import glob
from tqdm import tqdm
from torch.autograd import Variable
from models.configTrain import configTrain
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image as PIL_Image


def save_checkpoint(state, is_best, fold):
    """
    save model which have better performance on validation data
    ----------------------------------------------------------------------
    param:
        state: information include model_name\epoch\model\precision\loss\optimizer
        is_best: determine whether to save
        fold:
    return:
        None
    -----------------------------------------------------------------------
    """
    filename = configTrain.weights + configTrain.model_name + os.sep +str(fold) + os.sep + "_checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best:
        message = configTrain.best_models + configTrain.model_name+ os.sep +str(fold)  + os.sep + 'model_best.pth.tar'
        print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"], message))
        with open("./logs/%s.txt"%configTrain.model_name,"a") as f:
            print("Get Better top1 : %s saving weights to %s"%(state["best_precision1"], message), file=f)
        shutil.copyfile(filename, message)


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    ----------------------------------------------------------------------
    param:
        output: the predict results
        target: label
        topK: K top predictions
    return:
        accuracy
    -----------------------------------------------------------------------
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_files(root, mode):
    """
    get image's path and label with pandas
    ----------------------------------------------------------------------
    param:
        root: path to data
        mode: mode to category(train, test or validation)
    return:
        image file's path and label saved with pandas
    -----------------------------------------------------------------------
    """
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test":
        # for train and val
        all_data_path, labels = [], []
        image_folders = list(map(lambda x: root+x, os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x: glob(x+"/*"), image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split("\\")[-2]))
        all_files = pd.DataFrame({"filename": all_data_path, "label": labels})
        return all_files
    else:
        print("check the mode please!")


def collate_fn(batch):
    """
    define a function to merges a list of samples to form a mini-batch
    ----------------------------------------------------------------------
    param:
        batch: mini-batch
    return:
        batch in tensor form
    -----------------------------------------------------------------------
    """
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = configTrain.lr * (0.1 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def schedule(current_epoch, current_lrs, **logs):
        lrs = [1e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5]
        epochs = [0, 1, 6, 8, 12]
        for lr, epoch in zip(lrs, epochs):
            if current_epoch >= epoch:
                current_lrs[5] = lr
                if current_epoch >= 2:
                    current_lrs[4] = lr * 1
                    current_lrs[3] = lr * 1
                    current_lrs[2] = lr * 1
                    current_lrs[1] = lr * 1
                    current_lrs[0] = lr * 0.1
        return current_lrs


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
       lr += [param_group['lr']]

    # assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr


def evaluate(val_loader, model, criterion, epoch):
    """
    evaluate accuracy on validation data set
    ----------------------------------------------------------------------
    param:
        val_loader: data load by DataLoader provided by torch
        model: current epoch model
        criterion: loss function
        epoch: current epoch
    return:
        loss and precision
    -----------------------------------------------------------------------
    """
    # define meters
    losses = AverageMeter()
    top1 = AverageMeter()

    # progress bar
    val_progressor = ProgressBar(mode="Val  ", epoch=epoch, total_epoch=configTrain.epochs, model_name=configTrain.model_name, total=len(val_loader))

    # switch to evaluate mode and confirm model has been transfered to cuda
    model.cuda()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            val_progressor.current = i
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            precision1, precision2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(precision1[0], input.size(0))
            val_progressor.current_loss = losses.avg
            val_progressor.current_top1 = top1.avg
            val_progressor()
        val_progressor.done()
    return [losses.avg, top1.avg]


class ProgressBar(object):
    """ progress bar vision class"""
    DEFAULT = "Progress: %(bar)s %(percent)3d%%"

    def __init__(self, mode, epoch=None, total_epoch=None, current_loss=None, current_top1=None, model_name=None,
                 total=None, current=None, width=50, symbol=">", output=sys.stderr):
        assert len(symbol) == 1

        self.mode = mode
        self.total = total
        self.symbol = symbol
        self.output = output
        self.width = width
        self.current = current
        self.epoch = epoch
        self.total_epoch = total_epoch
        self.current_loss = current_loss
        self.current_top1 = current_top1
        self.model_name = model_name

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        bar = "[" + self.symbol * size + " " * (self.width - size) + "]"

        args = {
            "mode": self.mode,
            "total": self.total,
            "bar": bar,
            "current": self.current,
            "percent": percent * 100,
            "current_loss": self.current_loss,
            "current_top1": self.current_top1,
            "epoch": self.epoch + 1,
            "epochs": self.total_epoch
        }
        message = "\033[1;32;40m%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s\033[0m  [Current: Loss %(current_loss)f Top1: %(current_top1)f ]  %(current)d/%(total)d \033[1;32;40m[ %(percent)3d%% ]\033[0m" % args
        self.write_message = "%(mode)s Epoch:  %(epoch)d/%(epochs)d %(bar)s  [Current: Loss %(current_loss)f Top1: %(current_top1)f ]  %(current)d/%(total)d [ %(percent)3d%% ]" % args
        print("\r" + message, file=self.output, end="")

    def done(self):
        self.current = self.total
        self()
        print("", file=self.output)
        with open("./logs/%s.txt" % self.model_name, "a") as f:
            print(self.write_message, file=f)


class FaceDataset(Dataset):
    """
        redefine Dataset from torch.utils.data.Dataset
        two functions have been rewrite: getitem len
        define transforms for different dataset
    """
    def __init__(self, label_list, transforms=None, train=True, test=False):
        self.test = test
        self.train = train
        imgs = []
        if self.test:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs
        else:
            for index, row in label_list.iterrows():
                imgs.append((row["filename"], row["label"]))
            self.imgs = imgs
        if transforms is None:
            if self.test or not self.train:
                self.transforms = T.Compose([
                    T.Resize((configTrain.img_weight, configTrain.img_height)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
                    ])
            else:
                self.transforms = T.Compose([
                    T.Resize((configTrain.img_weight, configTrain.img_height)),
                    T.RandomHorizontalFlip(0.5),
                    # T.RandomVerticalFlip(0.5),
                    # T.RandomAffine(45),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
                    ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        if self.test:
            filename = self.imgs[index]
            # img = cv2.imread(filename)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = PIL_Image.open(filename)
            img = self.transforms(img)
            return img, filename
        else:
            filename, label = self.imgs[index]
            # img = cv2.imread(filename)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = PIL_Image.open(filename)
            img = self.transforms(img)
            return img, label

    def __len__(self):
        return len(self.imgs)