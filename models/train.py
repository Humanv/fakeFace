import os
import random
import time
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from torch import nn, optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from timeit import default_timer as timer
from IPython import embed
from pipelineTrain import *
from MosNet import *
from Xception import Xception
from configTrain import configTrain

# set cudnn
# os.environ["CUDA_VISIBLE_DEVICES"] = configTrain.gpus
# torch.backends.cudnn.benchmark = True
# warnings.filterwarnings('ignore')
random.seed(configTrain.seed)
np.random.seed(configTrain.seed)
torch.manual_seed(configTrain.seed)
torch.cuda.manual_seed_all(configTrain.seed)


def main():
    fold = 0
    # mkdirs
    if not os.path.exists(configTrain.weights):
        os.mkdir(configTrain.weights)
    if not os.path.exists(configTrain.best_models):
        os.mkdir(configTrain.best_models)
    if not os.path.exists(configTrain.logs):
        os.mkdir(configTrain.logs)
    if not os.path.exists(configTrain.weights + configTrain.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(configTrain.weights + configTrain.model_name + os.sep + str(fold) + os.sep)
    if not os.path.exists(configTrain.best_models + configTrain.model_name + os.sep + str(fold) + os.sep):
        os.makedirs(configTrain.best_models + configTrain.model_name + os.sep + str(fold) + os.sep)
    # get model and optimizer
    # model = MosNet()
    model = Xception()
    # model = torch.nn.DataParallel(model)
    model.cuda()

    # set optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=configTrain.lr, amsgrad=False, weight_decay=configTrain.weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()

    # some parameters
    start_epoch = 0
    best_precision1 = 0
    best_precision_save = 0
    resume = False

    # restart the training process
    if resume:
        checkpoint = torch.load(configTrain.best_models + str(fold) + "/model_best.pth.tar")
        start_epoch = checkpoint["epoch"]
        fold = checkpoint["fold"]
        best_precision1 = checkpoint["best_precision1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # read image files
    train_data_list = get_files(configTrain.train_data, "train")
    val_data_list = get_files(configTrain.val_data, "val")

    # load train_data and validation_data with DataLoader
    train_dataloader = DataLoader(FaceDataset(train_data_list), batch_size=configTrain.batch_size, shuffle=True,
                                  pin_memory=True)
    val_dataloader = DataLoader(FaceDataset(val_data_list, train=False), batch_size=configTrain.batch_size * 2,
                                shuffle=True, pin_memory=False)

    # Decays the learning rate by gamma every step_size epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=configTrain.stepLR_size, gamma=0.1)

    # define metrics for loss and accuracy
    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    valid_loss = [np.inf, 0, 0]

    # set the module in training mode
    model.train()

    # train
    for epoch in range(start_epoch, configTrain.epochs):
        scheduler.step(epoch)
        train_progressor = ProgressBar(mode="Train", epoch=epoch, total_epoch=configTrain.epochs,
                                       model_name=configTrain.model_name, total=len(train_dataloader))
        # global iter
        for iter, (input, target) in enumerate(train_dataloader):
            # switch to continue train process
            train_progressor.current = iter

            # get loss and accuracy
            model.train()
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(input)
            loss = criterion(output, target)

            precision1_train, precision2_train = accuracy(output, target, topk=(1, 2))

            # update batch's loss and accuracy to total
            train_losses.update(loss.item(), input.size(0))
            train_top1.update(precision1_train[0], input.size(0))

            # set loss and accuracy to progress bar
            train_progressor.current_loss = train_losses.avg
            train_progressor.current_top1 = train_top1.avg
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # progress bar vision
            train_progressor()
        train_progressor.done()

        # evaluate every epoch
        valid_loss_accuracy = evaluate(val_dataloader, model, criterion, epoch)

        # determine whether to save model
        is_best = valid_loss_accuracy[1] > best_precision1

        # save the best precision
        best_precision1 = max(valid_loss_accuracy[1], best_precision1)
        try:
            best_precision_save = best_precision1.cpu().data.numpy()
        except:
            pass
        save_checkpoint({
            "epoch": epoch + 1,
            "model_name": configTrain.model_name,
            "state_dict": model.state_dict(),
            "best_precision1": best_precision1,
            "optimizer": optimizer.state_dict(),
            "fold": fold,
            "valid_loss": valid_loss_accuracy,
        }, is_best, fold)


if __name__ =="__main__":
    main()


