# %load main.py
import os 
import time 
import json 
import argparse
import torch 
import torchvision
import random
import numpy as np 
from data import FaceDataset
from tqdm import tqdm 
from torch import nn
from torch import optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import ResNet
from mean_variance_loss import MeanVarianceLoss
from mean_variance_loss import MeanVarianceLoss2
import cv2
#from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter

#LAMBDA_1 = 0.2
#LAMBDA_2 = 0.05
START_AGE = 0
END_AGE = 69
VALIDATION_RATE= 0.1

random.seed(2019)
np.random.seed(2019)
torch.manual_seed(2019)

#DEV_CUDA = 'cuda:1'
# change DEV_CUDA in mean_varaince_loss.py 


def ResNet34(num_classes):

    model = resnet34(pretrained=True)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    model.lossRatio = nn.Sequential(
        nn.Linear(512, 3),
        nn.Softmax(dim=1)
    )
    return model



def train(train_loader, model, criterion1, criterion2, optimizer, epoch, result_directory):

    model.train()
    running_loss = 0.
    running_mean_loss = 0.
    running_variance_loss = 0.
    running_softmax_loss = 0.
    interval = 10
    for i, sample in enumerate(train_loader):
        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        output, _ = model(images)
        mean_loss, variance_loss = criterion1(output, labels)
        softmax_loss = criterion2(output, labels)
        loss = mean_loss + variance_loss + softmax_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_mean_loss += mean_loss.data
        running_variance_loss += variance_loss.data
        running_softmax_loss += softmax_loss.data
        if (i + 1) % interval == 0:
            print('[%d, %5d] mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f'
                  % (epoch, i, running_mean_loss / interval,
                     running_variance_loss / interval,
                     running_softmax_loss / interval,
                     running_loss / interval))
            with open(os.path.join(result_directory, 'log'), 'a') as f:
                f.write('[%d, %5d] mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f\n'
                        % (epoch, i, running_mean_loss / interval,
                           running_variance_loss / interval,
                           running_softmax_loss / interval,
                           running_loss / interval))
            running_loss = 0.
            running_mean_loss = 0.
            running_variance_loss = 0.
            running_softmax_loss = 0.
            
        return running_mean_loss, running_variance_loss, running_softmax_loss, running_loss
            
## RETURN of evaluate
#    return mean_loss_val / len(val_loader),\    # mean
#        variance_loss_val / len(val_loader),\   # variance
#        softmax_loss_val / len(val_loader),\    # soft-max
#        loss_val / len(val_loader),\            # total loss
#        mae / len(val_loader)                   # mae


def evaluate(val_loader, model, criterion1, criterion2):
    model.cuda()
    model.eval()
    loss_val = 0.
    mean_loss_val = 0.
    variance_loss_val = 0.
    softmax_loss_val = 0.
    mae = 0.
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            image = sample['image'].cuda()
            label = sample['label'].cuda()
            output, _ = model(image)
            mean_loss, variance_loss = criterion1(output, label)
            softmax_loss = criterion2(output, label)
            loss = mean_loss + variance_loss + softmax_loss
            loss_val += loss.data
            mean_loss_val += mean_loss.data
            variance_loss_val += variance_loss.data
            softmax_loss_val += softmax_loss.data
            m = nn.Softmax(dim=1)
            output_softmax = m(output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    return mean_loss_val / len(val_loader),\
        variance_loss_val / len(val_loader),\
        softmax_loss_val / len(val_loader),\
        loss_val / len(val_loader),\
        mae / len(val_loader)


def test(test_loader, model):
    model.cuda()
    model.eval()
    mae = 0.
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample['image'].cuda()
            label = sample['label'].cuda()
            output, _ = model(image)
            m = nn.Softmax(dim=1)
            output = m(output)
            a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            mae += np.absolute(pred - sample['label'].cpu().data.numpy())
    return mae / len(test_loader)


def predict(model, image):

    model.eval()
    with torch.no_grad():
        image = image.astype(np.float32) / 255.
        image = np.transpose(image, (2,0,1))
        img = torch.from_numpy(image).cuda()
        output, _ = model(img[None])
        m = nn.Softmax(dim=1)
        output = m(output)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output * a).sum(1, keepdim=True).cpu().data.numpy()
        pred = np.around(mean)[0][0]
    return pred


def get_image_list(image_directory, leave_sub, validation_rate):
    
    train_val_list = []
    test_list = []
    for fn in os.listdir(image_directory):
        filepath = os.path.join(image_directory, fn)
        subject = int(fn[:3])
        if subject == leave_sub:
            test_list.append(filepath)
        else:
            train_val_list.append(filepath)
    num = len(train_val_list)
    index_val = np.random.choice(num, int(num * validation_rate), replace=False)
    train_list = []
    val_list = []
    for i, fp in enumerate(train_val_list):
        if i in index_val:
            val_list.append(fp)
        else:
            train_list.append(fp)

    return train_list, val_list, test_list


def get_args():

    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=16)
    parser.add_argument('-l', '--loss_type', type=int, default=1)
    parser.add_argument('-i', '--image_directory', type=str)
    parser.add_argument('-ls', '--leave_subject', type=int)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-l1', '--lambda1', type=float, default=0.2)
    parser.add_argument('-l2', '--lambda2', type=float, default=0.05)
    parser.add_argument('-e', '--epoch', type=int, default=0)
    parser.add_argument('-r', '--resume', type=str, default=None)
    parser.add_argument('-rd', '--result_directory', type=str, default=None)
    parser.add_argument('-pi', '--pred_image', type=str, default=None)
    parser.add_argument('-pm', '--pred_model', type=str, default=None)
    parser.add_argument('-cu', '--cuda', type=int, default=0)
    return parser.parse_args()

def main():
    
    args = get_args()
    #if args.cuda == 1:
    #    DEV_CUDA = 'cuda:1'
    #Does not work properly.
    #  RuntimeError: Expected tensor for argument #1 'input' to have the same device as tensor for argument #2 'weight'; but device 0does not equal 1 #(while checking arguments for cudnn_convolution)
    if args.cuda:
        torch.cuda.set_device(args.cuda) 
    LAMBDA_1 = args.lambda1
    LAMBDA_2 = args.lambda2
    if args.epoch > 0:
        batch_size = args.batch_size
        if args.result_directory is None:
            args.result_directory = 'result-DMTL'+ \
                                    '-l'+str(args.loss_type) + \
                                    f'-lr={int(args.learning_rate*100000):06d}' + \
                                    f'-l1={int(args.lambda1*1000):04d}' + \
                                    f'-l2={int(args.lambda2*1000):04d}' + \
                                    f'-e{args.epoch:06d}'            
            print('result_directory = ', args.result_directory)
            #raise ValueError('stop')
            
        if args.result_directory is not None:
        #else:
            while os.path.isdir(args.result_directory):
                args.result_directory = args.result_directory + "_"
            if not os.path.exists(args.result_directory):
                os.mkdir(args.result_directory)
        print('result_directory modified to = ', args.result_directory)
        writer_train = SummaryWriter(args.result_directory+'train') # tensorboard
        writer_val = SummaryWriter(args.result_directory+'val') # tensorboard
        writer_test = SummaryWriter(args.result_directory+'test') # tensorboard
        
        train_filepath_list, val_filepath_list, test_filepath_list\
            = get_image_list(args.image_directory, args.leave_subject, VALIDATION_RATE)
        transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomApply(
                [torchvision.transforms.RandomAffine(degrees=10, shear=16),
                 torchvision.transforms.RandomHorizontalFlip(p=1.0),
                ], p=0.5),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.RandomCrop((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        train_gen = FaceDataset(train_filepath_list, transforms_train)
        train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])
        val_gen = FaceDataset(val_filepath_list, transforms)
        val_loader = DataLoader(val_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)

        test_gen = FaceDataset(test_filepath_list, transforms)
        test_loader = DataLoader(test_gen, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
        
        #https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        def forward(self, x):
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            xout = torch.flatten(x, 1)
            y = self.fc(xout)
            lossR = self.lossRatio(xout)

            return y,lossR

        ResNet.forward = forward

        model = ResNet34(END_AGE - START_AGE + 1)
        model.cuda()

        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=0.9, weight_decay=1e-4)
        if args.loss_type == 1:
            criterion1 = MeanVarianceLoss(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE).cuda()
        elif args.loss_type == 2:
            criterion1 = MeanVarianceLoss2(LAMBDA_1, LAMBDA_2, START_AGE, END_AGE).cuda()
        else:
            raise ValueError('loss_type should be either 1 or 2')
        criterion2 = torch.nn.CrossEntropyLoss().cuda()

        # scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        best_val_mae = np.inf
        best_val_loss = np.inf
        best_mae_epoch = -1
        best_loss_epoch = -1
        for epoch in range(args.epoch):
            if epoch == 10:
                for param in model.parameters():
                    param.requires_grad = True
            scheduler.step(epoch)
            mean_loss_train, variance_loss_train, softmax_loss_train, loss_train = \
                train(train_loader, model, criterion1, criterion2, optimizer, epoch, args.result_directory)
            mean_loss, variance_loss, softmax_loss, loss_val, mae = evaluate(val_loader, model, criterion1, criterion2)
            mae_test = test(test_loader, model)
            print('epoch: %d, mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f, mae: %3f' %
                  (epoch, mean_loss, variance_loss, softmax_loss, loss_val, mae))
            print('epoch: %d, test_mae: %3f' % (epoch, mae_test))
            with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                f.write('epoch: %d, mean_loss: %.3f, variance_loss: %.3f, softmax_loss: %.3f, loss: %.3f, mae: %3f\n' %
                        (epoch, mean_loss, variance_loss, softmax_loss, loss_val, mae))
                f.write('epoch: %d, mae_test: %3f\n' % (epoch, mae_test))
                
            #writer.add_scalar('mae/val', mae, epoch)
            #writer.add_scalar('mae/test', mae_test, epoch)
            writer_val.add_scalar('mae', mae, epoch)
            writer_test.add_scalar('mae', mae_test, epoch)
            
            #writer.add_scalar('total-loss/train', loss_train, epoch)
            #writer.add_scalar('total-loss/val', loss_val, epoch)
            writer_train.add_scalar('total-loss', loss_train, epoch)
            writer_val.add_scalar('total-loss', loss_val, epoch)
            
            #writer.add_scalar('loss/train-mean', mean_loss_train, epoch)
            #writer.add_scalar('loss/val-mean', mean_loss, epoch)
            writer_train.add_scalar('loss/mean', mean_loss_train, epoch)
            writer_val.add_scalar('loss/mean', mean_loss, epoch)
            
            #writer.add_scalar('loss/train-var', variance_loss_train, epoch)
            #writer.add_scalar('loss/val-var', variance_loss, epoch)
            writer_train.add_scalar('loss/var', variance_loss_train, epoch)
            writer_val.add_scalar('loss/var', variance_loss, epoch)
            
            #writer.add_scalar('loss/train-soft', softmax_loss_train, epoch)
            #writer.add_scalar('loss/val-soft', softmax_loss, epoch)
            writer_train.add_scalar('loss/soft', softmax_loss_train, epoch)
            writer_val.add_scalar('loss/soft', softmax_loss, epoch)
            
            #writer.add_scalars('loss/trainAndVal', {'train':loss_train, 'val':loss_val}, epoch)
                
            if best_val_mae > mae:
                best_val_mae = mae
                best_mae_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.result_directory, "model_best_mae"))
            if best_val_loss > loss_val:
                best_val_loss = loss_val
                best_loss_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.result_directory, "model_best_loss"))            
            with open(os.path.join(args.result_directory, 'log'), 'a') as f:
                f.write('best_loss_epoch: %d, best_val_loss: %f, best_mae_epoch: %d, best_val_mae: %f\n'
                        % (best_loss_epoch, best_val_loss, best_mae_epoch, best_val_mae))
            print('best_loss_epoch: %d, best_val_loss: %f, best_mae_epoch: %d, best_val_mae: %f'
                  % (best_loss_epoch, best_val_loss, best_mae_epoch, best_val_mae))
    if args.pred_image and args.pred_model:
        model = ResNet34(END_AGE - START_AGE + 1)
        model.cuda()
        img = cv2.imread(args.pred_image)
        resized_img = cv2.resize(img, (224, 224))
        model.load_state_dict(torch.load(args.pred_model))
        pred = predict(model, resized_img)
        print('Age: ' + str(int(pred)))
        cv2.putText(img, 'Age: ' + str(int(pred)), (int(img.shape[1]*0.1), int(img.shape[0]*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        name, ext = os.path.splitext(args.pred_image)
        cv2.imwrite(name + '_result.jpg', img)
        
        
if __name__ == "__main__":
    main()
