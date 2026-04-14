# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, CCPDDataloader, LPRDataLoader
from model.STNet import build_STNet
from model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import os

def parse_img_dirs(img_dirs):
    if isinstance(img_dirs, str):
        return [item.strip() for item in img_dirs.split(',') if item.strip()]
    return [str(item).strip() for item in img_dirs if str(item).strip()]

def resolve_img_dirs(img_dirs):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(script_dir, ".."))
    resolved_dirs = []

    for raw_path in parse_img_dirs(img_dirs):
        expanded_path = os.path.expanduser(raw_path)
        candidate_paths = [expanded_path]
        if not os.path.isabs(expanded_path):
            candidate_paths.append(os.path.join(src_root, expanded_path))

        matched_path = None
        for candidate in candidate_paths:
            abs_candidate = os.path.abspath(candidate)
            if os.path.isdir(abs_candidate):
                matched_path = abs_candidate
                break

        if matched_path is None:
            raise ValueError(
                f"找不到数据目录: {raw_path}. "
                f"可用逗号分隔多个目录，例如 "
                f"'dataset/CCPD2019,dataset/CCPD2020/ccpd_green/train'"
            )
        resolved_dirs.append(matched_path)

    if not resolved_dirs:
        raise ValueError("未提供有效的数据目录。")

    return resolved_dirs

def init_lprnet_weights(lprnet):
    for module in lprnet.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

def init_stnet_weights(stnet):
    identity_head = stnet.fc_loc[2]
    for module in stnet.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.Linear) and module is not identity_head:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    # Keep the affine regressor initialized to identity so STN starts stable.
    identity_head.weight.data.zero_()
    identity_head.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--max_epoch',type=int, default=5, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--train_img_dirs', default="dataset/CCPD2019,dataset/CCPD2020", help='comma-separated train image directories')
    parser.add_argument('--test_img_dirs', default="dataset/CCPD_test", help='comma-separated test image directories')
    parser.add_argument('--dataset_type', default='ccpd', choices=['generic', 'ccpd'], help='dataset format for dataloader')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--learning_rate',type=float, default=0.0001, help='base value of learning rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=128, help='training batch size.')
    parser.add_argument('--test_batch_size', default=120, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--save_interval', default=2000, type=int, help='interval for save model state dict')
    parser.add_argument('--test_interval', default=2000, type=int, help='interval for evaluate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='LPRNet/weights/', help='Location to save checkpoint models')
    parser.add_argument('--pretrained_model_LPR', default='/home/fiatiustitia/RK3568_LPR/src/LPRNet/weights/Final_LPRNet_model.pth', help='pretrained LPRNet base model')
    parser.add_argument('--pretrained_model_STN', default='/home/fiatiustitia/RK3568_LPR/src/LPRNet/weights/Final_STNet_model.pth', help='pretrained STNet base model')
    # parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def train():
    args = get_parser()

    T_length = 18 # args.lpr_max_len
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    stnet = build_STNet(phase=args.phase_train)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    stnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model_LPR:
        lprnet.load_state_dict(torch.load(args.pretrained_model_LPR, map_location=device))
        print("load LPR pretrained model successful!")
    else:
        init_lprnet_weights(lprnet)
        print("initial net weights successful!")
    
    if args.pretrained_model_STN:
        stnet.load_state_dict(torch.load(args.pretrained_model_STN, map_location=device))
        print("load STN pretrained model successful!")
    else:
        init_stnet_weights(stnet)
        print("initial net weights successful!")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(list(stnet.parameters()) + list(lprnet.parameters()), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = resolve_img_dirs(args.train_img_dirs)
    test_img_dirs = resolve_img_dirs(args.test_img_dirs)
    loader_cls = CCPDDataloader if args.dataset_type == 'ccpd' else LPRDataLoader
    train_dataset = loader_cls(train_img_dirs, args.img_size, args.lpr_max_len)
    test_dataset = loader_cls(test_img_dirs, args.img_size, args.lpr_max_len)

    print(f'dataset_type={args.dataset_type}')
    print(f'train_img_dirs={train_img_dirs}')
    print(f'test_img_dirs={test_img_dirs}')

    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
            loss_val = 0
            epoch += 1

        if iteration !=0 and iteration % args.save_interval == 0:
            torch.save(lprnet.state_dict(), args.save_folder + 'LPRNet_' + '_iteration_' + repr(iteration) + '.pth')
            torch.save(stnet.state_dict(), args.save_folder + 'STNet_' + '_iteration_' + repr(iteration) + '.pth')

        if (iteration + 1) % args.test_interval == 0:
            Greedy_Decode_Eval(stnet, lprnet, test_dataset, args)
            stnet.train()
            lprnet.train()

        start_time = time.time()
        # load train data
        images, labels, lengths = next(batch_iterator)
        # labels = np.array([el.numpy() for el in labels]).T
        # print(labels)
        # get ctc parameters
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        # update lr
        lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

        if args.cuda:
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(labels, requires_grad=False).cuda()
        else:
            images = Variable(images, requires_grad=False)
            labels = Variable(labels, requires_grad=False)

        # forward
        images = stnet(images)
        logits = lprnet(images)
        log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
        # print(labels.shape)
        log_probs = log_probs.log_softmax(2).requires_grad_()
        # log_probs = log_probs.detach().requires_grad_()
        # print(log_probs.shape)
        # backprop
        optimizer.zero_grad()
        loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        if loss.item() == np.inf:
            continue
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
        end_time = time.time()
        if iteration % 20 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                  'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))
    # final test
    print("Final test Accuracy:")
    Greedy_Decode_Eval(stnet, lprnet, test_dataset, args)

    # save final parameters
    torch.save(lprnet.state_dict(), args.save_folder + 'Final_LPRNet_model.pth')
    torch.save(stnet.state_dict(), args.save_folder + 'Final_STNet_model.pth')

def Greedy_Decode_Eval(stnet, lprnet, datasets, args):
    stnet.eval()
    lprnet.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load train data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label.cpu().numpy())
            start += length

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        with torch.no_grad():
            prebs = lprnet(stnet(images))
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(targets[i]):
                Tn_1 += 1
                continue
            if (np.asarray(targets[i]) == np.asarray(label)).all():
                Tp += 1
            else:
                Tn_2 += 1

    Acc = Tp * 1.0 / (Tp + Tn_1 + Tn_2)
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))


if __name__ == "__main__":
    train()
