# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from data.load_data import CHARS, CHARS_DICT, LPRDataLoader,CCPDDataloader
from PIL import Image, ImageDraw, ImageFont
from model.LPRNet import build_lprnet
from model.STNet import build_STNet
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

import cv2

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="dataset/CCPD_test", help='the test images path')
    parser.add_argument('--dataset_type', default='ccpd', choices=['generic', 'ccpd'], help='dataset format for dataloader')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=10, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=False, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--decode_method', default="greedy", help='choose method to perform ctc decode')
    parser.add_argument('--topk', default=3, type=int, help='choose top-k results to output in beam decode mode')
    parser.add_argument('--pretrained_model_LPR', default='LPRNet/weights/Final_LPRNet_model.pth', help='pretrained LPRNet model')
    parser.add_argument('--pretrained_model_STN', default='LPRNet/weights/Final_STNet_model.pth', help='pretrained STNet model')

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
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def test():
    args = get_parser()

    stnet = build_STNet(phase=args.phase_train)
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    stnet.to(device)
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if not args.pretrained_model_STN or not args.pretrained_model_LPR:
        print("[Error] Can't found pretrained model, please check!")
        return False
    stnet.load_state_dict(torch.load(args.pretrained_model_STN, map_location=device))
    lprnet.load_state_dict(torch.load(args.pretrained_model_LPR, map_location=device))
    print("load pretrained model successful!")

    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    test_dataset =  CCPDDataloader(test_img_dirs.split(','), args.img_size, args.lpr_max_len) if args.dataset_type == 'ccpd' else LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    try:
        decode_method = str(args.decode_method).lower()
        if decode_method.startswith("beam"):
            Beam_Decode_Eval(stnet, lprnet, test_dataset, args)
        else:
            Greedy_Decode_Eval(stnet, lprnet, test_dataset, args)
    finally:
        cv2.destroyAllWindows()

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
        imgs = images.numpy().copy()

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
            # show image and its predict label
            if args.show:
                show(imgs[i], label, targets[i])
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

def _log_sum_exp(*log_probs):
    valid = [p for p in log_probs if p != -np.inf]
    if not valid:
        return -np.inf
    max_log = max(valid)
    return max_log + np.log(sum(np.exp(p - max_log) for p in valid))

def _ctc_prefix_beam_search(log_probs, beam_size, blank_id):
    beam_size = max(1, int(beam_size))
    beam = {(): (0.0, -np.inf)}  # prefix -> (prob_end_blank, prob_end_non_blank)

    for t in range(log_probs.shape[0]):
        next_beam = {}
        for prefix, (p_blank, p_non_blank) in beam.items():
            for c in range(log_probs.shape[1]):
                p = log_probs[t, c]
                if c == blank_id:
                    n_blank, n_non_blank = next_beam.get(prefix, (-np.inf, -np.inf))
                    n_blank = _log_sum_exp(n_blank, p_blank + p, p_non_blank + p)
                    next_beam[prefix] = (n_blank, n_non_blank)
                    continue

                end_char = prefix[-1] if prefix else None
                extended = prefix + (int(c),)
                n_blank, n_non_blank = next_beam.get(extended, (-np.inf, -np.inf))
                if c == end_char:
                    # Repeated char only extends prefix from blank states.
                    n_non_blank = _log_sum_exp(n_non_blank, p_blank + p)
                else:
                    n_non_blank = _log_sum_exp(n_non_blank, p_blank + p, p_non_blank + p)
                next_beam[extended] = (n_blank, n_non_blank)

                if c == end_char:
                    # Repeated char from non-blank keeps the same collapsed prefix.
                    n_blank, n_non_blank = next_beam.get(prefix, (-np.inf, -np.inf))
                    n_non_blank = _log_sum_exp(n_non_blank, p_non_blank + p)
                    next_beam[prefix] = (n_blank, n_non_blank)

        sorted_beam = sorted(
            next_beam.items(),
            key=lambda x: _log_sum_exp(x[1][0], x[1][1]),
            reverse=True
        )
        beam = dict(sorted_beam[:beam_size])

    return sorted(
        [(list(prefix), _log_sum_exp(p_blank, p_non_blank)) for prefix, (p_blank, p_non_blank) in beam.items()],
        key=lambda x: x[1],
        reverse=True
    )[:beam_size]

def _is_valid_plate_seq(seq):
    if len(seq) < 7 or len(seq) > 8:
        return False
    return all(CHARS[int(c)] not in ("O", "-") for c in seq)

def _select_valid_prediction(pred_topk):
    for seq in pred_topk:
        if _is_valid_plate_seq(seq):
            return seq
    return []

def Beam_Decode_Eval(stnet,lprnet,datasets,args):
    stnet.eval()
    lprnet.eval()
    beam_size = max(1, int(args.topk))
    blank_id = len(CHARS) - 1
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Top1_Tp = 0
    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    t1 = time.time()
    for i in range(epoch_size):
        # load test data
        images, labels, lengths = next(batch_iterator)
        start = 0
        targets = []
        for length in lengths:
            label = labels[start:start+length]
            targets.append(label.cpu().numpy())
            start += length
        imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        with torch.no_grad():
            prebs = lprnet(stnet(images))
            log_probs = F.log_softmax(prebs, dim=1).permute(0, 2, 1).cpu().detach().numpy()

        for sample_idx in range(log_probs.shape[0]):
            beam_results = _ctc_prefix_beam_search(log_probs[sample_idx], beam_size, blank_id)
            if len(beam_results) == 0:
                beam_results = [([], -np.inf)]
            pred_topk = [item[0] for item in beam_results]
            top1_pred = pred_topk[0]
            selected_pred = _select_valid_prediction(pred_topk)
            target = [int(v) for v in targets[sample_idx].tolist()]
            target_tuple = tuple(target)

            if args.show:
                show(imgs[sample_idx], selected_pred, targets[sample_idx])
                topk_text = ["".join(CHARS[c] for c in seq) for seq in pred_topk]
                selected_text = "".join(CHARS[c] for c in selected_pred)
                target_text = "".join(CHARS[c] for c in target)
                print("[Beam] target: {}, top{}: {}, selected: {}".format(target_text, beam_size, topk_text, selected_text))

            if tuple(top1_pred) == target_tuple:
                Top1_Tp += 1

            if tuple(selected_pred) == target_tuple:
                Tp += 1
            else:
                if len(selected_pred) != len(target):
                    Tn_1 += 1
                else:
                    Tn_2 += 1

    total = Tp + Tn_1 + Tn_2
    if total == 0:
        print("[Info] Beam(top-{}) Test Accuracy: 0.0 [0:0:0:0]".format(beam_size))
    else:
        topk_acc = Tp * 1.0 / total
        top1_acc = Top1_Tp * 1.0 / total
        print("[Info] Beam(top-{}) Test Accuracy: {} [{}:{}:{}:{}]".format(beam_size, topk_acc, Tp, Tn_1, Tn_2, total))
        print("[Info] Beam(top-1) Accuracy: {}".format(top1_acc))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / len(datasets), len(datasets)))

def show(img, label, target):
    WINDOW_NAME = "test"
    WINDOW_POSITION = (100, 100)  # keep imshow window fixed on screen
    img = np.transpose(img, (1, 2, 0))
    img *= 128.
    img += 127.5
    img = img.astype(np.uint8)
    lb = ""
    for i in label:
        lb += CHARS[i]
    tg = ""
    for j in target.tolist():
        tg += CHARS[int(j)]

    flag = "F"
    if lb == tg:
        flag = "T"
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    img = cv2ImgAddText(img, lb, (0, 0))
    if (tg != lb):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(WINDOW_NAME, WINDOW_POSITION[0], WINDOW_POSITION[1])
        cv2.imshow(WINDOW_NAME, img)
        print("target: ", tg, " ### {} ### ".format(flag), "predict: ", lb)
        cv2.waitKey()
        cv2.destroyAllWindows()

def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
    if (isinstance(img, np.ndarray)):  # detect opencv format or not
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    draw.text(pos, text, textColor, font=fontText)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    test()
