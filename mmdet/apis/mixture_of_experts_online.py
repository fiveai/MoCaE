# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import torch
from mmcv.image import tensor2imgs

from mmdet.core import encode_mask_results, multiclass_nms, bbox2result
import numpy as np
import pickle, os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


class model_cal(torch.nn.Module):
    def __init__(self, in_size, out_size=80, prior_prob=0.10):
        super(model_cal, self).__init__()
        self.linear = torch.nn.Linear(in_size, in_size)
        torch.nn.init.normal_(self.linear.weight.data, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.linear.bias.data, mean=0.0, std=0.01)
        self.bn1 = torch.nn.BatchNorm1d(in_size)
        self.linear2 = torch.nn.Linear(in_size, out_size)
        torch.nn.init.normal_(self.linear2.weight.data, mean=0.0, std=0.01)
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.linear2.bias.data.fill_(bias_init)


    def forward(self, x):
        out1 = self.bn1(torch.relu(self.linear(x)))
        out = torch.sigmoid(self.linear2(out1))
        return out



class model_lin(torch.nn.Module):
    def __init__(self, in_size, out_size=80, prior_prob=0.10):
        super(model_lin, self).__init__()
        #self.linear = torch.nn.Linear(in_size, in_size)
        #self.bn1 = torch.nn.BatchNorm1d(in_size)
        self.linear2 = torch.nn.Linear(in_size, out_size)
        torch.nn.init.normal_(self.linear2.weight.data, mean=0.0, std=0.01)
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.linear2.bias.data.fill_(bias_init)


    def forward(self, x):
        #out1 = self.bn1(torch.relu(self.linear(x)))
        out = torch.sigmoid(self.linear2(x))
        return out

class model_logits(torch.nn.Module):
    def __init__(self, in_size=80, out_size=80, prior_prob=0.10):
        super(model_logits, self).__init__()
        #self.linear = torch.nn.Linear(in_size, in_size)
        #self.bn1 = torch.nn.BatchNorm1d(in_size)
        self.linear2 = torch.nn.Linear(in_size, out_size)
        torch.nn.init.normal_(self.linear2.weight.data, mean=0.0, std=0.0001)
        self.linear2.bias.data.fill_(0)

        self.linear3 = torch.nn.Linear(in_size, out_size, bias=False)
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        torch.nn.init.normal_(self.linear3.weight.data, mean=bias_init, std=0.0001)
        #bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        #self.linear3.bias.data.fill_(bias_init)


    def forward(self, x):
        a = self.linear2(x)
        b = self.linear3(x)
        #out1 = self.bn1(torch.relu(self.linear(x)))
        out = torch.sigmoid(a*x[:, :-1] + b)
        return out

class model_roi(torch.nn.Module):
    def __init__(self, in_size, out_size=80, prior_prob=0.50):
        super(model_roi, self).__init__()

        # Conv Layer for features

        self.conv1 = torch.nn.Conv2d(in_channels=256, out_channels=20, kernel_size=(3, 3))
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(20)

        # Fully connected and flatt
        #self.linear = torch.nn.Linear(in_size, in_size)
        #self.bn1 = torch.nn.BatchNorm1d(in_size)
        self.linear2 = torch.nn.Linear(in_size, 500)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(500)

        self.linear3 = torch.nn.Linear(500, out_size)
        torch.nn.init.normal_(self.linear3.weight.data, mean=0.0, std=0.0001)
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.linear3.bias.data.fill_(bias_init)

        self.linear4 = torch.nn.Linear(500, out_size)
        torch.nn.init.normal_(self.linear4.weight.data, mean=0.0, std=0.0001)
        bias_init = float(-np.log((1 - 1e-3) / 1e-3))
        self.linear4.bias.data.fill_(bias_init)


    def forward(self, logits, features):
        out_feat = self.bn1(self.relu1(self.conv1(features)))
        out_feat = torch.flatten(out_feat, start_dim=1)

        out_logit = self.bn2(self.relu2(self.linear2(logits)))

        a = torch.sigmoid(self.linear3(out_logit)) + (1-torch.sigmoid(self.linear3(out_feat)))

        b = torch.sigmoid(self.linear4(out_logit)) + torch.sigmoid(self.linear4(out_feat))

        #out1 = self.bn1(torch.relu(self.linear(x)))
        out = torch.sigmoid(a* logits + b)

        return out
def predict_prob(calibrator, scores, num_classes):
    predicted_ious = []
    for cls in range(num_classes):
        det_scores = scores[:, cls]
        predicted_ious.append(np.clip(calibrator[cls].predict(det_scores.reshape(-1, 1)), 0, 1))

    predicted_ious = np.vstack(predicted_ious).T
    return predicted_ious
    
def mixture_of_experts(model,
                    data_loader,
                    cfg,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    dim=5):
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    #model_names = ['rs_rcnn', 'atss']
    model_names = ['paa']
    model_name = 'faster_rcnn'
    model_names = ['faster_rcnn', 'rs_rcnn']
    #model_names = ['atss']
    model_names = ['rs_rcnn', 'atss']

    #model_names = ['faster_rcnn', 'rs_rcnn', 'atss', 'paa']
    calibration_type = 'IR'
    class_agnostic_calibration = True


    if 'faster_rcnn' in model_names or 'rs_rcnn' in model_names:
        max_per_img = cfg['model']['test_cfg']['rcnn']['max_per_img']
        score_thr = cfg['model']['test_cfg']['rcnn']['score_thr']
        nms_cfg = cfg['model']['test_cfg']['rcnn']['nms']
    else:

        max_per_img = cfg['model']['test_cfg']['max_per_img']
        score_thr = cfg['model']['test_cfg']['score_thr']
        nms_cfg = cfg['model']['test_cfg']['nms']

    #score_thr = 0.001
    num_classes = 80
    ensemble_detections = cfg['ensemble_detections']

    num_epochs = 2
    gamma = 2.0
    lr = 0.05
    wd = 0.0005

    epoch = 0
    i = 200

    if calibration_type == 'NN' or calibration_type == 'NN_net':
        

        calibration_dir_name = "calibration/" + model_name +"/calibrators/" + 'lr_' + str(lr) + '_wd_' + str(wd) + '_gamma_' + str(wd) + '_epochs_' + str(num_epochs)

        #calibration_dir_name = "calibration/faster_rcnn/calibrators/" + 'nonlinear_lr_' + str(lr) + '_wd_' + str(wd) + '_gamma_' + str(wd) + '_epochs_' + str(num_epochs)
        calibration_file = calibration_dir_name + '/epoch_' + str(epoch) + 'iteration_' + str(i) + '.pkl'
        #calibration_file = "calibration/faster_rcnn/calibrators/NN.pkl"
        if os.path.exists(calibration_file):
            calibrator = model_roi(80, 80).cuda()
            calibrator.load_state_dict(torch.load(calibration_file))
            calibrator.eval()

    else:
        calibrator = []
        for model_name in model_names:
            print(model_name)
            if class_agnostic_calibration:
                calibration_file = "calibration/" + model_name + "/calibrators/" + calibration_type + '_class_agnostic_clip' '.pkl'
            else:
                calibration_file = "calibration/" + model_name + "/calibrators/" + calibration_type + '_class_wise_clip' '.pkl'

            if os.path.exists(calibration_file):
                if calibration_type == 'NN':
                    calibrator = model_lin(1024, 80).cuda()

                    calibrator.load_state_dict(torch.load(calibration_file))
                    calibrator.eval()
                else:
                    with open(calibration_file, 'rb') as f:
                        calibrator.append(pickle.load(f))
            else:
                print('no calibrator, need to implement identity')

    cocoGt = COCO(cfg['data']['test']['ann_file'])
    if calibration_type == 'oracle':
        filename_test = "probability_logs/raw_detections/" + model_name + "/all_test.npy"
        dets = np.load(filename_test)

    idx = 0
    features_folder = "calibration/" + model_names[0] + "/features/val/"
    # features_folder = "calibration/" + model_name + "/logits/val/"

    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result_list = []

            # Read from file and concat
            for j, dir in enumerate(ensemble_detections):
                if 'counter' in cocoGt.dataset['images'][i]:
                    counter = cocoGt.dataset['images'][i]['counter']
                else:
                    counter = i
                x = np.load(dir + str(counter) + ".npy")
                if j == 0:
                    num_dets = x.shape[0]
                    if calibration_type == 'oracle':
                        x[:, :num_classes] = dets[idx: idx+num_dets, num_classes:]
                        idx += num_dets
                        scores = torch.from_numpy(x[:, :num_classes]).cuda().type(torch.cuda.FloatTensor)
                    elif calibration_type == 'IR' or calibration_type == 'LR':
                        s_ = predict_prob(calibrator[j], x[:, :num_classes], num_classes)
                        x[:, :num_classes] = s_
                        scores = torch.from_numpy(x[:, :num_classes]).cuda().type(torch.cuda.FloatTensor)
                    elif calibration_type == 'NN':
                        feature_path = features_folder + str(counter) + '.npy'
                        feature = torch.from_numpy(np.load(feature_path)).cuda().reshape(-1, 1024)
                        scores = calibrator(feature)
                    elif calibration_type == 'NN_net':
                        raw_detections, features = model(return_loss=False, rescale=True, **data)
                        raw_detections = torch.stack(raw_detections)[:, :, :num_classes].reshape(-1, num_classes)
                        logits = -torch.log((1 - raw_detections) / raw_detections)
                        scores = calibrator(logits, features)
                    else: # identity
                        scores = torch.from_numpy(x[:, :num_classes]).cuda().type(torch.cuda.FloatTensor)

                    # Padding for NMS
                    padding = scores.new_zeros(scores.shape[0], 1)
                    scores = torch.cat([scores, padding], dim=1)

                    bboxes = torch.from_numpy(x[:, num_classes:]).cuda().type(torch.cuda.FloatTensor)
                    if bboxes.shape[1] == 4:
                        bboxes = bboxes[:, None].expand(scores.size(0), num_classes, 4).reshape(-1, num_classes * 4)
                else:
                    num_dets = x.shape[0]
                    if calibration_type == 'oracle':
                        x[:, :num_classes] = dets[idx: idx+num_dets, num_classes:]
                        idx += num_dets
                        temp_scores = torch.from_numpy(x[:, :num_classes]).cuda().type(torch.cuda.FloatTensor)
                    elif calibration_type == 'IR' or calibration_type == 'LR':
                        s_ = predict_prob(calibrator[j], x[:, :num_classes], num_classes)
                        x[:, :num_classes] = s_
                        temp_scores = torch.from_numpy(x[:, :num_classes]).cuda().type(torch.cuda.FloatTensor)
                    elif calibration_type == 'NN':
                        feature_path = features_folder + str(counter) + '.npy'
                        feature = torch.from_numpy(np.load(feature_path)).cuda().reshape(-1, 1024)
                        temp_scores = calibrator(feature)
                    else: # identity
                        temp_scores = torch.from_numpy(x[:, :num_classes]).cuda().type(torch.cuda.FloatTensor)

                    padding = scores.new_zeros(temp_scores.shape[0], 1)
                    temp_scores = torch.cat([temp_scores, padding], dim=1)

                    temp_bboxes = torch.from_numpy(x[:, num_classes:]).cuda().type(torch.cuda.FloatTensor)
                    if temp_bboxes.shape[1] == 4:
                        temp_bboxes = temp_bboxes[:, None].expand(temp_scores.size(0), num_classes, 4).reshape(-1, num_classes * 4)

                    scores = torch.cat((scores, temp_scores))
                    bboxes = torch.cat((bboxes, temp_bboxes))

            # Send to NMS
            det_bboxes, det_labels, nms_time = multiclass_nms(bboxes, scores, score_thr, nms_cfg, max_per_img)

            # Convert to list
            raw_result = (det_bboxes, det_labels)
            result_list.append(raw_result)

            result = [
                bbox2result(det_bboxes, det_labels, num_classes, dim)
                for det_bboxes, det_labels in result_list
            ]

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
