# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import copy
import time
import warnings
import numpy as np

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)

from tools.analysis_tools.robustness_eval import get_results
import pdb
import random

SEED = 2
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def coco_eval_with_return(result_files,
                          result_types,
                          coco,
                          max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in ['proposal', 'bbox', 'segm', 'keypoints']

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    eval_results = {}
    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        if res_type == 'segm' or res_type == 'bbox':
            metric_names = [
                'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10',
                'AR100', 'ARs', 'ARm', 'ARl'
            ]
            eval_results[res_type] = {
                metric_names[i]: cocoEval.stats[i]
                for i in range(len(metric_names))
            }
        else:
            eval_results[res_type] = cocoEval.stats

    return eval_results


def voc_eval_with_return(result_file,
                         dataset,
                         iou_thr=0.5,
                         logger='print',
                         only_ap=True):
    det_results = mmcv.load(result_file)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    mean_ap, eval_results = eval_map(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger=logger)

    if only_ap:
        eval_results = [{
            'ap': eval_results[i]['ap']
        } for i in range(len(eval_results))]

    return mean_ap, eval_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--corruptions',
        type=str,
        nargs='+',
        default='benchmark',
        choices=[
            'all', 'benchmark', 'noise', 'blur', 'weather', 'digital',
            'holdout', 'None', 'gaussian_noise', 'shot_noise', 'impulse_noise',
            'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow',
            'frost', 'fog', 'brightness', 'contrast', 'elastic_transform',
            'pixelate', 'jpeg_compression', 'speckle_noise', 'gaussian_blur',
            'spatter', 'saturate'
        ],
        help='corruptions')
    parser.add_argument(
        '--severities',
        type=int,
        nargs='+',
        default=[0],
        help='corruption severity levels')
    parser.add_argument(
        '--final-prints',
        type=str,
        nargs='+',
        choices=['P', 'mPC', 'rPC'],
        default=['P', 'mPC', 'rPC'],
        help='corruption benchmark metric to print at the end')
    parser.add_argument(
        '--final-prints-aggregate',
        type=str,
        choices=['all', 'benchmark'],
        default='all',
        help='aggregate all results or only those for benchmark corruptions')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        breakpoint()
        init_dist(args.launcher, **cfg.dist_params)

    if 'test_time_modifications' in cfg.data.keys():
        if 'all' in cfg.data.test_time_modifications.corruptions:
            corruptions = [
                'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate',
                'jpeg_compression', 'speckle_noise', 'gaussian_blur', 'spatter',
                'saturate'
            ]
            severities = cfg.data.test_time_modifications.severities
        elif 'benchmark' in cfg.data.test_time_modifications.corruptions:
            corruptions = [
                'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
                'defocus_blur', 'motion_blur', 'gaussian_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate',
                'jpeg_compression'
            ]
            severities = cfg.data.test_time_modifications.severities
        elif 'noise' in cfg.data.test_time_modifications.corruptions:
            corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise']
        elif 'blur' in cfg.data.test_time_modifications.corruptions:
            corruptions = [
                'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'
            ]
            severities = cfg.data.test_time_modifications.severities
        elif 'weather' in cfg.data.test_time_modifications.corruptions:
            corruptions = ['snow', 'frost', 'fog', 'brightness']
            severities = cfg.data.test_time_modifications.severities
        elif 'digital' in cfg.data.test_time_modifications.corruptions:
            corruptions = [
                'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
            ]
            severities = cfg.data.test_time_modifications.severities
        elif 'holdout' in cfg.data.test_time_modifications.corruptions:
            corruptions = ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
            severities = cfg.data.test_time_modifications.severities
        elif 'None' in cfg.data.test_time_modifications.corruptions:
            corruptions = ['None']
            severities = [0]
        else:
            corruptions = cfg.data.test_time_modifications.corruptions
            severities = [0]
    else:
        corruptions = ['None']
        severities = [0]

    if 'mixup' in cfg.data.keys():
        corruptions = ['mixup']
        severities = [-1]
        num_examples = cfg.data.mixup.num_examples


    rank, _ = get_dist_info()
    aggregated_results = {}
    aggregated_results['random'] = {}
    for sev_i, corruption_severity in enumerate(severities):
        # print info
        print(f'\nTesting at severity {corruption_severity}')

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=32, dist=distributed, shuffle=False)

        # in case the test dataset is concatenated
        if isinstance(cfg.data.test, dict):
            if 'test_mode' not in cfg.data.test:
                cfg.data.test.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                if 'test_mode' not in ds_cfg:
                    ds_cfg.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }

        test_data_cfg = copy.deepcopy(cfg.data.test)

        # If id apply corruption
        if corruption_severity > 0:
            corruption_trans = dict(
                type='RandomCorrupt',
                corruptions=corruptions,
                severity=corruption_severity)
            # TODO: hard coded "1", we assume that the first step is
            # loading images, which needs to be fixed in the future
            test_data_cfg['pipeline'].insert(1, corruption_trans)
        # If ood apply mixup
        elif corruption_severity == -1:
            test_data_cfg = {'type': 'MultiImageMixDataset', 'dataset': {'type': 'CocoDataset', 'ann_file': 'data/coco/annotations/general_od_ood_test.json', 'img_prefix': 'data/', 'pipeline': [{'type': 'LoadImageFromFile'}], 'test_mode': True, 'filter_empty_gt': False}, 'max_refetch': 1, 'pipeline': [{'type': 'OODMixUp', 'ratio_range': (0.75, 1.0), 'corruptions': ['gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise', 'defocus_blur', 'motion_blur', 'gaussian_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'], 'severity_range': (1, 5), 'offset_ctr': sev_i}, {'type': 'MultiScaleFlipAug', 'img_scale': (1333, 800), 'flip': False, 'transforms': [{'type': 'Resize', 'keep_ratio': True}, {'type': 'RandomFlip'}, {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'type': 'DefaultFormatBundle'}, {'type': 'Collect', 'keys': ['img']}]}], 'test_mode': True}

        rank, _ = get_dist_info()
        # allows not to create
        if args.work_dir is not None and rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

        # build the dataloader
        dataset = build_dataset(test_data_cfg)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        if not distributed:
            model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
            # Uncertainty Module should add uncertainty values in outputs
            outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                      args.show_score_thr)
        else:
            model = build_ddp(
                model,
                cfg.device,
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False)
            # Uncertainty Module should add uncertainty values in outputs
            outputs = multi_gpu_test(
                model, data_loader, args.tmpdir, args.gpu_collect
                or cfg.evaluation.get('gpu_collect', False))

        if args.out and rank == 0:
            eval_results_filename = (
                osp.splitext(args.out)[0] + '_results' +
                osp.splitext(args.out)[1])
            eval_types = args.eval
            file_name = args.out[:-4] + '_severity_' + str(sev_i) + args.out[-4:]
            if cfg.dataset_type == 'VOCDataset':
                if eval_types:
                    for eval_type in eval_types:
                        if eval_type == 'bbox':
                            test_dataset = mmcv.runner.obj_from_dict(
                                cfg.data.test, datasets)
                            logger = 'print' if args.summaries else None
                            mean_ap, eval_results = \
                                voc_eval_with_return(
                                    args.out, test_dataset,
                                    args.iou_thr, logger)
                            aggregated_results['random'][
                                corruption_severity] = eval_results
                        else:
                            print('\nOnly "bbox" evaluation \
                            is supported for pascal voc')
            else:
                if eval_types:
                    print(f'Starting evaluate {" and ".join(eval_types)}')
                    if eval_types == ['proposal_fast']:
                        result_file = args.out
                    else:
                        if not isinstance(outputs[0], dict):
                            if test_data_cfg['type'] == 'MultiImageMixDataset':
                                result_files = dataset.dataset.results2json(
                                    outputs, file_name)
                            else:
                                result_files = dataset.results2json(
                                    outputs, file_name)
                        else:
                            for name in outputs[0]:
                                print(f'\nEvaluating {name}')
                                outputs_ = [out[name] for out in outputs]
                                result_file = args.out
                                + f'.{name}'
                                result_files = dataset.results2json(
                                    outputs_, result_file)

                    if test_data_cfg['ann_file'] != 'data/robustod/annotations/ood.json':
                        eval_results = coco_eval_with_return(
                            result_files, eval_types, dataset.coco)

                    aggregated_results['random'][
                        corruption_severity] = eval_results
                else:
                    print('\nNo task was selected for evaluation;'
                          '\nUse --eval to select a task')

if __name__ == '__main__':
    main()
