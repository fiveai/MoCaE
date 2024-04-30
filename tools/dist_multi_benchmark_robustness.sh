#!/usr/bin/env bash

CONFIG_DIR="configs/robustness/test/"
#CONFIGS=("atss_r50_fpn_straug_3x_robustodood.py" "atss_r50_fpn_straug_3x_robustodgenod.py" "rs_faster_rcnn_r50_fpn_straug_3x_robustodgenod.py" "rs_faster_rcnn_r50_fpn_straug_3x_robustodood.py" "faster_rcnn_r50_fpn_straug_3x_robustodgenod.py" "faster_rcnn_r50_fpn_straug_3x_robustodood.py" "deformable_detr_r50_16x2_50e_robustodood.py" "deformable_detr_r50_16x2_50e_robustodgenod.py" "nll_faster_rcnn_r50_fpn_straug_3x_robustodgenod.py" "nll_faster_rcnn_r50_fpn_straug_3x_robustodood.py")
#CHECKPOINTS=("work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/rs_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/rs_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/nll_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/nll_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth")
#CONFIGS=("es_faster_rcnn_r50_fpn_straug_3x_robustodgenod.py" "es_faster_rcnn_r50_fpn_straug_3x_robustodood.py")
#CHECKPOINTS=("work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth")
#CONFIGS=("atss_r50_fpn_straug_3x_coco.py" "atss_r50_fpn_straug_3x_obj365ood.py" "atss_r50_fpn_straug_3x_svhnood.py" "atss_r50_fpn_straug_3x_inatood.py" "rs_faster_rcnn_r50_fpn_straug_3x_coco.py" "rs_faster_rcnn_r50_fpn_straug_3x_obj365ood.py" "rs_faster_rcnn_r50_fpn_straug_3x_svhnood.py" "rs_faster_rcnn_r50_fpn_straug_3x_inatood.py" "faster_rcnn_r50_fpn_straug_3x_coco.py" "faster_rcnn_r50_fpn_straug_3x_obj365ood.py" "faster_rcnn_r50_fpn_straug_3x_svhnood.py" "faster_rcnn_r50_fpn_straug_3x_inatood.py" "deformable_detr_r50_16x2_50e_coco.py" "deformable_detr_r50_16x2_50e_obj365ood.py" "deformable_detr_r50_16x2_50e_svhnood.py" "deformable_detr_r50_16x2_50e_inatood.py" "nll_faster_rcnn_r50_fpn_straug_3x_coco.py"  "es_faster_rcnn_r50_fpn_straug_3x_coco.py")
#CHECKPOINTS=("work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/rs_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/rs_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/rs_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/rs_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/nll_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth")

# CONFIGS=("nll_faster_rcnn_r50_fpn_straug_3x_obj365ood.py" "nll_faster_rcnn_r50_fpn_straug_3x_svhnood.py" "nll_faster_rcnn_r50_fpn_straug_3x_inatood.py"  "es_faster_rcnn_r50_fpn_straug_3x_obj365ood.py" "es_faster_rcnn_r50_fpn_straug_3x_svhnood.py" "es_faster_rcnn_r50_fpn_straug_3x_inatood.py")
# CHECKPOINTS=("work_dirs/nll_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/nll_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/nll_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth")

#CONFIGS=("atss_noaux_r50_fpn_straug_3x_coco.py" "atss_noaux_r50_fpn_straug_3x_robustodgenod.py")
#CHECKPOINTS=("work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth")

#CONFIGS=("deformable_detr_r50_16x2_50e_svhndigitsood.py" "atss_r50_fpn_straug_3x_svhndigitsood.py")
#CHECKPOINTS=("work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth")

#CONFIGS=("atss_r50_fpn_1x_cocowithgt.py" "atss_r50_fpn_1x_cocowithnogt.py" "faster_rcnn_r50_fpn_1x_cocowithgt.py" "faster_rcnn_r50_fpn_1x_cocowithnogt.py")
#CHECKPOINTS=("work_dirs/atss_r50_fpn_1x_coco/epoch_12.pth" "work_dirs/atss_r50_fpn_1x_coco/epoch_12.pth" "work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth" "work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth")

# CONFIGS=("atss_r50_fpn_1x_robustodgenod.py" "atss_r50_fpn_1x_robustodood.py")
# CHECKPOINTS=("work_dirs/atss_r50_fpn_1x_coco/epoch_12.pth" "work_dirs/atss_r50_fpn_1x_coco/epoch_12.pth")
#
# CONFIGS=("faster_rcnn_r50_fpn_1x_coco.py" "faster_rcnn_r50_fpn_1x_robustodgenod.py" "faster_rcnn_r50_fpn_1x_robustodood.py")
# CHECKPOINTS=("work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth" "work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth" "work_dirs/faster_rcnn_r50_fpn_1x_coco/epoch_12.pth")

#CONFIGS=("faster_rcnn_r50_fpn_straug_3x_robustodavod_ood.py" "atss_r50_fpn_straug_3x_robustodavod_ood.py")
#CHECKPOINTS=("work_dirs/faster_rcnn_r50_fpn_straug_3x_nuimages/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_nuimages/epoch_36.pth" )

#CONFIGS=("rs_faster_rcnn_r50_fpn_straug_3x_cocoremoveobjects.py" "deformable_detr_r50_16x2_50e_cocoremoveobjects.py" "atss_r50_fpn_straug_3x_cocoremoveobjects.py" "atss_r50_fpn_straug_3x_robustodavod_ood.py")
#CHECKPOINTS=("work_dirs/rs_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/deformable_detr_r50_16x2_50e_coco/epoch_50.pth" "work_dirs/atss_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_nuimages/epoch_36.pth")

#CONFIGS=("faster_rcnn_r50_fpn_straug_3x_nuimages.py" "atss_r50_fpn_straug_3x_nuimages.py")
#CHECKPOINTS=("work_dirs/faster_rcnn_r50_fpn_straug_3x_nuimages/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_nuimages/epoch_36.pth" )

#CONFIGS=("faster_rcnn_r50_fpn_straug_3x_nuimagesremoveobjects.py" "atss_r50_fpn_straug_3x_nuimagesremoveobjects.py")
#CHECKPOINTS=("work_dirs/faster_rcnn_r50_fpn_straug_3x_nuimages/epoch_36.pth" "work_dirs/atss_r50_fpn_straug_3x_nuimages/epoch_36.pth")

#CONFIGS=("faster_rcnn_r50_fpn_3x_coco.py" "faster_rcnn_r50_fpn_3x_cocoremoveobjects.py" "faster_rcnn_r50_fpn_3x_cocowithgt.py" "faster_rcnn_r50_fpn_3x_robustodgenod.py" "faster_rcnn_r50_fpn_3x_robustodood.py")
#CHECKPOINTS=("work_dirs/faster_rcnn_r50_fpn_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_3x_coco/epoch_36.pth" "work_dirs/faster_rcnn_r50_fpn_3x_coco/epoch_36.pth")

CONFIGS=("faster_rcnn_r50_fpn_3x_robustodgenod.py")
CHECKPOINTS=("work_dirs/faster_rcnn_r50_fpn_3x_coco/epoch_36.pth")

#CONFIGS=("es_faster_rcnn_r50_fpn_straug_3x_cocowithgt.py" "es_faster_rcnn_r50_fpn_straug_3x_cocoremoveobjects.py")
#CHECKPOINTS=("work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth" "work_dirs/es_prob_faster_rcnn_r50_fpn_straug_3x_coco/epoch_36.pth")

GPUS=2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

for i in "${!CONFIGS[@]}"; do
  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
	python -m torch.distributed.launch \
    		--nnodes=$NNODES \
    		--node_rank=$NODE_RANK \
	    	--master_addr=$MASTER_ADDR \
	    	--nproc_per_node=$GPUS \
	    	--master_port=$PORT \
	    	$(dirname "$0")/analysis_tools/benchmark_robustness.py \
	    	"$CONFIG_DIR${CONFIGS[i]}"\
	    	"${CHECKPOINTS[i]}" \
	    	--eval bbox \
	    	--out "detections/${CONFIGS[i]:0:-3}.pkl" \
	    	--launcher pytorch \
	    	${@:4} \
	    	>> "results/${CONFIGS[i]:0:-3}.txt"
	done