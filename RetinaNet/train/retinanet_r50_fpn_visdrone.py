USE_MMDET = True
_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    # '../_base_/datasets/visdrone_challenge_det_original.py', # cocodataset,
    '../_base_/datasets/visdrone_challenge_det_original_augment.py', # cocodataset
    # '../_base_/datasets/visdrone_challenge_det.py', # visdronechallengedataset
    '../_base_/default_runtime.py'
]
model = dict(
    detector=dict(
        bbox_head=dict(num_classes=5),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  
            '~/sia/mmtracking/checkpoints/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth' # https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/README.md
            # '~/sia/mmtracking/tutorial_exps/retina_8b_50e_adamw/epoch_30.pth'
        )))
checkpoint_config = dict(interval=1)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 16