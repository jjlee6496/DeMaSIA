_base_ = [
    './yolox_test.py',
]

dataset_type = 'MOTChallengeDataset'
data_root = 'data/visdrone/'
work_dir = './work_dirs/yolox_notruc_mixup_4b_2e'

img_scale = (1920, 1080)  # weight, height
batch_size = 1

# some hyper parameters
# training settings
max_epochs = 2
num_last_epochs = 1
#interval = 2

backend_args = None

detector = _base_.model
detector.pop('data_preprocessor')
detector.bbox_head.update(dict(num_classes=5))
detector.test_cfg.nms.update(dict(iou_threshold=0.7))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'  # noqa: E501
)
del _base_.model

model = dict(
    type='ByteTrack',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        pad_size_divisor=32,
        # in bytetrack, we provide joint train detector and evaluate tracking
        # performance, use_det_processor means use independent detector
        # data_preprocessor. of course, you can train detector independently
        # like strongsort
        use_det_processor=True,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(576, 1024),
                size_divisor=32,
                interval=10)
        ]),
    detector=detector,
    tracker=dict(
        type='ByteTracker',
        motion=dict(type='KalmanFilter'),
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Resize',
        scale=img_scale,
        keep_ratio=True,
        clip_object_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadTrackAnnotations'),
        ]),
    dict(type='PackTrackInputs')
]

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=[
                dict(
                    type='CocoDataset',
                    data_root=data_root,
                    ann_file='no_truc/VisDrone2019-MOT-train_cocoformat.json',
                    data_prefix=dict(img=''),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    metainfo=dict(classes=('pedestrian','car','van','truck','bus')),
                    pipeline=[
                        dict(
                            type='LoadImageFromFile',
                            backend_args=_base_.backend_args),
                        dict(type='LoadAnnotations', with_bbox=True),
                    ]),
            ]),
        pipeline=train_pipeline))

val_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    # video_based
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type='TrackImgSampler'),  # image_based
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='no_truc/VisDrone2019-MOT-val_cocoformat.json',
        data_prefix=dict(img_path=''),
        test_mode=True,
        pipeline=test_pipeline))

"""
test_dataloader = val_dataloader
"""
test_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    # video_based
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type='TrackImgSampler'),  # image_based
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='no_truc/VisDrone2019-MOT-test-dev_cocoformat.json',
        data_prefix=dict(img_path=''),
        test_mode=True,
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
base_lr = 0.001 / 8 * batch_size
optim_wrapper = dict(optimizer=dict(lr=base_lr))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_begin=2,
    val_interval=2)

"""
# learning policy
param_scheduler = [
    dict(
        # use quadratic formula to warm up 1 epochs
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True),
    dict(
        # use cosine lr from 1 to 70 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=1,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        # use fixed lr during last 10 epochs
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]
"""

# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]


custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', 
                   interval=1, 
                    max_keep_ckpts=20),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer', )

# evaluator
val_evaluator = [dict(type='MOTChallengeMetric', 
#                      postprocess_tracklet_cfg=[dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)]
                        ),
                dict(type='CocoVideoMetric')]


test_evaluator = [dict(type='MOTChallengeMetric', 
#                      postprocess_tracklet_cfg=[dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)]
                        ),
                dict(type='CocoVideoMetric')]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (4 samples per GPU)
auto_scale_lr = dict(base_batch_size=1)

del detector
del _base_.tta_model
del _base_.tta_pipeline
del _base_.train_dataset
