_base_ = [
    '../../_base_/models/retinanet_r50_fpn.py',
    '../../_base_/datasets/visdrone_challenge.py', '../../_base_/default_runtime.py'
]

model = dict(
    type='ByteTrack',
    detector=dict(
        # input_size=img_scale,
        # random_size_range=(18, 32),
        bbox_head=dict(num_classes=5),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '~/mmtracking/tutorial_exps/retina_byte_16b_2e_mixup/epoch_2.pth',
            # '~/mmtracking/tutorial_exps/retina_byte_16b_2e_notrunc_mosaic_mixup/epoch_2.pth',
            # '~/mmtracking/tutorial_exps/retina_byte_16b_2e_mosaic_mixup/epoch_2.pth',
            # '~/mmtracking/tutorial_exps/retina_byte_32b_2e_noaug/epoch_2.pth',
            # '~/mmtracking/tutorial_exps/retina_byte_32b_2e_notrunc_noaug/epoch_2.pth',
            
                            )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))
