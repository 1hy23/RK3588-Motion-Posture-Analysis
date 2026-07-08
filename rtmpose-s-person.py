_base_ = ['mmpose::_base_/default_runtime.py']

# 数据集类型及路径
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/root/autodl-tmp/coco'

# 人体关键点检测数据集-元数据
# 人体关键点检测数据集-元数据
dataset_info = dict(
    dataset_name='coco',
    paper_info=dict(
        author='JYUAN',
        title='Squat Keypoints Detection',
        container='OpenMMLab',
        year='2025',
    ),
    keypoint_info={
        0: dict(name='nose', id=0, color=[255, 0, 0], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[0, 255, 0], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[0, 0, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[255, 255, 0], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[0, 255, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[255, 0, 255], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[128, 0, 0], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 128, 0], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[0, 0, 128], type='upper', swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[128, 128, 0], type='upper', swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[0, 128, 128], type='upper', swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[128, 0, 128], type='lower', swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[64, 0, 0], type='lower', swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 64, 0], type='lower', swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[0, 0, 64], type='lower', swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[64, 64, 0], type='lower', swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[0, 64, 64], type='lower', swap='left_ankle')
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[100, 150, 200]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[200, 100, 150]),
        2: dict(link=('right_ankle', 'right_knee'), id=2, color=[150, 120, 100]),
        3: dict(link=('right_knee', 'right_hip'), id=3, color=[100, 200, 150]),
        4: dict(link=('left_hip', 'right_hip'), id=4, color=[150, 100, 200]),
        5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[200, 150, 100]),
        6: dict(link=('right_shoulder', 'right_hip'), id=6, color=[100, 150, 100]),
        7: dict(link=('left_shoulder', 'right_shoulder'), id=7, color=[150, 200, 150]),
        8: dict(link=('left_shoulder', 'left_elbow'), id=8, color=[200, 100, 100]),
        9: dict(link=('right_shoulder', 'right_elbow'), id=9, color=[100, 200, 100]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[150, 100, 100]),
        11: dict(link=('right_elbow', 'right_wrist'), id=11, color=[100, 150, 150]),
        12: dict(link=('left_eye', 'right_eye'), id=12, color=[200, 200, 100]),
        13: dict(link=('nose', 'left_eye'), id=13, color=[100, 100, 200]),
        14: dict(link=('nose', 'right_eye'), id=14, color=[150, 150, 150]),
        15: dict(link=('left_eye', 'left_ear'), id=15, color=[200, 150, 150]),
        16: dict(link=('right_eye', 'right_ear'), id=16, color=[150, 200, 200]),
        17: dict(link=('left_ear', 'left_shoulder'), id=17, color=[100, 100, 100]),
        18: dict(link=('right_ear', 'right_shoulder'), id=18, color=[200, 200, 200])
    },
    joint_weights=[1.0] * 17,  # 17个关键点的权重
    sigmas=[0.025] * 17  # 17个关键点的标准差
)

# 获取关键点个数
NUM_KEYPOINTS = len(dataset_info['keypoint_info'])
dataset_info['joint_weights'] = [1.0] * NUM_KEYPOINTS
dataset_info['sigmas'] = [0.025] * NUM_KEYPOINTS

# 训练超参数
max_epochs = 500 # 训练 epoch 总数
val_interval = 5 # 每隔多少个 epoch 保存一次权重文件
train_cfg = {'max_epochs': max_epochs, 'val_interval': val_interval}
train_batch_size = 64 # 增大批量大小以充分利用24GB显存
val_batch_size = 34 # 增大验证批量大小
stage2_num_epochs = 50 # 启用第二阶段训练以提高精度
base_lr = 8e-3 # 因为批量大小增加，相应增加学习率
randomness = dict(seed=21)

# 优化器
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 学习率
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=20),
    dict(
        # use cosine lr from 210 to 420 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=1024)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(12, 12),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# 不同输入图像尺寸的参数搭配
# input_size=(256, 256),
# sigma=(12, 12)
# in_featuremap_size=(8, 8)
# input_size可以换成 256、384、512、1024，三个参数等比例缩放
# sigma 表示关键点一维高斯分布的标准差，越大越容易学习，但精度上限会降低，越小越严格，对于人体、人脸等高精度场景，可以调小，RTMPose 原始论文中为 5.66

# 不同模型的 config： https://github.com/open-mmlab/mmpose/tree/dev-1.x/projects/rtmpose/rtmpose/body_2d_keypoint

# 模型：RTMPose-S
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-s_imagenet_600e-ea671761.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=512,
        out_channels=NUM_KEYPOINTS,
        input_size=codec['input_size'],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# 模型：RTMPose-M
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         _scope_='mmdet',
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
#         deepen_factor=0.67,
#         widen_factor=0.75,
#         out_indices=(4, ),
#         channel_attention=True,
#         norm_cfg=dict(type='SyncBN'),
#         act_cfg=dict(type='SiLU'),
#         init_cfg=dict(
#             type='Pretrained',
#             prefix='backbone.',
#             checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'
#         )),
#     head=dict(
#         type='RTMCCHead',
#         in_channels=768,
#         out_channels=NUM_KEYPOINTS,
#         input_size=codec['input_size'],
#         in_featuremap_size=(8, 8),
#         simcc_split_ratio=codec['simcc_split_ratio'],
#         final_layer_kernel_size=7,
#         gau_cfg=dict(
#             hidden_dims=256,
#             s=128,
#             expansion_factor=2,
#             dropout_rate=0.,
#             drop_path=0.,
#             act_fn='SiLU',
#             use_rel_bias=False,
#             pos_enc=False),
#         loss=dict(
#             type='KLDiscretLoss',
#             use_target_weight=True,
#             beta=10.,
#             label_softmax=True),
#         decoder=codec),
#     test_cfg=dict(flip_test=True))

## 模型：RTMPose-L
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         _scope_='mmdet',
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
#         deepen_factor=1.,
#         widen_factor=1.,
#         out_indices=(4, ),
#         channel_attention=True,
#         norm_cfg=dict(type='SyncBN'),
#         act_cfg=dict(type='SiLU'),
#         init_cfg=dict(
#             type='Pretrained',
#             prefix='backbone.',
#             checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'
#         )),
#     head=dict(
#         type='RTMCCHead',
#         in_channels=1024,
#         out_channels=NUM_KEYPOINTS,
#         input_size=codec['input_size'],
#         in_featuremap_size=(8, 8),
#         simcc_split_ratio=codec['simcc_split_ratio'],
#         final_layer_kernel_size=7,
#         gau_cfg=dict(
#             hidden_dims=256,
#             s=128,
#             expansion_factor=2,
#             dropout_rate=0.,
#             drop_path=0.,
#             act_fn='SiLU',
#             use_rel_bias=False,
#             pos_enc=False),
#         loss=dict(
#             type='KLDiscretLoss',
#             use_target_weight=True,
#             beta=10.,
#             label_softmax=True),
#         decoder=codec),
#     test_cfg=dict(flip_test=True))

# # 模型：RTMPose-X
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True),
#     backbone=dict(
#         _scope_='mmdet',
#         type='CSPNeXt',
#         arch='P5',
#         expand_ratio=0.5,
#         deepen_factor=1.33,
#         widen_factor=1.25,
#         out_indices=(4, ),
#         channel_attention=True,
#         norm_cfg=dict(type='SyncBN'),
#         act_cfg=dict(type='SiLU'),
#         init_cfg=dict(
#             type='Pretrained',
#             prefix='backbone.',
#             checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-x_8xb256-rsb-a1-600e_in1k-b3f78edd.pth'
#         )),
#     head=dict(
#         type='RTMCCHead',
#         in_channels=1280,
#         out_channels=NUM_KEYPOINTS,
#         input_size=codec['input_size'],
#         in_featuremap_size=(32, 32),
#         simcc_split_ratio=codec['simcc_split_ratio'],
#         final_layer_kernel_size=7,
#         gau_cfg=dict(
#             hidden_dims=256,
#             s=128,
#             expansion_factor=2,
#             dropout_rate=0.,
#             drop_path=0.,
#             act_fn='SiLU',
#             use_rel_bias=False,
#             pos_enc=False),
#         loss=dict(
#             type='KLDiscretLoss',
#             use_target_weight=True,
#             beta=10.,
#             label_softmax=True),
#         decoder=codec),
#     test_cfg=dict(flip_test=True))

backend_args = dict(backend='local')

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    # dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.8, 1.2], rotate_factor=30),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='ChannelShuffle', p=0.5),
            dict(type='CLAHE', p=0.5),
            # dict(type='Downscale', scale_min=0.7, scale_max=0.9, p=0.2),
            dict(type='ColorJitter', p=0.5),
            dict(
                type='CoarseDropout',
                max_holes=4,
                max_height=0.3,
                max_width=0.3,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='annotations/train_coco.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dataset_info,
        data_mode=data_mode,
        ann_file='annotations/val_coco.json',
        data_prefix=dict(img='val/'),
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

default_hooks = {
    'checkpoint': {'save_best': 'PCK','rule': 'greater','max_keep_ckpts': 2},
    'logger': {'interval': 1}
}

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = [
    dict(type='CocoMetric', ann_file=data_root + '/annotations/val_coco.json'),
    dict(type='PCKAccuracy'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 1])
]

test_evaluator = val_evaluator
