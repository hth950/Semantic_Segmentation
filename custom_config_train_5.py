norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            '/home/mmsegmentation/checkpoints/swin_base_patch4_window12_384_22k.pth'
        )),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=32,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,avg_non_ignore=True,
            # class_weight=[0.1417, 0.3514, 4.7055, 8.606, 576.0494, 17857.5312, 29.0366, 87.1099, 23.3432, 10.969, 32.9475, 
            # 5.3514, 40.4017, 0.9936, 0.6036, 0.7028, 0.2773, 11.0368, 17.2537, 0.5324, 
            # 0.2011, 0.8016, 12.0172, 17857.5312, 4.8121, 44.3115, 25.9935, 14.7339, 0.3753, 0.5741, 17857.5312, 1.7224]
            )),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=32,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,avg_non_ignore=True,
            # class_weight=[0.1417, 0.3514, 4.7055, 8.606, 576.0494, 17857.5312, 29.0366, 87.1099, 23.3432, 10.969, 32.9475, 
            # 5.3514, 40.4017, 0.9936, 0.6036, 0.7028, 0.2773, 11.0368, 17.2537, 0.5324, 
            # 0.2011, 0.8016, 12.0172, 17857.5312, 4.8121, 44.3115, 25.9935, 14.7339, 0.3753, 0.5741, 17857.5312, 1.7224]
            )),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CustomDataset'
data_root = '/data/36-4/'
img_norm_cfg = dict(
    mean=[112.71, 108.37, 106.84], std=[49.51, 43.76, 43.01], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[112.71, 108.37, 106.84],
        std=[49.51, 43.76, 43.01],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[112.71, 108.37, 106.84],
                std=[49.51, 43.76, 43.01],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='CustomDataset',
        data_root='/data/36-4/',
        img_dir='/data/36-4/img_dir/',
        ann_dir='/data/36-4/ann_dir/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=True),
            dict(type='Resize', img_scale=(512, 512)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[112.71, 108.37, 106.84],
                std=[49.51, 43.76, 43.01],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split='splits/train.txt'),
    val=dict(
        type='CustomDataset',
        data_root='/data/36-4/',
        img_dir='/data/36-4/img_val/',
        ann_dir='/data/36-4/ann_val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[112.71, 108.37, 106.84],
                        std=[49.51, 43.76, 43.01],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/val.txt'),
    test=dict(
        type='CustomDataset',
        data_root='/data/36-4/',
        img_dir='/data/36-4/img_test/',
        ann_dir='/data/36-4/ann_test/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[112.71, 108.37, 106.84],
                        std=[49.51, 43.76, 43.01],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/test.txt'))
log_config = dict(
    interval=200, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/mmsegmentation/checkpoints/swin_base_patch4_window12_384_22k.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=48000)
checkpoint_config = dict(by_epoch=False, interval=1600)
evaluation = dict(interval=1600, metric='mIoU', pre_eval=True)
checkpoint_file = '/home/mmsegmentation/checkpoints/swin_base_patch4_window12_384_22k.pth'
work_dir = '/data/result/36-4/test/'
seed = 42
gpu_ids = range(0, 1)
device = 'cuda'
