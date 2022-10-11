_base_ = [
    './upernet_swin_base_patch4_window12_512x512_160k_ade20k_'
    'pretrain_384x384_1K.py'
]
checkpoint_file = '/home/mmsegmentation/checkpoints/swin_base_patch4_window12_384_22k.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)))
