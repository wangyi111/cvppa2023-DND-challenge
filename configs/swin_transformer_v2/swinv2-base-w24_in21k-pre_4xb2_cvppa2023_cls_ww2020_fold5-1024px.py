_base_ = [
    '../_base_/models/swin_transformer_v2/base_384.py',
    '../_base_/datasets/cvppa2023_cls_ww2020_fold5.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

train_dataloader = dict(
    batch_size=2,)
val_dataloader = dict(
    batch_size=2,)


model = dict(
    type='ImageClassifier',
    backbone=dict(
        img_size=1024,
        window_size=[24, 24, 24, 12],
        drop_path_rate=0.2,
        #pretrained_window_sizes=[12, 12, 12, 6],
        pretrained_window_sizes=[24, 24, 24, 12],
        init_cfg=dict(type='Pretrained', checkpoint="/p/project/hai_ssl4eo/wang_yi/mmpretrain/work_dirs/swinv2-base-w24_in21k-pre_4xb2_cvppa2023_cls_ww2020_wr2021-1024px/epoch_26.pth", prefix='backbone.')
        ),
    head=dict(
        type='LinearClsHead',
        num_classes=7,
        in_channels=1024,
        #init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
        #loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        #topk=(1),
        #cal_acc=False
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
    )
    
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=10,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-7, by_epoch=True, begin=10)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=8)