# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=7,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='Resize', scale=896),
    dict(type='RandomResizedCrop', scale=1024, crop_ratio_range=(0.4, 1.6), aspect_ratio_range=(0.8, 1.2)),
    #dict(type='CenterCrop', crop_size=896),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.01,
        max_area_ratio=0.1,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    #dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='Resize', scale=1024),
    #dict(type='CenterCrop', crop_size=896),
    dict(type='PackInputs'),
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/cvppa2023_cls/WW2020',
        ann_file='train_fold1.txt',
        with_label=True,
        classes=['unfertilized', '_PKCa', 'N_KCa', 'NP_Ca', 'NPK_', 'NPKCa', 'NPKCa+m+s'],
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/cvppa2023_cls/WW2020',
        ann_file='val_fold1.txt',
        with_label=True,
        classes=['unfertilized', '_PKCa', 'N_KCa', 'NP_Ca', 'NPK_', 'NPKCa', 'NPKCa+m+s'],
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/cvppa2023_cls/WW2020',
        ann_file='test_base.txt',
        data_prefix='images',
        with_label=True,
        classes=['unfertilized', '_PKCa', 'N_KCa', 'NP_Ca', 'NPK_', 'NPKCa', 'NPKCa+m+s'],
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = val_evaluator


