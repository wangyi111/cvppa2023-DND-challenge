import torch
import os

'''
def get_model_from_sd(state_dict, base_model):
    feature_dim = state_dict['classification_head.weight'].shape[1]
    num_classes = state_dict['classification_head.weight'].shape[0]
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()
    model.load_state_dict(state_dict)
    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    return torch.nn.DataParallel(model,  device_ids=devices)
'''
### uniform soup
'''
model_paths = [
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_wr2021_fold1-896px_pre2020_cutmix/epoch_44.pth",
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_wr2021_fold2-896px_pre2020_cutmix/epoch_37.pth",
  #"swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_wr2021_fold3-896px_pre2020_cutmix/epoch_39.pth",
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_wr2021_fold4-896px_pre2020_cutmix/epoch_46.pth",
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_wr2021_fold5-896px_pre2020_cutmix/epoch_44.pth"
]
'''
model_paths = [
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_ww2020_fold1-896px_pre2021_cutmix/epoch_47.pth",
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_ww2020_fold2-896px_pre2021_cutmix/epoch_41.pth",
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_ww2020_fold3-896px_pre2021_cutmix/epoch_48.pth",
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_ww2020_fold4-896px_pre2021_cutmix/epoch_29.pth",
  "swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_ww2020_fold5-896px_pre2021_cutmix/epoch_46.pth"
]


NUM_MODELS = len(model_paths)

for j, model_path in enumerate(model_paths):

    print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

    assert os.path.exists(model_path)
    ckpt = torch.load(model_path)
    state_dict = ckpt['state_dict']
    if j == 0:
        uniform_soup = {k : v * (1./NUM_MODELS) for k, v in state_dict.items()}
    else:
        uniform_soup = {k : v * (1./NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

ckpt_out = {'state_dict':uniform_soup}
#torch.save(ckpt_out,'swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_wr2021-896px/5fold_uniform_soup.pth')        
torch.save(ckpt_out,'swinv2-base-w16_in21k-pre_4xb4_cvppa2023_cls_ww2020-896px/5fold_uniform_soup.pth')    
