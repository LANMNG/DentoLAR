"""Fold 4: same architecture, preprocessing and seed as the experiment."""
_base_ = ['./sapiens_teeth.py']
train_dataloader = dict(dataset=dict(datasets=[dict(type='CocoDataset', data_root='data/teeth', data_mode='topdown', ann_file='annotations/fold_4/annotations_train.json', data_prefix=dict(img='imagesTr_clean/'))]))
val_dataloader = dict(dataset=dict(datasets=[dict(type='CocoDataset', data_root='data/teeth', data_mode='topdown', ann_file='annotations/fold_4/annotations_val.json', data_prefix=dict(img='imagesTr_clean/'))]))
