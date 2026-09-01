# %%
import os
import numpy as np
import h5py
import rawpy
import torch
import glob
from tqdm import tqdm
# %%
def find_nef_with_glob(root_folder):
    return glob.glob(
        os.path.join(root_folder, '**/*.nef'), 
        recursive=True
    )
# %%
raw_data_path = r'xxxxxxxxx'
nef_list = find_nef_with_glob(raw_data_path)
# %%
train_count = 0
train_img_size = (2160, 3840) # [H, W]: 2867
train_img_save_path = r'xxxxxxxxx'
valid_count = 0
valid_img_size = (2700, 4800) # [H, W]: 258
valid_img_save_path = r'xxxxxxxxx'
# %%
loop = tqdm(enumerate(nef_list), total=len(nef_list))
for i, img_path in loop:
    raw = rawpy.imread(img_path)
    if raw.color_desc == b'RGBG':
        raw_img = raw.raw_image_visible
        if raw_img.shape[0] <= train_img_size[0] or raw_img.shape[1] <= train_img_size[1]:
            continue
        
        # 1. read raw data
        raw_img = torch.tensor(raw_img, dtype=torch.float32) #! raw_img
        raw_color = torch.tensor(raw.raw_colors_visible, dtype=torch.uint8) #! raw_color
        black_level = raw.black_level_per_channel #! black_level            
        white_level = raw.white_level #! white_level
        # white balance
        raw_wb_matrix = raw.camera_whitebalance
        if raw_wb_matrix[0] == 0.0:
            raw_wb_matrix = raw.daylight_whitebalance
        if raw_wb_matrix[3] == 0.0:
            raw_wb_matrix[3] = raw_wb_matrix[1]
        # color matrix
        if raw.color_matrix[:3, :3].all():
            raw_color_matrix = raw.color_matrix
            raw_color_matrix = raw_color_matrix[:3, :3]
        else:
            xyz_srgb_matrix = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
                                            [-0.9692660, 1.8760108, 0.0415560],
                                            [0.0556434, -0.2040259, 1.0572252]], dtype=torch.float32) # 'D65'
            # xyz_srgb_matrix = torch.tensor([[3.1338561, -1.6168667, -0.4906146],
            #                                 [-0.9787684, 1.9161415, 0.0334540],
            #                                 [0.0719453, -0.2289914, 1.4052427]], dtype=torch.float32) # 'D50'
            xyz_srgb_matrix = xyz_srgb_matrix / torch.sum(xyz_srgb_matrix, dim=-1, keepdim=True)
            rgb_xyz_matrix = torch.linalg.inv(torch.tensor(raw.rgb_xyz_matrix[:3, :3], dtype=torch.float32))
            rgb_xyz_matrix = rgb_xyz_matrix / torch.sum(rgb_xyz_matrix, dim=-1, keepdim=True)
            raw_color_matrix = torch.einsum('ij,jk->ik', xyz_srgb_matrix, rgb_xyz_matrix).cpu().numpy()
        
        # 2. normalize
        black_level_mask = torch.zeros_like(raw_img)
        for i in range(len(black_level)):
            if torch.any(torch.eq(raw_color, i)):
                black_level_mask = black_level_mask.masked_fill(torch.eq(raw_color, i), black_level[i])
        raw_img = raw_img - black_level_mask
        raw_img = torch.div(raw_img, white_level - black_level_mask).clip(1e-8, 1.0) #! normalized_raw
            
        if train_count >= 3600 and raw_img.shape[0] >= valid_img_size[0] and raw_img.shape[1] >= valid_img_size[1]:
            left, upper = (raw_img.shape[0] - valid_img_size[0]) // 2, (raw_img.shape[1] - valid_img_size[1]) // 2
            
            save_raw_img = (raw_img[left:left+valid_img_size[0], upper:upper+valid_img_size[1]]).numpy()
            save_raw_color = (raw_color[left:left+valid_img_size[0], upper:upper+valid_img_size[1]]).numpy()
            raw_wb_matrix = np.array(raw_wb_matrix)
            raw_color_matrix = np.array(raw_color_matrix)
            
            with h5py.File(f'{valid_img_save_path}/{os.path.basename(img_path).split('.')[0]}.h5', 'w') as hf:
                img_dset = hf.create_dataset(
                    'raw_img',
                    shape=(valid_img_size[0], valid_img_size[1]),
                    dtype=np.float32,
                    data=save_raw_img
                )
                color_dset = hf.create_dataset(
                    'raw_color',
                    shape=(valid_img_size[0], valid_img_size[1]),
                    dtype=np.uint8,
                    data=save_raw_color
                )
                wb_dset = hf.create_dataset(
                    'raw_wb_matrix',
                    shape=(4, ),
                    dtype=np.float32,
                    data=raw_wb_matrix
                )
                color_matrix_dset = hf.create_dataset(
                    'raw_color_matrix',
                    shape=(3, 3),
                    dtype=np.float32,
                    data=raw_color_matrix
                )
            valid_count += 1
                
        else:
            left, upper = (raw_img.shape[0] - train_img_size[0]) // 2, (raw_img.shape[1] - train_img_size[1]) // 2

            save_raw_img = (raw_img[left:left+train_img_size[0], upper:upper+train_img_size[1]]).numpy()
            save_raw_color = (raw_color[left:left+train_img_size[0], upper:upper+train_img_size[1]]).numpy()
            raw_wb_matrix = np.array(raw_wb_matrix)
            raw_color_matrix = np.array(raw_color_matrix)
            
            with h5py.File(f'{train_img_save_path}/{os.path.basename(img_path).split(".")[0]}.h5', 'w') as hf:
                img_dset = hf.create_dataset(
                    'raw_img',
                    shape=(train_img_size[0], train_img_size[1]),
                    dtype=np.float32,
                    data=save_raw_img
                )
                color_dset = hf.create_dataset(
                    'raw_color',
                    shape=(train_img_size[0], train_img_size[1]),
                    dtype=np.uint8,
                    data=save_raw_color
                )
                wb_dset = hf.create_dataset(
                    'raw_wb_matrix',
                    shape=(4, ),
                    dtype=np.float32,
                    data=raw_wb_matrix
                )
                color_matrix_dset = hf.create_dataset(
                    'raw_color_matrix',
                    shape=(3, 3),
                    dtype=np.float32,
                    data=raw_color_matrix
                )
            train_count += 1
                
    loop.set_description(f'Train: {train_count}, Valid: {valid_count}')
# %%
