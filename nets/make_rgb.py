# %%
import os
import numpy as np
import h5py
from PIL import Image
# %%
def create_h5_dataset(image_path, output_path, img_size=(3840, 2160)):
    image_paths = []
    for img_name in os.listdir(image_path):
        img_path = os.path.join(image_path, img_name)
        image_paths.append(img_path)
    
    nums = 0
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        if img.height > img.width:
            img = img.transpose(Image.ROTATE_90)
        if img.height >= img_size[1] and img.width >= img_size[0]:
            nums += 1
            print(f'img id: {i}, Total: {nums}')
            
            left, upper = (img.width - img_size[0]) // 2, (img.height - img_size[1]) // 2
            img = img.crop([left, upper, left + img_size[0], upper + img_size[1]])
            img_array = np.array(img)    
            
            with h5py.File(f'{output_path}/{str(i+1)}.h5', 'w') as hf:
                img_dset = hf.create_dataset(
                    'img', 
                    shape=(img_size[1], img_size[0], 3),
                    dtype=np.uint8
                )
                img_dset[...] = img_array
    print('valid nums:', nums)
# %%
create_h5_dataset(
    image_path='xxxxxxxxx',
    output_path='xxxxxxxxx',
    img_size=(3600, 2700)
)
# %%
create_h5_dataset(
    image_path='xxxxxxxxx',
    output_path='xxxxxxxxx',
    img_size=(4800, 3600)
)
# %%
