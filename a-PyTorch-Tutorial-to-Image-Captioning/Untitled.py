
# coding: utf-8

# In[1]:


import torch
import torchvision
print(f'Torch version: {torch.__version__}')


# In[2]:


import numpy as np
from torchvision import transforms


# In[3]:


import h5py
import tqdm


# In[4]:


import os
import re
from PIL import Image


# # Check h5 file format

# In[5]:


import sys
sys.path.append('../visdial-challenge-starter-pytorch/')
sys.path.append('../visdial-challenge-starter-pytorch/visdialch/')


# In[6]:


from data.readers import ImageFeaturesHdfReader


# In[7]:


val_reader = ImageFeaturesHdfReader(
    '../visdial-challenge-starter-pytorch/data/features_faster_rcnn_x101_val.h5')
test_reader = ImageFeaturesHdfReader(
    '../visdial-challenge-starter-pytorch/data/features_faster_rcnn_x101_test.h5')
train_reader = ImageFeaturesHdfReader(
    '../visdial-challenge-starter-pytorch/data/features_faster_rcnn_x101_train.h5')


# In[8]:


print(f'Train has {len(train_reader)} samples')
print(f'Valid has {len(val_reader)} samples')
print(f'Test has {len(test_reader)} samples')


# In[9]:


len(set([id for reader in [train_reader, val_reader, test_reader] for id in reader.image_id_list ]))


# ### VGG features

# In[31]:


import h5py
import os


# In[84]:


file = h5py.File('../visdial-challenge-starter-pytorch/data/data_img_vgg16_pool5.h5','r')


# In[85]:


print(file['images_train'].shape, len(os.listdir('../visdial-challenge-starter-pytorch/data/images/train2014/')))
print(file['images_val'].shape, len(os.listdir('../visdial-challenge-starter-pytorch/data/images/val2014/')))
print(file['images_test'].shape, len(os.listdir('../visdial-challenge-starter-pytorch/data/images/test2014/')))
print(file['images_test'].shape, len(os.listdir('../visdial-challenge-starter-pytorch/data/images/VisualDialog_test2018/')))


# In[96]:


82783+40504


# In[95]:


len(train_reader)


# In[ ]:


print(file['images_train'].shape, 
      len(os.listdir('../visdial-challenge-starter-pytorch/data/images/train2014/')),
      len(os.listdir('../visdial-challenge-starter-pytorch/data/images/val2014/'))
     )
print(file['images_test'].shape, len(os.listdir('../visdial-challenge-starter-pytorch/data/images/VisualDialog_test2018/')))


# In[83]:


(
    len(os.listdir('../visdial-challenge-starter-pytorch/data/images/train2014/'))+
    len(os.listdir('../visdial-challenge-starter-pytorch/data/images/val2014/'))+
    len(os.listdir('../visdial-challenge-starter-pytorch/data/images/VisualDialog_val2018/'))
)


# In[65]:


#list(file.keys())


# In[7]:


#reader.__dict__.keys()


# In[10]:


#reader[reader.keys()[0]].shape


# # Load images

# In[10]:


splits = (
        'train2014',
        'test2014',
        'val2014'
        'VisualDialog_test2018',
        'VisualDialog_val2018',
)


# In[11]:


available_image_ids = {
    float(re.findall(r'([\d]+)\.jpg', f)[0])
    :
    os.path.join(f'../visdial-challenge-starter-pytorch/data/images/{split}/{f}')
    for split in (
        'train2014',
        'test2014',
        'val2014',
        'VisualDialog_test2018',
        'VisualDialog_val2018',
    )
    for f in os.listdir(f'../visdial-challenge-starter-pytorch/data/images/{split}/')
    }
len(available_image_ids)


# In[12]:


img = Image.open(available_image_ids[list(available_image_ids.keys())[40]])


# In[13]:


image_ids_with_features = [
    key
    for reader in [
        train_reader, 
        #test_reader,
        #val_reader,
    ]
    for key in reader.keys()
]
len(image_ids_with_features)


# # Load Model

# In[14]:


checkpoint_path = '../visdial-challenge-starter-pytorch/data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'


# In[15]:


model = torch.load(checkpoint_path)


# In[16]:


img_encoder = model['encoder']


# In[17]:


device = torch.device('cuda')


# In[18]:


transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# In[19]:


def get_image_features(img_id):
    img_path = available_image_ids.get(img_id)
    image = Image.open(img_path)
    
    # 1-channel to 3-channel
    if image.mode != 'RGB':
        w, h = image.size
        ima = Image.new('RGB', (w,h))
        data = zip(image.getdata(), image.getdata(), image.getdata())
        ima.putdata(list(data))
        image = ima
        
    features_tensor = img_encoder.resnet(transform(image).to(device).unsqueeze(0))
    return features_tensor.squeeze().detach().cpu().numpy()


# In[20]:


def get_images_features(img_ids):
    images_tensors = list()
    for img_id in img_ids:
        img_path = available_image_ids.get(img_id)
        image = Image.open(img_path)

        # 1-channel to 3-channel
        if image.mode != 'RGB':
            w, h = image.size
            ima = Image.new('RGB', (w,h))
            data = zip(image.getdata(), image.getdata(), image.getdata())
            ima.putdata(list(data))
            image = ima
        images_tensors.append(transform(image).to(device).unsqueeze(0))
    images_tensor = torch.cat(images_tensors, dim=0)
    return img_encoder.resnet(images_tensor).squeeze().detach().cpu().numpy()


# In[21]:


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[ ]:


with h5py.File('../visdial-challenge-starter-pytorch/data/features_thomas_train.h5', mode='w') as h5py_file:
    h5py_file.create_dataset(name='features', shape=(len(train_reader), 2048, 8, 8))
    h5py_file.create_dataset(name='image_id', shape=(len(train_reader),))
    h5py_file.attrs.create(name='split', data=np.string_('train'))
    
    ix = 0
    
    batch_size=16
    total=int(np.ceil(len(train_reader.keys())/batch_size))
    for img_ids in tqdm.tqdm(chunks(train_reader.keys(), batch_size), total=total):
        features = get_images_features(img_ids)
        for img_id, img_features in zip(img_ids, features):
            h5py_file['features'][ix] = img_features
            h5py_file['image_id'][ix] = img_id
            ix += 1


# In[ ]:


with h5py.File('../visdial-challenge-starter-pytorch/data/features_thomas_test.h5', mode='w') as h5py_file:
    h5py_file.create_dataset(name='features', shape=(len(test_reader), 2048, 8, 8))
    h5py_file.create_dataset(name='image_id', shape=(len(test_reader),))
    h5py_file.attrs.create(name='split', data=np.string_('test'))
    
    ix = 0
    
    batch_size=16
    total=int(np.ceil(len(test_reader.keys())/batch_size))
    for img_ids in tqdm.tqdm(chunks(test_reader.keys(), batch_size), total=total):
        features = get_images_features(img_ids)
        for img_id, img_features in zip(img_ids, features):
            h5py_file['features'][ix] = img_features
            h5py_file['image_id'][ix] = img_id
            ix += 1


# In[ ]:


with h5py.File('../visdial-challenge-starter-pytorch/data/features_thomas_val.h5', mode='w') as h5py_file:
    h5py_file.create_dataset(name='features', shape=(len(val_reader), 2048, 8, 8))
    h5py_file.create_dataset(name='image_id', shape=(len(val_reader),))
    h5py_file.attrs.create(name='split', data=np.string_('val'))
    
    ix = 0
    
    batch_size=16
    total=int(np.ceil(len(val_reader.keys())/batch_size))
    for img_ids in tqdm.tqdm(chunks(val_reader.keys(), batch_size), total=total):
        features = get_images_features(img_ids)
        for img_id, img_features in zip(img_ids, features):
            h5py_file['features'][ix] = img_features
            h5py_file['image_id'][ix] = img_id
            ix += 1


# ### Test if correct

# In[25]:


reader = ImageFeaturesHdfReader('../visdial-challenge-starter-pytorch/data/features_thomas_val.h5')


# In[29]:


val_reader.keys()[:10]


# # --
