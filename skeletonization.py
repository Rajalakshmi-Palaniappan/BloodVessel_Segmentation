#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kimimaro
import numpy as np


# In[ ]:


#convert tiff to numpy


# In[2]:


import PIL


# In[3]:


from PIL import Image


# In[ ]:


img = Image.open('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/1683_raw_tiff_(red)0596.tif')


# In[ ]:


img.show()


# In[ ]:


img_arr = np.array(img)


# In[ ]:


print(img_arr.shape)


# In[ ]:


np.save('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/numpyfile.npy', img_arr)


# In[ ]:





# In[ ]:


labels = np.load("/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/numpyfile.npy")


# In[ ]:


skels = kimimaro.skeletonize(
  Labels_arr, 
  teasar_params={
    'scale': 3,
    'const': 4, # physical units
    'pdrf_exponent': 4,
    'pdrf_scale': 100000,
    #'soma_detection_threshold': 1100, # physical units
    #'soma_acceptance_threshold': 3500, # physical units
    #'soma_invalidation_scale': 1.0,
    #'soma_invalidation_const': 300, # physical units
    'max_paths': 1000, # default None
  },
  # object_ids=[ ... ], # process only the specified labels
  # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
  # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
  dust_threshold=0, # skip connected components with fewer than this many voxels
  anisotropy=(1,1,1), # default True
  fix_branching=True, # default True
  fix_borders=True, # default True
  #fill_holes=False, # default False
  #fix_avocados=False, # default False
  progress=True, # default False, show progress bar
  parallel=1, # <= 0 all cpu, 1 single process, 2+ multiprocess
  parallel_chunk_size=50, # how many skeletons to process before updating progress bar
)


# In[ ]:


34708---> no of labels
 


# In[5]:


from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import tifffile 
import napari


# In[6]:


binary_data = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_binary_result.tif')


# In[ ]:


print(binary_data.shape)


# In[ ]:


binary_data_arr = np.array(binary_data)


# In[ ]:


print(binary_data_arr.shape)


# In[ ]:


# Connected component labelling using skimage


# In[ ]:


Labels = measure.label(binary_data_arr, background=0)


# In[ ]:


tifffile.imsave('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/Labels.tif', Labels)


# In[ ]:


viewer = napari.Viewer()


# In[ ]:


Labels = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/Labels.tif')


# In[ ]:


Labels_arr = np.array(Labels)


# In[ ]:


print(Labels_arr.shape)


# In[7]:


Labels = measure.label(binary_data, background=0)


# In[ ]:


# the output would be 32 bit. Do not downsample it as the number of labels is more to fit into 8bit or 16bit image


# In[9]:


tifffile.imwrite('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/Labels_no_array_input.tif', Labels)


# In[ ]:




