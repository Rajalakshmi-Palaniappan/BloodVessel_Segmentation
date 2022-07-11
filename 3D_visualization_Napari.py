#!/usr/bin/env python
# coding: utf-8

# In[2]:


gui qt5


# In[3]:


import napari
import skimage
import sys
import os
import cv2
from napari_ome_zarr import napari_get_reader
import PIL
from PIL import Image
import tifffile
import numpy as np
from PIL import TiffImagePlugin
import zarr as z
import tifffile as tiff


# In[4]:


import apoc
import os
from skimage.io import imread
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt


# In[ ]:


napari.run()


# In[ ]:


input_directory = "/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec/"


# In[ ]:





# In[ ]:


sorted(os.listdir(input_directory))


# In[ ]:


output = '/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff/'


# In[ ]:


for filename in sorted(os.listdir(input_directory)):
    filepath = '/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec/' + filename
    if filename.endswith(".bmp"):
     img = Image.open(filepath).convert('RGB')
     #tiff.imwrite(file_name, img, bigtiff=True, photometric='rgb')
     img.save('/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff/' + filename.replace('.bmp' , '.tiff'), format='TIFF', compression='tiff_lzw')


# In[32]:



bf2raw_dir = '/home/rajalakshmi/Segmentation/bioformats2raw-0.2.0/'

bf2raw = os.path.join(bf2raw_dir, "bioformats2raw")

tifffile = "/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/result_merged.tif"

os.chdir(bf2raw_dir)

cmd = 'bioformats2raw /home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/result_merged.tif /home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/result_merged.zarr/' 

print('Command String : ', cmd)


# In[33]:


os.system(cmd)


# In[34]:


def napari_view():
    path = '/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/result_merged.zarr/0'
    
    viewer = napari.Viewer()
    
    viewer.open('/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/result_merged.zarr/0')
    napari.run()
    
napari_view()


# In[ ]:


print('PIL',PIL.__version__)


# In[ ]:


print(apoc.__version__)


# In[ ]:


def napari_view():
    path = '/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/trainning_slices(red).tif'
    
    viewer = napari.Viewer()
    
    viewer.open('/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/trainning_slices(red).tif')
   
    
    napari.run()
    
napari_view()


# In[31]:


def napari_view():
    path = '/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff_merged_red.zarr/0'
    
    viewer = napari.Viewer()
    
    viewer.open('/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff_merged_red.zarr/0', channel_axis = 1)
    
    napari.run()
    
napari_view()


# In[5]:


from apoc import PixelClassifier
from skimage.io import imshow, imread

#test image
image0 = imread('/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff_merged_red1180.tif')
imshow(image0)

#trainned classifier model
segmenter = PixelClassifier(opencl_filename='/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/PixelClassifier.cl')

#imply model on test data
result = PixelClassifier(opencl_filename='/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/PixelClassifier.cl').predict(image=image0)

cle.imshow(result)


# In[7]:


#segmentation with Ilastik - install ilastik and move it to segmentation folder

ilastik_dir = '/home/rajalakshmi/Segmentation/ilastik-1.4.0b27post1-gpu-Linux'

#Trian the random forest classifier with few slices (as individual images) from the whole stack
#save the project as specific classifier model

ilastik_project = '/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/classifier_ilastik.ilp'

#path for test files

input_dir = '/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff_red/'

#make test files as input files for batch processing

infiles =os.listdir(input_dir)


# In[ ]:


print (infiles)


# In[30]:


os.chdir(ilastik_dir)
for infile in infiles:
    
    if infile[-4:] != '.tif':
        print ("skipping %s".format(infile))
        continue

    
    # probabilities, simple segmentation, uncertainity, labels
    
    export_source_type = "probabilities"
    

    
    #refer ilastik headless mode documentation for building the command
    
    command = './run_ilastik.sh --headless --project="%s" --export_source="%s" --output_format="tif" --output_filename_format="/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/ilastik_output/{nickname}_results.tif" --raw_data="%s%s"' %(
        ilastik_project,
        export_source_type,
        input_dir,
        infile)
    
    
    #run the command
    
    os.system(command) 

   
    


# In[ ]:


command = './run_ilastik.sh --headless    --project=/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/classifier_ilastik.ilp    --export_source=probabilities    --output_format=tif    --output_filename_format=/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/ilastik_output/{nickname}_results.tif   '
   


# In[ ]:




