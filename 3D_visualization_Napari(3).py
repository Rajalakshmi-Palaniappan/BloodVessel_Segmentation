#!/usr/bin/env python
# coding: utf-8

# In[1]:


gui qt5


# In[2]:


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


# In[3]:


import apoc
import os
from skimage.io import imread, imshow
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
from apoc import PixelClassifier


# In[ ]:


napari.run()


# In[ ]:


input_directory = "/mnt/md0/rajalakshmi/Kristin_Data/PE-CT1_Microfil_Perfusions-2020_07_13/2020_07_13_Hearts_wo_Aorta/1738-Heart_wo_aorta-downsampled/"


# In[ ]:


sorted(os.listdir(input_directory))


# In[ ]:


output = "/mnt/md0/rajalakshmi/Kristin_Data/PE-CT1_Microfil_Perfusions-2020_07_13/2020_07_13_Hearts_wo_Aorta/1738/1738_raw_Tif/"


# In[ ]:


for filename in sorted(os.listdir(input_directory)):
    filepath = '/mnt/md0/rajalakshmi/Kristin_Data/PE-CT1_Microfil_Perfusions-2020_07_13/2020_07_13_Hearts_wo_Aorta/1738-Heart_wo_aorta-downsampled/' + filename
    if filename.endswith(".bmp"):
     img = Image.open(filepath).convert('RGB')
     #tiff.imwrite(file_name, img, bigtiff=True, photometric='rgb')
     img.save('/mnt/md0/rajalakshmi/Kristin_Data/PE-CT1_Microfil_Perfusions-2020_07_13/2020_07_13_Hearts_wo_Aorta/1738/1738_raw_Tif/' + filename.replace('.bmp' , '.tiff'), format='TIFF', compression='tiff_lzw')


# In[7]:



bf2raw_dir = '/mnt/md0/rajalakshmi/bioformats2raw-0.2.0/'

bf2raw = os.path.join(bf2raw_dir, "bioformats2raw")

tifffile = "/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/output2_merged.tif"

os.chdir(bf2raw_dir)

cmd = 'bioformats2raw /mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/output2_merged.tif /mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/output2_merged.zarr' 

print('Command String : ', cmd)


# In[8]:


os.system(cmd)


# In[ ]:


#Napari 3D visualization:


# In[9]:


def napari_view():
    path = '/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/output2_merged.zarr/0'
    
    viewer = napari.Viewer()
    viewer.open('/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/output2_merged.zarr/0', channel_axis=1)
    napari.run()
    
napari_view()


# In[ ]:


print('PIL',PIL.__version__)


# In[ ]:


print(apoc.__version__)


# In[ ]:


# Napari GUI training pixel classifier


# In[ ]:


def napari_view():
    path = '/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/trainning_slices_red0000.tif'
    
    viewer = napari.Viewer()
    
    viewer.open('/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/trainning_slices_red0000.tif')
   
    
    napari.run()
    
napari_view()


# In[ ]:


def napari_view():
    path = '/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff_merged_red.zarr/0'
    
    viewer = napari.Viewer()
    
    viewer.open('/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff_merged_red.zarr/0', channel_axis = 1)
    
    napari.run()
    
napari_view()


# In[ ]:


# Napari implying trianned classifier on test images


# In[ ]:


from apoc import PixelClassifier
from skimage.io import imshow, imread

#test image
image0 = imread('/home/rajalakshmi/Segmentation/mv_heart/2020_01_21_Rat_4_Heart_Rec_tiff_merged_red.tif')

#trainned classifier model
segmenter = PixelClassifier(opencl_filename='/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/PixelClassifier2.cl')

#imply model on test data
result = PixelClassifier(opencl_filename='/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/PixelClassifier2.cl').predict(image=image0)

tifffile.imsave('/mnt/md0/rajalakshmi/napari_results/results.tif', result)


# In[ ]:


# Napari implying trianned classifier on a folder of test images (for loop for batch processing)


# In[6]:


#segmenter = PixelClassifier(opencl_filename='/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/PixelClassifier2.cl')

input_dir = '/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/2020_01_21_Rat_4_Heart_Rec_tiff_red/'

infiles = os.listdir(input_dir)

for filename in infiles:
    filepath = imread('/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/2020_01_21_Rat_4_Heart_Rec_tiff_red/' + filename)
    if filename[-4:] != '.tif':
        print ("skipping %s".format(infile))
        continue
        
    result = PixelClassifier(opencl_filename='/mnt/md0/rajalakshmi/bioformats2raw-0.2.0/PixelClassifier.cl').predict(image = filepath)
    
    tifffile.imsave('/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/output2/' + filename, result)   


# In[ ]:


#Ilastik Segmentation and batch processing using headless mode (Trainned the classifier using Ilastik software):


# In[ ]:


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


# In[ ]:


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


#Napari - training multiple images without GUI (python interface)
#refer: https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/20a_pixel_classification/apoc_train_on_folders.html


# In[ ]:


def napari_view():
    path = '/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/trainning_slices_red0000.tif'
    
    viewer = napari.Viewer()
    
    viewer.open('/home/rajalakshmi/Segmentation/mv_heart/trainned_classifier/trainning_slices_red0000.tif')
   
    
    napari.run()
    
napari_view()


# In[ ]:


images = '/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/image_folder/'
masks = '/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/mask_folder/'


# In[ ]:


file_list = os.listdir(images)

fig, axs = plt.subplots(1, 3, figsize=(15,15))
for i, filename in enumerate(file_list):
    image = imread(images + filename)
    cle.imshow(image, plot=axs[i])
plt.show()

file_list = os.listdir(masks)

fig, axs = plt.subplots(1, 3, figsize=(15,15))
for i, filename in enumerate(file_list):
    mask = imread(masks + filename)
    cle.imshow(mask, plot=axs[i])
plt.show()


# In[ ]:


Trainning_classifier = "PixelClassifier.cl"
apoc.erase_classifier(Trainning_classifier)
segmenter = apoc.PixelClassifier(opencl_filename=Trainning_classifier, max_depth = 10, num_ensembles = 50)

# setup feature set used for training
features = apoc.PredefinedFeatureSet.object_size_1_to_5_px.value

# train classifier on folders
apoc.train_classifier_from_image_folders(
    segmenter, 
    features, 
    image = images, 
    ground_truth = masks)


# In[ ]:


segmenter = apoc.PixelClassifier(opencl_filename=Trainning_classifier)


# In[ ]:


input_dir = '/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/2020_01_21_Rat_4_Heart_Rec_tiff_red/'

infiles = os.listdir(input_dir)

for filename in infiles:
    filepath = imread('/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/2020_01_21_Rat_4_Heart_Rec_tiff_red/' + filename)
    if filename[-4:] != '.tif':
        print ("skipping %s".format(infile))
        continue
        
    result = PixelClassifier(opencl_filename='PixelClassifier.cl').predict(image = filepath)
    
    tifffile.imsave('/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/output/' + filename, result)   


# In[4]:


segmenter = PixelClassifier(opencl_filename='/mnt/md0/rajalakshmi/bioformats2raw-0.2.0/PixelClassifier.cl')


# In[ ]:


image0 = imread('/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/trainning_slices_red0001.tif')


# In[ ]:


result = segmenter.predict(image=image0)


# In[ ]:


imshow(result)


# In[ ]:


tifffile.imsave('/mnt/md0/rajalakshmi/napari_results/Napari_training_folder/test_data/result.tif',result)


# In[ ]:




