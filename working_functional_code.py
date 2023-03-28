#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import napari
import skimage
import sys
import os
import cv2
import PIL
from PIL import Image
import tifffile
import numpy as np
from PIL import TiffImagePlugin
import zarr as z
import tifffile as tiff

import apoc
from skimage.io import imread
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
from apoc import PixelClassifier
from pyclesperanto_prototype import imshow
import bioformats2raw
from napari_skimage_regionprops import regionprops_table, add_table, get_table
import pandas as pd
import glob
import csv
import pyclesperanto_prototype as cle

import pyvista
import argparse
import itk
from distutils.version import StrictVersion as VS

import kimimaro
from skimage import measure
from skimage import filters
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import tifffile 
import napari
import pandas as pd
import os

import plotly.graph_objects as go


# In[ ]:


# convert bmp to tiff


# In[ ]:


input_directory = "/Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/data_1/"
os.remove("/Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/data_1/.DS_Store")
sorted(os.listdir(input_directory))


# In[ ]:


path = "/Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/data_rawTiff"

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)


# In[ ]:


for filename in sorted(os.listdir(input_directory)):
    filepath = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/data_1/' + filename
    if filename.endswith(".bmp"):
        img = Image.open(filepath)
     #tiff.imwrite(file_name, img, bigtiff=True, photometric='rgb')
        img.save('/Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/data_rawTiff/' + filename.replace('.bmp' , '.tiff'), format='TIFF', compression='tiff_lzw')


# In[ ]:


#view it if needed
viewer = napari.Viewer()


# In[ ]:


#convert tiff sequence to multipage tiff using FIJI


# In[ ]:


#convert multipage tiff to compressed format zarr: just for visulization 


# In[ ]:


bf2raw_dir = '/Users/ramyarajalakshmi/Documents/bioformats2raw-0.2.0/'

bf2raw=os.path.join(bf2raw_dir, "bioformats2raw")

tifffile = "/Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/26053_eye1/26053_eye1_raw.tif"

os.chdir(bf2raw_dir)

cmd='bioformats2raw /Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/26053_eye1/26053_eye1_raw.tif /Users/ramyarajalakshmi/Documents/Segmentation/mv_eyes/method_1/26053_eye1/26053_eye1_raw.zarr'

print('Command String : ', cmd)


# In[ ]:


os.system(cmd)


# In[ ]:


#view it if needed
viewer = napari.Viewer()


# In[ ]:


#Train random forest - requirements: vesselness filter, manual labelling


# In[ ]:


#vesselness filter for the data - use multipage tiff as input to the below filter obtained from simpleITK- Multiscale hessian based vesselness filter
#run this script in workstation, in laptop the kernel will die


# In[ ]:


import argparse

import itk
from distutils.version import StrictVersion as VS

if VS(itk.Version.GetITKVersion()) < VS("5.0.0"):
    print("ITK 5.0.0 or newer is required.")
    sys.exit(1)

    
input_image = "/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/kristin_data/1649/1649_rawTiff.tif"
output_image = "/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/kristin_data/1649/1649_rawTiff_vesselness.tif"
sigma_minimum = 1.0
sigma_maximum = 10.0
number_of_sigma_steps = 10


input_image = itk.imread(input_image, itk.F)

ImageType = type(input_image)
Dimension = input_image.GetImageDimension()
HessianPixelType = itk.SymmetricSecondRankTensor[itk.D, Dimension]
HessianImageType = itk.Image[HessianPixelType, Dimension]

objectness_filter = itk.HessianToObjectnessMeasureImageFilter[
    HessianImageType, ImageType
].New()
objectness_filter.SetBrightObject(True)
objectness_filter.SetScaleObjectnessMeasure(False)
objectness_filter.SetAlpha(0.5)
objectness_filter.SetBeta(1.0)
objectness_filter.SetGamma(5.0)
objectness_filter.SetObjectDimension(1)

multi_scale_filter = itk.MultiScaleHessianBasedMeasureImageFilter[
    ImageType, HessianImageType, ImageType
].New()
multi_scale_filter.SetInput(input_image)
multi_scale_filter.SetHessianToMeasureFilter(objectness_filter)
multi_scale_filter.SetSigmaStepMethodToLogarithmic()
multi_scale_filter.SetSigmaMinimum(sigma_minimum)
multi_scale_filter.SetSigmaMaximum(sigma_maximum)
multi_scale_filter.SetNumberOfSigmaSteps(number_of_sigma_steps)

OutputPixelType = itk.UC
OutputImageType = itk.Image[OutputPixelType, Dimension]

rescale_filter = itk.RescaleIntensityImageFilter[ImageType, OutputImageType].New()
rescale_filter.SetInput(multi_scale_filter)

itk.imwrite(rescale_filter.GetOutput(), output_image)


# In[ ]:


# open the tiff sequence obtained above and choose certain slices for annotation
#perform annotations using napari
#save the labels seperately


# In[ ]:


raw_image = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/images/1683_0596.tif')


# In[ ]:


vesselness = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/itk_vesselness/1683_0596.tif')


# In[ ]:


annotation = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/masks-single channel/1683_0596.tif')


# In[ ]:


feature_images = [
    raw_image,
    #gaussian(raw_image, sigma=1),
    #gaussian(raw_image, sigma=5),
    #sobel(gaussian(raw_image, sigma=1)),
    #sobel(gaussian(raw_image, sigma=5)),
    #vesselness,
]
tifffile.imsave('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/features_train/1683_0596.tif',feature_images)


# In[ ]:


features_train = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/features_train/'
os.remove('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/features_train/.DS_Store')


# In[ ]:


Trainning_classifier = "PixelClassifier.cl"
apoc.erase_classifier(Trainning_classifier)
segmenter = apoc.PixelClassifier(opencl_filename=Trainning_classifier, max_depth = 10, num_ensembles = 50)

# setup feature set used for training
features = "original"

# train classifier on folders
apoc.train_classifier_from_image_folders(
    segmenter, 
    features, 
    image = features_train, 
    ground_truth = masks)


# In[ ]:


#prepare the test data


# In[ ]:


test_images = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_raw_tiff/'


# In[ ]:


filename_list = sorted(os.listdir('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_raw_tiff/'))
for filename in filename_list:
    raw_image = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_raw_tiff/' + filename)
    vesselness = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/vesselness/' + filename)
    
    feature_images = [raw_image, vesselness]
        
    tifffile.imwrite('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_vesselness_test/' + filename, feature_images)


# In[ ]:


#implementing model on test data


# In[ ]:


input_dir = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_vesselness_test/'
os.remove('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_vesselness_test/.DS_Store')


# In[ ]:


input_dir = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_vesselness_test/'

infiles = os.listdir(input_dir)

for filename in infiles:
    filepath = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_vesselness_test/' + filename)

        
    result = PixelClassifier(opencl_filename='PixelClassifier.cl').predict(image = filepath)
    
    tifffile.imwrite('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/result/' + filename, result)   


# In[ ]:


#now the binarization is done and the next step is skeletonization


# In[ ]:


binary_data = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/multilabel/vesselness/test/1683_binary_result.tif')


# In[ ]:


binary_data_arr = np.array(binary_data)


# In[ ]:


print(binary_data_arr.shape)


# In[ ]:


Labels = measure.label(binary_data_arr, background=0)


# In[ ]:


tifffile.imsave('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/Labels.tif', Labels)


# In[ ]:


Labels = imread('/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/Labels.tif')


# In[ ]:


Labels = Labels.astype(np.uint32)


# In[ ]:


Labels = np.transpose(Labels)


# In[ ]:


print(Labels.shape)


# In[ ]:


skels = kimimaro.skeletonize(
  Labels, 
  teasar_params={
    'scale': 1.1,
    'const': 50, # set scale = 1.1 and const = 10 for the data with blood vessels
    'pdrf_exponent': 4,
    'pdrf_scale': 100000,
    'soma_detection_threshold': 10, # physical units
    'soma_acceptance_threshold': 10, # physical units
    'soma_invalidation_scale': 1.0,
    'soma_invalidation_const': 10, # physical units
    'max_paths': None, # default None
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


print ("Length : %d" % len (skels))


# In[ ]:


skeleton_folder = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/skeletons_scale1.1Const50/'


# In[ ]:


sample_name = 'skeleton'


# In[ ]:


df = pd.DataFrame(list(skels.items()),columns = ['a','b'])


# In[ ]:


print (df)


# In[ ]:


labels = np.array(df['a'])


# In[ ]:


print(labels.shape)


# In[ ]:


def skeleton_to_swc(skel, outfn):
    with open(outfn, "w") as f:
        f.write(skel.to_swc())


# In[ ]:


for label in labels:
    if label == 0:
        continue
    skel = skels[label]
    skel_swc_fn = os.path.join(skeleton_folder, sample_name + "_%i.swc" % label)
    skeleton_to_swc(skel, skel_swc_fn)


# In[1]:


# for sanity check visualize the skeletons using: "https://neuroinformatics.nl/HBP/morphology-viewer/"


# In[ ]:


#for further analysis and visulaization of the skeletons pplotly was used


# In[ ]:


def read_swc(file_path):
    nodes = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            columns = line.split()
            if len(columns) != 7:
                print(f"Error: line {i+1} in file {file_path} has {len(columns)} columns instead of 7")
                continue
            node_id = int(columns[0])
            node_type = int(columns[1])
            x, y, z = map(float, columns[2:5])
            radius = float(columns[5])
            parent_id = int(columns[6])
            nodes.append((node_id, node_type, x, y, z, radius, parent_id))
    return nodes

def get_branch_points(nodes):
    branch_points = []
    for node in nodes:
        node_id = node[0]
        parent_id = node[6]
        children = [n for n in nodes if n[6] == node_id]
        if parent_id != -1 and len(children) > 1:
            branch_points.append(node)
    return branch_points

def find_end_nodes(nodes):
    end_nodes = []
    for node in nodes:
        node_id = node[0]
        children = [n for n in nodes if n[6] == node_id]
        if len(children) == 0:
            end_nodes.append(node)
    return end_nodes

def plot_skeleton(nodes):
    # Get branch points
    branch_points = get_branch_points(nodes)
    branch_point_ids = set(node[0] for node in branch_points)
    end_nodes = find_end_nodes(nodes)
    end_nodes_ids = set(node[0] for node in end_nodes)
    special_nodes = branch_points + end_nodes
    special_nodes_ids = set(node[0] for node in special_nodes)

    # Creating the 3d scatter plot with the branchpoints highlighted in red and remaining nodes in blue
    fig = go.Figure(data=[go.Scatter3d(
        x=[node[2] for node in nodes],
        y=[node[3] for node in nodes],
        z=[node[4] for node in nodes],
        mode='markers',
        marker=dict(
            #size=[node[5] for node in nodes],
            #size =0.5,
            size=[5 if node[0] in special_nodes_ids else 2 for node in nodes],
            color=['red' if node[0] in special_nodes_ids else 'blue' for node in nodes],  
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    

    # Set axis labels and title
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectratio=dict(x=1, y=1, z=1),
    ),
        title='Skeleton for label 1'
    )
    fig.show()

def get_nodes_between_red_points(nodes):
    branch_points = get_branch_points(nodes)
    branch_point_ids = set(node[0] for node in branch_points)
    end_nodes = find_end_nodes(nodes)
    end_nodes_ids = set(node[0] for node in end_nodes)
    special_nodes = branch_points + end_nodes
    special_nodes_ids = set(node[0] for node in special_nodes)
    #print(len(special_nodes))
    red_points = [node for node in nodes if node[0] in special_nodes_ids]
    nodes_between = []
    for i in range(len(red_points)-1):
        start_index = nodes.index(red_points[i])
        end_index = nodes.index(red_points[i+1])
        nodes_between.append(abs(end_index - start_index) - 1)
    return nodes_between

def find_longest_branch(nodes):
    branch_lengths = get_nodes_between_red_points(nodes)
    branch_points = get_branch_points(nodes)
    branch_point_ids = set(node[0] for node in branch_points)
    end_nodes = find_end_nodes(nodes)
    end_nodes_ids = set(node[0] for node in end_nodes)
    special_nodes = branch_points + end_nodes
    special_nodes_ids = set(node[0] for node in special_nodes)
    
    
    #find longest_branch
    longest_branch_length = max(branch_lengths)
    print(longest_branch_length)
    red_points = [node for node in nodes if node[0] in special_nodes_ids]
    longest_branch_nodes =[]
    nodes_between = []
    for i in range(len(red_points)-1):
        start_index = nodes.index(red_points[i])
        end_index = nodes.index(red_points[i+1])
        nodes_between = (end_index - start_index) - 1
        if nodes_between == longest_branch_length:
            print(nodes_between)
            longest_branch_nodes = nodes[start_index + 1:end_index]
            break
    return(longest_branch_nodes)
    

def plot_skeleton_with_longest_branch(nodes):
    # Get branch points
    branch_points = get_branch_points(nodes)
    branch_point_ids = set(node[0] for node in branch_points)
    end_nodes = find_end_nodes(nodes)
    end_nodes_ids = set(node[0] for node in end_nodes)
    special_nodes = branch_points + end_nodes
    special_nodes_ids = set(node[0] for node in special_nodes)
    longest_branch_nodes = find_longest_branch(nodes)
    longest_branch_node_set = set(node[0] for node in longest_branch_nodes)
    print(len(longest_branch_node_set))
    print(len(longest_branch_nodes))
    
    
    # Creating the 3d scatter plot with the special_nodes highlighted in red, nodes belong to the longest_branch in green and remaining nodes in blue
    fig = go.Figure(data=[go.Scatter3d(
        x=[node[2] for node in nodes],
        y=[node[3] for node in nodes],
        z=[node[4] for node in nodes],
        mode='markers',
        marker=dict(
            size=[5 if node[0] in special_nodes_ids else 2 if node[0] in longest_branch_node_set else 1 for node in nodes],
            color=['red' if node[0] in special_nodes_ids else 'green' if node[0] in longest_branch_node_set else 'blue' for node in nodes],  
            colorscale='Viridis',
            opacity=0.8
        )
    )])
    fig.show()



if __name__ == '__main__':
    swc_file_path = '/Users/ramyarajalakshmi/Documents/Segmentation/mv_heart/skeletonization/skeletons/skeleton_1.swc'
    nodes = read_swc(swc_file_path)
    plot_skeleton(nodes)
    nodes_between_red_points = get_nodes_between_red_points(nodes)
    #print(nodes_between_red_points)
    #print(len(nodes_between_red_points))
    plot_skeleton_with_longest_branch(nodes)

