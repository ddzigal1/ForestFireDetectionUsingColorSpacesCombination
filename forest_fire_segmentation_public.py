'''
 Forest Fire Detection Using Color Spaces Combination
    Copyright (C) 2019  Džemil Džigal, Amila Akagić

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import glob
import time
import os
import pycm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss
from pycm import *
from time import time

##############################################################################

# Examples Directory structure
# CFDB/             - dataset directory (e.g. Corsican Fire DataBase) https://feuxdeforet.universita.corsica/?lang=en
#      Original/    - original files in *.png, other formats are also supported
#      Results/     - image segmentation results, black/white image
#      Fire/        - only fire pixels
#      NotFire/     - only non-fire pixels, while fire pixels are displayed in green
#      GroundTruth/ - Ground truth images saved as filename_gt.png

original_dir = 'CFDB/Original/'
results_dir = 'CFDB/Results/'
fire_dir = 'CFDB/Fire/'
notfire_dir = 'CFDB/NotFire/'
ground_truth_dir = 'CFDB/GroundTruth/'

original_list = sorted(glob.glob(original_dir+'*.png'))

num_imgs = len(original_list)
total_accuracy = 0.0
total_f1_score = 0.0
total_recall = 0.0
total_specificity = 0.0
total_precision = 0.0
total_fallout = 0.0

for f in range(num_imgs): 
    start_time = time()
    filename = os.path.basename(original_list[f])
    gt_img_name = os.path.basename(original_list[f])[0:3]+"_gt.png"
    # We first load the RGB image and extract the Hue and Value channels:
    print("Reading image", filename)
    bgr_img = cv2.imread(original_list[f])
    gt_img = cv2.imread(ground_truth_dir+gt_img_name, cv2.IMREAD_GRAYSCALE)
    # Convert BGR to RGB
    print("Converting image... ")
    result = bgr_img
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.imread(original_list[f], cv2.IMREAD_GRAYSCALE)

    # Blur image
    rgb_img = cv2.blur(rgb_img,(7,7))

    # Split R, G, B channels
    r,g,b = cv2.split(rgb_img)

    # Convert RGB image to HLS
    hls_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    height = rgb_img.shape[0]
    width = rgb_img.shape[1]

    # Channel separation for HLS
    h2 = hls_img[:, :, 0]

    # HWB creation - we use it only to display an image. It is not necessary for our method.
    hwb_img = np.zeros([height,width,3],dtype=np.uint8)
    hwb_img.fill(0) # or img[:] = 255
    hwb_img[:, :, 0] = h2       # H component same as in HLS
    for y in range(0, height):
            for x in range(0, width):
            	hwb_img[y, x, 1] = min(rgb_img[y, x, 0], min(rgb_img[y, x, 1], rgb_img[y, x, 2])) # W component
            	hwb_img[y, x, 2] = 1 - max(rgb_img[y, x, 0], max(rgb_img[y, x, 1], rgb_img[y, x, 2]))  # B component

    # HSV, HLS, HWB image representation
    hsv2_img = np.zeros([height,width,3],dtype=np.float)
    hsv2_img.fill(0) 

    hls2_img = np.zeros([height,width,3],dtype=np.float)
    hls2_img.fill(0) 

    hwb2_img = np.zeros([height,width,3],dtype=np.float)
    hwb2_img.fill(0) 

    for y in range(0, height):
            for x in range(0, width):
            	r = rgb_img[y, x, 0] / 255 
            	g = rgb_img[y, x, 1] / 255	
            	b = rgb_img[y, x, 2] / 255

            	# V component 
            	maxx = max(r,g,b)
            	minn = min(r,g,b)
            	l2 = (maxx + minn)/2
            	delta = maxx - minn
            	v = maxx
            	diff = v - min(r, g, b)

            	if (diff == 0):
            		h = 0
            		s1 = 0
            	else:
            		s1 = diff / v
            		if (r == maxx): 
            			h = (g - b) / (diff * 6)
            		elif (g == maxx):
            			h = 1/3 + (b - r) / (diff * 6)
            		elif (b == maxx):
            			h = 2/3 + (r - g) / (diff * 6)
            	if (h < 0):
            		h = h + 1
            	elif (h > 1): 
            		h = h - 1
            	if (maxx == minn):
            		s2 = 0
            	elif (l2 <= 0.5):
            		s2 = delta / (maxx + minn)
            	else:
            		s2 = delta / (2 - maxx - minn)

            	hsv2_img[y, x, 0] = h * 360
            	hsv2_img[y, x, 1] = s1 * 100
            	hsv2_img[y, x, 2] = v * 100
            	hls2_img[y, x, 0] = h * 360
            	hls2_img[y, x, 1] = l2 * 100
            	hls2_img[y, x, 2] = s2 * 100
            	hwb2_img[y, x, 0] = h * 360
            	hwb2_img[y, x, 1] = (1-s1)*v * 100
            	hwb2_img[y, x, 2] = (1-v) * 100

    hsl2_img = np.zeros([height,width,3],dtype=np.float)
    hsl2_img.fill(0) 
    hsl2_img[:, :, 0] = hls2_img[:, :, 0]
    hsl2_img[:, :, 1] = hls2_img[:, :, 2]
    hsl2_img[:, :, 2] = hls2_img[:, :, 1]

    # Channel separation for HWB
    h1 = hsv2_img[:, :, 0]
    l2 = hls2_img[:, :, 1]
    s2 = hls2_img[:, :, 2]
    w3 = hwb2_img[:, :, 1]
    b3 = hwb2_img[:, :, 2]

    print("Calculating average values... ")
    SAvg = np.average(s2)
    WAvg = np.average(w3)
    BAvg = np.average(b3)

    print("SAvg: ", SAvg, "WAvg: ", WAvg, "BAvg: ", BAvg)

    print("Image segmentation... ")
    # Creation of a new hybrid image...
    hybrid_img = np.zeros([height,width],dtype=np.uint8)
    hybrid_img.fill(0) 


    for y in range(0, height):
            for x in range(0, width):
                if (s2[y][x] >= 65):
                    if ((h1[y][x]>=330 or h1[y][x]<=65 and h1[y][x]>=0) and (l2[y][x]>70 or b3[y][x]<BAvg and w3[y][x] < WAvg or w3[y][x]>WAvg and s2[y][x]>SAvg) or w3[y][x]>=98 and b3[y][x]<=2):
                       hybrid_img[y][x] = 255   # white
                    else:
                       hybrid_img[y][x] = 0     # black
                else:
                    hybrid_img[y][x] = 0        # black


    # Remove small connected regions.
    # Find all your connected components (white blobs in your image).
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(hybrid_img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # The following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # Minimum size of particles we want to keep (number of pixels).
    # Here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    min_size = 150  

    img2 = np.zeros([height,width],dtype=np.uint8)
    img2_inv = np.zeros([height,width],dtype=np.uint8)
    img2.fill(0)
    img2_inv.fill(255)
    # For every component in the image, you keep it only if it's above min_size.
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
            img2_inv[output == i + 1] = 0

    end_time = time()

    print("Writing results... ")

    cv2.imwrite(results_dir+filename,img2)  # Segmentation results

    masked_image = cv2.bitwise_and(bgr_img,bgr_img,mask = img2)
    cv2.imwrite(fire_dir+filename,masked_image) # What is detected as fire

    result = rgb_img
    result[img2!=0] = (0,255,0)
    cv2.imwrite(notfire_dir+filename,result) # What is detected as not fire


    # Metrics
    # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4533825/
    tn, fp, fn, tp = confusion_matrix(gt_img.flatten(),img2.flatten()).ravel()
    f1_score = float((2 * tp) / ((2 * tp) + fp + fn))
    accuracy = float((tn+tp)/(fn+fp+1+tn+tp))
    print(filename, ",", width, "x", height, ",", height*width, ", ", accuracy, ",", f1_score, ", tn:, ", tn, ", fp: ,", fp, ", fn: , ", fn, ", tp:, ", tp, ", Time: ,", end_time - start_time)
    recall = float(tp/(tp+fn+1))
    specificity = float(tn/(tn+fp+1))
    fallout = float(1-tn/(tn+fp+1))
    precision = float(tp/(tp+fp+1))
    #print("Sensitivity/Recall = ",recall)
    #print("Specificity = ",specificity)
    #print("Fallout = ", fallout)
    #print("Precision = ", precision)
    total_accuracy += accuracy
    total_f1_score += f1_score 
    total_recall += recall
    total_specificity += specificity
    total_fallout += fallout
    total_precision += precision

print("Results of the entire dataset:")
print("Total accuracy:", total_accuracy/num_imgs, "Total F1 score:",total_f1_score/num_imgs)
print("Total recall:",total_recall, "Total specificity:",total_specificity, "Total fallout:",total_fallout, "Total precision:", total_precision)
