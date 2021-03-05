import os
import subprocess
import glob
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer, Metric
from i3drsgm import I3DRSGM, I3DRSGMAppAPI

# Path to download datasets
DATASET_FOLDER = os.path.join(os.getcwd(),"datasets")
# Display loaded scene data to OpenCV window
DISPLAY_IMAGES = True
# I3DRALSC exe path
I3DRALSC_EXE_PATH = "..\\install\\bin\\i3dralsc_driver.exe"
# Results filepath
RESULTS_CSV_PATH = "eval_results.csv"

# Matcher parameters
MIN_DISP = 0
DISP_RANGE = 16*250
WINDOW_SIZE = 5
PYRAMID_LEVEL = 6
INTERP = True

# Initalise I3DRSGM
license_files = glob.glob("*.lic")
if (len(license_files) <= 0):
    raise Exception("Failed to find license file in script directory.")
license_file = license_files[0]
i3drsgm = I3DRSGM(license_file)
# Check initalisation was success
if not i3drsgm.isInit():
    raise Exception("Failed to initalise I3DRSGM")

# Create dataset folder
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

all_scenes = []
for scene_info in Dataset.get_training_scene_list():
    all_scenes.append(scene_info.scene_name)

if (not os.path.exists('..\\data\\Middlesbury')):
    os.mkdir('..\\data\\Middlesbury')

if (not os.path.exists('..\\output')):
    os.mkdir('..\\output')
if (not os.path.exists('..\\output\\Middlesbury')):
    os.mkdir('..\\output\\Middlesbury')

metric_list = [" "]
metric_list.extend(Metric.get_metrics_list())
with open(RESULTS_CSV_PATH, mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(metric_list)

# Download datasets from middlebury servers
# will only download it if it hasn't already been downloaded
for scene_name in all_scenes:
    print("Downloading data for scene '"+scene_name+"'...")
    Dataset.download_scene_data(scene_name,DATASET_FOLDER)

    # Load scene data from downloaded folder
    print("Loading data for scene '"+scene_name+"'...")
    scene_data = Dataset.load_scene_data(scene_name,DATASET_FOLDER,DISPLAY_IMAGES)
    # Scene data class contains the following data:
    left_image = scene_data.left_image
    right_image = scene_data.right_image
    ground_truth_disp_image = scene_data.disp_image
    ndisp = scene_data.ndisp

    input_folder_path = '..\\data\\Middlesbury\\'+scene_name+'\\'
    left_image_path = input_folder_path+'left.png'
    right_image_path = input_folder_path+'right.png'
    out_folder_path = '..\\output\\Middlesbury\\'+scene_name+'\\'
    disp_image_path = out_folder_path + 'disparity.tiff'
    disp_colormap_image_path = out_folder_path + 'disparity_colourmap.tiff'

    if (not os.path.exists(input_folder_path)):
        os.mkdir(input_folder_path)

    
    disp_range = DISP_RANGE

    orig_dtype = ground_truth_disp_image.dtype
    ground_truth_disp_image = ground_truth_disp_image.astype(np.float32)
    ground_truth_disp_image = ground_truth_disp_image.astype(orig_dtype)

    # Get test data image dims
    new_image_height = left_image.shape[0]
    new_image_width = left_image.shape[1]
    print("{},{}".format(new_image_height,new_image_width))
    if new_image_height != image_height or new_image_width != image_width:
        # Re generate i3drsgm if image height is different
        i3drsgm.close()
        i3drsgm = I3DRSGM(license_file)
        # Set matcher parameters
        res = i3drsgm.setWindowSize(WINDOW_SIZE)
        res &= i3drsgm.setMinDisparity(MIN_DISP)
        res &= i3drsgm.setDisparityRange(disp_range)
        res &= i3drsgm.setPyamidLevel(PYRAMID_LEVEL)
        res &= i3drsgm.enableInterpolation(INTERP)
        image_height, image_width = new_image_height, new_image_width

    # Start timer
    timer = Timer()
    timer.start()

    # Stereo match image pair
    print("Running I3DRSGM on images...")
    valid, test_disp_image = i3drsgm.forwardMatch(left_image,right_image)

    if (valid and test_disp_image is not None):
        # Record elapsed time for match
        elapsed_time = timer.elapsed()

        # I3DRSGM result is negative, invert it
        test_disp_image = -test_disp_image.astype(np.float32)

        # Non matched pixel have value of 99999, replace with zero (zero is ignored in evaluation)
        test_disp_image = test_disp_image.astype(np.float32)
        test_disp_image[test_disp_image==99999]=0.0
        test_disp_image[test_disp_image<=0]=0.0
        test_disp_image[test_disp_image>=ndisp]=ndisp
        test_disp_image = np.nan_to_num(test_disp_image, nan=0.0,posinf=0.0,neginf=0.0)
        test_disp_image = test_disp_image.astype(ground_truth_disp_image.dtype)

        # Record elapsed time for simulated match
        elapsed_time = timer.elapsed()
        # Format match result into expected format for use in evaluation
        match_result = MatchData.MatchResult(
            left_image,right_image,ground_truth_disp_image,test_disp_image,elapsed_time,ndisp)
        # Evalulate match results against all Middlebury metrics
        metric_result_list = Eval.eval_all_metrics(match_result)
        results_row = [scene_name]
        for metric_result in metric_result_list:
            # Print metric and result
            print("{}: {}".format(metric_result.metric,metric_result.result))
            results_row.append(metric_result.result)

        with open(RESULTS_CSV_PATH, mode='a') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(results_row)

        plt.figure(1)
        plt.imshow(ground_truth_disp_image)
        plt.figure(2)
        plt.imshow(test_disp_image)
        plt.show()
    else:
        print("Matching failed")