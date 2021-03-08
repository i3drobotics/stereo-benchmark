import os
import subprocess
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from stereomideval.structures import MatchData
from stereomideval.dataset import Dataset
from stereomideval.eval import Eval, Timer, Metric

# Path to download datasets
DATASET_FOLDER = os.path.join(os.getcwd(),"datasets")
# Display loaded scene data to OpenCV window
DISPLAY_IMAGES = True
STEREO_MATCHER = "BM" #BM or SGBM

# Create dataset folder
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

all_scenes = []
for scene_info in Dataset.get_training_scene_list():
    all_scenes.append(scene_info.scene_name)

default_min_disp = 0
default_num_disparities = 25
default_block_size = 11
default_uniqueness_ratio = 10
default_texture_threshold = 5
default_speckle_size = 0
default_speckle_range = 500

if STEREO_MATCHER == "BM":
    RESULTS_CSV_PATH = "cvbm_eval_results.csv"
    cv_matcher = cv2.StereoBM_create()
    calc_block = (2 * default_block_size + 5)
    cv_matcher.setBlockSize(calc_block)
    cv_matcher.setMinDisparity(default_min_disp)
    cv_matcher.setNumDisparities(16*(default_num_disparities+1))
    cv_matcher.setUniquenessRatio(default_uniqueness_ratio)
    cv_matcher.setTextureThreshold(default_texture_threshold)
    cv_matcher.setSpeckleWindowSize(default_speckle_size)
    cv_matcher.setSpeckleRange(default_speckle_range)
elif STEREO_MATCHER == "SGBM":
    RESULTS_CSV_PATH = "cvsgbm_eval_results.csv"
    cv_matcher = cv2.StereoSGBM_create()
    calc_block = (2 * default_block_size + 5)
    cv_matcher.setBlockSize(calc_block)
    cv_matcher.setMinDisparity(default_min_disp)
    cv_matcher.setNumDisparities(16*(default_num_disparities+1))
    cv_matcher.setUniquenessRatio(default_uniqueness_ratio)
    cv_matcher.setSpeckleWindowSize(default_speckle_size)
    cv_matcher.setSpeckleRange(default_speckle_range)
else:
    raise Exception("Invalid matcher name. Must be BM or SGBM")

metric_list = [" "]
metric_list.extend(Metric.get_metrics_list())
metric_list.append("bad200_maskerr")
metric_list.append("rms_maskerr")
with open(RESULTS_CSV_PATH, mode='w', newline='') as results_file:
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

    # Start timer
    timer = Timer()
    timer.start()

    left_image_grey = cv2.cvtColor(left_image,cv2.COLOR_BGR2GRAY)
    right_image_grey = cv2.cvtColor(right_image,cv2.COLOR_BGR2GRAY)
    print("Running stereo match...")
    test_disp_image = cv_matcher.compute(left_image_grey, right_image_grey)
    # Record elapsed time for simulated match
    elapsed_time = timer.elapsed()
    test_disp_image = test_disp_image.astype(np.float32) / 16.0

    if (test_disp_image is not None):
        test_disp_image = test_disp_image.astype(np.float32)
        test_disp_image[test_disp_image==99999]=0.0
        test_disp_image[test_disp_image<=0]=0.0
        test_disp_image = np.nan_to_num(test_disp_image, nan=0.0,posinf=0.0,neginf=0.0)
        test_disp_image[test_disp_image>=ndisp]=ndisp
        test_disp_image = test_disp_image.astype(ground_truth_disp_image.dtype)
        if (scene_data == "Teddy" or scene_data == "Art"):
            test_disp_image = np.rint(test_disp_image)

        ground_truth_disp_image[ground_truth_disp_image<=0]=0.0
        ground_truth_disp_image = np.nan_to_num(ground_truth_disp_image, nan=0.0,posinf=0.0,neginf=0.0)
        ground_truth_disp_image[ground_truth_disp_image>=ndisp]=ndisp

        ground_truth_mask_invalid = ground_truth_disp_image.copy()
        ground_truth_mask_invalid[test_disp_image==0] = 0.0

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

        match_result_mask_invalid = MatchData.MatchResult(
            left_image,right_image,ground_truth_mask_invalid,test_disp_image,elapsed_time,ndisp)
        # Evalulate match results against all Middlebury metrics
        metric_result_list_invalid = Eval.eval_all_metrics(match_result_mask_invalid)

        for metric_result in metric_result_list_invalid:
            if (metric_result.metric == "bad200" or metric_result.metric == "rms"):
                metric_name = metric_result.metric+"_maskerr"
                # Print metric and result
                print("{}: {}".format(metric_name,metric_result.result))
                results_row.append(metric_result.result)

        with open(RESULTS_CSV_PATH, mode='a', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(results_row)

        #plt.figure(1)
        #plt.imshow(ground_truth_disp_image)
        #plt.figure(2)
        #plt.imshow(test_disp_image)
        #plt.show()
    else:
        print("Matching failed")
        results_row = [scene_name]
        with open(RESULTS_CSV_PATH, mode='a', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(results_row)