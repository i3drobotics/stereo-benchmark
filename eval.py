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
from i3drsgm import I3DRSGM

def eval(i3drsgm,dataset_folder,display_images,min_disp,disp_range,window_size,pyramid_level,interp):
    # Results filepath
    if (interp):
        RESULTS_CSV_PATH = "eval_results_wInterp.csv"
    else:
        RESULTS_CSV_PATH = "eval_results.csv"

    metric_list = [" "]
    metric_list.extend(Metric.get_metrics_list())
    metric_list.append("bad200_maskerr")
    metric_list.append("rms_maskerr")
    with open(RESULTS_CSV_PATH, mode='w', newline='') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        results_writer.writerow(metric_list)

    all_scenes = []
    for scene_info in Dataset.get_training_scene_list():
        all_scenes.append(scene_info.scene_name)

    image_height, image_width = 0,0
    # Download datasets from middlebury servers
    # will only download it if it hasn't already been downloaded
    for scene_name in all_scenes:
        print("Downloading data for scene '"+scene_name+"'...")
        Dataset.download_scene_data(scene_name,dataset_folder)

        # Load scene data from downloaded folder
        print("Loading data for scene '"+scene_name+"'...")
        scene_data = Dataset.load_scene_data(scene_name,dataset_folder,display_images)
        # Scene data class contains the following data:
        left_image = scene_data.left_image
        right_image = scene_data.right_image
        ground_truth_disp_image = scene_data.disp_image
        ndisp = scene_data.ndisp

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
            res = i3drsgm.setWindowSize(window_size)
            res &= i3drsgm.setMinDisparity(min_disp)
            res &= i3drsgm.setDisparityRange(disp_range)
            res &= i3drsgm.setPyamidLevel(pyramid_level)
            res &= i3drsgm.enableInterpolation(interp)
            image_height, image_width = new_image_height, new_image_width

        # Start timer
        timer = Timer()
        timer.start()

        # Stereo match image pair
        print("Running I3DRSGM on images...")
        valid, test_disp_image = i3drsgm.forwardMatch(left_image,right_image)
        # Record elapsed time for simulated match
        elapsed_time = timer.elapsed()

        if (valid and test_disp_image is not None):
            # Record elapsed time for match
            elapsed_time = timer.elapsed()

            # I3DRSGM result is negative, invert it
            test_disp_image = -test_disp_image.astype(np.float32)

            # Non matched pixel have value of 99999, replace with zero (zero is ignored in evaluation)
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
    i3drsgm.close()

if __name__=="__main__":
    # Path to download datasets
    DATASET_FOLDER = os.path.join(os.getcwd(),"datasets")
    # Display loaded scene data to OpenCV window
    DISPLAY_IMAGES = True

    # Matcher parameters
    MIN_DISP = 0
    DISP_RANGE = 16*120
    WINDOW_SIZE = 5
    PYRAMID_LEVEL = 6

    # Create dataset folder
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)

    # Initalise I3DRSGM
    license_files = glob.glob("*.lic")
    if (len(license_files) <= 0):
        raise Exception("Failed to find license file in script directory.")
    license_file = license_files[0]
    i3drsgm = I3DRSGM(license_file)
    # Check initalisation was success
    if not i3drsgm.isInit():
        raise Exception("Failed to initalise I3DRSGM")

    # evaluate with and without interpolation
    eval(
        i3drsgm,
        dataset_folder=DATASET_FOLDER,
        display_images=DISPLAY_IMAGES,
        min_disp=MIN_DISP,
        disp_range=DISP_RANGE,
        window_size=WINDOW_SIZE,
        pyramid_level=PYRAMID_LEVEL,
        interp=True
    )
    eval(
        i3drsgm,
        dataset_folder=DATASET_FOLDER,
        display_images=DISPLAY_IMAGES,
        min_disp=MIN_DISP,
        disp_range=DISP_RANGE,
        window_size=WINDOW_SIZE,
        pyramid_level=PYRAMID_LEVEL,
        interp=False
    )