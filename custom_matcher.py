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

# initalise matcher
default_min_disp = 0
default_num_disparities = 25
default_block_size = 3
default_uniqueness_ratio = 15
default_texture_threshold = 5
default_speckle_size = 0
default_speckle_range = 500

RESULTS_CSV_PATH = "cvbm_downfill_eval_results.csv"
cv_matcher = cv2.StereoBM_create()
calc_block = (2 * default_block_size + 5)
cv_matcher.setBlockSize(calc_block)
cv_matcher.setMinDisparity(default_min_disp)
cv_matcher.setNumDisparities(16*(default_num_disparities+1))
cv_matcher.setUniquenessRatio(default_uniqueness_ratio)
cv_matcher.setTextureThreshold(default_texture_threshold)
cv_matcher.setSpeckleWindowSize(default_speckle_size)
cv_matcher.setSpeckleRange(default_speckle_range)

metric_list = [" "]
metric_list.extend(Metric.get_metrics_list())
metric_list.append("bad200_maskerr")
metric_list.append("rms_maskerr")
with open(RESULTS_CSV_PATH, mode='w', newline='') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    results_writer.writerow(metric_list)

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def compute_custom_match(left_image, right_image):
    window_size = 128
    step_size = 32
    print("Running custom matcher...")
    # loop over a sliding window of the left image
    for (image_win_x, image_win_y, image_window) in sliding_window(right_image, stepSize=step_size, windowSize=(window_size, window_size)):
        # if the window does not meet our desired window size, ignore it
        if image_window.shape[0] != window_size or image_window.shape[1] != window_size:
            continue

        #right_display = right_image.copy()
        #cv2.rectangle(right_display, (image_win_x, image_win_y), (image_win_x + window_size, image_win_y + window_size), 255, 2)
        #right_display = cv2.resize(right_display,(640,480))
        #cv2.imshow("Right window", right_display)
        
        for (search_win_x, search_win_y, search_window) in sliding_window(left_image, stepSize=step_size, windowSize=(window_size, window_size)):
            # if the window does not meet our desired window size, ignore it
            if search_window.shape[0] != window_size or search_window.shape[1] != window_size:
                continue
            
            # only search along single horizontal sweep
            if search_win_y > 0:
                break
            
            offset_search_win_y = search_win_y + image_win_y
            offset_search_win_x = search_win_x + image_win_x

            if (offset_search_win_x > left_image.shape[1] - window_size):
                break

            #left_display = left_image.copy()
            #cv2.rectangle(left_display, (offset_search_win_x, offset_search_win_y), (offset_search_win_x + window_size, offset_search_win_y + window_size), 255, 2)
            #left_display = cv2.resize(left_display,(640,480))
            #cv2.imshow("Left search", left_display)
            #cv2.waitKey(1)
    left_shape = left_image.shape
    disp = np.zeros((left_shape[0],left_shape[1]))
    return disp

def compute_match(cv_matcher, left_image, right_image):
    img_shape = left_image.shape
    downscale_target = 500
    if (img_shape[1] < downscale_target):
        downscale_factor = 1
    else:
        downscale_factor = int(round(float(img_shape[1])/float(downscale_target)))
    downscale_amount = 1.0/float(downscale_factor)
    # compute disparity on source size images
    cv_matcher.setMinDisparity(0)
    cv_matcher.setNumDisparities(16*25)
    orig_block_size = 5
    cv_matcher.setBlockSize((2 * orig_block_size + 5))
    disp_orig = cv_matcher.compute(left_image, right_image)
    disp_orig = disp_orig.astype(np.float32)
    up_size = (img_shape[1],img_shape[0])
    # downsample images
    left_image_down = cv2.resize(left_image,None,fx=downscale_amount,fy=downscale_amount,interpolation=cv2.INTER_AREA)
    right_image_down = cv2.resize(right_image,None,fx=downscale_amount,fy=downscale_amount,interpolation=cv2.INTER_AREA)
    # compute disparity on downsamples images
    downscale_block_size = 5
    cv_matcher.setMinDisparity(0)
    cv_matcher.setBlockSize((2 * downscale_block_size + 5))
    downscale_num_disparities = 16*int(25*downscale_amount)
    if (downscale_num_disparities < 16*3):
        downscale_num_disparities = 16*3
    cv_matcher.setNumDisparities(downscale_num_disparities)
    disp_down = cv_matcher.compute(left_image_down, right_image_down)
    disp_down = disp_down.astype(np.float32)

    plt.figure(4)
    plt.imshow(disp_down)

    # upscale disparity and correct disparity value
    disp_up = cv2.resize(disp_down, up_size,interpolation=cv2.INTER_CUBIC) * downscale_factor
    # compute valid pixel mask for full size disparity image
    mask_valid = np.zeros(disp_orig.shape,np.uint8)
    mask_valid[disp_orig!=-16] = 255
    # compute inverse mask
    not_mask_valid = cv2.bitwise_not(mask_valid)

    img1_bg = cv2.bitwise_and(disp_orig,disp_orig, mask = mask_valid)
    img2_fg = cv2.bitwise_and(disp_up,disp_up, mask = not_mask_valid)
    disp = cv2.add(img1_bg,img2_fg)

    return disp

def compute_match_orig(cv_matcher, left_image, right_image):
    orig_block_size = 5
    cv_matcher.setMinDisparity(0)
    cv_matcher.setNumDisparities(16*25)
    cv_matcher.setBlockSize((2 * orig_block_size + 5))
    disp = cv_matcher.compute(left_image, right_image)
    return disp

def compute_match_intensity(cv_matcher, left_image, right_image):
    # normalise left image
    cv2.normalize(left_image, left_image, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(right_image, right_image, 0, 255, cv2.NORM_MINMAX)
    # create masks of left image based on intensity
    mask1_intensity_l = cv2.inRange(left_image, 0, 50)
    mask2_intensity_l = cv2.inRange(left_image, 50, 100)
    mask3_intensity_l = cv2.inRange(left_image, 100, 150)
    mask4_intensity_l = cv2.inRange(left_image, 150, 200)
    mask5_intensity_l = cv2.inRange(left_image, 200, 255)

    # create masks of right image based on intensity
    mask1_intensity_r = cv2.inRange(right_image, 0, 50)
    mask2_intensity_r = cv2.inRange(right_image, 50, 100)
    mask3_intensity_r = cv2.inRange(right_image, 100, 150)
    mask4_intensity_r = cv2.inRange(right_image, 150, 200)
    mask5_intensity_r = cv2.inRange(right_image, 200, 255)

    left_image_1 = cv2.bitwise_and(left_image, left_image, mask = mask1_intensity_l)
    right_image_1 = cv2.bitwise_and(right_image, right_image, mask = mask1_intensity_r)
    left_image_2 = cv2.bitwise_and(left_image, left_image, mask = mask2_intensity_l)
    right_image_2 = cv2.bitwise_and(right_image, right_image, mask = mask2_intensity_r)
    left_image_3 = cv2.bitwise_and(left_image, left_image, mask = mask3_intensity_l)
    right_image_3 = cv2.bitwise_and(right_image, right_image, mask = mask3_intensity_r)
    left_image_4 = cv2.bitwise_and(left_image, left_image, mask = mask4_intensity_l)
    right_image_4 = cv2.bitwise_and(right_image, right_image, mask = mask4_intensity_r)
    left_image_5 = cv2.bitwise_and(left_image, left_image, mask = mask5_intensity_l)
    right_image_5 = cv2.bitwise_and(right_image, right_image, mask = mask5_intensity_r)

    disp_orig = cv_matcher.compute(left_image, right_image)
    disp_1 = cv_matcher.compute(left_image_1, right_image_1)
    disp_2 = cv_matcher.compute(left_image_2, right_image_2)
    disp_3 = cv_matcher.compute(left_image_3, right_image_3)
    disp_4 = cv_matcher.compute(left_image_4, right_image_4)
    disp_5 = cv_matcher.compute(left_image_5, right_image_5)

    mask1_valid_l = np.zeros(disp_1.shape,np.uint8)
    mask2_valid_l = np.zeros(disp_1.shape,np.uint8)
    mask3_valid_l = np.zeros(disp_1.shape,np.uint8)
    mask4_valid_l = np.zeros(disp_1.shape,np.uint8)
    mask5_valid_l = np.zeros(disp_1.shape,np.uint8)
    mask1_valid_l[disp_1!=-16] = 255
    mask2_valid_l[disp_2!=-16] = 255
    mask3_valid_l[disp_3!=-16] = 255
    mask4_valid_l[disp_4!=-16] = 255
    mask5_valid_l[disp_5!=-16] = 255

    #mask1_valid_l = mask1_valid_l + mask1_intensity_l
    #mask2_valid_l = mask2_valid_l + mask2_intensity_l
    #mask3_valid_l = mask3_valid_l + mask3_intensity_l
    #mask4_valid_l = mask4_valid_l + mask4_intensity_l
    #mask5_valid_l = mask5_valid_l + mask5_intensity_l

    not_mask1_intensity_l = cv2.bitwise_not(mask1_valid_l)
    not_mask2_intensity_l = cv2.bitwise_not(mask2_valid_l)
    not_mask3_intensity_l = cv2.bitwise_not(mask3_valid_l)
    not_mask4_intensity_l = cv2.bitwise_not(mask4_valid_l)
    not_mask5_intensity_l = cv2.bitwise_not(mask5_valid_l)

    img1_bg = cv2.bitwise_and(disp_1,disp_1,mask = not_mask2_intensity_l)
    img2_fg = cv2.bitwise_and(disp_2,disp_2,mask = mask2_valid_l)

    disp = cv2.add(img1_bg,img2_fg)

    img2_bg = cv2.bitwise_and(disp,disp,mask = not_mask3_intensity_l)
    img3_fg = cv2.bitwise_and(disp_3,disp_3,mask = mask3_valid_l)

    disp = cv2.add(img2_bg,img3_fg)

    img3_bg = cv2.bitwise_and(disp,disp,mask = not_mask4_intensity_l)
    img4_fg = cv2.bitwise_and(disp_4,disp_4,mask = mask4_valid_l)

    disp = cv2.add(img3_bg,img4_fg)

    img4_bg = cv2.bitwise_and(disp,disp,mask = not_mask5_intensity_l)
    img5_fg = cv2.bitwise_and(disp_5,disp_5,mask = mask5_valid_l)

    disp = cv2.add(img4_bg,img5_fg)

    return disp

def clean_disp(disp,ndisp):
    disp = disp.astype(np.float32)
    disp[disp==99999]=0.0
    disp[disp<=0]=0.0
    disp = np.nan_to_num(disp, nan=0.0,posinf=0.0,neginf=0.0)
    disp[disp>=ndisp]=ndisp
    return disp

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
    test_disp_image_orig = compute_match_orig(cv_matcher, left_image_grey, right_image_grey)
    test_disp_image = compute_match(cv_matcher, left_image_grey, right_image_grey)
    #test_disp_image = compute_custom_match(left_image,right_image)
    # Record elapsed time for simulated match
    elapsed_time = timer.elapsed()
    test_disp_image = test_disp_image.astype(np.float32) / 16.0
    test_disp_image_orig = test_disp_image_orig.astype(np.float32) / 16.0

    if (test_disp_image is not None):
        test_disp_image = clean_disp(test_disp_image,ndisp)
        test_disp_image_orig = clean_disp(test_disp_image_orig,ndisp)
        test_disp_image = test_disp_image.astype(ground_truth_disp_image.dtype)
        test_disp_image_orig = test_disp_image_orig.astype(ground_truth_disp_image.dtype)
        if (scene_data == "Teddy" or scene_data == "Art"):
            test_disp_image = np.rint(test_disp_image)
            test_disp_image_orig = np.rint(test_disp_image_orig)

        ground_truth_disp_image[ground_truth_disp_image<=0]=0.0
        ground_truth_disp_image = np.nan_to_num(ground_truth_disp_image, nan=0.0,posinf=0.0,neginf=0.0)
        ground_truth_disp_image[ground_truth_disp_image>=ndisp]=ndisp

        ground_truth_mask_invalid = ground_truth_disp_image.copy()
        ground_truth_mask_invalid[test_disp_image_orig==0] = 0.0

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

        plt.figure(1)
        plt.imshow(ground_truth_disp_image)
        plt.figure(2)
        plt.imshow(test_disp_image_orig)
        plt.figure(3)
        plt.imshow(test_disp_image)
        #plt.show()
    else:
        print("Matching failed")
        results_row = [scene_name]
        with open(RESULTS_CSV_PATH, mode='a', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(results_row)