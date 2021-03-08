import os
import subprocess
import csv
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
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

RESULTS_CSV_PATH = "cm_eval_results.csv"
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

def sliding_window(image, stepSize, windowSize, xOffset=0, yOffset=0, xMax=None, yMax=None):
	# slide a window across the image
    if (xMax is None):
        xMax = image.shape[1]
    if (yMax is None):
        yMax = image.shape[0]
    for y in range(yOffset, yMax-windowSize[1], stepSize[1]):
        for x in range(xOffset, xMax-windowSize[0], stepSize[0]):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def compute_custom_match(left_image, right_image):
    window_size = (24,24)
    step_size = (12,12)
    max_disp = 200
    minScore = 1.0
    match_method = cv2.TM_CCORR_NORMED # TM_SQDIFF / TM_SQDIFF_NORMED / TM_CCORR / TM_CCORR_NORMED / TM_CCOEFF / TM_CCOEFF_NORMED
    print("Running custom matcher...")
    # downsample images
    downscale_factor = 1
    downscale_amount = 1.0/float(downscale_factor)
    up_size = (left_image.shape[1],left_image.shape[0])
    left_image_down = cv2.resize(left_image,None,fx=downscale_amount,fy=downscale_amount,interpolation=cv2.INTER_AREA)
    right_image_down = cv2.resize(right_image,None,fx=downscale_amount,fy=downscale_amount,interpolation=cv2.INTER_AREA)

    left_shape = left_image_down.shape
    disp = np.zeros((left_shape[0],left_shape[1]))
    # loop over a sliding window of the left image
    for (image_win_x, image_win_y, image_window) in sliding_window(right_image_down, stepSize=step_size, windowSize=window_size):
        # right_window = right_image_down.copy()
        # cv2.rectangle(right_window, (image_win_x, image_win_y), (image_win_x + window_size[1], image_win_y + window_size[0]), 255, 2)
        # right_window_display = cv2.resize(right_window,(640,480))
        # cv2.imshow("Rect window", right_window_display)
        # cv2.waitKey(1)

        # cv2.imshow("Image window", image_window)
        # cv2.waitKey(1)

        mask_crop = left_image_down[image_win_y:image_win_y+window_size[1],image_win_x:image_win_x+window_size[0]+int(max_disp * downscale_amount)].copy()

        # cv2.imshow("Match mask", mask_crop)
        # cv2.waitKey(1)

        #process_timer.start()
        result = cv2.matchTemplate(mask_crop, image_window, match_method)
        #print("match template: {}".format(process_timer.elapsed()))
        #process_timer.start()
        cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
        #print("normalise & minmax: {}".format(process_timer.elapsed()))
        if (match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED):
            matchLoc = minLoc
            matchScore = 1-_minVal
        else:
            matchLoc = maxLoc
            matchScore = _maxVal

        #process_timer.start()
        pixel_disp = matchLoc[0]
        #print(pixel_disp)
        if pixel_disp < (max_disp * downscale_amount) and matchScore >= minScore:
            disp[image_win_y:image_win_y + window_size[1],image_win_x+pixel_disp:image_win_x + window_size[0]+pixel_disp] = pixel_disp
        #print("check valid disparity: {}".format(process_timer.elapsed()))

            #if (image_win_x == 0):

        # left_window = left_image_down.copy()
        # cv2.rectangle(left_window, (image_win_x + pixel_disp, image_win_y), (image_win_x + pixel_disp + window_size[0], image_win_y + window_size[1]), 255, 2)
        # left_window = cv2.resize(left_window,(640,480))
        # cv2.imshow("Window found", left_window)
        # cv2.waitKey(1)

        # if (image_win_x == 0):
        #     plt.figure(1)
        #     plt.imshow(left_window)
        #     plt.figure(2)
        #     plt.imshow(disp)
        #     plt.show()

    # upscale disparity and correct disparity value
    disp_up = cv2.resize(disp, up_size, interpolation=cv2.INTER_CUBIC) * downscale_factor

    disp_up = disp_up * 16
    return disp_up

def compute_downinfill_match(cv_matcher, left_image, right_image):
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
    #test_disp_image_orig = compute_match_orig(cv_matcher, left_image_grey, right_image_grey)
    #test_disp_image = compute_match(cv_matcher, left_image_grey, right_image_grey)
    test_disp_image = compute_custom_match(left_image_grey,right_image_grey)
    # Record elapsed time for simulated match
    elapsed_time = timer.elapsed()
    test_disp_image = test_disp_image.astype(np.float32) / 16.0
    #test_disp_image_orig = test_disp_image_orig.astype(np.float32) / 16.0

    if (test_disp_image is not None):
        test_disp_image = clean_disp(test_disp_image,ndisp)
        #test_disp_image_orig = clean_disp(test_disp_image_orig,ndisp)
        test_disp_image = test_disp_image.astype(ground_truth_disp_image.dtype)
        #test_disp_image_orig = test_disp_image_orig.astype(ground_truth_disp_image.dtype)
        if (scene_data == "Teddy" or scene_data == "Art"):
            test_disp_image = np.rint(test_disp_image)
            #test_disp_image_orig = np.rint(test_disp_image_orig)

        ground_truth_disp_image[ground_truth_disp_image<=0]=0.0
        ground_truth_disp_image = np.nan_to_num(ground_truth_disp_image, nan=0.0,posinf=0.0,neginf=0.0)
        ground_truth_disp_image[ground_truth_disp_image>=ndisp]=ndisp

        ground_truth_mask_invalid = ground_truth_disp_image.copy()
        ground_truth_mask_invalid[ground_truth_disp_image==0] = 0.0

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
        #plt.figure(2)
        #plt.imshow(test_disp_image_orig)
        plt.figure(3)
        plt.imshow(test_disp_image)
        #plt.show()
    else:
        print("Matching failed")
        results_row = [scene_name]
        with open(RESULTS_CSV_PATH, mode='a', newline='') as results_file:
            results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            results_writer.writerow(results_row)