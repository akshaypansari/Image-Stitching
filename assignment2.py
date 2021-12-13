import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from function import *
import sys
#width
DEBUG = False
SHOW_IMAGE = False
SIZE = (480, 360)#SIZE has width * height 
DETECTOR = 'sift'
base_image_multiplier = 1
if (len(sys.argv)==3):
    DETECTOR = sys.argv[2]
    print("Using {} as detector".format(DETECTOR))
elif(len(sys.argv)==4):
    DETECTOR = sys.argv[2]
    base_image_multiplier = int(sys.argv[3])
    print("Using {} as detector,  {} as multiplier ".format(DETECTOR, base_image_multiplier))
IMG_FOLDER_NAMES = [sys.argv[1]]
OUTFOLDER_NAME = "/"
# IMG_FILE_NAME = "sys.argv[1]"

if __name__ == '__main__':
    # print("We are using folder named {}".format(IMG_FOLDER_NAME))
    for IMG_FOLDER_NAME in IMG_FOLDER_NAMES:
        print("We are using folder named {}".format(IMG_FOLDER_NAME))

        images_name, images_coloured , images_grey = read_images(IMG_FOLDER_NAME, size = SIZE, grey_scale = False)
        # if(len(images_grey)==0):
        #     print("Youhave passed grey_scale in read_images as False. Thought no need to worry. you can't use the grey scale images now.")

    #kps and features are list of list keypoints for iamgs

        kps, features = find_keypoints_features (images_coloured, DETECTOR)

        #create matcher
        #display the matches and verify which all are correct
        edgelist, frequent_id  = find_matches_for_image_list(images_name, images_coloured, features)

        master_img = find_max_width_height(images_coloured, frequent_id,base_image_multiplier, size = SIZE)

        #create a big image and send this iamge to the center of that 
        output_image = create_panorama(images_coloured, frequent_id, edgelist, master_img, DETECTOR, good_distance = 0.65)

        # cv2.imwrite(OUTFOLDER_NAME+IMG_FOLDER_NAME+'_' +IMG_FILE_NAME,output_image)
        cv2.imwrite(IMG_FOLDER_NAME+'_' +"out.jpg",output_image)
        # img_yuv = cv2.cvtColor(output_image, cv2.COLOR_BGR2YUV)

        # # equalize the histogram of the Y channel
        # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # # convert the YUV image back to RGB format
        # img_yuv = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 
  
        # # stacking images side-by-side 
        # res = np.hstack((output_image, img_yuv)) 
          
        # # show image input vs output 
        # show_image( img_yuv,0) 
        # show_image( res,0) 
        print("done")






'''



img1 = cv2.imread('2/1.jpg')#query image
# img_ = cv2.resize(img_, (0,0), fx=1, fy=1)
img1 = cv2.resize(img1, WIDTH)
# img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('2/2.jpg')#train image
# img = cv2.resize(img, (0,0), fx=1, fy=1)
img2 = cv2.resize(img2, WIDTH)
# img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# detect and extract features from the image
def find_keypoints_features(list_of_images, name_of_descriptor):
    if name_of_descriptor == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif name_of_descriptor == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif name_of_descriptor == 'brisk':
        descriptor = cv2.BRISK_create()
    elif name_of_descriptor == 'akaze':
        descriptor = cv2.BRISK_create()
    elif name_of_descriptor == 'orb':
        descriptor = cv2.ORB_create()
    kps = []
#     print("kps_shape", kps.shape)
    features = []
#     print("features_shape", features.shape)
    for i in range(len(list_of_images)):
        (kps_, features_) = descriptor.detectAndCompute(list_of_images[i], None)
        kps.append(kps_)
        features.append(features_)
    return (kps, features)
# '''

# def createMatcher(name_of_descriptor,crossCheck):
#     "Create and return a Matcher Object "
#     '''# For SIFT and SURF OpenCV recommends using Euclidean distance. For other feature extractors like ORB and BRISK,
#     Hamming distance is suggested.
#     crossCheck bool parameter indicates whether the two features have to match each other to be considered valid
#     for a pair of features (f1, f2) to considered valid, 
#     f1 needs to match f2 and f2 has to match f1 as the closest match as well'''
    
#     if name_of_descriptor == 'sift' or name_of_descriptor == 'surf':
#         bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
#     elif name_of_descriptor == 'orb' or name_of_descriptor == 'brisk' or name_of_descriptor=='akaze':
#         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
#     return bf
# kps, fs =  find_keypoints_features([img1, img2], "orb")
# # kp2, f2 =  find_keypoints_features([img2], "orb")
# # print(kps)
# cv2.imshow('original_image_left_keypoints1',cv2.drawKeypoints(img1,kps[0],None))
# cv2.imshow('original_image_left_keypoints2',cv2.drawKeypoints(img2,kps[1],None))

# cv2.waitKey(1000)
# cv2.destroyAllWindows()
# #create_matcher
# bf = createMatcher('orb', crossCheck=True)
# matches = bf.match(fs[0],fs[1])
# matches = sorted(matches, key = lambda x:x.distance)
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    flags = 2)
# img3 = cv2.drawMatches(img1,kps[0],img2,kps[1],matches,None )
# # plt.imshow(img3),plt.show()
# cv2.imshow('orb',img3)

# cv2.waitKey(10000)
# cv2.destroyAllWindows()


# # Extract location of good matches
# points1 = np.zeros((len(matches), 2), dtype=np.float32)
# points2 = np.zeros((len(matches), 2), dtype=np.float32)

# for i, match in enumerate(matches):
#     points1[i, :] = kps[0][match.queryIdx].pt
#     points2[i, :] = kps[1][match.trainIdx].pt

# # Find homography
# h, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
# # print(len(matches))
# print("h", h)
# dst = cv2.warpPerspective(img1,h,(img2.shape[1] + img1.shape[1], img2.shape[0]+img2.shape[0]))
# cv2.imshow("dst", dst)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
# dst[0:img2.shape[0],0:img2.shape[1]] = img2
# cv2.imshow("original_image_stitched.jpg", dst)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
# def trim(frame):
#     #crop top
#     if not np.sum(frame[0]):
#         return trim(frame[1:])
#     #crop top
#     if not np.sum(frame[-1]):
#         return trim(frame[:-2])
#     #crop top
#     if not np.sum(frame[:,0]):
#         return trim(frame[:,1:])
#     #crop top
#     if not np.sum(frame[:,-1]):
#         return trim(frame[:,:-2])
#     return frame
# cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #cv2.imsave("original_image_stitched_crop.jpg", trim(dst))

