import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx
from networkx.algorithms import tree
import queue

#1 m 2 and 4 are working and keep sets

DEBUG = False
SHOW_IMAGE = False
SIZE = (300,200)#SIZE has width * height 
# DETECTOR = 'sift'
SHOWTIME = 1000

base_image_multiplier  = 2

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def show_image(img, showtime = 1000):
    cv2.imshow('img',img)
    cv2.waitKey(showtime)
    cv2.destroyAllWindows()


def read_images(img_dir, size = SIZE, grey_scale = False):
    '''Read images from the directory and then read the images'''
    img_list = []
    for files in os.listdir(img_dir):
        if(files[-4:]!='.jpg'):
            continue
        img_list.append(img_dir+'/'+ files)
    if DEBUG:
        print(img_list)

    images_colour = []
    images_grey = []
    for image_name in img_list:
        img_c = cv2.imread(image_name)
        img_c = cv2.resize(img_c, size)
        images_colour.append(img_c)
        if DEBUG:
            print( img_c.shape)
            show_image(img_c)
        if grey_scale:
            img_g = cv2.cvtColor(img_c,cv2.COLOR_BGR2GRAY)
            images_grey.append(img_g)
    if DEBUG: 
        print(len(images_grey))
    return (img_list, images_colour, images_grey)

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
    for i, image  in enumerate(list_of_images):
        (kps_, features_) = descriptor.detectAndCompute(list_of_images[i], None)
        kps.append(kps_)
        features.append(features_)
    return (kps, features)


def show_draw_keypoints(image_list, keypoints_list):
    '''It contains list of keypoints for each image computed by one of the descriptors'''
    for i, kps in enumerate(keypoints_list):

        cv2.imshow('original_image_left_keypoints'+str(i),cv2.drawKeypoints(image_list[i],kps,None))
        cv2.waitKey(SHOWTIME)
        cv2.destroyAllWindows()

def find_matches_for_image_list(img_names, img_list, descriptors):
    G = nx.Graph()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    good_distance = 0.65
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matchings = []
# make a function which tells that we have some error
    for i, des1 in enumerate(descriptors):
        for j , des2 in enumerate(descriptors):
            if(j>i):
                print(i,j)
                matches = flann.knnMatch(des1, des2 ,k=2)
                good = []
                for m,n in matches:
                    if m.distance < good_distance*n.distance:
                        good.append(m)
                print("leng of good mathces  {} {} = ".format(img_names[i], img_names[j]) , str(len(good)))

                G.add_edge(i, j , weight = len(good))
                matching_list = [img_names[i], img_names[j], len(good)]
                matchings.append(matching_list)
    matchings.sort(key =lambda x: x[2], reverse = True)
    print(matchings)
    if DEBUG:
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 10]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 10]
        pos = nx.spring_layout(G)  # positions for all nodes

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # edges
        nx.draw_networkx_edges(G, pos, edgelist=elarge,
                               width=6)
        nx.draw_networkx_edges(G, pos, edgelist=esmall,
                               width=6, alpha=0.5, edge_color='b', style='dashed')

        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        # labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
        plt.axis('off')
        plt.show()
    mst = tree.maximum_spanning_edges(G, algorithm='kruskal', data=False)
    edgelist = list(mst)
    sorted(sorted(e) for e in edgelist)
    print(edgelist)
    node_frequency = {}
    for u, v in edgelist:
        if(u not in node_frequency.keys()):
            node_frequency[u]=1
        else:
            node_frequency[u] +=1
        if(v not in node_frequency.keys()):
            node_frequency[v] = 1
        else:
            node_frequency[v] +=1
    print(node_frequency)
    # print()
    image_index_with_most_freq = max(node_frequency.keys(), key=(lambda key: node_frequency[key]))

    return edgelist, image_index_with_most_freq


def find_max_width_height(img_list,frequent_id, base_image_multiplier = 1,  size = SIZE):
    height = 0
    width = 0
    for i, image in enumerate(img_list):

        height += image.shape[0]
        width += image.shape[1]
        print("height = {} and width = {}".format(height, width))

    max_height = int(base_image_multiplier*height )
    max_width =int( base_image_multiplier* width )
    img = np.zeros([max_height, max_width, 3],dtype=np.uint8)
    img.fill(0) # or img[:] = 255
    start_filling_x = max_width//2 - size[0]//2
    start_filling_y = max_height//2 - size[1]//2
    print(start_filling_x, start_filling_y)

    print("shape of image is {} and max width and height is {} {} and start_filling_x and start_filling_y is {} {} ".format((max_width, max_height), max_width, max_height, start_filling_x, start_filling_y))
    img[start_filling_y:start_filling_y+size[1], start_filling_x:start_filling_x+size[0]] = img_list[frequent_id]
    print("along y ",start_filling_y,start_filling_y+size[1],"along x " ,  start_filling_x,start_filling_x+size[0])
    show_image(cv2.resize(img,(1080, 720)), 2000)
    # show_image(img, 0)
    return img


def blending (img1, img2, alpha = 0.5):
    # master_img = cv2.bitwise_or(img1, img2)
    master_img = np.zeros(img1.shape, np.uint8)
    for i in range(img1.shape[0]):
        if(np.sum(img1[i,:,:])==0 and np.sum(img2[i,:,:])==0):
            continue
        for j in range(img1.shape[1]):
            if(np.sum(img1[i,j,:])==0 and np.sum(img2[i, j, :])==0):
                master_img[i,j,:] = img1[i,j,:]
            elif (np.sum(img1[i,j,:])==0 and np.sum(img2[i, j, :])>0):
                master_img[i,j,:] = img2[i,j,:]
            elif (np.sum(img1[i,j,:])>0 and np.sum(img2[i, j, :])==0):
                master_img[i,j,:] = img1[i,j,:]
            else:
                master_img[i,j,:] = alpha*img1[i,j,:]+(1-alpha)*img2[i,j,:]

    return master_img




def create_panorama(images_coloured, frequent_id, edgelist, master_img, descriptor = "sift", good_distance = 0.6):
    Edges = queue.Queue(len(edgelist))
    for edge in edgelist:
        Edges.put(edge)
    done_image = [frequent_id]
    while(not Edges.empty()):
        (u,v) = Edges.get()
        if ((u in done_image and v not in done_image) or (v in done_image and u not in done_image)):
            if(u in done_image):
                done_image.append(v)
                img1 = images_coloured[v]
                print(v)
            else:
                done_image.append(u)
                img1 = images_coloured[u]
                print(u)

            img2 = master_img
            kps, features  = find_keypoints_features([img1, img2], descriptor)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            # good_distance = 0.60
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(features[0], features[1] ,k=2)
            good = []
            for m,n in matches:
                if m.distance < good_distance*n.distance:
                    good.append(m)

            print("good_points = ", len(good))
            src_pts = np.float32([ kps[0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kps[1][m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist() 
            print(M)

            dst = cv2.warpPerspective(img1,M,(img2.shape[1], img2.shape[0]))
            show_image(cv2.resize(dst, (720, 480)), SHOWTIME)
            print(dst.shape)
            # have to make a mask and then do bitwise opeation            
            # master_img = cv2.addWeighted(dst,0.5,img2,0.5,0).astype('uint8') 

            # master_img = cv2.bitwise_or(img2, dst)
            master_img = blending(img2, dst)
            show_image(cv2.resize(master_img,(1080, 720)), SHOWTIME)         
        else:
            Edges.put((u,v))

    show_image(trim(cv2.resize(master_img,(1080, 720))), 5*SHOWTIME)
    return trim(cv2.resize(master_img,(1080, 720)))


#**********************************************TRY  *******************************************************





