import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from ReadCameraModel import ReadCameraModel

fx, fy, cx, cy, _, LUT = ReadCameraModel('./Oxford_dataset_reduced/model')

K = np.eye(3)
K[0,0] = fx
K[1,1] = fy
K[0,2] = cx
K[1,2] = cy

images =  sorted(os.listdir('./Oxford_dataset_reduced/images'))[1:]

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

counter = 0
poses = []


for filename in images:
    prev_img = cv2.cvtColor(cv2.imread(os.path.join('./Oxford_dataset_reduced/images', filename),flags=-1), cv2.COLOR_BayerGR2BGR)

    if counter > 0:
        img = cv2.cvtColor(cv2.imread(os.path.join('./Oxford_dataset_reduced/images', filename),flags=-1), cv2.COLOR_BayerGR2BGR)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        
        knn_match = bf.knnMatch(querry_desc,descriptors,k=2)
        points1 = []
        points2 = []

        for i,j in knn_match:
            if i.distance < 0.73*j.distance:
                points2.append(keypoints[i.trainIdx].pt)
                points1.append(final_points[i.queryIdx].pt)
                
        points1 = np.int32(points1)
        points2 = np.int32(points2)
        feature, mask = cv2.findFundamentalMat(points1,points2,cv2.FM_LMEDS,confidence=.99)
        
        E = np.dot(np.dot(K.T,feature), K)
        poses.append(cv2.recoverPose(E, points1, points2, K))
        
      
    if counter > 0:
        final_points = keypoints
        querry_desc = descriptors
    else:
        final_points, querry_desc = sift.detectAndCompute(prev_img,None)
    
    counter += 1


matrix = np.eye(4)
o = np.array([[0, 0, 0, 1]]).T

x = []
y = []
z = []

for _, R, t, _ in poses:
    temp = (np.concatenate((R, t), axis=1))
    temp_matrix = np.concatenate((temp, np.array([[0, 0, 0, 1]])))
    matrix = np.dot(matrix, np.linalg.inv(temp_matrix))
    dot = np.dot(matrix, o)
    x.append(dot[0])
    z.append(dot[1])
    y.append(dot[2])
    
x = np.array(x)
y = np.array(y)
z = np.array(z)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_zlim3d(-40, 40)
ax.scatter3D(x, y, z)
plt.show()

plt.plot(x, y, linewidth=5)
plt.show()