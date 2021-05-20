import cv2
import numpy as np

img = cv2.imread('./image/1.jfif')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray,None)

# img=cv2.drawKeypoints(gray,kp)

# cv2.imwrite('sift_keypoints.jpg',img)


cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('sift_keypoints_ditto.jpg',img)

kp,des=sift.compute(gray,kp)

print('number of key-points:',len(kp))
print('shape of SIFT features:',des.shape)