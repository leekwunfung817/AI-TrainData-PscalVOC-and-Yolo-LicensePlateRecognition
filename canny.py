import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import mean, sqrt, square, arange

def process_cell(str,img=None):
	listi = str.split("-")
	for x in listi:
		if x=='grey':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		if x=='medianBlur':
			img = cv2.medianBlur(img,5)
		if x=='adaptiveThreshold':
			img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

		if x=='HistogramEqualization':
			equ = cv2.equalizeHist(img)
			# img = np.hstack((img,equ)) #stacking images side-by-side
	return img

def process(path,order):
	img = cv2.imread(path,1)
	# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# img = cv2.medianBlur(img,5)
	# img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
	img=process_cell(order,img)
	# img = cv2.resize(img, (800, 600))
	print('path:',path,img.shape)
	h,w=img.shape
	# edges = cv2.Canny(img,h,w,th)


	# cv2.imshow('My Image', img)
	# # cv2.imshow('My edges', edges)
	# cv2.waitKey(500)
	# cv2.destroyAllWindows()
	return img


	# plt.subplot(121)
	# plt.imshow(img)

	# plt.title('Original Image')
	# plt.xticks([]), plt.yticks([])
	# plt.subplot(122)
	# plt.imshow(edges,cmap = 'gray')
	# plt.title('Edge Image')
	# plt.xticks([])
	# plt.yticks([])

	# plt.show()

import sys
print(sys.argv)
# exit()

import argparse

# folder='D:\\CPOS AI\\OD_OCR\\data\\'
order=sys.argv[1]
from_folder=sys.argv[2]
to_folder=sys.argv[3]

# process_cell(order)
# exit()

# path=folder+'17_1_i_h_01649433___License Plate.jpg'
# canny(path,0)
# canny(path,50)
# canny(path,100)
# canny(path,150)
# canny(path,200)
# canny(path,255)
# exit()

import os
if not os.path.exists(to_folder):
	os.makedirs(to_folder)

for filename in os.listdir(from_folder):
	if filename[-4:]=='.jpg':
		print('filename:',filename)
		# rms = sqrt(mean(square(img)))
		# avg = np.average(img)
		
		cv2.imwrite(to_folder+filename, process(from_folder+filename,order) ) 


		# canny(folder+filename,50)
		# canny(folder+filename,100)
		# canny(folder+filename,150)
		# canny(folder+filename,200)
		# canny(folder+filename,255)


'''
cd /d "D:/CPOS AI/"
python canny.py "grey-medianBlur,adaptiveThreshold" "D:/CPOS AI/ImagePreprocessing/Origin/" "D:/CPOS AI/ImagePreprocessing/Canny/" 
python canny.py "grey-medianBlur" "D:/CPOS AI/ImagePreprocessing/Origin/" "D:/CPOS AI/ImagePreprocessing/grey-medianBlur/" 
python canny.py "grey" "D:/CPOS AI/ImagePreprocessing/Origin/" "D:/CPOS AI/ImagePreprocessing/grey/" 
python canny.py "grey-adaptiveThreshold" "D:/CPOS AI/ImagePreprocessing/Origin/" "D:/CPOS AI/ImagePreprocessing/grey-adaptiveThreshold/" 
python canny.py "grey-HistogramEqualization" "D:/CPOS AI/ImagePreprocessing/Origin/" "D:/CPOS AI/ImagePreprocessing/grey-HistogramEqualization/" 


'''