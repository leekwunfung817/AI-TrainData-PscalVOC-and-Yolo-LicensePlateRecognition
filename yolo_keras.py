# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3

# pip3 install matplotlib
# pip3 install flask

import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def r_predefined_classes(fun_dir):
	i = 0
	count = 0
	arr={}
	for line in open(fun_dir+'/data/predefined_classes.txt','r').readlines():
		arr[line.strip()]=i
		i+=1
	return arr

def predefined_classes(fun_dir):
	i = 0
	count = 0
	arr=[]
	for line in open(fun_dir+'/data/predefined_classes.txt','r').readlines():
		arr.append(line.strip())
		i+=1
	return arr

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1


		self.info = {
			'xmin':xmin,
			'ymin':ymin,
			'xmax':xmax,
			'ymax':ymax,
			'objness':objness,
			'classes':classes,
			'self.label':self.label,
			'self.score':self.score,
		}

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh

	for i in range(grid_h*grid_w):
		row = i // grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	# if union is 0:
	# 	return float(intersect)
	return float(intersect) / union

def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	# print(' >>>>> ',range(nb_class))
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		# print(' >>>>> >>>>> ',range(len(sorted_indices)))
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			# print(' >>>>> >>>>> >>>>> ',range(i+1, len(sorted_indices)))
			# print(' >>>>> >>>>> >>>>> ',nms_thresh)
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]

				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0

def image_preprocess(image):
	# <comment 1> print(' image @@ --',image,image.size)
	# convert to numpy array
	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image

# load and prepare an image
def load_image_pixels(filename, shape):
	# load the image to get its shape
	image = load_img(filename)
	width, height = image.size
	# load the image with the required size
	image = load_img(filename, target_size=shape)
	# <comment 1> print('before image_preprocess:::::',image.size)
	image = image_preprocess(image)

	return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(img, v_boxes, v_labels, v_scores):
	# # load the image
	# data = pyplot.imread(filename)
	# # plot the image
	# pyplot.imshow(data)
	# # get the context for drawing boxes
	# ax = pyplot.gca()
	# # plot each box
	arr=[]
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1

		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='white')
		# <comment 1> print('rect:',rect)

		(x,y)=rect.xy
		if x<0 or y<0:
			continue

		a_arr={
			'x1':x1, 
			'y1':y1, 
			# 'x2':x2,
			# 'y2':y2, 
			# 'x':x,
			# 'y':y,
			'width':width,
			'height':height,
			'labels':v_labels[i], 
			'scores':v_scores[i]
		}
		# <comment 1> print('a_arr:',a_arr)
		arr.append(a_arr)
		
		# draw the box
		# ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		# pyplot.text(x1, y1, label, color='white')
	# show the plot
	# pyplot.show()
	return arr






model_buffer = {}

def predict_bytes(
	h5_file,
	image,
	labels,
	input_w,
	input_h
):
	# <comment 1> print('predict_bytes ========== ========== ========== ========== ========== ')

	# print('image @@@@@@@@@@ ',image)
	# <comment 1> print('image @@@@@@@@@@ ',image.size)
	# <comment 1> print('image @@@@@@@@@@ ',image.shape)
	if h5_file not in model_buffer:
		# <comment 1> print('model_buffer before ',model_buffer)
		model_buffer[h5_file] = load_model(h5_file, compile=False)
		# <comment 1> print('model_buffer after ',model_buffer)

	# <comment 1> print('load_model @@@@@@@@@@ ')
	
	# make prediction
	yhat = model_buffer[h5_file].predict(image)
	# print('yhat',yhat,' @@@@@ @@@@@ @@@@@ @@@@@ @@@@@ ')

	# <comment 1> print('predict @@@@@@@@@@ ')

	# summarize the shape of the list of arrays
	# <comment 1> print('shape:',[a.shape for a in yhat])
	# define the anchors
	anchors = [[81,82,  135,169,  344,319], [10,14,  23,27,  37,58]]
	# define the probability threshold for detected objects
	class_threshold = 0.6
	boxes = list()
	for i in range(len(yhat)):
		# decode the output of the network
		boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
	# print('predict data:',boxes)
	# correct the sizes of the bounding boxes for the shape of the image
	# print('image:',image)
	# <comment 1> print('image.shape:',image.shape)
	a,image_w,image_h,d=image.shape
	correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
	# suppress non-maximal boxes
	do_nms(boxes, 0.5)
	# get the details of the detected objects
	v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
	# summarize what we found

	# result = []
	# for i in range(len(v_boxes)):
	# 	info=v_boxes[i].info
	# 	print(v_labels[i], v_scores[i], info)
		
	# 	info['scores']=v_scores[i]
	# 	info['label']=v_labels[i]
	# 	del info['classes']
	# 	result.append(info)
	
	# draw what we found
	# <comment 1> print('labels:::',labels)
	# <comment 1> print('v_labels:::',v_labels)
	return draw_boxes(image, v_boxes, v_labels, v_scores)
def predict(
	h5_file,
	photo_filename,

	# define the labels
	labels,
	# define the expected input shape for the model
	input_w,
):
	# load and prepare image
	image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
	predict_bytes(
		h5_file,
		image, 
		labels,
		input_w,input_h
	)

# http://flask.pocoo.org/docs/patterns/fileuploads/
import os
import io
import numpy as np

import codecs, json 
import base64

from matplotlib import cm

from PIL import Image, ImageDraw 
from PIL import ImageFont
font = ImageFont.load_default()

from datetime import datetime

def FlaskImageProcess(h5_path,PIL_Image,nn_w,nn_h,labels):

	ori_PIL_Image = PIL_Image.copy()
	(ori_w,ori_h)=PIL_Image.size

	img = PIL_Image.resize((nn_w, nn_h),Image.ANTIALIAS)
	predict_Image = img.copy()
	img = image_preprocess(img)
	# print('predict_Image',predict_Image)

	timestamp = datetime.now()

	arr=predict_bytes(h5_path,img,labels,nn_w,nn_h)

	# <comment 1> print('predict duartion 1:',datetime.now() - timestamp)
	timestamp = datetime.now()
	# <comment 1> print('img:',img.shape)

	# <comment 1> print('output:',arr)
	for ele in arr:
		x1=ele['x1']
		y1=ele['y1']
		width=ele['width']
		height=ele['height']

		label=ele['labels']
		scores=ele['scores']

		
		ImageDraw.Draw(predict_Image).rectangle([(x1,y1),(x1+width,y1+height)], outline ="red")

		w_p = (ori_w/nn_w)
		x1=x1*w_p
		width=width*w_p

		h_p = (ori_h/nn_h)
		y1=y1*h_p
		height=height*h_p

		ele['x1']=x1
		ele['y1']=y1
		ele['width']=width
		ele['height']=height
		draw=ImageDraw.Draw(PIL_Image)

		draw.text(tuple((x1+width,y1+height)), label+' ('+str(scores)+')', font=font)
		draw.rectangle([(x1,y1),(x1+width,y1+height)], outline ="red")

		draw.text(tuple((x1,y1)), '\n('+str(int(x1))+','+str(int(y1))+')', font=font)
		draw.text(tuple((x1,y1+height)), '\n('+str(int(x1))+','+str(int(y1+height))+')', font=font)
		draw.text(tuple((x1+width,y1)), '\n('+str(int(x1+width))+','+str(int(y1))+')', font=font)
		draw.text(tuple((x1+width,y1+height)), '\n('+str(int(x1+width))+','+str(int(y1+height))+')', font=font)

	# <comment 1> print('predict duartion 2:',datetime.now() - timestamp)
	timestamp = datetime.now()

	demo_PIL_Image = PIL_Image.copy()

	# <comment 1> print('predict duartion 3:',datetime.now() - timestamp)
	timestamp = datetime.now()

	return (predict_Image,ori_PIL_Image,demo_PIL_Image,arr)


