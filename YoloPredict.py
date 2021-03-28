# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle


def predefined_classes():
	i = 0
	count = 0
	arr=[]
	for line in open('predefined_classes.txt','r').readlines():
		arr[i]=line.strip()
		i+=1
	return arr


clp_h5_path = 'D:\\CPOS License Plate Recognition\\8.0\\cpos_clp\\cpos-clp-yolov3-tiny-applicat.h5'
clp_w = 224
clp_h = 224
clp_labels = [
	"Car",
	"License Plate"
]
ocr_h5 = 'D:\\CPOS License Plate Recognition\\8.0\\cpos_ocr\\cpos-ocr-3-tiny-yolov3_final.h5'
ocr_w = 224
ocr_h = 160
ocr_labels = [
	"0",
	"1",
	"2",
	"3",
	"4",
	"5",
	"6",
	"7",
	"8",
	"9",
	"A",
	"B",
	"C",
	"D",
	"E",
	"F",
	"G",
	"H",
	"J",
	"K",
	"L",
	"M",
	"N",
	"P",
	"R",
	"S",
	"T",
	"U",
	"V",
	"W",
	"X",
	"Y",
	"Z"
]

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
		row = i / grid_w
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
	print(' image @@ --',image,image.size)
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
	print('before image_preprocess:::::',image.size)
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
		print('rect:',rect)

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
		print('a_arr:',a_arr)
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
	print('predict_bytes ========== ========== ========== ========== ========== ')

	# print('image @@@@@@@@@@ ',image)
	print('image @@@@@@@@@@ ',image.size)
	print('image @@@@@@@@@@ ',image.shape)
	if h5_file not in model_buffer:
		print('model_buffer before ',model_buffer)
		model_buffer[h5_file] = load_model(h5_file, compile=False)
		print('model_buffer after ',model_buffer)

	print('load_model @@@@@@@@@@ ')
	
	# make prediction
	yhat = model_buffer[h5_file].predict(image)
	print('yhat',yhat,' @@@@@ @@@@@ @@@@@ @@@@@ @@@@@ ')

	print('predict @@@@@@@@@@ ')

	# summarize the shape of the list of arrays
	print('shape:',[a.shape for a in yhat])
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
	print('image.shape:',image.shape)
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
from flask import Flask, request, redirect, url_for, send_from_directory
import _thread
import numpy as np

from PIL import Image #PIL pakage name is Pillow 
# from werkzeug import secure_filename
import codecs, json 
import base64

from matplotlib import cm

from PIL import Image, ImageDraw 
from PIL import ImageFont
font = ImageFont.load_default()

from datetime import datetime

app = Flask(__name__)

def FlaskImageProcess(h5_path,PIL_Image,nn_w,nn_h,labels):

	ori_PIL_Image = PIL_Image.copy()
	(ori_w,ori_h)=PIL_Image.size

	img = PIL_Image.resize((nn_w, nn_h),Image.ANTIALIAS)
	predict_Image = img.copy()
	img = image_preprocess(img)
	# print('predict_Image',predict_Image)

	timestamp = datetime.now()

	arr=predict_bytes(h5_path,img,labels,nn_w,nn_h)

	print('predict duartion 1:',datetime.now() - timestamp)
	timestamp = datetime.now()
	print('img:',img.shape)

	print('output:',arr)
	for ele in arr:
		x1=ele['x1']
		y1=ele['y1']
		width=ele['width']
		height=ele['height']

		labels=ele['labels']
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

		ImageDraw.Draw(PIL_Image).rectangle([(x1,y1),(x1+width,y1+height)], outline ="red")
		ImageDraw.Draw(PIL_Image).text(tuple((x1+width,y1+height)), labels+' ('+str(scores)+')', font=font)

	print('predict duartion 2:',datetime.now() - timestamp)
	timestamp = datetime.now()

	demo_PIL_Image = PIL_Image.copy()

	print('predict duartion 3:',datetime.now() - timestamp)
	timestamp = datetime.now()

	return (predict_Image,ori_PIL_Image,demo_PIL_Image,arr)

def ocr_process(filename,ocr,i):
	print('License Plate detected!!!!!')
	(ocr_predict_Image,ocr_ori_PIL_Image,ocr_demo_PIL_Image,ocr_arr) = FlaskImageProcess(ocr_h5,ocr,ocr_w,ocr_h,ocr_labels)

	ocr_path='ocr/'+filename+'_'+'License Plate'+'_'+str(i)+'.jpg'
	ocr_ori_PIL_Image.save(ocr_path)
	ocr_demo_PIL_Image.save(ocr_path+'.demo.jpg')
	ocr_predict_Image.save(ocr_path+'.nn.demo.jpg')

	avg_h=0
	avg_w=0
	for ocr_ele in ocr_arr:
		avg_h=avg_h+ocr_ele['height']
		avg_w=avg_w+ocr_ele['width']
	avg_h=avg_h/len(ocr_arr)
	avg_w=avg_w/len(ocr_arr)

	max_h=avg_h/2

	print('avg_h',avg_h)
	print('avg_w',avg_w,'\n\n')

	sort_arr={}
	for ocr_ele in ocr_arr:
		center_x=ocr_ele['x1']+(ocr_ele['width']/2)
		sort_arr[center_x]=ocr_ele
		# center_y=ocr_ele['y1']+(ocr_ele['height']/2)

	ocr_min_list=[]
	while True:

		min_=9999999999
		for ocr_i in range(len(ocr_arr)):
			ocr_ele=ocr_arr[ocr_i]

			center_x=ocr_ele['x1']+(ocr_ele['width']/2)
			if min_>center_x and center_x not in ocr_min_list:
				min_=center_x
		if min_==9999999999:
			break
		ocr_min_list.append(min_)


	print('\n\n ocr_min_list = ',ocr_min_list,' \n\n')
	ocr_ele_list=[]
	for ocr_min in ocr_min_list:
		for ocr_ele in ocr_arr:
			center_x=ocr_ele['x1']+(ocr_ele['width']/2)
			if center_x==ocr_min:
				ocr_ele_list.append(ocr_ele)
	print('\n\n ocr_ele_list = ',ocr_ele_list,' \n\n')

	ocr_result={}
	cur_char=0
	cur_y=None
	for ocr_ele in ocr_ele_list:
		center_y=ocr_ele['y1']+(ocr_ele['height']/2)
		if cur_y is None:
			cur_y=center_y
		dif=center_y-cur_y
		
		if dif<0:
			dif=dif*-1
		if dif>max_h:
			if center_y>cur_y:
				cur_char+=1
			else:
				cur_char-=1
		key=str(cur_char)+'_'

		if key not in ocr_result:
			ocr_result[key]=ocr_ele['labels']
		else:
			ocr_result[key]+=ocr_ele['labels']
	print('ocr_result:',ocr_result)


	# for 
	# 	ocr_range = range(len(ocr_arr))
	# 	for ocr_i in ocr_range:
	# 		print('ocr_i,ocr_arr',ocr_i,len(ocr_arr))
	# 		ocr_ele=ocr_arr[ocr_i]

	# 		center_x=ocr_ele['x1']+(ocr_ele['width']/2)
	# 		if center_x==min_:
	# 			ocr_ele_list.append(ocr_arr.pop(ocr_i))
	# 			ocr_range = range(len(ocr_arr))



def clp_process(filename,clp):
	(clp_predict_Image,ori_PIL_Image,demo_PIL_Image,arr) = FlaskImageProcess(clp_h5_path,clp,clp_w,clp_h,clp_labels)

	print('FlaskImageProcess:')
	print('FlaskImageProcess:',arr)

	clp_path='clp/'+filename
	ori_PIL_Image.save(clp_path)
	demo_PIL_Image.save(clp_path+'.demo.jpg')
	clp_predict_Image.save(clp_path+'.nn.demo.jpg')

	print('\n')
	ocr_list=[]
	i=1
	for ele in arr:
		crop_config = (ele['x1'],ele['y1'],ele['x1']+ele['width'],ele['y1']+ele['height'])
		print('ele:',ele,'\n')
		ocr = ori_PIL_Image.crop(crop_config)

		if ele['labels']=='License Plate':
			ele['result']=ocr_process(filename,ocr,i)
			ocr_list.append(ele)
		i=i+1
	return json.dumps(ocr_list)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['file']
		filename = file.filename
		return clp_process(filename,Image.open(file.stream))
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form action="" method=post enctype=multipart/form-data>
		<input type=file name=file>
		<input type=submit value=Upload>
	</form>
	
	'''
	# <img src="''' + u'data:img/jpeg;base64,'+data64.decode('utf-8') + '''">

FlaskImageProcess(
	ocr_h5,
	Image.open('D:\\CPOS License Plate Recognition\\8.0\\cpos_clp\\ocr\\21_1_i_h__20190703073338_68807221.jpg_License Plate_1.jpg'),
	ocr_w,
	ocr_h,
	ocr_labels
)
FlaskImageProcess(
	clp_h5_path,
	Image.open('D:\\train_data_raw\\34_1_i_h__20200712115929_69855180.jpg'),
	clp_w,
	clp_h,
	clp_labels
)
# FlaskImageProcess(
# 	clp_h5_path,
# 	Image.open('D:\\train_data_raw\\34_1_i_h__20200711143728_69990723.jpg'),
# 	clp_w,
# 	clp_h,
# 	clp_labels
# )

def thread1():
	while True:
		input_line = input()
		print('input_line:',input_line)
		predict(
			clp_h5_path,
			input_line
		)
		pass
_thread.start_new_thread(thread1,())

app.run(host= '0.0.0.0',debug=True)

