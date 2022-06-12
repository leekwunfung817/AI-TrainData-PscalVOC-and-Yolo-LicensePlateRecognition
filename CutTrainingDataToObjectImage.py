import os
import cv2

import hashlib

txtPath='data.unrecognized.20211207/'
targetFullPath=txtPath.replace('/','.cut/')
if not os.path.exists(targetFullPath):
	os.makedirs(targetFullPath)

	
files = os.listdir(txtPath)
if txtPath=='.':
	txtPath=''
xmlPath=txtPath
picPath=txtPath

hsh = cv2.img_hash.BlockMeanHash_create()


def predefined_classes():
	i = 0
	count = 0
	arr={}
	for line in open('data/predefined_classes.txt','r').readlines():
		arr[str(i)]=line.strip()
		i+=1
	return arr

dict_=predefined_classes()
print('dict_:',dict_)
for i, name in enumerate(files):
	if not '.txt' in name:
		continue
	fullPath = txtPath+name
	image_path_name=name[0:-4]
	fullPathImg = picPath+image_path_name+".jpg"
	txtFile=open(fullPath,"r")
	txtList = txtFile.readlines()
	img = cv2.imread(fullPathImg)
	if img is None:
		continue
	print('fullPath:',fullPath)
	print('fullPathImg:',fullPathImg)

	Pheight,Pwidth,Pdepth=img.shape
	box_row=''
	print(' ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ')

	print('Pheight Pwidth')
	print(txtPath,name,Pheight,Pwidth)
	for i in txtList:
		print(' ===== ===== ===== ===== =====')
		oneline = i.strip().split(" ")

		c_num=oneline[0]

		c_x=float(oneline[1])
		c_y=float(oneline[2])
		o_w=float(oneline[3])
		o_h=float(oneline[4])

		# print('c_x c_y o_w o_h:',(c_x,c_y,o_w,o_h))

		x_min=c_x-(o_w/2)
		y_min=c_y-(o_h/2)
		x_max=c_x+(o_w/2)
		y_max=c_y+(o_h/2)

		# print('x_min y_min x_max y_max',(x_min,y_min,x_max,y_max))

		x_min*=Pwidth
		y_min*=Pheight
		x_max*=Pwidth
		y_max*=Pheight

		# print('x_min y_min x_max y_max',(x_min,y_min,x_max,y_max))

		x_min=int(x_min)
		y_min=int(y_min)
		x_max=int(x_max)
		y_max=int(y_max)

		crop_img = img[y_min:y_max, x_min:x_max]
		crop_img_hash = hsh.compute(crop_img)[0]

		# crop_img_hash = ''.join( crop_img_hash.to_bytes(2, 'big') ).decode('utf-8')

		# bl = b''
		# for int_ in crop_img_hash:
		# 	bl+=int_.tobytes('A')

		crop_img_hash = hashlib.md5(crop_img_hash).hexdigest()
		print('x_min y_min x_max y_max',(x_min,y_min,x_max,y_max),crop_img.shape,crop_img_hash)

		cv2.imwrite(targetFullPath+crop_img_hash+'_'+dict_[c_num]+'.jpg',crop_img)

		cv2.imshow("cropped", crop_img)
		cv2.waitKey(1)

		# if x_min<0:
		# 	x_min=0
		# if y_min<0:
		# 	y_min=0

		# box_txt=''
		# box_txt+=str(x_min)
		# box_txt+=','+str(y_min)
		# box_txt+=','+str(x_max)
		# box_txt+=','+str(y_max)

		# box_txt+=','+c_num
		# box_row+=' '+box_txt
		# print('x_min',x_min)