from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import os
import cv2

print('cv2.__version__',cv2.__version__)
# exit()

def predefined_classes():
	i = 0
	count = 0
	arr={}
	for line in open('../data/predefined_classes.txt','r').readlines():
		arr[str(i)]=line.strip()
		i+=1
	return arr

r_labels=predefined_classes()
print('r_labels:',r_labels)
files = os.listdir('.')
for i, name in enumerate(files):
	if not'.jpg' in name:
		continue
	img_name = name
	img = cv2.imread(name)
	name=name.replace('.jpg','.txt',1)
	if not os.path.isfile(name):
		continue
	print('name:',name)
	txtFile=open(name,"r")
	txtList = txtFile.readlines()
	if img is None:
		continue
	Pheight,Pwidth,Pdepth=img.shape
	box_row=''
	print(' ===== ===== ===== ===== ===== ===== ===== ===== ===== ===== ')
	print(name,Pheight,Pwidth)

	im = Image.open(img_name)
	draw = ImageDraw.Draw(im)

	for i in txtList:
		print(' ===== ===== ===== ===== =====')
		oneline = i.strip().split(" ")
		c_num=oneline[0]
		c_x=float(oneline[1])
		c_y=float(oneline[2])
		o_w=float(oneline[3])
		o_h=float(oneline[4])
		print('c_x c_y o_w o_h:',(c_x,c_y,o_w,o_h))
		x_min=c_x-(o_w/2)
		y_min=c_y-(o_h/2)
		x_max=c_x+(o_w/2)
		y_max=c_y+(o_h/2)
		print('x_min y_min x_max y_max',(x_min,y_min,x_max,y_max))
		x_min*=Pwidth
		y_min*=Pheight
		x_max*=Pwidth
		y_max*=Pheight
		print('x_min y_min x_max y_max',(x_min,y_min,x_max,y_max))
		x_min=int(x_min)
		y_min=int(y_min)
		x_max=int(x_max)
		y_max=int(y_max)
		print('x_min y_min x_max y_max',(x_min,y_min,x_max,y_max))
		# cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
		draw.rectangle((x_min,y_min,x_max,y_max),outline=(255,0,0),width=3)
		draw.text((x_min,y_min),r_labels[c_num],fill='white')

	save_fp = '../demo/'+img_name
	im.save(save_fp)
	print(save_fp)
	# cv2.imshow('../demo/'+img_name, img)
	# cv2.imwrite(save_fp, img)
