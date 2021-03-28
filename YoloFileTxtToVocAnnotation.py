import os
import cv2

if os.path.isfile("voc_annotation.txt"):
	os.remove("voc_annotation.txt")

def predefined_classes():
	i = 0
	count = 0
	arr={}
	for line in open('../data/predefined_classes.txt','r').readlines():
		arr[str(i)]=line.strip()
		i+=1
	return arr

def makexml(txtPath):
	files = os.listdir(txtPath)
	if txtPath=='.':
		txtPath=''
	xmlPath=txtPath
	picPath=txtPath
	dict=predefined_classes()
	voc_annotation=open('voc_annotation.txt', 'a+')
	for i, name in enumerate(files):
		if not '.txt' in name:
			continue
		txtFile=open(txtPath+name,"r")
		txtList = txtFile.readlines()
		image_path_name=name[0:-4]
		img = cv2.imread(picPath+image_path_name+".jpg")
		if img is None:
			continue
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

			if x_min<0:
				x_min=0
			if y_min<0:
				y_min=0

			box_txt=''
			box_txt+=str(x_min)
			box_txt+=','+str(y_min)
			box_txt+=','+str(x_max)
			box_txt+=','+str(y_max)

			box_txt+=','+c_num
			box_row+=' '+box_txt
			print('x_min',x_min)
		box_row+='\n'
		fp=xmlPath+image_path_name+".xml"
		row=image_path_name+box_row
		voc_annotation.write(row)
		# break
		print('row',row)
	voc_annotation.close()

makexml(".")