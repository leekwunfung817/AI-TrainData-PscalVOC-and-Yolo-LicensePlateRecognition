from xml.dom.minidom import Document
import os
import cv2

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
	for i, name in enumerate(files):
		xmlBuilder = Document()
		annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签

		xmlBuilder.appendChild(annotation)

		folder = xmlBuilder.createElement("folder")#folder标签
		folderContent = xmlBuilder.createTextNode(txtPath)
		folder.appendChild(folderContent)
		annotation.appendChild(folder)

		filename = xmlBuilder.createElement("filename")#filename标签
		filenameContent = xmlBuilder.createTextNode(name[0:-4]+".jpg")
		filename.appendChild(filenameContent)
		annotation.appendChild(filename)

		source = xmlBuilder.createElement("source")  # size标签
		database = xmlBuilder.createElement("database")  # size标签
		Unknown = xmlBuilder.createTextNode('Unknown')
		database.appendChild(Unknown)
		source.appendChild(database)
		annotation.appendChild(source)
		

		path = xmlBuilder.createElement("path")#filename标签
		pathContent = xmlBuilder.createTextNode(txtPath+name[0:-4]+".jpg")
		path.appendChild(pathContent)
		annotation.appendChild(path)

		if not '.txt' in name:
			continue
		print('txtPath+name:',txtPath,name)
		txtFile=open(txtPath+name,"r")
		txtList = txtFile.readlines()
		# print('picPath+name[0:-4]',picPath+name[0:-4])
		img = cv2.imread(picPath+name[0:-4]+".jpg")
		if img is None:
			continue
		# print('img.shape',img)
		Pheight,Pwidth,Pdepth=img.shape

		size = xmlBuilder.createElement("size")  # size标签

		width = xmlBuilder.createElement("width")  # size子标签width
		widthContent = xmlBuilder.createTextNode(str(Pwidth))
		width.appendChild(widthContent)
		size.appendChild(width)

		height = xmlBuilder.createElement("height")  # size子标签height
		heightContent = xmlBuilder.createTextNode(str(Pheight))
		height.appendChild(heightContent)
		size.appendChild(height)

		depth = xmlBuilder.createElement("depth")  # size子标签depth
		depthContent = xmlBuilder.createTextNode(str(Pdepth))
		depth.appendChild(depthContent)
		size.appendChild(depth)

		annotation.appendChild(size)

		for i in txtList:
			oneline = i.strip().split(" ")
			# print('oneline',oneline)


			object = xmlBuilder.createElement("object")

			picname = xmlBuilder.createElement("name")
			nameContent = xmlBuilder.createTextNode(dict[oneline[0]])
			picname.appendChild(nameContent)
			object.appendChild(picname)

			pose = xmlBuilder.createElement("pose")
			poseContent = xmlBuilder.createTextNode("Unspecified")
			pose.appendChild(poseContent)
			object.appendChild(pose)

			truncated = xmlBuilder.createElement("truncated")
			truncatedContent = xmlBuilder.createTextNode("0")
			truncated.appendChild(truncatedContent)
			object.appendChild(truncated)

			difficult = xmlBuilder.createElement("difficult")
			difficultContent = xmlBuilder.createTextNode("0")
			difficult.appendChild(difficultContent)
			object.appendChild(difficult)

			bndbox = xmlBuilder.createElement("bndbox")

			xmin = xmlBuilder.createElement("xmin")
			mathData=int(((float(oneline[1]))*Pwidth+1)-(float(oneline[3]))*0.5*Pwidth)
			xminContent = xmlBuilder.createTextNode(str(mathData))
			xmin.appendChild(xminContent)
			bndbox.appendChild(xmin)

			ymin = xmlBuilder.createElement("ymin")
			mathData = int(((float(oneline[2]))*Pheight+1)-(float(oneline[4]))*0.5*Pheight)
			yminContent = xmlBuilder.createTextNode(str(mathData))
			ymin.appendChild(yminContent)
			bndbox.appendChild(ymin)

			xmax = xmlBuilder.createElement("xmax")
			mathData = int(((float(oneline[1]))*Pwidth+1)+(float(oneline[3]))*0.5*Pwidth)
			xmaxContent = xmlBuilder.createTextNode(str(mathData))
			xmax.appendChild(xmaxContent)
			bndbox.appendChild(xmax)

			ymax = xmlBuilder.createElement("ymax")
			mathData = int(((float(oneline[2]))*Pheight+1)+(float(oneline[4]))*0.5*Pheight)
			ymaxContent = xmlBuilder.createTextNode(str(mathData))
			ymax.appendChild(ymaxContent)
			bndbox.appendChild(ymax)

			object.appendChild(bndbox)

			annotation.appendChild(object)

		fp=xmlPath+name[0:-4]+".xml"
		# if os.path.isfile(fp):
		# 	continue
		f = open(fp, 'w')
		xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
		f.close()
	print('dict',dict)
makexml(".")