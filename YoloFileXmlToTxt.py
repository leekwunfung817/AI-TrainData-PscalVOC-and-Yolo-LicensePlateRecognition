import os
import sys

import xml.etree.ElementTree as Et

def r_predefined_classes():
	i = 0
	count = 0
	arr={}
	for line in open('../data/predefined_classes.txt','r').readlines():
		arr[line.strip()]=str(i)
		i+=1
	return arr

class VOC:
	"""
	Handler Class for VOC PASCAL Format
	"""

	def xml_indent(self, elem, level=0):
		i = "\n" + level * "\t"
		if len(elem):
			if not elem.text or not elem.text.strip():
				elem.text = i + "\t"
			if not elem.tail or not elem.tail.strip():
				elem.tail = i
			for elem in elem:
				self.xml_indent(elem, level + 1)
			if not elem.tail or not elem.tail.strip():
				elem.tail = i
		else:
			if level and (not elem.tail or not elem.tail.strip()):
				elem.tail = i

	def generate(self, data):
		predefined_classes=r_predefined_classes()
		print('r_predefined_classes:',r_predefined_classes)
		xml_list = {}
		for key in data:
			print('',key)
			element = data[key]
			xml_width =float(element["size"]["width"]) 
			xml_height = float(element["size"]["height"]) 
			str_b = ''
			for i in range(0, int(element["objects"]["num_obj"])):
				name = element["objects"][str(i)]["name"]
				str_b+=predefined_classes[name]+' '
				obj_xmin = element["objects"][str(i)]["bndbox"]["xmin"]
				obj_xmax = element["objects"][str(i)]["bndbox"]["xmax"]

				obj_ymin = element["objects"][str(i)]["bndbox"]["ymin"]
				obj_ymax = element["objects"][str(i)]["bndbox"]["ymax"]

				obj_h = obj_ymax-obj_ymin
				obj_w = obj_xmax-obj_xmin

				cen_x = (obj_xmin+obj_xmax)/2
				cen_y = (obj_ymax+obj_ymin)/2


				str_b+=str(cen_x/xml_width)+' '
				str_b+=str(cen_y/xml_height)+' '
				
				str_b+=str(obj_w/xml_width)+' '
				str_b+=str(obj_h/xml_height)+'\n'
			xml_list[key]=str_b
		return True, xml_list

	@staticmethod
	def save(xml_list, path):

		try:
			path = os.path.abspath(path)

			progress_length = len(xml_list)
			progress_cnt = 0
			# printProgressBar(0, progress_length, prefix='\nVOC Save:'.ljust(10), suffix='Complete', length=40)

			for key in xml_list:
				xml = xml_list[key]
				filepath = os.path.join(path, "".join([key, ".xml"]))
				ElementTree(xml).write(filepath)

				# printProgressBar(progress_cnt + 1, progress_length, prefix='VOC Save:'.ljust(15), suffix='Complete', length=40)
				progress_cnt += 1

			return True, None

		except Exception as e:

			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

			msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

			return False, msg

	@staticmethod
	def parse(path):
		print('check point 1')
		p = os.path.abspath(path)
		w = os.walk(p)
		print(w)
		paras = next(w)
		(dir_path, dir_names, filenames) = paras


		print('check point 2')
		
		data = {}
		# progress_length = len(filenames)
		progress_cnt = 0
		# printProgressBar(0, progress_length, prefix='\nVOC Parsing:'.ljust(15), suffix='Complete', length=40)
		for filename in filenames:
			if '.xml' in filename:

				xml = open(os.path.join(dir_path, filename), "r",encoding="ascii")
				# print(filename)
				tree = Et.parse(xml)
				root = tree.getroot()

				xml_size = root.find("size")
				size = {
					"width": xml_size.find("width").text,
					"height": xml_size.find("height").text,
					"depth": xml_size.find("depth").text

				}

				objects = root.findall("object")
				if len(objects) == 0:
					continue
					# return False, None
					# "number object zero"

				obj = {
					"num_obj": len(objects)
				}

				obj_index = 0
				for _object in objects:

					tmp = {
						"name": _object.find("name").text
					}

					xml_bndbox = _object.find("bndbox")
					bndbox = {
						"xmin": float(xml_bndbox.find("xmin").text),
						"ymin": float(xml_bndbox.find("ymin").text),
						"xmax": float(xml_bndbox.find("xmax").text),
						"ymax": float(xml_bndbox.find("ymax").text)
					}
					tmp["bndbox"] = bndbox
					obj[str(obj_index)] = tmp

					obj_index += 1

				annotation = {
					"size": size,
					"objects": obj
				}

				data[root.find("filename").text.split(".")[0]] = annotation

				# printProgressBar(progress_cnt + 1, progress_length, prefix='VOC Parsing:'.ljust(15), suffix='Complete', length=40)
				progress_cnt += 1

		return True, data

import json

def convert(dir):
	print('check point A')
	print('check point A1')
	voc = VOC()
	(bool,context)=voc.parse(dir)

	print('check point A2')
	print('check point A3')

	print('check point 1')
	y = json.dumps(context)
	# print('context')
	# print(y)
	# for file in context:
	#	 print('loop')
	#	 print(file)
	#	 if file in context:
	#		 print(context[file])
	#	 break
	print('check point B')
	(bool,xml_list)=voc.generate(context)

	# print(xml_list)
	for file in xml_list:
		fp=file+'.txt'
		if dir!='.':
			fp=dir+fp

		fc=xml_list[file]
		print('point e:',fp,fc)

		# if not os.path.isfile(fp):
		# if os.path.isfile(fp):
		open(fp,'w+').write(fc)

	# print(context)

	pass
convert(".")