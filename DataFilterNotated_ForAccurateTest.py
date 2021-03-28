import os
import helper
import sys

def getDataNotatList(od_path="."):
	file_list={}
	files = os.listdir(od_path)
	for i, file in enumerate(files):
		if '.' in file:
			continue
		if 'data' in file:
			# print(file)
			folder = os.listdir(od_path+'/'+file)
			for ii, sub_file in enumerate(folder):
				fp=file+'/'+sub_file
				if helper.NotatExist3(fp):
					if sub_file not in file_list:
						file_list[sub_file]=fp
	return file_list

def getPenddingList(od_path="."):
	file_list={}
	files = os.listdir(od_path)
	for i, file in enumerate(files):
		if '.' in file:
			continue
		if 'pendding' in file:
			folder = os.listdir(od_path+'/'+file)
			for ii, sub_file in enumerate(folder):
				fp=file+'/'+sub_file
				if helper.isJPG(fp):
					if sub_file not in file_list:
						file_list[sub_file]=fp
	return file_list

def getNonOverlappingPenddingList(od_path="."):
	file_list={}
	data_list = getDataNotatList(od_path)
	pendding_list = getPenddingList(od_path)
	for fn in pendding_list:
		fp = pendding_list[fn]
		if fn in data_list:
			print('overlap',fn)
		else:
			file_list[fn]=od_path+'/'+fp
	return file_list

def getODsNonOverlappingPenddingList():
	od_file_list={}
	count=0
	for od in sys.argv:
		if '.py' in od:
			continue
		file_list=getNonOverlappingPenddingList(od)
		for fn in file_list:
			if fn not in od_file_list:
				od_file_list[fn]=file_list[fn]
	return od_file_list


def writeODsNonOverlappingPenddingList(LIST_FILE_FN):
	if os.path.exists(LIST_FILE_FN):
		os.remove(LIST_FILE_FN)
	list_file = open(LIST_FILE_FN,'a+')
	nonOList=getODsNonOverlappingPenddingList()
	print(len(nonOList))
	for fn in nonOList:
		list_file.write(nonOList[fn]+'\n')

writeODsNonOverlappingPenddingList('DataFilterNotated_ForAccurateTest.list')
