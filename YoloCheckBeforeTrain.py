import os

count_arr = {}
files = os.listdir(".")
for i, jpg in enumerate(files):
	if not '.jpg' in jpg:
		continue
	txt=jpg.replace('.jpg','.txt',1)
	if not os.path.isfile(txt):
		continue

	txtFile=open(txt,"r")
	txtList = txtFile.readlines()
	for i in txtList:
		
		oneline = i.strip().split(" ")
		c_num=oneline[0]
		print(jpg,'c_num='+c_num)
		if c_num in count_arr:
			count_arr[c_num]+=1
		else:
			count_arr[c_num]=1
print('count_arr:',count_arr)