import os

cur_path=os.getcwd()+'/'
l = []
for jpg in os.listdir("."):
    if not jpg.endswith(".jpg"):
    	continue
    l.append(jpg)

train=open('jpg.list','w+')
i=0
for jpg in l:
	train.write(cur_path+jpg+'\n')
train.close()

