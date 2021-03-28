
from PIL import Image,ImageDraw,ImageFont
from yolo_keras import FlaskImageProcess,predefined_classes,r_predefined_classes

font = ImageFont.load_default()
ai_name='OD_CTR'
ctr_h5_path = ai_name+'/model.h5'
ctr_w = 224
ctr_h = 224
ctr_labels = predefined_classes(ai_name)
ctr_labels_r = r_predefined_classes(ai_name)

ai_name='OD_LPR'
clp_h5_path = ai_name+'/model.h5'
clp_w = 224
clp_h = 224
clp_labels = predefined_classes(ai_name)
clp_labels_r = r_predefined_classes(ai_name)

ai_name='OD_OCR'
ocr_h5 = ai_name+'/model.h5'
ocr_w = 224
ocr_h = 224
ocr_labels = predefined_classes(ai_name)
ocr_labels_r = r_predefined_classes(ai_name)

FlaskImageProcess(
	ocr_h5,
	Image.open('OD_OCR/data/17_1_i_h_01649433___License Plate.jpg'),
	ocr_w,
	ocr_h,
	ocr_labels
)

FlaskImageProcess(
	clp_h5_path,
	Image.open('OD_LPR/data/17_1_i_h_01649433_.jpg'),
	clp_w,
	clp_h,
	clp_labels
)

FlaskImageProcess(
	ctr_h5_path,
	Image.open('OD_LPR/data/17_1_i_h_01649433_.jpg'),
	ctr_w,
	ctr_h,
	ctr_labels
)

def toAnotation(ori_img,arr,path,labels_r):
	ori_w, ori_h = ori_img.size
	line=''
	for ele in arr:
		center_x=ele['x1']+(ele['width']/2)
		center_y=ele['y1']+(ele['height']/2)
		line+=str(labels_r[ele['labels']])+' '+str(center_x/ori_w)+' '+str(center_y/ori_h)+' '+str(ele['width']/ori_w)+' '+str(ele['height']/ori_h)+'\n'
	f=open(path+'.txt','w+')
	f.write(line)
	f.close()
	ori_img.save(path+'.jpg')

def ctr_process(filename,ctr,toNotat=True,h5_path=ctr_h5_path):
	ori_image=ctr.copy()
	(nn_demo_img,ori_img,demo_img,arr) = FlaskImageProcess(h5_path,ctr,ctr_w,ctr_h,ctr_labels)
	if toNotat:
		path='OD_CTR/pendding/'+filename.replace('.jpg','')
		toAnotation(ori_image,arr,path,ctr_labels_r)
	labels=[]
	for ele in arr:
		labels.append(ele['labels'])
	return ','.join(labels)

def ocr_process(filename,ocr,toNotat=True,h5_path=ocr_h5):
	(ocr_predict_Image,ocr_ori_PIL_Image,ocr_demo_PIL_Image,ocr_arr) = FlaskImageProcess(h5_path,ocr,ocr_w,ocr_h,ocr_labels)
	ori_w, ori_h = ocr_ori_PIL_Image.size
	ana_center_img=ocr_ori_PIL_Image.copy()
	draw=ImageDraw.Draw(ana_center_img)
	avg_h=0
	avg_w=0
	ocr_arr_len=len(ocr_arr)
	for ocr_ele in ocr_arr:
		avg_h=avg_h+ocr_ele['height']
		avg_w=avg_w+ocr_ele['width']
	if ocr_arr_len>0:
		avg_h=avg_h/ocr_arr_len
		avg_w=avg_w/ocr_arr_len
	max_h=avg_h/2
	
	sort_arr={}
	for ocr_ele in ocr_arr:
		center_x=ocr_ele['x1']+(ocr_ele['width']/2)
		center_y=ocr_ele['y1']+(ocr_ele['height']/2)
		sort_arr[center_x]=ocr_ele
		draw.text(tuple((center_x,center_y)), ocr_ele['labels'], font=font, fill=(255,0,0,255))
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
	ocr_ele_list=[]
	for ocr_min in ocr_min_list:
		for ocr_ele in ocr_arr:
			center_x=ocr_ele['x1']+(ocr_ele['width']/2)
			if center_x==ocr_min:
				ocr_ele_list.append(ocr_ele)
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
			if center_y<cur_y:
				cur_char+=1
			else:
				cur_char-=1

		key=str(cur_char)+'_'
		if key not in ocr_result:
			ocr_result[key]=ocr_ele['labels']
		else:
			ocr_result[key]+=ocr_ele['labels']
		cur_y=center_y
	ocr_txt=''
	key='2_'
	if key in ocr_result:
		ocr_txt+=ocr_result[key]
	key='1_'
	if key in ocr_result:
		ocr_txt+=ocr_result[key]
	key='0_'
	if key in ocr_result:
		ocr_txt+=ocr_result[key]
	key='-1_'
	if key in ocr_result:
		ocr_txt+=ocr_result[key]
	key='-2_'
	if key in ocr_result:
		ocr_txt+=ocr_result[key]
	ocr_result['ocr_txt']=ocr_txt
	if toNotat:
		ocr_path='OD_OCR/pendding/'+filename.replace('.jpg','')+'_'+ocr_txt
		if len(ocr_txt)>1:
			toAnotation(ocr_ori_PIL_Image,ocr_arr,ocr_path,ocr_labels_r)
	return ocr_result

def clp_process_test(filename,clp,toNotat=True,h5_path=clp_h5_path):
	(clp_predict_Image,ori_PIL_Image,demo_PIL_Image,arr) = FlaskImageProcess(h5_path,clp,clp_w,clp_h,clp_labels)
	for ele in arr:
		if ele['labels']=='License Plate':
			return 'License Plate'
	return ''

def clp_process(filename,clp,toNotat=True,h5_path=clp_h5_path):
	(clp_predict_Image,ori_PIL_Image,demo_PIL_Image,arr) = FlaskImageProcess(h5_path,clp,clp_w,clp_h,clp_labels)
	if toNotat:
		clp_path='OD_LPR/pendding/'+filename.replace('.jpg','')
		toAnotation(ori_PIL_Image,arr,clp_path,clp_labels_r)
	ocr_result_list=[]
	ocr_list=[]
	i=1
	for ele in arr:
		tl_x=ele['x1']
		tl_y=ele['y1']
		rl_x=ele['x1']+ele['width']
		rl_y=ele['y1']+ele['height']
		crop_config = (tl_x,tl_y,rl_x,rl_y)
		ocr = ori_PIL_Image.crop(crop_config)
		if ele['labels']=='License Plate':
			ele['result']=ocr_process(filename,ocr)
			if len(ele['result']['ocr_txt'])>0:
				ocr_result_list.append(ele['result']['ocr_txt'])
			ocr_list.append(ele)
		i=i+1
	return ','.join(ocr_result_list)

def lpr_process(filename,img):
	ctr=img.copy()
	clp=img.copy()
	return clp_process(filename,clp)+';'+ctr_process(filename,ctr)
