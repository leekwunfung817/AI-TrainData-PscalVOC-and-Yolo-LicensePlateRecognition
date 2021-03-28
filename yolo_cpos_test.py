# yolo_cpos_test.py

# run in project root CPOS AI
'''

cd /d "D:\CPOS AI\""
python yolo_cpos_test.py


For varify the accuracy of the model

'''





from difflib import SequenceMatcher

def similar(a, b):
	a=str(a)
	b=str(b)
	result=SequenceMatcher(None, a, b).ratio()
	print('Similarity:'+str(result)+'% ('+a+'->'+b+')')
	return result


# print(  similar('abc','ab')  )
# exit()


#compile each of model
from yolov3_darknet_to_keras import convert
import os

def complie_multi_weights(prefix,weights_paths):
	for weights_path in weights_paths:
		wfp=prefix+'backup/'+weights_path
		to_fp=prefix+weights_path+'.h5'
		print('compile ['+wfp+'] to ['+to_fp+']')
		if not os.path.isfile(to_fp):
			print('file not exist')
			convert(prefix+'model.cfg',wfp,to_fp)

# exit()


from yolo_cpos import clp_process_test,ctr_process,ocr_process
from PIL import Image

class AccuracyCount:
	"""docstring for ClassName"""
	total_case=0
	success_case=0
	def __init__(self):
		self.total_case = 0
		self.success_case = 0
		
	def correct_case(self,accuracy):
		self.total_case+=1
		self.success_case+=accuracy
	def print_accuray(self):
		print( 'Accuracy:'+str((self.success_case/self.total_case)*100)+'%' )



def test(data_file_name,h5_path,od):
	img=Image.open('./'+od+'/data/'+data_file_name+'.jpg')
	if od is 'OD_CTR':
		return ctr_process(data_file_name,img,False,h5_path)
	if od is 'OD_OCR':
		return ocr_process(data_file_name,img,False,h5_path)['ocr_txt']
	if od is 'OD_LPR':
		return clp_process_test(data_file_name,img,False,h5_path)


def test_each_h5(h5_path,od,test_subject):
	print(' ===== Testing H5 or weights ===== :',h5_path)
	acc=AccuracyCount()
	for fn in test_subject:
		
		result=test(fn,h5_path,od)
		print('>> Test:'+fn+' expect:'+test_subject[fn]+' result:'+result)
		if test_subject[fn] in result:
			acc.correct_case(1)
			print('Match')
		else:
			acc.correct_case( similar(test_subject[fn],result) )
	acc.print_accuray()

def test_model_accuracy(prefix,weights,test_subject,compile_weights=False):
	if compile_weights:
		complie_multi_weights(prefix+'/',weights)
	for weight in weights:
		test_each_h5(prefix+'/'+weight+'.h5',prefix,test_subject)


'''
test_model_accuracy(
	# prefix
	'OD_CTR',
	# weights
	[
		# 'model_10000.weights'
		# ,'model_20000.weights'
		# ,'model_30000.weights'
		# ,'model_40000.weights'
		# ,'model_50000.weights'
		'model_60000.weights'
		# Testing: OD_CTR/model_60000.weights.h5
		# Accuracy:93.54838709677419%
		# ,'model_61000.weights'
		# ,'model_62000.weights'
		# ,'model_63000.weights'
		# ,'model_64000.weights'
		# ,'model_64700.weights'
		# ,'model_66500.weights'
		# ,'model_69400.weights'
		# ,'model_79800.weights'
	],
	# private car
	{
		'11_1_i_h__20190621093202_27871908':'privatecar'
		,'11_1_i_h__20190621100000_08805205':'privatecar'
		,'11_1_i_h__20190623041031_17420771':'privatecar'
		,'11_1_i_h__20190625104309_10269503':'privatecar'
		,'11_1_i_h__20190625092456_09393799':'privatecar'

		,'20200619074729_08868264':'privatecar'
		,'20200619074918_07546349':'privatecar'
		,'20200619134258_28359670':'privatecar'
		,'20200619135303_85410316':'privatecar'
		,'20200620014159_0568228245':'privatecar'

		,'20200727091137_72993066':'privatecar'
		,'20200727132539_0264773913':'privatecar'
		,'20200727134048_89641235':'privatecar'
		,'20200727143120_0263200345':'privatecar'
		,'20200727150702_0407839351':'privatecar'

		,'20200727132349_87600313_S':'privatecar'
		,'20200727134048_89641235_S':'privatecar'
		,'20200727142748_33879786_S':'privatecar'
		,'20200727150702_0407839351_S':'privatecar'
		,'20200728165226_0262944297_S':'privatecar'

		,'20200620231451_0618808195_S':'privatecar'
		,'20200620235123_41293949_S':'privatecar'
		,'20200621021631_69768625_S':'privatecar'
		,'20200621032718_0559514261_S':'privatecar'
		,'20200621084048_71114328_S':'privatecar'

		,'20200727085607_0264773913_S':'privatecar'
		,'20200727085957_0212616393_S':'privatecar'
		,'20200727090100_71182965_S':'privatecar'
		,'20200727091131_4211335541_S':'privatecar'
		,'20200727091137_72993066_S':'privatecar'

		,'20191110100934_75041900':'taxi_mark'
		,'20191110162011_11366252':'taxi_mark'
		,'20191117160504_74770556':'taxi_mark'
		,'20191118072605_28573790':'taxi_mark'
		,'20200503181139_08657826':'taxi_mark'

		,'20200503183134_31382950':'taxi_mark'
		,'20200504141841_43878991':'taxi_mark'
		,'20200506110109_89154861':'taxi_mark'
		,'20200506111315_11021985':'taxi_mark'
		,'20200509111448_71368490':'taxi_mark'

		,'20200617091625_47582131':'taxi_mark'
		,'20200616195358_77227456':'taxi_mark'
		,'20200618112951_71759878':'taxi_mark'
		,'20200618203022_70543119':'taxi_mark'
		,'20200618104953_17978785':'taxi_mark'

		,'20200621101628_91717588_S':'taxi_mark'
		,'20200621173900_77227456_S':'taxi_mark'
		,'20200621184914_73198358_S':'taxi_mark'
		,'20200621191024_11241783_S':'taxi_mark'
		,'20200622101017_89136910_S':'taxi_mark'

		,'20200803170953_84792993_S':'taxi_mark'
		,'20200725184107_56127754_S':'taxi_mark'
		,'20200624134503_78984808_S':'taxi_mark'
		,'20200624132108_77287729_S':'taxi_mark'
		,'20200624120602_86407323_S':'taxi_mark'

		,'20200814174454_21473366':'taxi_mark'
		,'20200814114238_71759878':'taxi_mark'
		,'20200814081438_92829400':'taxi_mark'
		,'20200814081438_92829400':'taxi_mark'
		,'20200803170953_84792993':'taxi_mark'

		,'20200727111756_08057578':'vehicle'
		,'20200727111756_08057578_S':'vehicle'
	}
)

'''

test_model_accuracy(
	# prefix
	'OD_OCR',
	# weights
	[
		# 'model',
		# 'model_20200812',
		# 'model_20200817',
		# 'model_20200904',
		# 'model_20200905',
		# 'ocr_model_20200921.weights',
		# 'ocr_model_20200911.weights',
		# 'model_410600.train.hard.weights',
		# 'model_406600.train.middle.weights',
		# 'model_380000.train.soft.weights',
		# 'model_278500_20200904.weights',


		# 'model_97800.weights',

		# 'model_99400.weights', # Accuracy:86.73469387755102%
		# 'model_250000.weights', # Accuracy:86.73469387755102%
		# 'model_262500.weights', # Accuracy:87.77056277056276%
		# 'model_300000.weights', # Accuracy:85.71428571428571%
		# 'model_320000.weights', # Accuracy:85.71428571428571%
		# 'model_330000.weights', # Accuracy:86.73469387755102%
		# 'model_340000.weights', # Accuracy:85.71428571428571%
		# 'model_345000.weights', # Accuracy:85.71428571428571%
		# 'model_349000.weights', # Accuracy:85.3896103896104%
		# 'model_349900.weights', # Accuracy:85.71428571428571%
		'model_350000.weights', # Accuracy:88.09523809523809%
		# 'model_350100weights', # Accuracy:87.44588744588746%
		# 'model_351000.weights', # Accuracy:87.22527472527473%
		# 'model_353000.weights', # Accuracy:86.73469387755102%
		# 'model_359100.weights', # Accuracy:87.5%
		# 'model_355000.weights', # Accuracy:86.73469387755102%
		# 'model_360000.weights', # Accuracy:85.71428571428571%
		# 'model_365000.weights', # Accuracy:85.3896103896104%
		# 'model_370000.weights', # Accuracy:85.71428571428571%
		# 'model_375000.weights', # Accuracy:86.73469387755102%
		# 'model_380000.weights', # Accuracy:85.71428571428571%
		# 'model_385000.weights', # Accuracy:85.71428571428571%
		# 'model_390000.weights', # Accuracy:85.71428571428571%
		# 'model_391200.weights', # Accuracy:85.71428571428571%
		

		# 'model_244500.weights'
	],
	{
		'20200824083402_71702466_L37':'LG3087',
		'20200823231804_0408106455_1431':'NY1431',
		'20200823230242_4210812885_STT':'SATELL1T',
		'20200823224923_4211225077_PN':'PN1296',
		'20200823215900_11350515_959':'TH9599',
		'20200823215135_0830089001_U40':'UU4023',
		'20200823205723_4211335541_268':'CC6268',
		'20200823204937_0453654593_WJ':'WJ4797',
		'20200823200842_0265719001_UF8':'UF8519',
		'20200823190800_4211001253_P705':'NP5705',
		'20200823171710_0514313411_SUMW':'SUME0W',
		'20200823162623_0623194873_2123':'KF2123',
		'20200823155705_4211292949_F2':'PF127',
		'20200823155525_0477285538_WS':'WS4728',
		'20200823155506_0518801619_UR81':'UR8148',
		'20200823144901_4211228389_18':'VS1809',
		'20200823121744_0211060777_8285':'RV8285',
		'20200823121525_4213817125_DE0':'DE3880',
		'20200823090516_48948717_UKUN':'S1UKEUNG',
		'20200823084453_92074482_W5':'WW8575',
		'20200822192739_0320476537_S3':'SE3498',
		'17_1_i_h_11327945___License Plate':'TM4790',
		'17_1_i_h_11183954_.jpg_NN3792':'NN3792',
		'17_1_i_h_09564457__LH3305':'LH3305',
		'17_1_i_h_09107387_.jpg_PT4469':'PT4469',
		'4E74B481-EC70-4D57-A309-8AAAFF6F6B78.jpeg_WF7806':'WF7806',
		'11_1_i_h__20190622160830_06803091_11':'DX7113',
		'17_1_i_h_03969301_.jpg_RL7800':'RL7800'
	},
	True
)
'''

test_model_accuracy(
	# prefix
	'OD_LPR',
	# weights
	[
		'model_244500.weights'
	],
	{
		'17_1_i_h_03969301_':'RL7800:privatecar',
		'17_1_i_h_15741168_':'SKIVI:privatecar',
		'20200902190030_0268851145_S':'LX2046:privatecar',
		'20200902122014_94599136_S':'NU6154:privatecar',
		'20200902121314_0267685993_S':'VD874:privatecar',
		'20200902114239_04804126_S':'KS9126:taxi,taxi_mark',
		'20200902032900_0265675081_S':'UT8576:privatecar',
		'20200901221812_0266629289_S':'WL8605:privatecar',
		'20200901122723_0263965337_S':'UM1395:privatecar',
		'20200901085552_23865483_S':'BS1211:privatecar',
		'20200901084028_39962834_S':'SD5317:privatecar',
		'20200901082805_0264235273_S':'NS449:vehicle',
		'20200831200314_0266000217_S':'VN9230:privatecar',
		'20200823164248_17990848_S':'SX637:privatecar',
		'20200823152753_4214162901_S':'PJ3834:privatecar',
		'20200823144531_38548126_S':'PB2868:privatecar',
		'20200823142236_0320476537':'SE3498:privatecar',
		'20200823135413_0212616393_S':'TS3597:privatecar',
		'20200823131948_72139850_S':'WG4864:vehicle'
	}
)

'''





