
from flask import Flask, request, redirect, url_for, send_from_directory
app = Flask(__name__)
import _thread
from yolo_cpos import lpr_process,ctr_process,ocr_process

@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['file']
		filename = file.filename
		return lpr_process(filename,Image.open(file.stream))
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form action="" method=post enctype=multipart/form-data>
		<input type=file name=file>
		<input type=submit value=Upload>
	</form>
	'''
@app.route('/ocr', methods=['GET', 'POST'])
def upload_ocr():
	if request.method == 'POST':
		file = request.files['file']
		filename = file.filename
		return ocr_process(filename,Image.open(file.stream))
	return '''
	<!doctype html>
	<title>Upload new File</title>
	<h1>Upload new File</h1>
	<form action="" method=post enctype=multipart/form-data>
		<input type=file name=file>
		<input type=submit value=Upload>
	</form>
	'''

def cli_thread():
	while True:
		try:
			input_line = input()
			# <comment 1> print('input_line:',input_line)
			input_line=input_line.replace('"','')
			lpr_result = lpr_process(os.path.basename(input_line),Image.open(input_line))
			print('lpr_result:',lpr_result)
		except:
			print("An exception occurred")
_thread.start_new_thread(cli_thread,())

def http_thread():
	app.run(host= '0.0.0.0',port=5000)
_thread.start_new_thread(http_thread,())

import time
while True:
	time.sleep(3)
	pass
