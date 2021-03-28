import os

def NotatExist3(full_path):
	if '.jpg' not in full_path:
		return False
	raw=full_path.replace('.jpg','')
	xml=raw+'.xml'
	txt=raw+'.txt'
	if os.path.exists(xml) and os.path.exists(txt) and os.path.exists(full_path):
		return True
	return False


def isJPG(full_path):
	if '.jpg' not in full_path:
		return False
	return True

