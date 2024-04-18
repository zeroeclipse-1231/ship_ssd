import re
import os
import xml.etree.ElementTree as ET

#myself dataset path
annotation_folder = './VOC2007/Annotations/'
list = os.listdir(annotation_folder)

def file_name(file_dir):
	L = []
	for root, dirs, files in os.walk(file_dir):
		for file in files:
			if os.path.splitext(file)[1] == '.xml':
				L.append(os.path.join(root, file))
	return L

count = 0
xml_dirs = file_name(annotation_folder)

for i in range(0, len(xml_dirs)):
	#print(xml_dirs[i])
	annotation_file = open(xml_dirs[i]).read()
	root = ET.fromstring(annotation_file)
	if root is not None:
		label = root.find('name').text
	#print(label)
	count_label = count

	#get the pictures' width and height
	for size in root.findall('size'):
		label_width = int(size.find('width').text)
		label_height = int(size.find('height').text)

	#get the boundbox's width and height
	for obj in root.findall('object'):
		for bbox in obj.findall('bndbox'):
			label_xmin = int(bbox.find('xmin').text)
			label_ymin = int(bbox.find('ymin').text)
			label_xmax = int(bbox.find('xmax').text)
			label_ymax = int(bbox.find('ymax').text)
			if label_xmin<0 or label_xmax>label_width or label_ymin<0 or label_ymax>label_height:
				#judge the filename is not repeat
				if label_temp == label:
					continue
				print('--'*30)
				print(xml_dirs[i])   #print the xml's filename
				#print(label)
				print("width:",label_width)
				print("height:",label_height)
				print(label_xmin,label_ymin,label_xmax,label_ymax)
				print('--'*30)
				count = count+1
		label_temp = label

print("================================")
print(count)
