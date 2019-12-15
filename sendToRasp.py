import bluetooth as bt
import sys
from mlagents.envs.environment import UnityEnvironment
from socket import *
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util


def realSense(detection):
	print("realSense()")
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

	pipeline.start(config)

	MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
	MODEL_FILE = MODEL_NAME + '.tar.gz'
	DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

	PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

	PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

	NUM_CLASSES = 90

	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
		file_name = os.path.basename(file.name)
		if 'frozen_inference_graph.pb' in file_name:
			tar_file.extract(file, os.getcwd())

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(
		label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	def load_image_into_numpy_array(image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			while not detection:
				frames = pipeline.wait_for_frames()
				depth_frame = frames.get_depth_frame()
				depth = depth_frame
				color_frame = frames.get_color_frame()
				depth_image = np.asanyarray(depth_frame.get_data())
				color_image = np.asanyarray(color_frame.get_data())
				depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
				detect_frame = color_image
		    
				image_np_expanded = np.expand_dims(color_image, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')
				(boxes, scores, classes, num_detections) = sess.run(
				[boxes, scores, classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})
				vis_util.visualize_boxes_and_labels_on_image_array(
					color_image,np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),category_index,
					use_normalized_coordinates=True,
					line_thickness=8)

				#cv2.imshow('RealSense', detect_frame)
				#cv2.imshow('RealSense-Depth', depth_colormap)
		
				for index, value in enumerate(classes[0]):
					if scores[0, index] > 0.8:
						ycenter = (boxes[0][index][0]+boxes[0][index][2])*300
						xcenter = (boxes[0][index][1]+boxes[0][index][3])*400
						dist = depth.get_distance((int)(xcenter), (int)(ycenter))
						if category_index.get(value)['name']=='car':
							print("car detected")
							detection = True
						
				if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
	pipeline.stop()
	return category_index.get(value)['name'], xcenter, ycenter, dist

def calculatorXYZ(x,y,z):
	print("calculatorXYZ()")
	unityX = round(((x*3/320)-6),2)
	unityY = round((10-(y*5.5/720)),2)
	unityZ = round((-10-((z-1.2)*4.5/0.6)),2)
	return (str(unityX)+"/"+str(unityY)+"/"+str(unityZ))	

class UnityEnv():
	def __init__(self):
		self.clientSock = socket(AF_INET, SOCK_STREAM)
		self.clientSock.connect(('0.0.0.0', 55565))

	def envClose(self):
		#self.env.close()
		self.clientSock.close()

	def socketUnity(self,xyz):
		print("UnityEnv.socketUnity()")
		try:
			print("xyz:",xyz, type(xyz))
			self.clientSock.send(xyz.encode())
			self.data = self.clientSock.recv(100)
			print(self.data)
			if self.data:
				print("send")
			return self.data.decode('utf-8')
		except IOError:
			return 'x'
	

def calculatorAngle(data, objName):
	print("calculatorAngle()")
	if data:
		dataSplit = data.split('/')
		print("data",data)
		print("dataSplit",dataSplit)
		dataSplit = np.array([dataSplit[0],dataSplit[1],dataSplit[2],dataSplit[3]])
		dataSplit = dataSplit.astype(np.float)
		raBD = round((2.5-((dataSplit[0]-260)*9/160)),1)
		if (dataSplit[1]+70) > 200:
			dataSplit[1] -= 290
		else:
			dataSplit[1] += 70
		raBS = round((9.6-(dataSplit[1]*3/55)),1)
		if (dataSplit[2]+106) > 200:
			dataSplit[2] -= 254
		else:
			dataSplit[2] += 106
		raSE = round((13.4-(dataSplit[2]*5.5/107.1)),1)
		if (dataSplit[3]+10) > 200:
			dataSplit[3] -= 270
		else:
			dataSplit[3] += 90
		raEW = round((2+(dataSplit[3]*4.5/100)),1)
		return "1"+"/"+str(raBD)+"/"+str(raBS)+"/"+str(raSE)+"/"+str(raEW)
	else:
		return 0
	
def bluetoothRasp(data):
	print("bluetoothRasp()")
	clnt_socket = bt.BluetoothSocket(bt.RFCOMM)
	clnt_socket.connect(("DC:A6:32:4F:B9:71",1))
	clnt_socket.send(data)
	time.sleep(20)
	clnt_socket.close()

if __name__ == '__main__':
	print("1")
	objName, camX, camY, camZ = realSense(False)
	print("2")
	sendXYZ = calculatorXYZ(camX,camY,camZ)
	print("3")
	unityEnv = UnityEnv()
	print("4")
	unityAngle = unityEnv.socketUnity(sendXYZ)
	print(unityAngle)
	print("5")
	if unityAngle != 'x':
		print("6")
		sendRasp = calculatorAngle(unityAngle, objName)
		print("sendData:",sendRasp);
		print("7")
		bluetoothRasp(sendRasp)
		print("8")
	
	


