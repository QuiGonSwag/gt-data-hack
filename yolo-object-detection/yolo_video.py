# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import json
import math
import pandas as pd


def euclidean_distance(p1, p2):
	return np.linalg.norm(np.array(p1) - np.array(p2))


def find_nearest(new_player, old_frame):
	closest_player_ndx = -1
	nearest_distance = float('inf')
	for old_ndx, old_player in old_frame.items():
		if old_player is None:
			continue
		topLeftDistance = euclidean_distance((new_player['X'],new_player['Y']),(old_player['X'],old_player['Y']))
		bottomRightDistance = euclidean_distance((new_player['X']+new_player['W'],new_player['Y']+new_player['H']),(old_player['X']+old_player['W'],old_player['Y']+old_player['H']))
		if np.mean([topLeftDistance,bottomRightDistance]) < nearest_distance:
			nearest_distance = np.mean([topLeftDistance,bottomRightDistance])
			closest_player_ndx = old_ndx
			
	return closest_player_ndx

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def withinTrap(X,Y):
	point = Point(X,Y)
	polygon = Polygon([(0, 1080),(0,582),(627,225),(1259,225),(1920,650),(1920,1080)])
	return polygon.contains(point)
        

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

outputArray = []
playerDatabase = {}
frames = -1

# loop over frames from the video file stream
while True:
	frames +=1
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		print("OMG YESSSSS",H,W)

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []


	

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	print("BOXES", boxes)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	print("IDXs",idxs)


	players = {}
	playercnt = 0

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			

			text = LABELS[classIDs[i]]
			# draw a bounding box rectangle and label on the frame


			if (str(text) == "person" and w < 500 and withinTrap(x,y)):
				if outputArray:
					nearestNeighbor = find_nearest({'X':x,'Y':y,'W':w,'H':h}, playerDatabase)
					players[playercnt] = {'X':x,'Y':y,'W':w,'H':h, 'closestPlayer': nearestNeighbor}
					playerDatabase[nearestNeighbor] = {'X':x,'Y':y,'W':w,'H':h}
				else:
					players[playercnt] = {'X':x,'Y':y,'W':w,'H':h, 'closestPlayer': -1}
					playerDatabase[playercnt] = {'X':x,'Y':y,'W':w,'H':h}

				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				cv2.putText(frame, str(players[playercnt]['closestPlayer']), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

				playercnt += 1	

				# print("Name:%s X:%d Y:%d W:%d H:%d" %(str(text),x,y,w,h))
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)
	# print(players)
	outputArray.append(playerDatabase)
	if (frames >= 1000):
		break
	# print(frames)

# release the file pointers
with open('data.json', 'w') as json_file:
    json.dump(outputArray, json_file)

print("[INFO] cleaning up...")
writer.release()
vs.release()