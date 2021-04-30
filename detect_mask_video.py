# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob  from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence 
		confidence = detections[0, 0, i, 2]

		
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(x1, y1, x2, y2) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(x1, y1) = (max(0, x1), max(0, y1))
			(x2, y2) = (min(w - 1, x2), min(h - 1, y2))

			#resize it to 224x224, and preprocess it
			face = frame[y1:y2, x1:x2]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((x1, y1, x2, y2))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"G:\Mini_project_new\Mask_detection\face_detector\deploy.prototxt"
weightsPath = r"G:\Mini_project_new\Mask_detection\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("G:\Mini_project_new\Mask_detection\mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	
	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()