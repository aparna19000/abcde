import numpy as np
import cv2


# Load the tracker
tracker = cv2.legacy.TrackerCSRT_create()

confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

video = cv2.VideoCapture("bb2.mp4")

# Flag variable to show whether basketball is detected or not
detected = False


def drawBox(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    # Draw rectangle and display a notification Tracking
    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


while True:
    check, image = video.read()
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    dimensions = image.shape[:2]
    H, W = dimensions

    # Detect the ball only if not detected earlier
    if detected == False:
        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416))
        yoloNetwork.setInput(blob)

        layerName = yoloNetwork.getUnconnectedOutLayersNames()
        layerOutputs = yoloNetwork.forward(layerName)

        boxes = []
        confidences = []
        classIds = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]

                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIds.append(classId)

        detectionNMS = cv2.dnn.NMSBoxes(
            boxes, confidences, confidenceThreshold, NMSThreshold)

        if (len(detectionNMS) > 0):
            for i in detectionNMS.flatten():

                if labels[classIds[i]] == "sports ball":
                    x, y, w, h = boxes[i]

                    color = (255, 0, 0)

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    # Initialise the tracker on the img and the bounding box
                    tracker.init(image, boxes[i])
                    # Changing flag variable value to true
                    detected = True
    else:
        # Update the tracker on the img and the bounding box
        trackerInfo = tracker.update(image)
        success = trackerInfo[0]
        bbox = trackerInfo[1]

        # Call drawBox() if the tracking is successful
        if success:
            drawBox(image, bbox)
        else:
            cv2.putText(image, "Lost", (75, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Image', image)
    cv2.waitKey(1)

    key = cv2.waitKey(25)
    if key == 32:
        print("Stopped")
        break
