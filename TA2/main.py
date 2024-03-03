import numpy as np
import cv2

# Trajectory array Coords
xCoords = []
yCoords = []

tracker = cv2.legacy.TrackerCSRT_create()

confidenceThreshold = 0.3
NMSThreshold = 0.1

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'

labels = open(labelsPath).read().strip().split('\n')

yoloNetwork = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

video = cv2.VideoCapture("bb2.mp4")


detected = False


def drawBox(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# Define the function to track the goal
def goalTrack(img, bbox):
    x = int(bbox[0])
    y = int(bbox[1])
    w = int(bbox[2])
    h = int(bbox[3])

    # Get the CENTER Points of the Bounding Box
    c1 = x + int(w/2)
    c2 = y + int(h/2)

    # Append the center points c1 and c2 to xCoords and yCoords respectively
    xCoords.append(c1)
    yCoords.append(c2)

    # Draw the circles for the previous center points
    for i in range(len(xCoords)-1):
        cv2.circle(img, (xCoords[i], yCoords[i]), 2, (0, 0, 255), 5)


while True:
    check, image = video.read()
    image = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    dimensions = image.shape[:2]
    H, W = dimensions

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

                    tracker.init(image, boxes[i])
                    detected = True
    else:
        trackerInfo = tracker.update(image)
        success = trackerInfo[0]
        bbox = trackerInfo[1]

        if success:
            drawBox(image, bbox)
        else:
            cv2.putText(image, "Lost", (75, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Call goalTrack() function
        goalTrack(image, bbox)

    cv2.imshow('Image', image)
    cv2.waitKey(1)

    key = cv2.waitKey(25)
    if key == 32:
        print("Stopped")
        break
