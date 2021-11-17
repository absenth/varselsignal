import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time


stream_server = "192.168.1.139"
stream_user = "streamuser"      # Put something better here.
stream_pass = "passwd123"       # Put something better here.
stream_port = "1050"            # Set to your RTSP Server Port
stream_channel = "602"          # This could be several channels, if we wanted to monitor more than one?



with open('security-cameras.names', 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]


def detect_people(frame, outs, height, width, confidence_min):
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if confidence > confidence_min:
                print(f"Confidence: {confidence}, min {confidence_min}")
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    return boxes, confidences, class_ids


def put_boxes(frame, boxes, confidences, class_ids):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.5)
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    font = cv2.FONT_HERSHEY_PLAIN

    # Do the person Detection
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            print(boxes[i])
            label = str(CLASSES[class_ids[i]])
            if label == 'person':
                color = colors[class_ids[i]]
                cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness = -1)
                # cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def main(confidence_min):
    frame_counter = 1
    cap = cv2.VideoCapture(f"rtsp://{stream_user}:{stream_pass}@{stream_server} / Streaming/Channels/{stream_channel}")

    if cap.isOpened() is False:
        raise BaseException("Error opening video stream")

    net = cv.dnn.readNet('security-cameras_best.weights', 'yolov4-tiny.cfg')

    while True:
        frames_counter = frames_counter + 1
        ret, image = cap.read()
        print('Is an image: ', ret)
        if not ret:
            print("No return, breaking")
            break

        scale = 40
        Width = int(image.shape[1] * scale / 100) - 100
        Height = int(image.shape[0] * scale / 100) - 500
        dim = (Width, Height)
        print("resized")

        image = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)

        blob = cv2.dnn.blobFromImage(image, scale/100, (480, 256), True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

        outs = net.forward(output_layers)
        print("Detecting People")
        boxes, confidences, class_ids = detect_people(image, outs, Height, Width, confidence_min)
        image = put_boxes(image, boxes, confidences, class_ids)

        print("showing image")
        cv2.imshow('VIDEO', image)

        if cv2.waitKey(25) & oxFF == ord('q'):
            break
        print('to next frame\n')

    print(f"Total Frames: {frames_counter}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == " __main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--confidence", type=float, dest="confidence", default=0.1)
    args = parser.parse_args()

    main(args.confidence)




'''
    while(1):

        ret, frame = vcap.read()
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)

https://pastebin.com/M5fvXFTy

### OLD STUFF -- This is for reference only.
  classes = None
  with open('security-cameras.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

  net = cv.dnn.readNet('security-cameras_best.weights', 'yolov4-tiny.cfg')
  net.setInput(cv.dnn.blobFromImage(vcap, 0.00392, (480,256), (0,0,0), True, crop=False))
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
  outs = net.forward(output_layers)


  class_ids = []
  confidences = []
  boxes = []
  Width = vcap.shape[1]
  Height = vcap.shape[0]
  for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.1:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


  indices = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
  #check if is people detection
  for i in indices:
    i = i[0]
    box = boxes[i]
    if class_ids[i]==0:
        label = str(classes[class_id])
  #        cv.rectangle(vcap, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
  #        cv.putText(vcap, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv.rectangle(vcap, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (255, 255, 0), 1)
        cv.putText(vcap, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
'''
