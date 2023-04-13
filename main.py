import cv2
import numpy as np
import os
import torch
from dotenv import load_dotenv
from ultralytics import YOLO


def get_creds():
    load_dotenv()
    stream_server = os.getenv('STREAM_SERVER')
    stream_user = os.getenv('STREAM_USER')
    stream_pass = os.getenv('STREAM_PASS')
    stream_port = os.getenv('STREAM_PORT')
    stream_channel = os.getenv('STREAM_CHANNEL')

    return stream_server, stream_user, stream_pass, stream_port, stream_channel


def main():
    stream_server, stream_user, stream_pass, stream_port, stream_channel = get_creds()
    stream_url = f'rtsp://{stream_user}:{stream_pass}@{stream_server}:{stream_port}/Streaming/Channels/{stream_channel}'

    cap = cv2.VideoCapture(stream_url)
    model = YOLO('yolov8l.pt')
    class_names = model.names
    desired_class = 'person'

    processing_frame = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # skip to the next iteration if there was an issue reading the frame

        if processing_frame:
            continue

        processing_frame = True

        results = model(frame, device='mps')
        result = results[0]
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')
        classes = np.array(result.boxes.cls.cpu(), dtype='int')
        confidences = np.array(result.boxes.conf.cpu(), dtype='float')
        for bbox, cls, confidence in zip(bboxes, classes, confidences):
            (x, y, x2, y2) = bbox
            if cls != 0:
                continue
            if confidence < 0.50:
                continue
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_names[cls], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('IMG', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

        processing_frame = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
