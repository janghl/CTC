from reproduce import Reproduce as rr
import time
import cv2
import socket
import os
import threading
# r = rr()
video = cv2.VideoCapture('video.mp4')
frame_count = 0
while True:         
    ret, frame = video.read()
    frame_count += 1
    if not ret:             
        break
    os.mkdir('foo') if not os.path.isdir('foo') else None
    outputPath = os.path.join("foo", f'frame_{frame_count}.jpg')
    cv2.imwrite(outputPath, frame)
