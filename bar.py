import time
import cv2
import socket
import os
import threading
import re

def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0


def Encapsulation(directory='receiver',output='output.mp4'):
    frame_files=[f for f in os.listdir(directory) if f.startswith('frame_') and f.endswith('.png')]
    frame_files.sort(key=numerical_sort)
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    fps=30
    #print(frame_files)
    frame_width,frame_height=cv2.imread(os.path.join(directory,frame_files[0])).shape[:2]
    video=cv2.VideoWriter(output,fourcc,fps,(frame_height,frame_width),True)

    for frame_file in frame_files:
        frame_path=os.path.join(directory,frame_file)
        frame=cv2.imread(frame_path)
        video.write(frame)
    video.release()
        
    
