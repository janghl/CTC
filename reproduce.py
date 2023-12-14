import time
import cv2
import socket
import os
import threading
import argparse
import codec
import torch
from models import model_CTC
from codec import _enc,_dec
import base64
import shutil

PORT = 8080
WAIT_TIME = 1.5
class Reproduce:
    def __init__(self, args):
        net = model_CTC(N=192).to("cuda")
        ckpt = torch.load("ctc.pt")["state_dict"]
        net.load_state_dict(ckpt)
        net.update()
        self.net = net
        self.ip = socket.gethostbyname(socket.gethostname())
        self.addr = (self.ip, PORT)
        self.source = args.source
        self.dest = args.dest
        os.mkdir(self.dest) if not os.path.isdir(self.dest) else None
        self.interval = args.interval/1000
        sender_thread = threading.Thread(target=self.sender)
        receiver_thread = threading.Thread(target=self.receiver)
        sender_thread.start()
        receiver_thread.start()
        sender_thread.join()
        receiver_thread.join()
        
    def clear_dir(self, folder):
        if os.path.exists(folder):
            for content in os.listdir(folder):
                if os.path.isfile(f'{folder}/{content}'):
                    os.remove(f'{folder}/{content}')
                    # print(f'{folder}/{content}')
                elif os.path.isdir(f'{folder}/{content}'):
                    self.clear_dir(self, os.path.join(folder, content))
                    os.rmdir(folder)
                    # print(folder)
        else:
            os.mkdir(folder)
        
    def sender(self):
        self.clear_dir('sender')
        with open('sender/log', 'w') as f:
            video = cv2.VideoCapture(self.source)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.addr)
                f.write('connected to receiver!')
                frame_count = 0
                # for each frame
                while True:         
                    startTime = time.time()
                    ret, frame = video.read()
                    frame_count += 1
                    bin_count = 0
                    # no more frames left
                    if not ret:             
                        s.sendall("task finished!")
                        f.write(f"task finished with frame = {frame_count}!")
                        break
                    outputPath = os.path.join("sender", f'frame_{frame_count}.jpg')
                    cv2.imwrite(outputPath, frame)
                    args = [f"--input-file={outputPath}", "--cuda", f"--mode=enc", f"--save-path=sender"] 
                    _enc(args, self.net)
                    for binFile in os.listdir("sender/bits"):
                        fileSize = os.path.getsize(f'sender/bits/{binFile}')
                        block_count = fileSize//1024 if fileSize%1024==0 else fileSize//1024+1
                        s.sendall(f"frame={frame_count},fname={binFile},bin_count={bin_count},block_count={block_count}".encode())
                        if time.time()<startTime+self.interval:
                            with open(f'sender/bits/{binFile}', 'r') as b:
                                s.sendall(b.read())
                            bin_count += 1
                    s.sendall("begin decode!".encode())
                    time.sleep(WAIT_TIME*self.interval)
                    f.write(f"in frame {frame_count} I sent {bin_count} cipher texts out of 160! Ratio = {bin_count/160}")
                    self.clear_dir('sender')
        pass
    
    def receiver(self):
        self.clear_dir('receiver')
        with open('receiver/log', 'w') as f:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(self.addr)
                s.listen(10)
                conn, address = s.accept()
                f.write('connected to sender!')
                frame_count = 0
                while True:
                    message = conn.recv(1024).decode()
                    if message.startswith("task"):      # task finished
                        f.write(f"task finished with frame = {frame_count}")
                        break
                    elif message.startswith("frame"):
                        new_frame_count = message.split(',')[0].split('=')[1]
                        binFile = message.split(',')[1].split('=')[1]       # file name
                        bin_count = message.split(',')[2].split('=')[1]     # successfully sent bins
                        block_count = message.split(',')[3].split('=')[1]   # socket transfer in blocks
                        if new_frame_count != frame_count:
                            self.clear_dir("receiver/bits")
                            frame_count = new_frame_count
                        with open(binFile, "wb") as b:
                            for iter in range(block_count):
                                b.write(s.recv(1024))
                            f.write(f"in frame {frame_count} file {binFile} I wrote {block_count} blocks")
                    elif message.startswith("begin"):
                        args = [f"--cuda", f"--mode=dec", f"--save-path=receiver/bits"]  
                        _dec(args, self.net)
                        shutil.copyfile("receiver/bits/recon/q0160.png", os.path.join(self.dest, f"{frame_count}.png"))
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='source directory, e.g. video.mp4', default='video.mp4')
    parser.add_argument('-d', '--dest', help='output directory', default='receiver')
    parser.add_argument('-t', '--interval', help='max time interval, e.g. 30ms', type=int, default=30)
    args = parser.parse_args()
    reproduce = Reproduce(args)