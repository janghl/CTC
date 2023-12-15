import time
import cv2
import socket
from bar import Encapsulation
import os
import threading
import argparse
import codec
import torch
from models import model_CTC
from test import main
import base64
import shutil

PORT = 8080
class Reproduce:
    def __init__(self, args):
        self.debug=True
        net = model_CTC(N=192).to("cuda")
        ckpt = torch.load("ctc.pt")["state_dict"]
        net.load_state_dict(ckpt)
        net.update()
        self.net = net
        self.ip = socket.gethostbyname(socket.gethostname())
        self.addr = (self.ip, PORT)
        self.source = args.source
        self.max_frame = args.max_frame
        self.dest = args.dest
        os.mkdir(self.dest) if not os.path.isdir(self.dest) else None
        self.interval = args.interval
        sender_thread = threading.Thread(target=self.sender)
        receiver_thread = threading.Thread(target=self.receiver)
        sender_thread.daemon = True
        receiver_thread.daemon = True
        sender_thread.start()
        receiver_thread.start()
        sender_thread.join()
        receiver_thread.join()
        
    def clear_dir(self, folder, exception=None):
        if os.path.exists(folder):
            for content in os.listdir(folder):
                if os.path.isfile(f'{folder}/{content}') and not content==exception:
                    os.remove(f'{folder}/{content}')
                    # print(f'{folder}/{content}')
                elif os.path.isdir(f'{folder}/{content}'):
                    self.clear_dir(os.path.join(folder, content))
                    os.rmdir(os.path.join(folder, content))
                    # print(folder)
        else:
            os.mkdir(folder)
            
    def padding(self, buffer, size=1024):
        return buffer+" ".encode('utf-8')*(1024-len(buffer))
        
    def sender(self):
        self.clear_dir('sender')
        with open('sender/log', 'w') as fs:
            video = cv2.VideoCapture(self.source)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(self.addr)
                fs.write('connected to receiver!\n')
                fs.flush()
                frame_count = 0
                # for each frame
                while True:         
                    startTime = time.time()
                    ret, frame = video.read()
                    frame_count += 1
                    fs.write(f'load frame {frame_count}!\n')
                    fs.flush()
                    bin_count = 0
                    # no more frames left
                    if not ret:             
                        s.sendall(self.padding("task finished!".encode('utf-8')))
                        fs.write(f"task finished with frame = {frame_count}!\n")
                        fs.flush()
                        break
                    outputPath = os.path.join("sender", f'frame_{frame_count}.jpg')
                    cv2.imwrite(outputPath, frame)
                    args = [f"--input-file={outputPath}", "--cuda", f"--mode=enc", f"--save-path=sender"] 
                    # _enc(args, self.net)
                    main(args, self.net)
                    for binFile in sorted(os.listdir("sender/bits")):
                        fileSize = os.path.getsize(f'sender/bits/{binFile}')
                        s.sendall(self.padding(f"frame={frame_count},fname={binFile},bin_count={bin_count},fileSize={fileSize},".encode('utf-8')))
                        if time.time()<startTime+self.interval:
                            with open(f'sender/bits/{binFile}', 'rb') as b:
                                while fileSize>0:
                                    if fileSize>1024:
                                        s.send(b.read(1024))
                                        fileSize -= 1024
                                    else:
                                        s.send(b.read(fileSize))
                                        fileSize -= fileSize
                            bin_count += 1
                        if self.debug:
                            fs.write(f"sent file {binFile} with count {bin_count}!\n")
                            fs.flush()
                    s.sendall(self.padding("begin decode!".encode('utf-8')))
                    fs.write(f"in frame {frame_count} I sent {bin_count} cipher texts out of 160! Ratio = {bin_count/160}\n")
                    fs.flush()
                    self.clear_dir('sender/bits')
                    if s.recv(1024).decode().startswith('success'):
                        fs.write(f"receiver decode success\n")
                        fs.flush()
                        
        pass
    
    def receiver(self):
        self.clear_dir('receiver')
        with open('receiver/log', 'w') as f:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(self.addr)
                s.listen(10)
                conn, address = s.accept()
                f.write('connected to sender!\n')
                f.flush()
                frame_count = 0
                while int(frame_count)<int(self.max_frame):
                    message = conn.recv(1024).decode('utf-8')
                    if self.debug:
                        f.write("message: "+message+"\n")
                        f.flush()
                    if message.startswith("task"):      # task finished
                        f.write(f"task finished with frame = {frame_count}\n")
                        f.flush()
                        break
                    elif message.startswith("frame"):
                        new_frame_count = message.split(',')[0].split('=')[1]
                        binFile = message.split(',')[1].split('=')[1]       # file name
                        bin_count = message.split(',')[2].split('=')[1]     # successfully sent bins
                        fileSize = message.split(',')[3].split('=')[1]   # socket transfer in blocks
                        fileSize = int(fileSize)
                        Size = int(fileSize)
                        if new_frame_count != frame_count:
                            self.clear_dir("receiver/bits")
                            frame_count = new_frame_count
                        with open(os.path.join("receiver/bits", binFile), "wb") as b:
                            while Size>0:
                                if Size>1024:
                                    b.write(conn.recv(1024))
                                    Size -= 1024
                                else:
                                    b.write(conn.recv(Size))
                                    Size -= Size
                            if self.debug:
                                f.write(f"in frame {frame_count} file {binFile} I wrote {fileSize} bytes\n")
                                f.flush()
                    elif message.startswith("begin"):
                        args = [f"--cuda", f"--mode=dec", f"--save-path=receiver"]  
                        # _dec(args, self.net)
                        main(args, self.net)
                        shutil.copyfile("receiver/recon/q0160.png", f"receiver/frame_{frame_count}.png")
                        self.clear_dir('receiver/bits')
                        self.clear_dir('receiver/recon')
                        f.write(f"decoded frame {frame_count} finished!\n")
                        f.flush()
                        conn.send(self.padding("success".encode()))
                        f.write(f"decode success\n")
                        f.flush()
                    
                Encapsulation()
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='source directory, e.g. video.mp4', default='video.mp4')
    parser.add_argument('-d', '--dest', help='output directory', default='receiver')
    parser.add_argument('-t', '--interval', help='max time interval, e.g. 30s', type=int, default=30)
    parser.add_argument('-m', '--max_frame', help='trancate video frames, e.g. 10', type=int, default=10)
    args = parser.parse_args()
    reproduce = Reproduce(args)
