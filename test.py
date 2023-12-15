import os
import sys
import argparse

import time
import torch

from models import model_CTC
from utils import path2torch, torch2img, psnr

torch.autograd.set_detect_anomaly(True)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument("--mode", type=str, choices=["enc", "dec"], default="enc")
    parser.add_argument("--save-path", type=str, default="results")
    parser.add_argument("--input-file", type=str, default=None)
    parser.add_argument("--recon-level", type=int, choices=list(range(1, 161)), default=160)
    parser.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args(argv)
    return args


def _enc(args, net):
    x = path2torch(args.input_file).to(args.device)
    save_path_enc = os.path.join(args.save_path, "bits")
    if not os.path.exists(save_path_enc): os.mkdir(save_path_enc)
    net.encode_and_save_bitstreams_ctc(x, save_path_enc)


def _dec(args, net):
    save_path_dec = os.path.join(args.save_path, "recon")
    if not os.path.exists(save_path_dec): os.mkdir(save_path_dec)
    dec_time, x_rec, bpp = net.reconstruct_ctc(args)
    torch2img(x_rec).save(f"{save_path_dec}/q{args.recon_level:04d}.png")

    print(f"dec time: {dec_time:.3f}, bpp: {bpp:.5f}", end=" ")

    if args.input_file is not None:
        x_in = path2torch(args.input_file).to(args.device)
        metric = psnr(x_in, x_rec)
        print(f", psnr: {metric:.4f}")


def main(argv, net):
    args = parse_args(argv)
    args.device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.mode == "enc":
        _enc(args, net)

    elif args.mode == "dec":
        _dec(args, net)

    else:
        raise ValueError(f"{args.mode} error: choose 'enc' or 'dec'.")

def get_directory_size(directory):
    total_size = 0

    for path, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)

    return total_size

if __name__ == "__main__":
    net = model_CTC(N=192).to("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("ctc.pt")["state_dict"]
    net.load_state_dict(ckpt)
    net.update()

    images_dir = "kodak"  # image directory
    total_enc_time = 0.0
    total_dec_time = 0.0
    total_target_size = 0.0

    image_files = os.listdir(images_dir)
    num_images = len(image_files)

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        target_path = os.path.join("results", os.path.splitext(image_file)[0])
        os.makedirs(target_path, exist_ok=True)
        image_path = os.path.join(images_dir, image_file)
        args = [f"--input-file={image_path}", "--cuda", f"--mode=enc", f"--save-path={target_path}"]  # 构造命令行参数
        enc_start_time = time.time()
        main(args, net)  # encode by calling main func
        enc_end_time = time.time()
        enc_time = enc_end_time - enc_start_time

        dec_start_time = time.time()
        args[2] = f"--mode=dec"  # modify mode as decoding
        main(args, net)  
        dec_end_time = time.time()
        dec_time = dec_end_time - dec_start_time

        total_enc_time += enc_time
        total_dec_time += dec_time
        total_target_size += get_directory_size(os.path.join(target_path, "bits"))
          
    total_original_size = get_directory_size(images_dir)
    avg_enc_time = total_enc_time / num_images
    avg_dec_time = total_dec_time / num_images
    avg_cr = 1 - total_target_size / total_original_size

    print(f"Average Encoding Time: {avg_enc_time:.3f} seconds")
    print(f"Average Decoding Time: {avg_dec_time:.3f} seconds")
    print(f"Average Compression rate: {avg_cr:.5f}")
