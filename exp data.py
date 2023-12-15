import statistics
import os
import re

os.chdir('CTC')
for num in [100, 130, 160]:
    with open(f'recon{num}/data_{num}', 'r') as file:
        lines = file.readlines()

    dec_times = []
    bpps = []
    enc_times = []

    for line in lines:
        if line.startswith("dec"):
            matches = re.findall(r"dec time: ([\d.]+), bpp: ([\d.]+) Enc ([\d.]+)sec", line)
            if matches:
                dec_time, bpp, enc_time = matches[0]
                
                dec_times.append(float(dec_time))
                bpps.append(float(bpp))
                enc_times.append(float(enc_time))

    dec_time_mean = statistics.mean(dec_times)
    bpp_mean = statistics.mean(bpps)
    enc_time_mean = statistics.mean(enc_times)

    dec_time_std = statistics.stdev(dec_times)
    bpp_std = statistics.stdev(bpps)
    enc_time_std = statistics.stdev(enc_times)

    with open(f'expData_{num}', 'w') as f:
        f.write(f"Dec Time Mean: {dec_time_mean}\n")
        f.write(f"BPP Mean: {bpp_mean}\n")
        f.write(f"Enc Time Mean: {enc_time_mean}\n")
        f.write(f"Dec Time Standard Deviation: {dec_time_std}\n")
        f.write(f"BPP Standard Deviation: {bpp_std}\n")
        f.write(f"Enc Time Standard Deviation: {enc_time_std}\n")