# uiuc computer vision final project

## Installation
Download [pre-trained model](https://drive.google.com/file/d/1q0IyOnOcl9E9Y07viYjmLmj3FkA3ZDMT/view?usp=sharing) parameters on the root path.

## Usage
### Encoding
```bash
  $ python codec.py --mode enc --save-path {path} --input-file {input image file} --cuda
```
"--cuda" is optional.

For exasmple, command below
```bash
  $ python codec.py --mode enc --save-path results --input-file sample.png --cuda
```
generates binary files in "results/bits".

### Decoding
```bash
  $ python codec.py --mode dec --save-path {path same with enc} --input-file {original image file} --recon-level {int} --cuda
```
"--input-file" is optional, used to calculate PSNR.

For example, command below
```bash
  $ python codec.py --mode dec --save-path results --input-file sample.png --recon-level 140 --cuda
```
prints metrics and saves reconstructed an image "results/recon/q0140.png".

### Experimenting with datasets
```bash
  $ python test.py 
```

### Simulating real time video transforming
```bash
  $ python reproduce.py -s {source directory} -d {target file name} -t {time interval for network condition} -m {truncate first few frames from input video}
```

For example, command below
```bash
  $ python reproduce.py -s video.mp4 -d output.mp4 -t 5 -m 30
```
