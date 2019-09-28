# ms_ssim_pytorch

The code was modified from https://github.com/One-sixth/ms_ssim_pytorch.  
Part of the code has been modified to make it fitting our request, clean up some codes, and add reductions.

## Great speed up in pytorch 1.2. It is strongly recommended to update to pytorch 1.2 !

# Speed up. Only test on GPU.
ssim1 is https://github.com/lizhengwei1992/MS_SSIM_pytorch/blob/master/loss.py 268fc76  
ssim2 is https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py 881d210  
ssim3 is https://github.com/VainF/pytorch-msssim/blob/master/pytorch_msssim/ssim.py b47c07c  
reference is https://github.com/One-sixth/ms_ssim_pytorch/blob/master/ssim.py 0f69f16  

My test environment: i7-8700 GTX1080Ti  

## SSIM
Test output  

pytorch 1.2  
```
SSIM Performance Compare

Testing ssim2
cuda time:        	 31086.12109375
perf_counter time:	 30.369310373964254

Testing ssim3
cuda time:        	 12339.9169921875
perf_counter time:	 12.317489090957679

Testing reference
cuda time:        	 10347.5712890625
perf_counter time:	 9.909248579991981

Testing ours
cuda time:        	 10345.275390625
perf_counter time:	 9.90704761899542

```

## MS-SSIM
Test output  

pytorch 1.2  
```
MS_SSIM Performance Compare

Testing ssim1
cuda time:        	 42780.546875
perf_counter time:	 42.74076483398676

Testing ssim3
cuda time:        	 17724.392578125
perf_counter time:	 17.709311129001435

Testing reference
cuda time:        	 14692.154296875
perf_counter time:	 14.678562582994346

Testing ours
cuda time:        	 14446.7724609375
perf_counter time:	 14.433157603023574

```


## Test speed by yourself
1. python PerformanceTest.py

# Other thing
Add parameter use_padding.  
When set to True, the gaussian_filter behavior is the same as https://github.com/Po-Hsun-Su/pytorch-ssim.  
This parameter is mainly used for MS-SSIM, because MS-SSIM needs to be downsampled.  
When the input image is smaller than 176x176, this parameter needs to be set to True to ensure that MS-SSIM works normally. (when parameter weight and level are the default)  

# Require
Pytorch >= 1.1  

if you want to test the code with animation. You also need to install some package.  
```
pip install imageio imageio-ffmpeg opencv-python
```

# Test code with animation
The test code is included in the PerformanceTest.py file, you can run the file directly to start the test.  

1. git clone https://github.com/dororojames/ms_ssim_pytorch.git
2. cd ms_ssim_pytorch  
3. python PerformanceTest.py

# Code Example.
```python
import torch
import ssim


im = torch.randint(0, 255, (5, 3, 256, 256), dtype=torch.float, device='cuda')
img1 = im / 255
img2 = img1 * 0.5

losser = ssim.SSIM(data_range=1., channel=3).cuda()
loss = losser(img1, img2).mean()

losser2 = ssim.MS_SSIM(data_range=1., channel=3).cuda()
loss2 = losser2(img1, img2).mean()

print(loss.item())
print(loss2.item())
```

# Animation
GIF is a bit big. Loading may take some time.  
Or you can download the mkv video file directly to view it, smaller and smoother.  
https://github.com/dororojames/ms_ssim_pytorch/blob/master/ssim_test.mkv
https://github.com/dororojames/ms_ssim_pytorch/blob/master/ms_ssim_test.mkv  

SSIM  
![ssim](https://github.com/dororojames/ms_ssim_pytorch/blob/master/ssim_test.gif)

MS-SSIM  
![ms-ssim](https://github.com/dororojames/ms_ssim_pytorch/blob/master/ms_ssim_test.gif)

# References
https://github.com/VainF/pytorch-msssim  
https://github.com/Po-Hsun-Su/pytorch-ssim  
https://github.com/lizhengwei1992/MS_SSIM_pytorch  
https://github.com/One-sixth/ms_ssim_pytorch
