from time import perf_counter

import cv2
import numpy as np
import torch

import baseline.no1_ms_ssim_lizhengwei1992_MS_SSIM_pytorch as ssim1
import baseline.no2_ssim_Po_Hsun_Su_pytorch_ssim as ssim2
import baseline.no3_ssim_VainF_pytorch_msssim as ssim3
import baseline.no4_ssim_One_sixth_pytorch_msssim as ssim4
from ssim import MS_SSIM, SSIM


def SimpleTest():
    print('Simple Test')
    im = torch.randint(0, 255, (5, 3, 256, 256)).float().cuda()
    img1 = im / 255
    img2 = img1 * 0.5

    losser = SSIM(data_range=1.).cuda()
    loss = losser(img1, img2)
    print(loss)

    losser2 = MS_SSIM(data_range=1.).cuda()
    loss2 = losser2(img1, img2)
    print(loss2)

    print(loss.mean().item())
    print(loss2.mean().item())


def Training(t_im, ssim_type, save_video=False, video_ext="gif"):
    print('Training %s' % ssim_type)
    if save_video:
        if video_ext == "gif":
            fps = 2
            out_wh = (t_im.size(-1)//2, t_im.size(-2)//2)
        else:
            fps = 5
            out_wh = (t_im.size(-1), t_im.size(-2))
        video_last_time = perf_counter()
        from imageio import get_writer as videoWriter
        video = videoWriter('%s_test.%s' % (ssim_type, video_ext), fps=fps)

    if ssim_type == "SSIM":
        losser = SSIM(data_range=1., channel=t_im.size(1)).cuda()
    else:
        losser = MS_SSIM(data_range=1., channel=t_im.size(1)).cuda()

    rand_im = torch.randint_like(t_im, 0, 255, dtype=torch.float32) / 255.
    rand_im.requires_grad = True
    optim = torch.optim.Adam([rand_im], 0.005, eps=1e-8)

    ssim_score, epoch, scores = 0, 0, []
    while ssim_score < 0.999:
        optim.zero_grad()
        loss = losser(rand_im, t_im)
        (-loss).sum().backward()
        ssim_score = loss.item()
        optim.step()

        epoch += 1
        print(epoch, ssim_score)
        scores.append(ssim_score*100)
        r_im = np.transpose(rand_im.detach().cpu().numpy().clip(0, 1) * 255,
                            [0, 2, 3, 1]).astype(np.uint8)[0]
        r_im = cv2.putText(r_im, '%s %f' % (ssim_type, ssim_score), (10, 30),
                           cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if save_video:
            if perf_counter() - video_last_time > 1. / fps:
                video_last_time = perf_counter()
                out_frame = cv2.cvtColor(r_im, cv2.COLOR_BGR2RGB)
                out_frame = cv2.resize(out_frame, out_wh,
                                       interpolation=cv2.INTER_AREA)
                if isinstance(out_frame, cv2.UMat):
                    out_frame = out_frame.get()
                video.append_data(out_frame)

        # cv2.imshow(ssim_type, r_im)
        # cv2.setWindowTitle(ssim_type, '%s %f' % (ssim_type, ssim_score))
        # cv2.waitKey(1)

    if save_video:
        video.close()

    return scores


def TrainingTest(imgfile='test_img.jpg', *args, **kwarg):
    import matplotlib.pyplot as plt
    print('Training Test')
    im = cv2.imread(imgfile, 1)
    t_im = torch.from_numpy(im).cuda().permute(2, 0, 1).float()[None] / 255.

    ssim_score = Training(t_im, "SSIM", *args, **kwarg)
    msssim_score = Training(t_im, "MS_SSIM", *args, **kwarg)

    curve = plt.figure()
    plt.plot(ssim_score, label="ssim")
    plt.plot(msssim_score, label="ms_ssim")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Score(%)")
    plt.title("SSIM Traing Score")
    plt.savefig("./curve.png")
    plt.close(curve)


def PerformanceTesting(losser):
    a = torch.randint(0, 255, (20, 3, 256, 256)).float().cuda() / 255.
    b = a * 0.5
    a.requires_grad = True
    b.requires_grad = True

    start_record = torch.cuda.Event(enable_timing=True)
    end_record = torch.cuda.Event(enable_timing=True)

    start_time = perf_counter()
    start_record.record()
    for _ in range(500):
        loss = losser(a, b).mean()
        loss.backward()
    end_record.record()
    end_time = perf_counter()

    torch.cuda.synchronize()

    print('cuda time:        \t', start_record.elapsed_time(end_record))
    print('perf_counter time:\t', end_time - start_time)


def ShowPerformance():
    print('Performance Testing SSIM')
    PerformanceTesting(SSIM(data_range=1.).cuda())
    print('Performance Testing MS_SSIM')
    PerformanceTesting(MS_SSIM(data_range=1.).cuda())


def PerformanceCompare(ssim_type="SSIM"):
    print('\n%s Performance Compare' % ssim_type)
    print()
    if ssim_type == "SSIM":
        losser1 = ssim2.SSIM(window_size=11, size_average=False)
        losser2 = ssim3.SSIM(win_size=11, win_sigma=1.5, data_range=1.,
                             size_average=False, channel=3)
        reference = ssim4.SSIM(data_range=1.)
        ours = SSIM(data_range=1.)
        names = ["losser2", "losser3", "reference", "ours"]
    else:
        losser1 = ssim1.MS_SSIM(size_average=False, max_val=1.)
        losser2 = ssim3.MS_SSIM(win_size=11, win_sigma=1.5,
                                data_range=1., size_average=False, channel=3)
        reference = ssim4.MS_SSIM(data_range=1.)
        ours = MS_SSIM(data_range=1.)
        names = ["losser1", "losser3", "reference", "ours"]

    for idx, losser in enumerate([losser1, losser2, reference, ours]):
        print('Testing %s' % names[idx])
        PerformanceTesting(losser.cuda())
        print()


if __name__ == '__main__':
    torch.cuda.manual_seed_all(0)
    SimpleTest()
    TrainingTest('test_img.jpg', save_video=True, video_ext="gif")
    # ShowPerformance()
    PerformanceCompare("SSIM")
