from time import perf_counter

import torch

from ssim import MS_SSIM, SSIM


def SimpleTest():
    print('Simple Test')
    im = torch.randint(0, 255, (5, 3, 256, 256),
                       dtype=torch.float, device='cuda')
    img1 = im / 255
    img2 = img1 * 0.5

    losser = SSIM(data_range=1.).cuda()
    loss = losser(img1, img2)
    print(loss)
    loss = loss.mean()

    losser2 = MS_SSIM(data_range=1.).cuda()
    loss2 = losser2(img1, img2)
    print(loss2)
    loss2 = loss2.mean()

    print(loss.item())
    print(loss2.item())


def TrainingTest():
    print('Training Test')
    import cv2
    import numpy as np
    import imageio

    out_test_video = False
    # 最好不要直接输出gif图，会非常大，最好先输出mkv文件后用ffmpeg转换到GIF
    video_use_gif = False

    im = cv2.imread('test_img1.jpg', 1)
    t_im = torch.from_numpy(im).cuda().permute(2, 0, 1).float()[None] / 255.

    if out_test_video:
        if video_use_gif:
            fps = 0.5
            out_wh = (im.shape[1]//2, im.shape[0]//2)
            suffix = '.gif'
        else:
            fps = 5
            out_wh = (im.shape[1], im.shape[0])
            suffix = '.mkv'
        video_last_time = perf_counter()
        video = imageio.get_writer('ssim_test'+suffix, fps=fps)

    # 测试ssim
    print('Training SSIM')
    rand_im = torch.randint_like(t_im, 0, 255, dtype=torch.float32) / 255.
    rand_im.requires_grad = True
    optim = torch.optim.Adam([rand_im], 0.003, eps=1e-8)
    losser = SSIM(data_range=1., channel=t_im.shape[1]).cuda()
    ssim_score = 0
    while ssim_score < 0.999:
        optim.zero_grad()
        loss = losser(rand_im, t_im)
        (-loss).sum().backward()
        ssim_score = loss.item()
        optim.step()
        r_im = np.transpose(rand_im.detach().cpu().numpy().clip(
            0, 1) * 255, [0, 2, 3, 1]).astype(np.uint8)[0]
        r_im = cv2.putText(r_im, 'ssim %f' % ssim_score,
                           (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if out_test_video:
            if perf_counter() - video_last_time > 1. / fps:
                video_last_time = perf_counter()
                out_frame = cv2.cvtColor(r_im, cv2.COLOR_BGR2RGB)
                out_frame = cv2.resize(
                    out_frame, out_wh, interpolation=cv2.INTER_AREA)
                if isinstance(out_frame, cv2.UMat):
                    out_frame = out_frame.get()
                video.append_data(out_frame)

        cv2.imshow('ssim', r_im)
        cv2.setWindowTitle('ssim', 'ssim %f' % ssim_score)
        cv2.waitKey(1)

    if out_test_video:
        video.close()

    # 测试ms_ssim
    if out_test_video:
        if video_use_gif:
            fps = 0.5
            out_wh = (im.shape[1]//2, im.shape[0]//2)
            suffix = '.gif'
        else:
            fps = 5
            out_wh = (im.shape[1], im.shape[0])
            suffix = '.mkv'
        video_last_time = time.perf_counter()
        video = imageio.get_writer('ms_ssim_test'+suffix, fps=fps)

    print('Training MS_SSIM')
    rand_im = torch.randint_like(t_im, 0, 255, dtype=torch.float32) / 255.
    rand_im.requires_grad = True
    optim = torch.optim.Adam([rand_im], 0.003, eps=1e-8)
    losser = MS_SSIM(data_range=1., channel=t_im.shape[1]).cuda()
    ssim_score = 0
    while ssim_score < 0.999:
        optim.zero_grad()
        loss = losser(rand_im, t_im)
        (-loss).sum().backward()
        ssim_score = loss.item()
        optim.step()
        r_im = np.transpose(rand_im.detach().cpu().numpy().clip(
            0, 1) * 255, [0, 2, 3, 1]).astype(np.uint8)[0]
        r_im = cv2.putText(r_im, 'ms_ssim %f' % ssim_score,
                           (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        if out_test_video:
            if perf_counter() - video_last_time > 1. / fps:
                video_last_time = perf_counter()
                out_frame = cv2.cvtColor(r_im, cv2.COLOR_BGR2RGB)
                out_frame = cv2.resize(
                    out_frame, out_wh, interpolation=cv2.INTER_AREA)
                if isinstance(out_frame, cv2.UMat):
                    out_frame = out_frame.get()
                video.append_data(out_frame)

        cv2.imshow('ms_ssim', r_im)
        cv2.setWindowTitle('ms_ssim', 'ms_ssim %f' % ssim_score)
        cv2.waitKey(1)

    if out_test_video:
        video.close()


def PerformanceTesting(losser):
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

    print('cuda time', start_record.elapsed_time(end_record))
    print('perf_counter time', end_time - start_time)


def ShowPerformance():
    print('Performance Testing SSIM')
    PerformanceTesting(SSIM(data_range=1.).cuda())
    print('Performance Testing MS_SSIM')
    PerformanceTesting(MS_SSIM(data_range=1.).cuda())


def PerformanceCompare(ssim_type="SSIM"):
    import baseline.no1_ms_ssim_lizhengwei1992_MS_SSIM_pytorch as ssim1
    import baseline.no2_ssim_Po_Hsun_Su_pytorch_ssim as ssim2
    import baseline.no3_ssim_VainF_pytorch_msssim as ssim3
    import baseline.no4_ssim_One_sixth_pytorch_msssim as ssim4
    print('\n%s Performance Compare' % ssim_type)
    print()
    if ssim_type == "SSIM":
        losser1 = ssim2.SSIM(window_size=11, size_average=False)
        losser2 = ssim3.SSIM(win_size=11, win_sigma=1.5, data_range=1.,
                             size_average=False, channel=3)
        reference = ssim4.SSIM(window_size=11, window_sigma=1.5,
                               data_range=1., channel=3, use_padding=False)
        ours = SSIM(window_size=11, window_sigma=1.5, scale=True,
                    data_range=1., channel=3, use_padding=False)
    else:
        losser1 = ssim1.MS_SSIM(size_average=False, max_val=1.)
        losser2 = ssim3.MS_SSIM(win_size=11, win_sigma=1.5,
                                data_range=1., size_average=False, channel=3)
        reference = ssim4.MS_SSIM(window_size=11, window_sigma=1.5,
                                  data_range=1., channel=3, use_padding=False)
        ours = MS_SSIM(window_size=11, window_sigma=1.5, scale=True,
                       data_range=1., channel=3, use_padding=False)

    print('testing losser1')
    PerformanceTesting(losser1.cuda())
    print()

    print('testing losser2')
    PerformanceTesting(losser2.cuda())
    print()

    print('testing reference')
    PerformanceTesting(reference.cuda())
    print()

    print('testing ours')
    PerformanceTesting(ours.cuda())
    print()


if __name__ == '__main__':
    torch.cuda.manual_seed_all(0)
    SimpleTest()
    # TrainingTest()
    a = torch.randint(0, 255, (20, 3, 256, 256),
                      dtype=torch.float32, device='cuda') / 255.
    b = a * 0.5
    a.requires_grad = True
    b.requires_grad = True
    # ShowPerformance()
    PerformanceCompare("SSIM")
