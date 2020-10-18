import cv2
import numpy as np
import torch
import torch.nn.functional as F
from callonce import printonce

def get_flow_with_size(img0, img1, dis, size=(68, 120)):
    tmp0 = np.array(img0)
    tmp1 = np.array(img1)
    tmp0 = cv2.resize(tmp0, size)
    tmp1 = cv2.resize(tmp1, size)
    #tmp0 = cv2.resize(tmp0, (34, 60))
    #tmp1 = cv2.resize(tmp1, (34, 60))

    tmp0_gray = cv2.cvtColor(tmp0, cv2.COLOR_BGR2GRAY)
    tmp1_gray = cv2.cvtColor(tmp1, cv2.COLOR_BGR2GRAY)
    # print(np.shape(tmp0))

    # print(np.shape(tmp0))
    flow = dis.calc(tmp0_gray, tmp1_gray, None,)

    reliable = get_reliable(flow, tmp0, tmp1)
    #print("rr", np.shape(reliable))
    # print(np.mean(reliable))

    flow = torch.Tensor(flow).cuda()
    flow = flow.permute(2, 0, 1)

    reliable = torch.Tensor(reliable).cuda()

    # print(flow.size())
    return flow, reliable


def get_reliable(flow, img1, img2):
    blank_img = np.ones_like(flow[:, :, 0])
    # print(np.shape(blank_img))
    blank_warp = warp_flow(blank_img, flow)
    re_blank = (blank_warp >= 1)
    #re_flow = (flow_bd <= 0.01 * flow_sq + 0.005)
    #re = np.logical_and(re_img,re_blank).astype("float32")
    re = re_blank.astype("float32")
    return re


def warp_flow(img, flow):
    flow = flow.copy()
    h, w = flow.shape[:2]
    flow = - flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def flow_warp(x, flow, padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (n, c, h, w)
        flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
        padding_mode (str): 'zeros' or 'border'

    Returns:
        Tensor: warped image or feature map
    """
    flow = - flow.cuda()
    printonce(x.size(),flow.size())
    assert x.size()[-2:] == flow.size()[-2:]
    n, _, h, w = x.size()
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    grid += flow
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    #grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, padding_mode=padding_mode)