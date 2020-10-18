import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import cv2
import torch.nn.functional as F
import time

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class SANet(nn.Module):
    
    def __init__(self, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))
        
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O

class Transform(nn.Module):
    def __init__(self, in_planes):
        super(Transform, self).__init__()
        self.sanet4_1 = SANet(in_planes = in_planes)
        self.sanet5_1 = SANet(in_planes = in_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, (3, 3))
    def forward(self, content4_1, style4_1):
        csc = self.merge_conv(self.merge_conv_pad(1.25 * self.sanet4_1(content4_1, style4_1) ))
        ccc = self.merge_conv(self.merge_conv_pad(1.25 * self.sanet4_1(content4_1, content4_1) ))
        print(ccc.size(), csc.size())
        out = ccc * 0.25 + csc * 0.75
        return out, csc

    def forward_with_flow(self, content4_1, style4_1, last_feat, reliable):
        csc = self.merge_conv(self.merge_conv_pad(1.25 * self.sanet4_1(content4_1, style4_1)))
        ccc = self.merge_conv(self.merge_conv_pad(1.25 * self.sanet4_1(content4_1, content4_1)))

        reliable = reliable * 0.75
        #print(reliable.size())
        csc = last_feat * reliable + csc * (1 - reliable) 

        out = ccc * 0.25 + csc * 0.75
        #out = csc
        return out, csc

def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


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
    print(x.size(),flow.size())
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


def get_motion_boundary(flow):
    # get x boundary
    h = np.shape(flow)[0]
    w = np.shape(flow)[1]

    flow_h = np.concatenate([np.zeros([1,w,2]),flow,np.zeros([1,w,2])])
    bd_h = []
    for idx in range(h):
        tmp_bd = -0.5*flow_h[idx]+0.5*flow_h[idx+2]
        bd_h.append(tmp_bd)
    bd_h = np.stack(bd_h,axis=0)
    
    flow_w = np.concatenate([np.zeros([h,1,2]),flow,np.zeros([h,1,2])],axis=1)
    bd_w = []
    for idx in range(w):
        tmp_bd = -0.5*flow_w[:,idx,:]+0.5*flow_w[:,idx+2,:]
        bd_w.append(tmp_bd)
    bd_w = np.stack(bd_w,axis=1)
    bd = np.sum(bd_h*bd_h + bd_w*bd_w, axis=2)
    #print(np.shape(bd))
    return bd

def warp_flow(img, flow):
    flow = flow.copy()
    h, w = flow.shape[:2]
    flow = - flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def get_reliable_old(flow, img1, img2):
    blank_img = np.ones_like(flow[:,:,0])
    print(np.shape(blank_img))
    blank_warp = warp_flow(blank_img, flow)
    img_warp = warp_flow(img1, flow)
    flow_warp = warp_flow(flow, flow) 

    img2 = img2.astype("float32")
    img_warp = img_warp.astype("float32")
    diff_img = np.mean(np.abs(img2-img_warp),axis=2)
    img2 = np.mean(img2,axis=2)

    re_img = (diff_img<=0.1*img2+10)
    re_blank = (blank_warp>=1)
    #re_flow = (flow_bd <= 0.01 * flow_sq + 0.005)
    #re = np.logical_and(re_img,re_blank).astype("float32")
    re = re_blank.astype("float32")
    return re

def get_reliable(flow, img1, img2):
    blank_img = np.ones_like(flow[:,:,0])
    #print(np.shape(blank_img))
    blank_warp = warp_flow(blank_img, flow)
    re_blank = (blank_warp>=1)
    #re_flow = (flow_bd <= 0.01 * flow_sq + 0.005)
    #re = np.logical_and(re_img,re_blank).astype("float32")
    re = re_blank.astype("float32")
    return re

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument('--content', type=str, default = 'input/chicago.jpg',
                    help='File path to the content image')
parser.add_argument('--style', type=str, default = 'style/style11.jpg',
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument('--steps', type=str, default = 1)
parser.add_argument('--vgg', type=str, default = 'models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default = 'models/decoder_iter_500000.pth')
parser.add_argument('--transform', type=str, default = 'models/transformer_iter_500000.pth')

# Additional options
parser.add_argument('--save_ext', default = '.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default = 'output',
                    help='Directory to save the output image(s)')

# Advanced options

args = parser.parse_args('')

# 污染了env
# args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(args.output):
    os.mkdir(args.output)

decoder = decoder
transform = Transform(in_planes = 512)
vgg = vgg

decoder.eval()
transform.eval()
vgg.eval()

# decoder.load_state_dict(torch.load(args.decoder))
# transform.load_state_dict(torch.load(args.transform))
# vgg.load_state_dict(torch.load(args.vgg))

norm = nn.Sequential(*list(vgg.children())[:1])
enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

norm.to(device)
enc_1.to(device)
enc_2.to(device)
enc_3.to(device)
enc_4.to(device)
enc_5.to(device)
transform.to(device)
decoder.to(device)

content_tf = test_transform()
style_tf = test_transform()

style_name = "wave"
#style_name = "candy_extend"
style_img = "style/" + style_name + ".jpg"
style = style_tf(Image.open(style_img))
style = style.to(device).unsqueeze(0)
style = style[:,:3,:,:]
Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
#Style5_1 = enc_5(Style4_1)

frame_root = "./data/frames/"
output_root = "./data/frames_out/"
video_name = 'mda-igggd9vkgz3j7n4t'#"mda-ig5h66qkz5gz889u"
#video_name = '1573639769'

out_path = output_root + video_name +"_" + style_name
try:
    os.mkdir(out_path)
except:
    print("dir is already created")

frame_list = os.listdir(frame_root + video_name)
frame_list.sort()

dis = cv2.DISOpticalFlow_create()

def get_flow(img0, img1, dis):
    tmp0 = np.array(img0)
    tmp1 = np.array(img1)
    tmp0 = cv2.resize(tmp0, (68, 120))
    tmp1 = cv2.resize(tmp1, (68, 120))
    #tmp0 = cv2.resize(tmp0, (34, 60))
    #tmp1 = cv2.resize(tmp1, (34, 60))

    tmp0_gray = cv2.cvtColor(tmp0,cv2.COLOR_BGR2GRAY)
    tmp1_gray = cv2.cvtColor(tmp1,cv2.COLOR_BGR2GRAY)
    #print(np.shape(tmp0))

    #print(np.shape(tmp0))
    flow = dis.calc(tmp0_gray, tmp1_gray, None,)

    reliable = get_reliable(flow, tmp0, tmp1)
    #print("rr", np.shape(reliable))
    #print(np.mean(reliable))

    flow = torch.Tensor(flow).cuda()
    flow = flow.permute(2,0,1)

    reliable = torch.Tensor(reliable).cuda()

    #print(flow.size())
    return flow, reliable





for idx, img_name in enumerate(frame_list):
    if idx == 0:
        img0 = Image.open(os.path.join(frame_root,video_name,img_name))
        img0 = img0.resize([540,960])
        #img0 = img0.resize([270,480])
        content = content_tf(img0)
        content = content.to(device).unsqueeze(0)
        #print(content.size())
        #break
        with torch.no_grad():
            Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
            #Content5_1 = enc_5(Content4_1)
            content, last_feat = transform(Content4_1, Style4_1)
            content = decoder(content)
            content.clamp(0, 255)
            content = content.cpu()
            output_name = "%s/%s" %(out_path, img_name)
            save_image(content, output_name)
            last_feat = last_feat.cpu().data.numpy()
        continue
    else:
        #print("hh")
        t0 = time.time()
        img1 = Image.open(os.path.join(frame_root,video_name,img_name))
        img1 = img1.resize([540,960])
        #img1 = img1.resize([270,480])
        flow, reliable = get_flow(img0, img1, dis)
        print("flow/re",time.time() - t0)
        t0 = time.time()

        last_feat = torch.Tensor(last_feat).cuda()
        warp_feat = flow_warp(last_feat,flow)

        print("warp", time.time() - t0)
        t0 = time.time()
        content = content_tf(img1)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
            #Content5_1 = enc_5(Content4_1)
            print("enc",time.time() - t0)
            t0 = time.time()
            content, last_feat = transform.forward_with_flow(Content4_1, Style4_1, warp_feat, reliable)
            print("trans",time.time() - t0)
            t0 = time.time()
            content = decoder(content)
            print("dec",time.time() - t0)
            t0 = time.time()
            #content = decoder(transform(Content4_1, Style4_1, Content5_1, Style5_1))
            content.clamp(0, 255)
            content = content.cpu()
            output_name = "%s/%s" %(out_path, img_name)
            save_image(content, output_name)
            last_feat = last_feat.cpu().data.numpy()

        img0 = img1
        print(time.time() - t0)

        reliable = reliable.cpu().data.numpy()
        cv2.imwrite("tmp/" + img_name, reliable * 255.)
    #break



