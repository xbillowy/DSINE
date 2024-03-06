import os
import sys
import glob
import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

import utils.utils as utils
from models.datasets import EasyVolCapDataset
from easyvolcap.utils.data_utils import save_image


if __name__ == '__main__':
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='dsine', help='model checkpoint')
    parser.add_argument('--data_root', type=str, default='data/iphone/room_412_whole', help='')
    parser.add_argument('--images_dir', type=str, default='images', help='')
    parser.add_argument('--normals_dir', type=str, default='normals', help='')
    parser.add_argument('--result_dir', type=str, default='data/result/iphone/room_412_whole', help='')
    parser.add_argument("--view_sample", type=int, default=[0, None, 1], nargs='+', help="sample from the rendering novel view list")
    parser.add_argument("--frame_sample", type=int, default=[0, 1, 1], nargs='+', help="sample from the frame list")
    parser.add_argument('--digit', type=int, default=6, help='')
    args = parser.parse_args()

    # Define the easyvolcap format datasets
    dataset = EasyVolCapDataset(data_root=args.data_root, images_dir=args.images_dir, view_sample=args.view_sample, frame_sample=args.frame_sample)
    # Define the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Define model
    device = torch.device('cuda')
    from models.dsine import DSINE
    model = DSINE(h=int(dataset.Hs[0, 0]), w=int(dataset.Ws[0, 0])).to(device)
    model.pixel_coords = model.pixel_coords.to(device)

    # Load the model checkpoint and set it to evaluation mode
    model = utils.load_checkpoint('./checkpoints/%s.pt' % args.ckpt, model)
    model.eval()

    # Define the normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Test on the dataloader
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            # Load the image
            img = sample['img'].to(device)
            # Pad the input image so that both the width and height are multiples of 32
            _, _, orig_H, orig_W = img.shape
            l, r, t, b = utils.pad_input(orig_H, orig_W)
            img = F.pad(img, (l, r, t, b), mode="constant", value=0.0)
            # Normalize the image
            img = normalize(img)

            # Load and modify the intrinsic
            ixt = sample['K'].to(device)
            ixt[:, 0, 2] += l
            ixt[:, 1, 2] += t

            # Predict the normals
            normal_map = model(img, intrins=ixt)[-1]
            normal_map = normal_map[:, :, t:t+orig_H, l:l+orig_W]

            # Post-process the predicted normals
            normal_map = -normal_map.cpu().detach().numpy()[0, :, :, :].transpose(1, 2, 0)  # (H, W, 3)
            normal_map = (normal_map + 1.0) / 2.0  # normalize the predicted normals to [0, 1] for saving

            # Save the predicted normals
            normal_pth = os.path.join(args.result_dir, args.normals_dir, f'{i:0{args.digit}}', f'{0:0{args.digit}}.jpg')
            os.makedirs(os.path.dirname(normal_pth), exist_ok=True)
            save_image(normal_pth, normal_map)
            print(f"Saved the predicted normals to {normal_pth}")
