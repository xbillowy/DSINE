import os
import cv2
import torch
import numpy as np
from PIL import Image
from glob import glob
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.easy_utils import read_camera
from easyvolcap.utils.math_utils import affine_inverse, affine_padding, torch_inverse_3x3, point_padding
from easyvolcap.utils.bound_utils import get_bound_2d_bound, get_bounds, monotonic_near_far, get_bound_3d_near_far
from easyvolcap.utils.data_utils import DataSplit, UnstructuredTensors, load_resize_undist_ims_bytes, load_image_from_bytes, as_torch_func, to_cuda, to_cpu, to_tensor, export_pts, load_pts, decode_crop_fill_ims_bytes, decode_fill_ims_bytes


class EasyVolCapDataset(Dataset):
    def __init__(self,
                 split: str = 'train',
                 data_root: str = 'data/iphone/room_412_whole',
                 intri_file: str = 'intri.yml',
                 extri_file: str = 'extri.yml',
                 images_dir: str = 'images',
                 cameras_dir: str = 'cameras',

                 ratio: float = 1.0,  # use original image size
                 center_crop_size: List[int] = [-1, -1],  # center crop image to this size, after resize
                 view_sample: list = [0, None, 1],
                 frame_sample: list = [0, None, 1],
                 ims_pattern: str = '{frame:06d}.jpg',
                 imsize_overwrite: List[int] = [-1, -1],  # overwrite the image size

                 # Image preprocessing & formatting
                 dist_opt_K: bool = True,  # use optimized K for undistortion (will crop out black edges), mostly useful for large number of images
                 encode_ext: str = '.jpg',
                 cache_raw: bool = True,
                 dist_mask: List[bool] = [1] * 5,
                 ):
        # Get the paths to the data
        self.data_root = data_root
        self.intri_file = intri_file
        self.extri_file = extri_file
        self.images_dir = images_dir
        self.cameras_dir = cameras_dir

        # The frame number & image size should be inferred from the dataset
        self.view_sample = view_sample
        self.frame_sample = frame_sample
        if self.view_sample[1] is not None: self.n_view_total = self.view_sample[1]
        else: self.n_view_total = len(os.listdir(join(self.data_root, self.images_dir)))  # total number of cameras before filtering
        if self.frame_sample[1] is not None: self.n_frames_total = self.frame_sample[1]
        else: self.n_frames_total = min([len(glob(join(self.data_root, self.images_dir, cam, '*'))) for cam in os.listdir(join(self.data_root, self.images_dir))])  # total number of images before filtering

        # Compute needed visual hulls & align all cameras loaded
        self.load_cameras()  # load and normalize all cameras (center lookat, align y axis)
        self.select_cameras()  # select repective cameras to use

        self.ims_pattern = ims_pattern
        self.dist_mask = dist_mask
        self.ratio = ratio
        self.imsize_overwrite = imsize_overwrite
        self.center_crop_size = center_crop_size
        self.split = split
        self.dist_opt_K = dist_opt_K
        self.encode_ext = encode_ext
        self.cache_raw = cache_raw

        # Load the image paths and the corresponding depth paths
        self.load_paths()  # load image files into self.ims
        # Load the images and the corresponding depths
        self.load_bytes()

        # Define the normalization transform
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_cameras(self):
        # Load camera related stuff like image list and intri, extri.
        # Determine whether it is a monocular dataset or multiview dataset based on the existence of root `extri.yml` or `intri.yml`
        # Multiview dataset loading, need to expand, will have redundant information
        if exists(join(self.data_root, self.intri_file)) and exists(join(self.data_root, self.extri_file)):
            self.cameras = read_camera(join(self.data_root, self.intri_file), join(self.data_root, self.extri_file))
            self.camera_names = np.asarray(sorted(list(self.cameras.keys())))  # NOTE: sorting camera names
            self.cameras = dotdict({k: [self.cameras[k] for i in range(self.n_frames_total)] for k in self.camera_names})
            # TODO: Handle avg processing

        # Monocular dataset loading, each camera has a separate folder
        elif exists(join(self.data_root, self.cameras_dir)):
            self.camera_names = np.asarray(sorted(os.listdir(join(self.data_root, self.cameras_dir))))  # NOTE: sorting here is very important!
            self.cameras = dotdict({
                k: [v[1] for v in sorted(
                    read_camera(join(self.data_root, self.cameras_dir, k, self.intri_file),
                                join(self.data_root, self.cameras_dir, k, self.extri_file)).items()
                )] for k in self.camera_names
            })
            # TODO: Handle avg export and loading for such monocular dataset
        else:
            raise NotImplementedError(f'Could not find {{{self.intri_file},{self.extri_file}}} or {self.cameras_dir} directory in {self.data_root}, check your dataset configuration')

        # Expectation:
        # self.camera_names: a list containing all camera names
        # self.cameras: a mapping from camera names to a list of camera objects
        # (every element in list is an actual camera for that particular view and frame)
        # NOTE: ALWAYS, ALWAYS, SORT CAMERA NAMES.
        self.Hs = torch.as_tensor([[cam.H for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ws = torch.as_tensor([[cam.W for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Ks = torch.as_tensor([[cam.K for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Rs = torch.as_tensor([[cam.R for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 3
        self.Ts = torch.as_tensor([[cam.T for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 3, 1
        self.Ds = torch.as_tensor([[cam.D for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F, 1, 5
        self.ts = torch.as_tensor([[cam.t for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.ns = torch.as_tensor([[cam.n for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.fs = torch.as_tensor([[cam.f for cam in self.cameras[k]] for k in self.camera_names], dtype=torch.float)  # V, F
        self.Cs = -self.Rs.mT @ self.Ts  # V, F, 3, 1
        self.w2cs = torch.cat([self.Rs, self.Ts], dim=-1)  # V, F, 3, 4
        self.c2ws = affine_inverse(self.w2cs)  # V, F, 3, 4

    def select_cameras(self):
        # Only retrain needed
        # Perform view selection first
        view_inds = torch.arange(self.Ks.shape[0])
        if len(self.view_sample) != 3: view_inds = view_inds[self.view_sample]  # this is a list of indices
        else: view_inds = view_inds[self.view_sample[0]:self.view_sample[1]:self.view_sample[2]]  # begin, start, end
        self.view_inds = view_inds
        if len(view_inds) == 1: view_inds = [view_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # Perform frame selection next
        frame_inds = torch.arange(self.Ks.shape[1])
        if len(self.frame_sample) != 3: frame_inds = frame_inds[self.frame_sample]
        else: frame_inds = frame_inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.frame_inds = frame_inds  # used by `load_smpls()`
        if len(frame_inds) == 1: frame_inds = [frame_inds]  # MARK: pytorch indexing bug, when length is 1, will reduce a dim

        # NOTE: if view_inds == [0,] in monocular dataset or whatever case, type(`self.camera_names[view_inds]`) == str, not a list of str
        self.camera_names = np.asarray([self.camera_names[view] for view in view_inds])  # this is what the b, e, s means
        self.cameras = dotdict({k: [self.cameras[k][int(i)] for i in frame_inds] for k in self.camera_names})  # reloading
        self.Hs = self.Hs[view_inds][:, frame_inds]
        self.Ws = self.Ws[view_inds][:, frame_inds]
        self.Ks = self.Ks[view_inds][:, frame_inds]
        self.Rs = self.Rs[view_inds][:, frame_inds]
        self.Ts = self.Ts[view_inds][:, frame_inds]
        self.Ds = self.Ds[view_inds][:, frame_inds]
        self.ts = self.ts[view_inds][:, frame_inds]
        self.Cs = self.Cs[view_inds][:, frame_inds]
        self.w2cs = self.w2cs[view_inds][:, frame_inds]
        self.c2ws = self.c2ws[view_inds][:, frame_inds]

    def load_paths(self):
        # Load image related stuff for reading from disk later
        # If number of images in folder does not match, here we'll get an error
        ims = [[join(self.data_root, self.images_dir, cam, self.ims_pattern.format(frame=i)) for i in range(self.n_frames_total)] for cam in self.camera_names]
        if not exists(ims[0][0]):
            ims = [[i.replace('.' + self.ims_pattern.split('.')[-1], '.JPG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.JPG', '.png') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [[i.replace('.png', '.PNG') for i in im] for im in ims]
        if not exists(ims[0][0]):
            ims = [sorted(glob(join(self.data_root, self.images_dir, cam, '*')))[:self.n_frames_total] for cam in self.camera_names]
        ims = [np.asarray(ims[i])[:min([len(i) for i in ims])] for i in range(len(ims))]  # deal with the fact that some weird dataset has different number of images
        self.ims = np.asarray(ims)  # V, N
        self.ims_dir = join(*split(dirname(self.ims[0, 0]))[:-1])  # logging only

        # TypeError: can't convert np.ndarray of type numpy.str_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint8, and bool.
        # MARK: Names stored as np.ndarray
        inds = np.arange(self.ims.shape[-1])
        if len(self.frame_sample) != 3: inds = inds[self.frame_sample]
        else: inds = inds[self.frame_sample[0]:self.frame_sample[1]:self.frame_sample[2]]
        self.ims = self.ims[..., inds]  # these paths are later used for reading images from disk

    def load_bytes(self):
        # Camera distortions are only applied on the ground truth image, the rendering model does not include these
        # And unlike intrinsic parameters, it has no direct dependency on the size of the loaded image, thus we directly process them here
        dist_mask = torch.as_tensor(self.dist_mask)
        self.Ds = self.Ds.view(*self.Ds.shape[:2], 5) * dist_mask  # some of the distortion parameters might need some manual massaging

        # Need to convert to a tight data structure for access
        ori_Ks = self.Ks
        ori_Ds = self.Ds
        ratio = self.imsize_overwrite if self.imsize_overwrite[0] > 0 else self.ratio  # maybe force size, or maybe use ratio to resize

        # Image pre cacheing (from disk to memory)
        self.ims_bytes, self.Ks, self.Hs, self.Ws = \
            load_resize_undist_ims_bytes(self.ims, ori_Ks.numpy(), ori_Ds.numpy(), ratio, self.center_crop_size,
                                         f'Loading imgs bytes for {blue(self.ims_dir)} {magenta(self.split)}',
                                         dist_opt_K=self.dist_opt_K, encode_ext=self.encode_ext)

        self.Ks = torch.as_tensor(self.Ks)
        self.Hs = torch.as_tensor(self.Hs)
        self.Ws = torch.as_tensor(self.Ws)

        if self.cache_raw:
            # To make memory access faster, store raw floats in memory
            self.ims_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.ims_bytes, desc=f'Caching imgs for {blue(self.data_root)} {magenta(self.split)}')])  # High mem usage
            if hasattr(self, 'mks_bytes'): self.mks_bytes = to_tensor([load_image_from_bytes(x, normalize=True) for x in tqdm(self.mks_bytes, desc=f'Caching mks for {blue(self.data_root)} {magenta(self.split)}')])
            if hasattr(self, 'dps_bytes'): self.dps_bytes = to_tensor([load_image_from_bytes(x, normalize=False) for x in tqdm(self.dps_bytes, desc=f'Caching dps for {blue(self.data_root)} {magenta(self.split)}')])

    @property
    def n_views(self): return len(self.cameras)

    @property
    def n_latents(self): return len(next(iter(self.cameras.values())))  # short for timestamp

    # NOTE: everything beginning with get are utilities for __getitem__
    # NOTE: coding convension are preceded with "NOTE"
    def get_indices(self, index):
        # These indices are relative to the processed dataset
        view_index, latent_index = index // self.n_latents, index % self.n_latents

        if len(self.view_sample) != 3: camera_index = self.view_sample[view_index]
        else: camera_index = view_index * self.view_sample[2] + self.view_sample[0]

        if len(self.frame_sample) != 3: frame_index = self.frame_sample[latent_index]
        else: frame_index = latent_index * self.frame_sample[2] + self.frame_sample[0]

        return view_index, latent_index, camera_index, frame_index

    def get_image_bytes(self, view_index: int, latent_index: int):
        im_bytes = self.ims_bytes[view_index * self.n_latents + latent_index]  # MARK: no fancy indexing

        return im_bytes

    def get_image(self, view_index: int, latent_index: int):
        # Load bytes (rgb, msk, wet, dpt)
        im_bytes = self.get_image_bytes(view_index, latent_index)
        rgb = None

        # Load image from bytes
        if self.cache_raw:
            rgb = torch.as_tensor(im_bytes)
        else:
            rgb = torch.as_tensor(load_image_from_bytes(im_bytes, normalize=True))  # 4-5ms for 400 * 592 jpeg, sooo slow

        return rgb

    def __getitem__(self, idx):
        # Prepare the output data
        data = dict()

        # Load the indices
        view_index, latent_index, camera_index, frame_index = self.get_indices(idx)
        data['view_index'], data['latent_index'], data['camera_index'], data['frame_index'] = view_index, latent_index, camera_index, frame_index

        # Load the camera parameters
        data['H'] = self.Ws[view_index, latent_index]
        data['W'] = self.Ws[view_index, latent_index]
        data['K'] = self.Ks[view_index, latent_index]  # 3, 3

        # Load the rgb image, depth and mask
        rgb = self.get_image(view_index, latent_index)
        # Deal with the order of the channels
        data['img'] = rgb.permute(2, 0, 1)  # 3, H, W

        return data

    def __len__(self):
        return self.n_views * self.n_latents  # there's no notion of epoch here
