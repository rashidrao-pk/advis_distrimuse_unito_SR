
# Standard Library
import os
import re
import json
import platform
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

# Scientific Computing
import numpy as np
from scipy.interpolate import griddata

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# Image Processing
import cv2
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import dilation, square
from skimage.segmentation import find_boundaries

# Deep Learning
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# Machine Learning
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.manifold import TSNE

# Local
import utils_model_CAD as utmC

# Globals
global device


##################################################################################
def get_params_paths():
    

    @dataclass
    class Parameters:
        dummy           : bool  = False
        model_depth     : int   = 3
        latent_dims     : int   = 64
        learning_rate_enc_dec   : float = None
        learning_rate_dis   : float = None
        target_size     : int   = 128
        batch_size      : int   = 64
        input_shape     : tuple = (None,None,None)
        model_name      : str = ''
        train_mode      : str = ''
        num_workers     : int   = 4
        pin_memory      : bool  = False
        persistent_workers : bool = False
        aug_type        : str     = 'full'
        
        
    @dataclass
    class Paths:
        dummy           : bool  = False
        path_datasets   : str  =   ''
        dataset_type    : str = ''
        path_dataset_selected: str =''
        train_dir : str = ''
        test_dir   : str = ''
        train_classes : str =''
        test_classes : str =''
        class_names_train : str = ''
        path_codes : str = ''
        path_models     : str  =   ''
        path_results    : str  =   ''
        history_fname   : str = ''

    params = Parameters()
    paths = Paths()

    params.input_shape= (3, params.target_size, params.target_size)
    params.target_size = (params.target_size, params.target_size)
    params.model_name    =  "VAE-GAN"
    paths.path_codes = os.getcwd()
    return params,paths
##################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def natural_sort_key(filename):
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]
##################################################################################
def get_map_coord():
    return {'PLeft': (238,58,360, 360) ,
        'RoboArm': (432, 0, 416,403),
        'ConvBelt': (495, 240,300,480),
        'PRight': (680, 58, 360,360),
        }
##################################################################################

def create_video_custom(paths,video_for='train', video_ext='mp4'
                             ,folder_path=None,save_path=None,
                              filter_frame_name='train', verbose=False, interval=10,repeat=10):
    """
    Create a video from a folder of images
    :param folder_path: Path to the folder containing the images
    :param video_ext: Extension of the video file
    :return: None
    """
    
    ls = folder_path.split('\\')[-1]
    if save_path is None:
        save_path = os.path.join(paths.path_codes, paths.dataset_type)
    
    if not os.path.exists(save_path):
        assert 'Path Not Found'
    last_id = 1 if video_for=='anomalous' else -1
    print(f'save_path\t{save_path}')
    print(f'folder_path\t{folder_path}')

    # Get the list of image filenames sorted
    images = sorted([img for img in os.listdir(folder_path) if img.endswith(".png") or img.endswith(".jpg")],  key=natural_sort_key)
    # images = sorted([x for x in images if filter_frame_name in x], key=natural_sort_key)
    print('images',images)
    allowed_images = images
    # allowed_images = sorted([x for x in images if filter_frame_name in x and int(x.split('.')[0].split('_')[last_id]) % interval == 0] , key=natural_sort_key)
    # print('allowed_images',allowed_images)
    
    # return
    # Get the first image to determine the frame size
    if len(allowed_images) < 1:
        print(allowed_images)
        assert 'Not enough images to create a video'
        return
    
    first_image = cv2.imread(os.path.join(folder_path, allowed_images[1]))
    height, width, _ = first_image.shape
    if verbose:
        print(f"First image: {allowed_images[0]}")
        print(f"Number of images: {len(allowed_images)}")
        print(f"Image size: {width}x{height}")
        print(f"Output video: video_{ls}.{video_ext}")
        print(f"folder_path {folder_path}")
        # print(f"Creating video for {ls}")
    # Define the codec and create a VideoWriter object
    if video_ext=='mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change this depending on the codec

    video_path = f'{save_path}/video_{interval}_{ls}.{video_ext}'
    output_video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))  # 30 is the frame rate

    # Loop through the images and add them to the video
    if verbose:
        allowed_images = tqdm(allowed_images)
    if interval==1:
        for image in allowed_images:

            img = cv2.imread(os.path.join(folder_path, image))
            for _ in range(repeat):
                output_video.write(img)

    else:
        for image in allowed_images:
            img = cv2.imread(os.path.join(folder_path, image))
            output_video.write(img)

    # Release the VideoWriter object
    output_video.release()
    if verbose:
        print(f'video_path ', video_path)
    print("Video creation completed!")
    
    del folder_path,output_video,img,fourcc,first_image,height, width,save_path,allowed_images,image

###########################################################################################
def create_video_from_frames(paths,suffix=None,data_type = 'full',video_for='train', video_ext='mp4',
                             predefined_folder=False,folder_path=None,fps =5,
                              filter_frame_name=None, verbose=False, interval=10,repeat=10,save_path_type='cloud'):
    """
    Create a video from a folder of images
    :param folder_path: Path to the folder containing the images
    :param video_ext: Extension of the video file
    :return: None
    """

    # Folder path where images are stored
    # folder_path = 'E:/Cloud/RashidPHD/Codes/DistriMuSe/AD_CAD/camera_main_images/test_64_0.0001_0.01'
    # folder_path = paths.path_results
    # folder_path = paths.path_results_fix
    if folder_path  is not None:
        print('Using predefined folder path')
        paths.path_results = folder_path
    #------------------------------------------------------
    if data_type == 'full' and not predefined_folder:
        folder_path = paths.path_results
    
    elif data_type == 'fixed' and not predefined_folder:
        folder_path = paths.path_results_fix
    elif data_type == 'fixed' and not predefined_folder:
        folder_path = paths.path_results_fix
    elif data_type == 'custom':
        folder_path=folder_path
    else:
        print('Using default folder path')
        folder_path = paths.video_savepath
    #------------------------------------------------------
    ls = folder_path.split('\\')[-1]

    #------------------------------------------------------
    if save_path_type=='local':
        save_path = paths.path_results_local
    elif save_path_type=='cloud':
        save_path = os.path.join(paths.path_codes, paths.dataset_type)
    
    if video_for=='test':
        folder_path = os.path.join(paths.path_codes,f"test_{suffix}" )
        save_path = os.path.join(paths.path_codes, paths.dataset_type)
    
    if video_for=='segmentation':
        folder_path = paths.path_results
        # folder_path = os.path.join(paths.path_codes,f"test_{suffix}" )
        save_path = os.path.join(paths.path_codes, paths.dataset_type)

    

    if not os.path.exists(save_path):
        assert 'Path Not Found'
    last_id = 1 if video_for=='anomalous' else -1

    print(f'save_path\t{save_path}')
    print(f'folder_path\t{folder_path}')
    # return
    # Get the list of image filenames sorted
    images = sorted([img for img in os.listdir(folder_path) if img.endswith(".png") or img.endswith(".jpg")],  key=natural_sort_key)
    if filter_frame_name is not None:
        images = sorted([x for x in images if filter_frame_name in x], key=natural_sort_key)
    
    print('images',images[0:5])
    if video_for=='test':
        allowed_images = images

    elif video_for=='video':
        allowed_images = sorted([x for x in images if filter_frame_name in x and int(x.split('.')[0].split('_')[last_id]) % interval == 0] , key=natural_sort_key)
    elif video_for=='segmentation':
        allowed_images = images
        # allowed_images = sorted([x for x in images if filter_frame_name in x and int(x.split('.')[0].split('_')[last_id]) % interval == 0] , key=natural_sort_key)
        pass
    else:
        allowed_images = sorted([x for x in images if filter_frame_name in x and int(x.split('.')[0].split('_')[last_id]) % interval == 0] , key=natural_sort_key)
        print('allowed_images',allowed_images)
    
    # return
    # Get the first image to determine the frame size
    if len(allowed_images) < 1:
        print(allowed_images)
        assert 'Not enough images to create a video'
        return
    # return
    first_image = cv2.imread(os.path.join(folder_path, allowed_images[1]))
    height, width, _ = first_image.shape
    if verbose:
        print(f"First image: {allowed_images[0]}")
        print(f"Number of images: {len(allowed_images)}")
        print(f"Image size: {width}x{height}")
        print(f"Output video: video_{ls}.{video_ext}")
        print(f"folder_path {folder_path}")
        # print(f"Creating video for {ls}")
    # Define the codec and create a VideoWriter object
    # return
    if video_ext=='mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change this depending on the codec

    video_path = f'{save_path}/video_{interval}_{ls}.{video_ext}'
    os.makedirs(save_path, exist_ok=True)
    output_video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))  # 30 is the frame rate

    # Loop through the images and add them to the video
    if verbose:
        allowed_images = tqdm(allowed_images)
    if interval==1:
        for image in allowed_images:

            img = cv2.imread(os.path.join(folder_path, image))
            for _ in range(repeat):
                output_video.write(img)

    else:
        for image in allowed_images:
            img = cv2.imread(os.path.join(folder_path, image))
            output_video.write(img)

    # Release the VideoWriter object
    output_video.release()
    if verbose:
        print(f'video_path ', video_path)
    print("Video creation completed!")
    
    del folder_path,output_video,img,fourcc,first_image,height, width,save_path,allowed_images,image
##################################################################################
##################################################################################
def get_paths(paths, verbose=False):
    import platform
    platform_node = platform.node()
    print(f'system -- OS({os.name}) - user({platform_node})', )
    if os.name=='nt':
        # from matplotlib import rc
        # rc('text',usetex=True)
        # rc('text.latex', preamble='\\usepackage{color}')
        platform_node = platform.node()
        if platform_node=='Rashid-Unito':
            paths.path_results = 'E:/Cloud/RashidPHD/Codes/DistriMuSe/AD_CAD_v3'
            paths.path_datasets_main        = r'D:/DS/VeleriaLab/'
            paths.path_results_local        = r'E:/PHD/datacloud_data/repos/AD_CAD_v3'
        elif platform_node=='DESKTOP-Q14PULG':
            paths.path_results = r'C:/rashid/RashidPHD/Codes/DistriMuSe/AD_CAD_v3'
            paths.path_datasets_main         = r'C:/DS/ValeriaLab'
            paths.path_results_local        = r'C:/rashid_data/codes/DistriMuSe'
    elif os.name=='posix':
        if 'epito' in platform_node:
            paths.path_results        = '/beegfs/home/mrashid/repos/AD_CAD_v3'
            paths.path_datasets_main  = '/beegfs/home/mrashid/datasets/ValeriaLab'
            paths.path_results_local  = '/beegfs/home/mrashid/repos/AD_CAD_v3'
        if 'distrimuse' in platform_node:
            paths.path_results        = os.getcwd()
            paths.path_datasets_main  = '/home/unito/data/DS/ValeriaLab'
            paths.path_results_local  = '/home/unito/data/'

    if verbose:
        print('OS type:', os.name)
        print(f'Code Running === {platform.node()} :: Windows')
        print(f'path_results :  {paths.path_results}')
        print(f'path_datasets_main : {paths.path_datasets_main}')
    return paths

#################################################################################################
#################################################################################################



class CustomCrop:
    """Custom masking based on subgroup"""
    def __init__(self, subgroup=None,map_coor=None, fill_color=(0, 0, 0)):
        self.subgroup = subgroup  # Store subgroup for cropping
        self.fill_color = fill_color  # Color to fill the masked region
        self.map_coor = map_coor

    def __call__(self, image):

        self.map_coor   
        # print(f'self.map_coor[self.subgroup :::: {self.map_coor[self.subgroup]}')
        # return
        if self.subgroup is None:
            return image
        else:
            x, y, w, h = self.map_coor[self.subgroup]
            image = image.crop((x, y, x+w, y+h))
        return image
        # assert False
        # image = image.convert("RGBA")
        

        # mask = Image.new("RGBA", image.size, self.fill_color + (255,))  # Create a filled mask
        draw = ImageDraw.Draw(mask)
        draw.rectangle([x, y, x + w, y + h], fill=(0, 0, 0, 0))  # Make ROI transparent
        
        # Apply mask
        image = Image.alpha_composite(image, mask).convert("RGB")
        return image
################################################################################################################################
################################################################################################################################


class LoadMaskedImage:
    """Apply a mask or crop an image based on the selected subgroup and mask type."""
    def __init__(self,paths=None,target_size=None, subgroup=None, subgroup_mask='box', map_coor=None, fill_color=(0, 0, 0),mask_image_name=6232):
        self.paths = paths
        self.subgroup = subgroup  # Subgroup to process
        self.subgroup_mask = subgroup_mask  # 'box' or 'mask'
        self.map_coor = map_coor  # Dictionary with bounding boxes
        self.fill_color = fill_color  # Color for the mask overlay
        self.mask_image_name = mask_image_name

        ###################################
        mask_path = os.path.join(paths.mask_dir, f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{subgroup_mask}_ext.png")
        print(mask_path)
        assert os.path.exists(mask_path), "Mask image does not exist!"
        self.mask = Image.open(mask_path).convert("L").resize(target_size, Image.LANCZOS)

        # mask_name = f'{self.paths.dataset_type}_{self.mask_image_name}_{self.subgroup}_{self.subgroup_mask}_ext.png'
        # mask_path = os.path.join(self.paths.mask_dir, mask_name)
        # # print('mask_path:::', mask_path, os.path.exists(mask_path))
        # if not os.path.exists(mask_path):
        #     print('IMAGE NOT EXIST')
        #     assert 'Error'
        # # print('mask_path ---',os.path.join(self.paths.mask_dir),self.subgroup)
        
        # # mask_path = os.path.join(self.map_coor[self.subgroup])  # Assuming path to mask is stored
        # self.mask = Image.open(mask_path).convert("L")  # Load mask as grayscale

        # # Ensure mask and image sizes match
        # self.mask = self.mask.resize(target_size, Image.LANCZOS)

        # self.mask = self.mask.point(lambda p: 255 if p > 128 else 0)  # Binarize

        # # Apply binary mask to image using multiplication
        # self.mask = self.mask.convert("L")
        ###################################
        
    def __call__(self, image):
        return Image.composite(image.convert("RGB"), Image.new("RGB", image.size, self.fill_color), self.mask)
    # def __call__(self, image):
    #     """Apply a mask or crop to the image."""
    #     if self.subgroup is None or self.map_coor is None:
    #         return image  # No masking, return original image

    #     # Read image as RGBA (to handle transparency)
    #     image = image.convert("RGBA")

    #     if self.subgroup in self.map_coor:
    #         x, y, w, h = self.map_coor[self.subgroup]
    #         if self.subgroup_mask == 'box':
    #             # Crop using bounding box coordinates
    #             image = image.crop((x, y, x + w, y + h))

    #         elif self.subgroup_mask == 'mask':

    #             # Load the binary mask for this subgroup
    #             # mask_name = f'{self.paths.dataset_type}_{self.mask_image_name}_{self.subgroup}_{self.subgroup_mask}_ext.png'
    #             # mask_path = os.path.join(self.paths.mask_dir, mask_name)
    #             # # print('mask_path:::', mask_path, os.path.exists(mask_path))
    #             # if not os.path.exists(mask_path):
    #             #     print('IMAGE NOT EXIST')
    #             #     assert 'Error'
    #             # # print('mask_path ---',os.path.join(self.paths.mask_dir),self.subgroup)

    #             # # mask_path = os.path.join(self.map_coor[self.subgroup])  # Assuming path to mask is stored
    #             # mask = Image.open(mask_path).convert("L")  # Load mask as grayscale

    #             # # Ensure mask and image sizes match
    #             # mask = mask.resize(image.size, Image.LANCZOS)
    #             image = image.convert("RGB")  # Ensure image is in RGB format

    #             # Convert mask to binary (black & white)
    #             # mask = mask.point(lambda p: 255 if p > 128 else 0)  # Binarize

    #             # # Apply binary mask to image using multiplication
    #             # mask = mask.convert("L")
    #             # print(image.size, self.mask.size)
    #             image = Image.composite(image, Image.new("RGB", image.size, self.fill_color), self.mask)

    #     return image
################################################################################################################################
### MaskedCrop
################################################################################################################################


class MaskedCrop:
    """Custom masking based on subgroup"""
    def __init__(self, subgroup=None,mask=None, fill_color=(0, 0, 0), verbose=False):
        self.subgroup = subgroup  # Store subgroup for cropping
        self.fill_color = fill_color  # Color to fill the masked region
        self.mask = mask
        self.verbose = verbose
        # print('mask',mask.dtype, mask.shape)
        assert len(mask.shape)==2 and mask.dtype==bool
        # compute the bounding box of the mask
        rows, cols = np.where(mask)
        self.x1, self.x2 = np.min(cols), np.max(cols)
        self.y1, self.y2 = np.min(rows), np.max(rows)
        # create the 3-channel mask for boolean operations
        cropped_mask_1ch = mask[self.y1:self.y2+1, self.x1:self.x2+1]  # Shape: (h, w)
        self.cropped_mask = np.stack([cropped_mask_1ch]*3, axis=-1)  # Shape: (h, w, 3)

    def __call__(self, image):

        # self.map_coor   
        # print(f'self.map_coor[self.subgroup :::: {self.map_coor[self.subgroup]}')
        # return
        if self.subgroup is None:
            return image
        else:
            if self.verbose:
                print('computed ',self.subgroup, self.x1,self.x2,self.y1,self.y2)
            image_np = np.array(image)  # Convert to NumPy array for masking
            image_np = image_np[self.y1:self.y2+1, self.x1:self.x2+1, :]  # Crop the image
            image_np *= self.cropped_mask  # Element-wise multiplication with the mask
            image = Image.fromarray(image_np.astype('uint8'))
            
        return image
    
    def uncrop(self, image_np):
        assert image_np.shape[0] == self.cropped_mask.shape[0] and image_np.shape[1] == self.cropped_mask.shape[1], "Image shape does not match mask shape"
        # reapply the boolean mask to image_np
        image_np = image_np * self.cropped_mask
        # rescale back to the original image size
        no_channels = image_np.shape[2]
        orig_image_np = np.zeros((self.mask.shape[0], self.mask.shape[1], no_channels), dtype=image_np.dtype)
        orig_image_np[self.y1:self.y2+1, self.x1:self.x2+1, :] = image_np

        return orig_image_np
    

################################################################################################################################
class CustomDrawRectangle:
    """Draw a rectangle on the image instead of cropping it."""
    def __init__(self, subgroup=None, map_coor=None, outline_color=(255, 0, 0), thickness=8):
        self.subgroup = subgroup  # Store subgroup for rectangle placement
        self.map_coor = map_coor  # Coordinates of the rectangle
        self.outline_color = outline_color  # Color of the rectangle outline
        self.thickness = thickness  # Thickness of the rectangle border

    def __call__(self, image):
        if self.subgroup is None or self.map_coor is None:
            return image

        draw = ImageDraw.Draw(image)
        x, y, w, h = self.map_coor[self.subgroup]

        # Draw the rectangle with specified thickness
        for i in range(self.thickness):
            draw.rectangle([x - i, y - i, x + w + i, y + h + i], outline=self.outline_color)

        return image
################################################################################################################################
################################################################################################################################
class DatasetWithFilename(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_filename=None):
        super().__init__(root, transform)
        
        # If a specific filename is given, filter the dataset
        if target_filename:
            self.imgs = [(path, label) for path, label in self.imgs if os.path.basename(path) == target_filename]
            self.samples = self.imgs  # Ensure compatibility with ImageFolder

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        filename = os.path.basename(path)
        return img, label, filename

##############################################################
def get_data_loaders(paths,params,#  train_dir,test_dir,
                    target_size=(128, 128), aug_type='min',
                    batch_size=64, mask_image_name=6232,
                    orig_image_size = (1280,720),
                    reorder_test_classes=False,
                    reorder_test_classes_opp=False,
                    subgroup=None,
                    num_workers=8, pin_memory=True,persistent_workers=True
                    ):
    
    
    if subgroup is None:
        subgroup = params.subgroup
    
    if paths.dataset_type == 'fronttop':
        # print('Using fronttop dataset')
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}_ext.png"
    else:
        # print(f'Using {paths.dataset_type} dataset')

        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}.png"
    # print('mask_path',mask_path)
    # if subgroup is not None:
    # print('aug_type ------>',aug_type)
    ###################################################
    ### Train transform
    ###################################################
    if aug_type=='min':
        transform_train = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif aug_type=='auto-mask':
        component_mask_path = os.path.join(paths.mask_dir, mask_path)
        if not os.path.exists(component_mask_path):
            print(f"Component mask not found: {component_mask_path}")
        component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
        component_mask = (component_mask > 128).astype(np.float32) #.point(lambda p: 1.0 if p 
        
        # print('component_mask.shape',component_mask.shape)

        transform_train = transforms.Compose([
            MaskedCrop(subgroup=params.subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
            transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # transform_train = transforms.Compose([
        # LoadMaskedImage(paths,(orig_image_size),subgroup=params.subgroup, subgroup_mask=params.subgroup_mask,
        #                  map_coor=params.map_coor, mask_image_name=mask_image_name),
        
        # # CustomCrop(params.subgroup, params.map_coor),
        # transforms.Resize(target_size),
        # # transforms.Lambda(center_crop_square),  # Manually crop to 1435x1435
        # transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # ])
    elif aug_type=='min-mask':
        transform_train = transforms.Compose([
        LoadMaskedImage(paths,(orig_image_size),subgroup=params.subgroup, subgroup_mask=params.subgroup_mask, map_coor=params.map_coor, mask_image_name=mask_image_name),
        CustomCrop(params.subgroup, params.map_coor),
        transforms.Resize(target_size),
        # transforms.Lambda(center_crop_square),  # Manually crop to 1435x1435
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif aug_type=='min-box':
        transform_train = transforms.Compose([
        CustomCrop(params.subgroup, params.map_coor),
        transforms.Resize(target_size),
        # transforms.Lambda(center_crop_square),  # Manually crop to 1435x1435
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif aug_type=='max-box':
        transform_train = transforms.Compose([
        CustomCrop(params.subgroup, params.map_coor),
        transforms.Resize(target_size),
        transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                shear=5,
                scale=(0.95, 1.0),
                fill=(0, 0, 0),  # Tuple for RGB fill for Robo
            ),
            # transforms.CenterCrop(target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif aug_type=='max-mask':
        transform_train = transforms.Compose([
        # CustomCrop(subgroup, map_coor),
        LoadMaskedImage(paths,(orig_image_size),subgroup=params.subgroup, subgroup_mask=params.subgroup_mask, map_coor=params.map_coor),
        transforms.Resize(target_size),
        transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                shear=5,
                scale=(0.95, 1.0),
                fill=(0, 0, 0),  # Tuple for RGB fill for Robo
            ),
            # transforms.CenterCrop(target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif aug_type=='rect':
        transform_train = transforms.Compose([
        CustomDrawRectangle(params.subgroup, params.map_coor,thickness=8),
        transforms.Resize(target_size),
        # transforms.Lambda(center_crop_square),  # Manually crop to 1435x1435
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif aug_type=='max':
        transform_train = transforms.Compose([
            transforms.Resize(target_size),
            #  transforms.CenterCrop(target_size),
            # transforms.RandomRotation(15, fill=(35, 35, 35)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                shear=5,
                scale=(0.95, 1.0),
                fill=(0, 0, 0),  # Tuple for RGB fill for Robo
            ),
            # transforms.CenterCrop(target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    ###################################################
    ### Test transform
    ###################################################
    if aug_type=='min-mask':
        # Test transform
        transform_test = transforms.Compose([
        LoadMaskedImage(paths,(orig_image_size),subgroup=params.subgroup, subgroup_mask=params.subgroup_mask, map_coor=params.map_coor),
        CustomCrop(params.subgroup, params.map_coor),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    elif aug_type=='auto-mask':
        component_mask_path = os.path.join(paths.mask_dir, mask_path)
        if not os.path.exists(component_mask_path):
            print(f"Component mask not found: {component_mask_path}")
        component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
        component_mask = (component_mask > 128).astype(np.float32) #.point(lambda p: 1.0 if p 
        print('component_mask.shape',component_mask.shape)

        transform_test = transforms.Compose([
            MaskedCrop(subgroup=params.subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
            transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    else:
        # Test transform
        transform_test = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # Create datasets
    train_dataset = datasets.ImageFolder(root=paths.train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=paths.test_dir, transform=transform_test)
    
    if reorder_test_classes:
        if reorder_test_classes_opp:
            original_class_to_idx = test_dataset.class_to_idx
            new_class_to_idx = { 0:"normal"}
            other_classes = [cls for cls in original_class_to_idx if cls != "normal"]
            # print('other_classes', other_classes)
            # Assign new indices to other classes (starting from 1)
            for i, cls in enumerate(other_classes, start=1):
                new_class_to_idx[i] = cls
            
            test_dataset.class_to_idx = new_class_to_idx
            # for path, label in test_dataset.samples:
                # print(path,' | ',label,' | ',new_class_to_idx)
                # print(path,label, new_class_to_idx[label])
            test_dataset.samples = [(path, new_class_to_idx[label]) for path, label in test_dataset.samples]
        else:
            original_class_to_idx = test_dataset.class_to_idx
            new_class_to_idx = { "normal":0}
            other_classes = [cls for cls in original_class_to_idx if cls != "normal"]
            # print('other_classes', other_classes)
            # Assign new indices to other classes (starting from 1)
            for i, cls in enumerate(other_classes, start=1):
                new_class_to_idx[cls] = i
            
            test_dataset.class_to_idx = new_class_to_idx
            idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
            test_dataset.samples = [(path, new_class_to_idx[idx_to_class[label]]) for path, label in test_dataset.samples]

        

    print('train_dataset', len(train_dataset))
    print('test_dataset', len(test_dataset))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=persistent_workers, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=pin_memory
    )

    return train_loader, test_loader, train_dataset, test_dataset

##################################################################################################


def get_data_loaders_for_contour(paths,params,#  train_dir,test_dir,
                     root_dir=None,
                     subgroup=None,
                    aug_type='min',
                    batch_size=64, 
                    mask_image_name=6232,
                    vebose=False,
                    num_workers=8,
                    shuffle=False,
                    pin_memory=True,persistent_workers=True
                    ):

    ###################################################
    ### Train transform
    ###################################################
    if subgroup is None:
        subgroup = params.subgroup

    if paths.dataset_type == 'fronttop':
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}_ext.png"
    else:
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}.png"
    if vebose:
        print(f'using {subgroup} group and {mask_path}')
    # if aug_type=='auto-mask':
    #     component_mask_path = os.path.join(paths.mask_dir, mask_path)
    #     if not os.path.exists(component_mask_path):
    #         print(f"Component mask not found: {component_mask_path}")
    #     component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
    #     component_mask = (component_mask > 128).astype(np.float32) #.point(lambda p: 1.0 if p 
    #     # print('component_mask.shape',component_mask.shape)

    #     transform_train = transforms.Compose([
    #         MaskedCrop(subgroup=subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
    #         transforms.Resize(params.target_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ])
    if aug_type=='auto-contour':
        component_mask_path = os.path.join(paths.mask_dir, mask_path)
        if not os.path.exists(component_mask_path):
            print(f"Component mask not found: {component_mask_path}")
        component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
        component_mask = (component_mask > 128).astype(np.float32) #.point(lambda p: 1.0 if p 
        transform_train = transforms.Compose([
            # MaskedCrop(subgroup=subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
            # transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # Create datasets
    if root_dir is None:
        root_dir = paths.train_dir
    
    dataset = datasets.ImageFolder(root=root_dir, transform=transform_train)

    print('dataset', len(dataset))

    # Create dataloaders
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=persistent_workers, drop_last=True
    )
    
    return data_loader, dataset

##############################
# 
#   
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) 
                          if os.path.isfile(os.path.join(root_dir, f))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed
        
        if self.transform:
            image = self.transform(image)
            
        # If you have labels, you would add them here
        # For now returning just the image
        return image
    
def get_data_loaders_for_contour_and_gt(paths,params,#  train_dir,test_dir,
                     root_dir=None,
                     subgroup=None,
                     image_to_safety_area= False,
                    aug_type='min',
                    batch_size=64, 
                    mask_image_name=6232,
                    verbose=False,
                    num_workers=8,
                    shuffle=False,
                    pin_memory=True,persistent_workers=True
                    ):
    
    ###################################################
    ### Train transform
    ###################################################
    if image_to_safety_area:
        if subgroup is None:
            subgroup = params.subgroup
        print('LOADING subgroup as --> ', subgroup)
        if paths.dataset_type == 'fronttop':
            mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}_ext.png"
        else:
            mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}.png"
        if verbose:
            print(f'using {subgroup} group and {mask_path}')

        if aug_type=='auto-contour':
            component_mask_path = os.path.join(paths.mask_dir, mask_path)
            if not os.path.exists(component_mask_path):
                print(f"Component mask not found: {component_mask_path}")
            component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
            component_mask = (component_mask > 128).astype(np.float32) #.point(lambda p: 1.0 if p 
    
    if image_to_safety_area:
        transform = transforms.Compose([
            MaskedCrop(subgroup=subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
            transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            # MaskedCrop(subgroup=subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
            # transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # Create datasets
    if root_dir is None:
        root_dir = paths.train_dir
    segmentation_masks_dir = paths.segments

    dataset_images = datasets.ImageFolder(root=root_dir, transform=transform)
    dataset_segmentation_maps = datasets.ImageFolder(root=segmentation_masks_dir, transform=transform)
    # dataset_segmentation_maps = CustomImageDataset(root_dir=segmentation_masks_dir, transform=transform_train)
    print('dataset', len(dataset_images))
    print('segmentation_masks', len(dataset_segmentation_maps))

    # Create dataloaders
    data_loader = DataLoader(
        dataset_images, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=persistent_workers, drop_last=True
    )
    segmentation_loader = DataLoader(
        dataset_segmentation_maps, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=persistent_workers, drop_last=True
    )
    return data_loader, dataset_images, segmentation_loader, dataset_segmentation_maps

##################################################################################################
## get_data_loaders_from_preprocessed   
##################################################################################################
def get_data_loaders_from_preprocessed(train_dir_processed_subgroup,
                                       augmentation_type='min',
                                       batch_size=64, 
                                       shuffle = True,
                                       drop_last = True,
                                       num_workers=8,
                                       pin_memory=True,
                                       persistent_workers=True,
                                       verbose = False,
                                       ):
    if verbose:
        print(f'----  DATA LOADER  ---- with ++ {augmentation_type} ++ Augmentation')

    if augmentation_type=='min':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    elif augmentation_type=='custom':
        transform_train = transforms.Compose([
            transforms.RandomAffine(
                degrees=0.01,
                translate=(0.01, 0.01),
                shear=0.1,
                scale=(0.99, 1.0),
                fill=(0, 0, 0),  # Tuple for RGB fill for Robo
            ),
            # transforms.CenterCrop(target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    else:
        assert f'{augmentation_type} NOT SUPPORTED'
    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_dir_processed_subgroup, transform=transform_train)       
    if verbose:
        print('train_dataset', len(train_dataset), ' from ', train_dir_processed_subgroup)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        persistent_workers=persistent_workers, drop_last=drop_last
    )

    # return train_loader, test_loader, train_dataset, test_dataset
    return train_loader, train_dataset
##################################################################################################
##################################################################################################
class DatasetWithFilename(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_filename=None, keep_folders=None):
        super().__init__(root, transform)
        
        keep_folders = set(keep_folders or [])

        # Keep only the specified folders (classes)
        if keep_folders:
            self.imgs = [
                (path, label) for path, label in self.imgs
                if os.path.basename(os.path.dirname(path)) in keep_folders
            ]
            self.samples = self.imgs  # Ensure compatibility

        # Optionally filter by filename
        if target_filename:
            self.imgs = [
                (path, label) for path, label in self.imgs
                if os.path.basename(path) == target_filename
            ]
            self.samples = self.imgs

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        filename = os.path.basename(path)
        return img, label, filename
#########################################################################

def get_test_loaders_from_processed_data_new(
        paths,
        params,
        data_type='test',
        sel_type='min-mask',
        target_size=(128, 128),
        batch_size=64,
        shuffle_data=False,
        mask_image_name=6232,
        subgroup=None,
        orig_image_size=(1280, 720),
        reorder_test_classes=False,
        keep_folders=None,
        num_workers=1,
        verbose=True,
        pin_memory=False,
        target_filename=None
    ):
    if keep_folders is None:
        keep_folders = []

    if subgroup is None:
        subgroup = params.subgroup

    # -------------------------------------------------------
    # Mask path selection (unchanged)
    # -------------------------------------------------------
    if paths.dataset_type == 'fronttop':
        if verbose:
            print('Using fronttop dataset')
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}_ext.png"
    else:
        if verbose:
            print(f'Using {paths.dataset_type} dataset')
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}.png"

    # -------------------------------------------------------
    # Default test transform
    # -------------------------------------------------------
    transform_test = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Optionally override with auto-mask transform
    if sel_type == 'auto-mask':
        component_mask_path = os.path.join(paths.mask_dir, mask_path)
        if not os.path.exists(component_mask_path):
            print(f"Component mask not found: {component_mask_path}")
        component_mask = np.array(Image.open(component_mask_path).convert("L"))
        component_mask = (component_mask > 128).astype(np.float32)

        if verbose:
            print('component_mask.shape', component_mask.shape)

        transform_test = transforms.Compose([
            # MaskedCrop(subgroup=params.subgroup, mask=component_mask.astype('bool')),
            # transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # -------------------------------------------------------
    # Choose directory: train or test
    # -------------------------------------------------------
    if data_type == 'test':
        data_dir = paths.test_dir
    else:
        data_dir = paths.train_dir

    test_dataset = DatasetWithFilename(
        root=data_dir,
        transform=transform_test,
        target_filename=target_filename,
        keep_folders=keep_folders
    )

    if verbose:
        print('data_dir \t', data_dir)

    # -------------------------------------------------------
    # (Optional) Reorder classes so that "normal" = 0
    # -------------------------------------------------------
    if reorder_test_classes:
        original_class_to_idx = test_dataset.class_to_idx
        new_class_to_idx = {"normal": 0}
        other_classes = [cls for cls in original_class_to_idx if cls != "normal"]

        # Assign new indices to other classes (starting from 1)
        for i, cls in enumerate(other_classes, start=1):
            new_class_to_idx[cls] = i

        test_dataset.class_to_idx = new_class_to_idx
        idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}

        # Remap labels in samples
        test_dataset.samples = [
            (path, new_class_to_idx[idx_to_class[label]])
            for path, label in test_dataset.samples
        ]

    # -------------------------------------------------------
    # NEW: sort samples in temporal order by frame index
    # -------------------------------------------------------
    def extract_frame_idx(path):
        """
        Extract an integer frame index from the filename.
        Example filenames:
          - frame_000123_PLeft.png  ->  123
          - img_0045.png            ->  45
        If no digits are found, return 0.
        """
        bn = os.path.basename(path)
        m = re.search(r'\d+', bn)
        return int(m.group()) if m else 0

    # Sort by frame index so that iteration follows video time
    test_dataset.samples.sort(key=lambda s: extract_frame_idx(s[0]))

    # -------------------------------------------------------
    # DataLoader
    # -------------------------------------------------------
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader, test_dataset



###########################################################################################################
def get_test_loaders_from_processed_data(paths,params,data_type='test',sel_type='min-mask', target_size=(128, 128),
                     batch_size=64, shuffle_data=False, 
                     mask_image_name=6232,
                     subgroup=None,
                     orig_image_size = (1280,720),
                     reorder_test_classes = False,
                     keep_folders = [],
                     num_workers=1, 
                     verbose = True,
                     pin_memory=False, target_filename=None):
    if subgroup is None:
        subgroup = params.subgroup

    if paths.dataset_type == 'fronttop':
        if verbose:
            print('Using fronttop dataset')
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}_ext.png"
    else:
        if verbose:
            print(f'Using {paths.dataset_type} dataset')
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}.png"

    # Test transform
    transform_test = transforms.Compose([
        transforms.Resize(target_size),
        # transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # if sel_type=='min':
    #     transform_test = transforms.Compose([
    #         transforms.Resize(target_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ])
    # el
    if sel_type=='auto-mask':
        component_mask_path = os.path.join(paths.mask_dir, mask_path)
        if not os.path.exists(component_mask_path):
            print(f"Component mask not found: {component_mask_path}")
        component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
        component_mask = (component_mask > 128).astype(np.float32) #.point(lambda p: 1.0 if p 
        if verbose:
            print('component_mask.shape',component_mask.shape)

        transform_test = transforms.Compose([
            # MaskedCrop(subgroup=params.subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
            # transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    # elif sel_type=='min-mask':
    #     transform_test = transforms.Compose([
    #         LoadMaskedImage(paths,(orig_image_size),subgroup=params.subgroup, subgroup_mask=params.subgroup_mask, map_coor=params.map_coor, mask_image_name=mask_image_name),
    #         CustomCrop(params.subgroup, params.map_coor),
    #         transforms.Resize(target_size),
    #         # transforms.Lambda(center_crop_square),  # Manually crop to 1435x1435
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ])

    # Load only the specific image if target_filename is provided
    if data_type=='test':
        data_dir  = paths.test_dir
    else:
        data_dir  = paths.train_dir
    # skip_folders = ['bad_folder1', 'bad_folder2']

    test_dataset = DatasetWithFilename(root=data_dir, transform=transform_test, target_filename=target_filename,
                                       keep_folders=keep_folders)
    if verbose:
        print('data_dir \t', data_dir)

    if reorder_test_classes:
        original_class_to_idx = test_dataset.class_to_idx
        new_class_to_idx = { "normal":0}
        other_classes = [cls for cls in original_class_to_idx if cls != "normal"]
        # print('other_classes', other_classes)
        # Assign new indices to other classes (starting from 1)
        for i, cls in enumerate(other_classes, start=1):
            new_class_to_idx[cls] = i
        
        test_dataset.class_to_idx = new_class_to_idx
        idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
        test_dataset.samples = [(path, new_class_to_idx[idx_to_class[label]]) for path, label in test_dataset.samples]
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle_data, 
        num_workers=0, pin_memory=pin_memory
    )
    # test_loader.batch_size = batch_size

    return test_loader,test_dataset
##################################################################################
def get_test_loaders(paths,params,data_type='test',sel_type='auto-mask', target_size=(128, 128),
                     batch_size=64, shuffle_data=False, 
                     test_dir = None,
                     mask_image_name=6232,
                     subgroup=None,
                     orig_image_size = (1280,720),
                     reorder_test_classes = False,
                     keep_folders = [],
                     num_workers=1, 
                     verbose = True,
                     pin_memory=False, target_filename=None):
    if subgroup is None:
        subgroup = params.subgroup

    if paths.dataset_type == 'fronttop':
        if verbose:
            print('Using fronttop dataset')
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}_ext.png"
    else:
        if verbose:
            print(f'Using {paths.dataset_type} dataset')
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{params.subgroup_mask}.png"

    # Test transform
    transform_test = transforms.Compose([
        transforms.Resize(target_size),
        # transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    if sel_type=='auto-mask':
        component_mask_path = os.path.join(paths.mask_dir, mask_path)
        if not os.path.exists(component_mask_path):
            print(f"Component mask not found: {component_mask_path}")
        component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
        component_mask = (component_mask > 128).astype(np.float32) #.point(lambda p: 1.0 if p 
        if verbose:
            print('component_mask.shape',component_mask.shape)

        transform_test = transforms.Compose([
            MaskedCrop(subgroup=params.subgroup, mask=component_mask.astype('bool')),   # Just the cropping part
            transforms.Resize(params.target_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    # Load only the specific image if target_filename is provided
    if test_dir is None:
        if data_type=='test':
            data_dir  = paths.test_dir
        else:
            data_dir  = paths.train_dir
    else:
        data_dir = test_dir
    # skip_folders = ['bad_folder1', 'bad_folder2']

    test_dataset = DatasetWithFilename(root=data_dir, transform=transform_test, target_filename=target_filename,
                                       keep_folders=keep_folders)
    if verbose:
        print('data_dir \t', data_dir)

    if reorder_test_classes:
        original_class_to_idx = test_dataset.class_to_idx
        new_class_to_idx = { "normal":0}
        other_classes = [cls for cls in original_class_to_idx if cls != "normal"]
        # print('other_classes', other_classes)
        # Assign new indices to other classes (starting from 1)
        for i, cls in enumerate(other_classes, start=1):
            new_class_to_idx[cls] = i
        
        test_dataset.class_to_idx = new_class_to_idx
        idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
        test_dataset.samples = [(path, new_class_to_idx[idx_to_class[label]]) for path, label in test_dataset.samples]
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle_data, 
        num_workers=0, pin_memory=pin_memory
    )
    # test_loader.batch_size = batch_size

    return test_loader,test_dataset


#################################################################################################
#################################################################################################
def show_and_save(file_name, real_batch,paths,data_type='tuple',img_count=2, destroy_fig=True,fontsize = 14):
    if data_type=='tuple':
        grid_b = real_batch[0]
    else:
        grid_b = real_batch
    img = make_grid((grid_b*0.5+0.5).cpu(),img_count)

    subgroup = file_name.split('_')[-1]
    
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = f"{paths.path_results}/{file_name}.png"
    
    fig = plt.figure(figsize=(15,5))
    
    # Adjust the position of the suptitle
    plt.title(file_name, fontweight='bold', fontsize = fontsize)  # y parameter controls the vertical position
    
    plt.imshow(npimg)
    plt.xticks([])
    plt.yticks([])
    
    # Adjust the layout to reduce the gap between the suptitle and the subplot
    plt.tight_layout()  # rect parameter controls the area that tight_layout considers
    
    plt.savefig(f, bbox_inches='tight', pad_inches=0.1)  # Save the figure with tight bounding box
    
    if destroy_fig:
        plt.close(fig)

# Example usage:
# show_and_save("example", img_tensor, "path/to/results")
    
    plt.show()
#################################################################################################
#################################################################################################
def plot_loss(loss_list):
    plt.figure(figsize=(10,5))
    plt.title("Loss During Training")
    plt.plot(loss_list,label="Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
############################################################################
def plot_test_images(test_loader, paths):
    # Get one batch from the dataloader
    real_batch, labels = next(iter(test_loader))

    # Convert tensors to numpy for visualization
    real_batch = real_batch.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5  # (B, C, H, W) → (B, H, W, C)
    labels = labels.cpu().numpy()

    # Dictionary to store one image per class
    class_samples = {}
    for img, lbl in zip(real_batch, labels):
        if lbl not in class_samples:
            class_samples[lbl] = img
        if len(class_samples) == len(set(labels)):  # Stop if we have one sample per class
            break

    # Plot the images
    fig, axes = plt.subplots(1, len(class_samples), figsize=(12, 3))
    for ax, (label, img) in zip(axes, class_samples.items()):
        ax.imshow(img)
        ax.set_title(f"{paths.test_classes[label]}", fontsize = 18)
        ax.axis("off")
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.tight_layout(pad=0.05)
    plt.savefig(f"{paths.path_results_local}/class_samples.png")

    plt.show()

############################################################################

def get_colormap(mode=1):
    if mode==1:
    # TUNED Colormap 
        colormap_anomaly_map = \
            LinearSegmentedColormap.from_list("colormap_anomaly_map", 
                                            [(0.0, 'white'),
                                            (0.9 / 2, 'skyblue'),
                                            (0.91 / 2, 'violet'),
                                            (1.0, 'red')
                                            ])
    else:
        colormap_anomaly_mapv1 = \
            LinearSegmentedColormap.from_list("colormap_anomaly_map", 
                                      [(0.0, '#ffffff'),
                                       (0.1, '#ff0000'),
                                       (1.0, '#03fcf8')])
    return colormap_anomaly_map

colormap_anomaly_map = get_colormap()
############################################################################
def plot_test_single(original, reconstructed, epoch, ttl='test', data_type='train', plot_suptitle=True,
                     save_path = None,batch = False,anomaly_score=None,
                     dpi = 150,
                     transparent_fig=True, plot_anomaly_scores=False, save_fig=True, destroy_fig=True, fontsize=12,
                     fontcolor='black', verbose_debug=False):
    print()
    os.makedirs(save_path,exist_ok=True)
    if plot_anomaly_scores:
        if anomaly_score is None:
            if not batch:
                anomaly_score = get_anomaly_score(original, reconstructed)[0]
            else:
                anomaly_score = get_anomaly_score(original, reconstructed)
    else:
        # print('debugggggggg,', anomaly_score)
        anomaly_score = anomaly_score[0]
            
    original = original.cpu().detach()
    reconstructed = reconstructed.cpu().detach()
    
    # Only plot one image (first image in the batch)
    orig_img = original[0]  # Extract the first image from the batch
    recon_img = reconstructed[0]  # Extract the first reconstructed image
    
    orig_img = orig_img * 0.5 + 0.5  # Unnormalize
    recon_img = recon_img * 0.5 + 0.5
    
    # Compute the difference (anomaly map)
    diff_img = torch.norm(orig_img - recon_img, dim=0)

    # Ensure the images are in HWC format for plotting (H, W, C)
    orig_img = orig_img.permute(1, 2, 0).numpy()  # Convert from C x H x W to H x W x C
    recon_img = recon_img.permute(1, 2, 0).numpy()  # Convert from C x H x W to H x W x C
    diff_img = diff_img.numpy()

    # Plot the original image, reconstructed image, and anomaly map
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(5, 2))  # Plot 3 images side by side
    
    axes[0].imshow(orig_img)
    axes[1].imshow(recon_img)
    axes[2].imshow(diff_img, cmap=colormap_anomaly_map, vmin=0, vmax=2)


    if plot_anomaly_scores:
            # print('anomaly_score :::::', anomaly_score)
            axes[2].text(10, 15, f'Score: {anomaly_score:.2f}', color=fontcolor, fontsize=fontsize, weight='bold')
    if plot_suptitle:
        plt.suptitle(f'Epoch {epoch}: {ttl}', fontsize=fontsize, color=fontcolor)
    for (ax,titl) in zip(axes,[ttl, "x'",'diff_map']):
        ax.set_title(f'{titl}', fontsize=fontsize, color=fontcolor)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    if save_fig:
        # Save the figure if needed
        plt.savefig(f'{save_path}/{ttl}.png', transparent=transparent_fig, dpi = dpi,bbox_inches='tight')
    if destroy_fig:
        plt.close(fig)  # Close the plot to free memory
    else:
        plt.show()  # Show the plot interactively
############################################################################

def imshow(img, axs=None):
    """ Function to show an image. """
    img = img / 2 + 0.5  # Unnormalize the image
    npimg = img.numpy()
    if axs is None:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    else:
        axs.imshow(np.transpose(npimg, (1, 2, 0)))
def img_CHW_HWC(img_tensor):
    return img_tensor.permute(1, 2, 0)  # (C, H, W) to (H, W, C)
############################################################################
def get_anomaly_score(data_batch,recon_batch):
    assert data_batch.shape == recon_batch.shape, f"Shape mismatch: {recon_batch.shape} vs {data_batch.shape}"

    # recon_batch = recon_.cpu().detach().numpy()
    # data_batch = data_.cpu().detach().numpy()
    data_batch = data_batch.to(device)
    recon_batch = recon_batch.to(device)
    # print(data_batch.shape, recon_batch.shape)
    abs_diff = torch.abs(recon_batch - data_batch)
    mean_diff = abs_diff.mean(dim=1)
    max_score = mean_diff.max(dim=-1).values.max(dim=-1).values
    return max_score

############################################################################
def plot_images(original, reconstructed, epoch, paths,ttl='train',data_type=None,anomaly_scores=None,save_path=None,cmap=None,
                plot_anomaly_scores=False,plot_suptitle=True, destroy_fig=False, save_fig=True, interval=10,fontsize=12,fontcolor='black'):
    '''
    data_, recon_, epoch, ttl = ttl, save_fig=save_fig, plot_anomaly_scores=plot_anomaly_scores,destroy_fig=destroy_fig
    | ------------------------------------------------------------------------------|
    |                                                                               |
    |   Plot a single original, reconstructed, and anomaly map image side by side   |
    | ------------------------------------------------------------------------------|
    |   original : input image                                                      |
    |   reconstructed: reconstructed image from VAE model                           |
    |   anomaly_map : anomaly map image                                             |
    |   epoch : epoch number                                                        |
    |   save_fig :  whether to save figure to disk              |   default: False  |
    '''
    # Detach tensors and move them to CPU for plotting
    if plot_anomaly_scores and anomaly_scores is None:
        anomaly_score = get_anomaly_score(reconstructed, original)
    else:
        anomaly_score = anomaly_scores
    # Detach tensors and move to CPU for plotting

    cmap='Reds' if cmap is None else colormap_anomaly_map
    original = original.cpu().detach()
    reconstructed = reconstructed.cpu().detach()
    if isinstance(ttl, str):
        if ttl=='train':
            cls_ttl = 'x: Normal'
        elif ttl =='test': 
            cls_ttl = 'x: Anomalous' 
        elif ttl !='train' and ttl !='test': 
            # cls_ttl = f'x: {ttl}'
            cls_ttl = ttl
    # elif isinstance(ttl, list):
    if data_type=='train':
        classes_name = paths.train_classes
    else:
        classes_name = paths.test_classes
    

    n_images = 3  # Number of images to display
    fig, axes = plt.subplots(nrows=1, ncols=n_images * 3, figsize=(10, 2))

    for i_id_i, (i_id,i) in enumerate(zip(ttl[0:n_images], range(n_images))):
        # Extract the i-th image from each set
        orig_img = original[i] * 0.5 + 0.5  # Unnormalize
        recon_img = reconstructed[i] * 0.5 + 0.5
        
        # Compute the anomaly map (difference between original and reconstructed images)
        diff_img = torch.norm(orig_img - recon_img, dim=0)

        # Plot each image in the appropriate column
        axes[i * 3].imshow(orig_img.permute(1, 2, 0).numpy())
        axes[i * 3 + 1].imshow(recon_img.permute(1, 2, 0).numpy())
        axes[i * 3 + 2].imshow(diff_img.numpy(), cmap=cmap, vmin = 0, vmax = 2)#cmap='hot')

        # Set titles for the first set only
        # if i == 0:
        # print(type(ttl))
        if isinstance(ttl, list):
            # print(ttl, test_classes[i_id][i_id] )
            axes[i * 3].set_title(classes_name[i_id])
        else:
            # print('---->', cls_ttl, i_id_i)
            axes[i * 3].set_title(cls_ttl)
        axes[i * 3 + 1].set_title("x'")
        axes[i * 3 + 2].set_title('m')

        # Overlay anomaly score if specified
        if plot_anomaly_scores:
            score = anomaly_score[i]
            axes[i * 3 + 2].text(10, 15, f'Score: {score:.2f}', color=fontcolor, fontsize=fontsize-2, weight='bold')

    # Turn off axes
    for ax in axes:
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    # for ax,ttl_ in zip(axes.flatten(),["x","x'","m"]):
    #     ax.set_title(ttl_)
    if plot_suptitle:
        plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0.05,wspace=0.05)
    # Save figure if specified
    if save_path is None:
        save_path = paths.path_results
    if save_fig and epoch % interval == 0:
        plt.savefig(f'{save_path}/{ttl}_{epoch}.png', bbox_inches='tight')
    
    # Close the figure if specified
    if destroy_fig:
        plt.close(fig)
        
    plt.show()

############################################################################

def plot_images_tracking(original_random, reconstructed_random, 
                         original_fixed, reconstructed_fixed, 
                         original_noise, reconstructed_noise, 
                         epoch, path_results, ttl='train', 
                         cmap = None,
                         plot_anomaly_scores=False, destroy_fig=False, 
                         save_fig=True, interval=10, fontsize=10, fontcolor='black'):
    '''
    Plot one image from each category (random, fixed, noise) in a single row showing:
    - Original image
    - Reconstructed image
    - Anomaly map (difference between original and reconstructed)
    '''
    original_random = original_random.cpu().detach()
    reconstructed_random = reconstructed_random.cpu().detach()
    original_fixed = original_fixed.cpu().detach()
    reconstructed_fixed = reconstructed_fixed.cpu().detach()
    original_noise = original_noise.cpu().detach()
    reconstructed_noise = reconstructed_noise.cpu().detach()

    categories = [(original_random, reconstructed_random, "Random"),
                  (original_fixed, reconstructed_fixed, "Fixed"),
                  (original_noise, reconstructed_noise, "Noise")]

    fig, axes = plt.subplots(nrows=1, ncols=3 * 3, figsize=(9, 1.5))  # 3 images (original, reconstructed, anomaly map)
    
    for c, (original, reconstructed, title_prefix) in enumerate(categories):
        orig_img = original[0] * 0.5 + 0.5  # Using the first image from each category
        recon_img = reconstructed[0] * 0.5 + 0.5
        diff_img = torch.norm(orig_img - recon_img, dim=0)

        base_idx = c * 3  # 3 columns: original, reconstructed, and anomaly map
        
        axes[base_idx].imshow(orig_img.permute(1, 2, 0).numpy())
        axes[base_idx].set_title(f"{title_prefix} x", fontsize=fontsize)
        
        axes[base_idx + 1].imshow(recon_img.permute(1, 2, 0).numpy())
        axes[base_idx + 1].set_title(f"{title_prefix} x'", fontsize=fontsize)
        
        axes[base_idx + 2].imshow(diff_img.numpy(), cmap='Reds' if cmap is None else colormap_anomaly_map, vmin=0, vmax=2)
        axes[base_idx + 2].set_title(f"{title_prefix} m", fontsize=fontsize)
        if plot_anomaly_scores:
            # print(title_prefix,reconstructed.shape, original.shape)
            anomaly_score = get_anomaly_score(reconstructed, original)[0]
            # print(f"Anomaly score: {anomaly_score}")
            axes[base_idx + 2].text(10, 15, f'Score: {anomaly_score:.2f}', color=fontcolor, fontsize=fontsize-2, weight='bold')

        for ax in [axes[base_idx], axes[base_idx + 1], axes[base_idx + 2]]:
            # ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle(f'Epoch {epoch}', fontsize=fontsize)
    plt.tight_layout(pad=0.05)
    plt.subplots_adjust(hspace=0.05,wspace=0.05)

    if save_fig and epoch % interval == 0:
        plt.savefig(f'{path_results}/{ttl}_{epoch}.png', bbox_inches='tight')

    if destroy_fig:
        plt.close(fig)
    
    plt.show()
def get_header(params,paths, verbose=False):
    log_separator_bold = "=" * 160 + "\n"
    log_separator_dash = "-" * 160 + "\n"
    log_messages = ""
    input_shape = f"{params.input_shape[0]}x{params.input_shape[1]}x{params.input_shape[2]}"
    log_code_header_stat = f"| {'dataset':^10} | {'camera':^10} | {'subgroup':^10} |{'epochs':^10} | {'latent_dims':^15} | {'input_shape':^15} | {'batch_size':^10} | {'LR_enc_dec':^15} | {'LR_dis':^15} |\n"
    log_code_header_dyn = f"| {paths.dataset_version:^10} | {paths.dataset_type:^10} | {params.subgroup:^10} | {params.epochs:^10} | {params.latent_dims:^15} | {f'{input_shape}':^15} | {params.batch_size:^10} | {params.learning_rate_enc_dec:^15} | {params.learning_rate_dis:^15} |\n"
    if verbose:
        print(log_separator_bold.strip())
        print(log_code_header_stat.strip())
        print(log_separator_dash.strip())
        print(log_code_header_dyn.strip())
    return log_code_header_stat,log_code_header_dyn
############################################################################
def create_log_file(params,paths,start_time, verbose=False, read_mode='w'):
    log_messages = ""
    log_separator_dash = "-" * 160 + "\n"
    log_separator_bold = "=" * 160 + "\n"
    log_messages += log_separator_dash

    if os.path.exists(paths.log_file_full) and os.path.getsize(paths.log_file_full) > 0:
        return log_messages
    
    log_save_data_msg = f"START CODE \n"

    log_code_started_msg = f"code started:\t{start_time}\n"

    # input_shape = f"{params.input_shape[0]}x{params.input_shape[1]}x{params.input_shape[2]}"
    # log_code_header_stat = f"| {'epochs':^10} | {'latent_dims':^15} | {'input_shape':^15} | {'batch_size':^10} | {'LR_enc_dec':^15} | {'LR_dis':^15} | {'beta_kl':^10} | {'beta_gan':^10} | {'reconstruction_loss_fn':^10} | {'adversarial_loss_fn':^10} |\n"
    # log_code_header_dyn = f"| {params.epochs:^10} | {params.latent_dims:^15} | {f'{input_shape}':^15} | {params.batch_size:^10} | {params.learning_rate_enc_dec:^15} | {params.learning_rate_dis:^15} | { params.beta_kl:^10} | {params.beta_gan:^10} | { params.reconstruction_loss_fn:^10} | {params.adversarial_loss_fn:^10} |\n"
    log_code_header_stat,log_code_header_dyn = get_header(params,paths, verbose=False)
    if verbose:
        print(log_separator_bold.strip())
        print(log_code_header_stat.strip())
        print(log_separator_dash.strip())
        print(log_code_header_dyn.strip())

    # log_table_header_msg =f"| {'epoch':<8} | {'avg_recon':<12.5} | {'avg_kl':<25.5} | {'avg_real':<12.5} | {'avg_fake':<12.5} | {'avg_vae':<15.5} | {'avg_gan':<15.5} | {'elapsed_time_str':<15} |\n"
    
    log_messages += log_separator_dash
    log_messages += f"{log_code_header_stat}\n"
    log_messages += log_separator_dash
    log_messages += f"{log_code_header_dyn}\n"
    log_messages += log_separator_dash
    # Append log messages
    log_messages += log_separator_bold
    # log_messages += log_header_main
    # log_messages += log_header_main_data

    log_messages += log_separator_dash
    log_messages += log_save_data_msg
    log_messages += log_code_started_msg
    log_messages += f"paths_results:\t{paths.path_results}\n"
    log_messages += f"paths_models:\t{paths.path_models}\n"
    # log_messages += f"paths_results:\t{paths.path_results}\n"

    log_messages += log_separator_dash
    # log_messages += log_table_header_msg

    # Print separator lines and log messages
    if verbose:
        print(log_separator_bold.strip())
        print(log_messages.strip())
        # print(log_header_main_data.strip())
    
        print(log_separator_bold.strip())
        print(log_save_data_msg.strip())
        print(log_code_started_msg.strip())
        # print(log_table_header_msg.strip())
        print(log_separator_dash.strip())
    return log_messages
############################################################################
def save_log_file(log_file_full, log_messages,read_mode='a', verbose=True):
    with open(log_file_full, read_mode) as log_file:
        log_file.write(log_messages)
    if verbose:
        print(f"Log file saved to {log_file_full}")
############################################################################
# def check_verify_paths(paths,suffix,save_path_type='local',create_path = True, verbose=False):
#     if save_path_type=='local':
#         base_dir = paths.path_codes_local
#     elif save_path_type=='cloud':
#         base_dir = paths.path_codes_cloud
#     paths.path_models = os.path.join(base_dir,f'models_{suffix}')
#     paths.path_results = os.path.join(base_dir,f'results_{suffix}')
#     paths.path_results_fix = os.path.join(base_dir,f'monitor_{suffix}')
#     paths.history_fname = f'history_train_{suffix}.csv'
    
#     if create_path:
#         os.makedirs(paths.path_models, exist_ok=True)
#         os.makedirs(paths.path_results, exist_ok=True)
#         os.makedirs(paths.path_results_fix, exist_ok=True)
#     if verbose:
#         print(f'path_models : {paths.path_models}')
#         print(f'path_results : {paths.path_results}')
#         print(f'path_results_fix : {paths.path_results_fix}')
#     return paths
############################################################################

def get_status_info(loss_history,params,paths):
    print('Model with Epochs \t-->',len(loss_history))
    print('Component \t\t-->',params.subgroup)
    print('Dataset \t\t-->',paths.dataset_type)
    print('Train Classes \t\t-->',paths.train_classes)
    print('Test Classes \t\t-->',paths.test_classes)

############################################################################

def get_initial_paths(paths,params, verbose=True):
    paths.path_codes = os.getcwd()

    paths.path_codes_local = os.path.join(paths.path_results_local,paths.dataset_type)
    paths.path_codes_cloud = os.path.join(paths.path_codes,paths.dataset_type)
    
    # paths.path_codes_local = os.path.join(paths.path_codes_local,params.subgroup)

    if verbose:
        print('current path:\t\t', os.getcwd())
        print('path_codes_local:\t', paths.path_codes_local)
        print('path_codes_cloud:\t', paths.path_codes_cloud)
        print('path_codes_local:\t', paths.path_codes_local)
        print('path_codes_local:\t', paths.path_codes_local)


    if not os.path.exists(paths.path_codes_local):
        os.makedirs(paths.path_codes_local)
        print('Creating path:\t\t', paths.path_codes_local)
    else:
        print('Path exists:\t\t', paths.path_codes_local)


    return paths



############################################################################
# def get_initial_paths(paths,params, verbose=True):
#     paths.path_codes = os.getcwd()

#     paths.path_codes_local = os.path.join(paths.path_results_local,paths.dataset_type)
#     paths.path_codes_cloud = os.path.join(paths.path_codes,paths.dataset_type)
    
#     # paths.path_codes_local = os.path.join(paths.path_codes_local,params.subgroup)

#     if verbose:
#         print('current path:\t\t', os.getcwd())
#         print('path_codes_local:\t', paths.path_codes_local)
#         print('path_codes_cloud:\t', paths.path_codes_cloud)
#         print('path_codes_local:\t', paths.path_codes_local)
#         print('path_codes_local:\t', paths.path_codes_local)


#     if not os.path.exists(paths.path_codes_local):
#         os.makedirs(paths.path_codes_local)
#         print('Creating path:\t\t', paths.path_codes_local)
#     else:
#         print('Path exists:\t\t', paths.path_codes_local)

#     return paths


############################################################################
def get_create_results_path(subgroup,params,args,paths,dir='scripts', save_path_type='local',models_dir = 'models',create_dirs=True, verbose=True):
    if save_path_type=='local':
        base_dir = paths.path_codes_local
    elif save_path_type=='cloud':
        base_dir = paths.path_results_cloud
    
    suffix = f'{subgroup}_{params.latent_dims}'
    if verbose:
        print('using suffix as -->', suffix)
    paths.history_fname = f'vae_gan_train_history_{subgroup}.csv'
    paths.path_models = os.path.join(base_dir,f'{models_dir}')
    paths.path_results = os.path.join(base_dir,f'training/{suffix}')
    # paths.results_test = os.path.join(base_dir, f'test/{suffix}')
    if args.save_figures:
        paths.path_results_fix = os.path.join(base_dir,f'monitor/{suffix}')

    ############################################################################
    
    paths.path_codes_cloud = paths.path_codes
    paths.path_codes_main = os.path.join(paths.path_codes,dir)

    # paths.path_models = os.path.join(paths.path_codes, 'models' )
    # paths.path_results = os.path.join(paths.path_codes,'results')

    paths.path_codes_local = os.path.join(paths.path_results_local,dir)
    paths.path_results_cloud = os.path.join(paths.path_codes_cloud,dir)
    
    ############################################################################
    if create_dirs:
        os.makedirs(paths.path_models, exist_ok=True)
        os.makedirs(paths.path_results, exist_ok=True)
        # os.makedirs(paths.results_test, exist_ok=True)
        if args.save_figures:
            os.makedirs(paths.path_results_fix, exist_ok=True)
    # log_file_full = f'{paths.path_codes}/log_{suffix}.txt'
    paths.log_file_full = f'{paths.path_results}/log_file_full.txt'
    if verbose:
        print('Component : ', subgroup)
        print('-'*120)
        print('base_dir : ', base_dir)
        print('path_models : ',paths.path_models)
        print('path_results : ',paths.path_results)
        # print('results_test : ',paths.results_test)
        if args.save_figures:
            print('path_results_fix : ',paths.path_results_fix)

    return suffix,paths

############################################################################

def plot_loss_sep(loss_history,params,paths,plot_type=3,save_fig=True, destroy_fig=True,verbose_print=False, plot_long_header=True, fontsize = 12):
    # === Plot losses ===
    if plot_type==3:
        fig, axs = plt.subplots(1, 3, figsize=(12, 5))
    elif plot_type==2:
        fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    ax = axs[0]

    ax.plot([l["recon_loss"] for l in loss_history], label="Reconstruction Loss", c='blue')
    ax.plot([l["kl_loss"] for l in loss_history], label="KL Loss", c='teal')
    ax.plot([l["beta_kl_loss"] for l in loss_history], label="beta * KL Loss", c='darkturquoise')
    ax.plot([l["gan_loss"] for l in loss_history], label="GAN Loss", c='green')
    ax.plot([l["beta_gan_loss"] for l in loss_history], label="beta * GAN Loss", c='limegreen')
    ax.plot([l["vae_loss"] for l in loss_history], label="VAE Loss", c='indianred')
    ax.plot([l["annealing_lambda"] for l in loss_history], label="Annealing", c='lightgray')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("VAE Loss (Log)")
    ax.set_yscale('log')
    ax.legend()
    ################
    ax = axs[1]
    ax.scatter(range(len(loss_history)), [l["dis_acc"] for l in loss_history], 
                    label="Accuracy(Dis)", c='orangered', alpha=0.5, marker='+')
    ax.scatter(range(len(loss_history)), [l["dis_F1"] for l in loss_history], 
                    label="F1(Dis)", c='royalblue', alpha=0.5, marker='x')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Discriminator Scores")
    ax.legend()
    ################
    if plot_type==3:
        ax = axs[2]
        ax.plot([l["gan_loss"] for l in loss_history], label="GAN Loss", c='green')
        ax.scatter(range(len(loss_history)), [l["disc_loss"] for l in loss_history], 
                    label="Discriminator Loss", c='purple', s=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("GAN Loss")
        ax.legend()
    

    header_top = f"Loss Progression (Epoch {len(loss_history)})"
    if plot_long_header:
                # Define column headers and widths
        column_headers = ["DS_v","camera","subgroup","Epochs", 
                        # "Beta KL", "Beta GAN", "Recon Loss", "Adv Loss"
                        ]
        column_widths = [10,10,10,10,
                        #   10, 10, 15, 15
                          ]

        # Extract values from params
        values = [paths.dataset_version,paths.dataset_type,params.subgroup,f'{len(loss_history)} ({params.epochs})', 
                #   params.beta_kl, params.beta_gan, params.reconstruction_loss_fn, params.adversarial_loss_fn
                  ]

        # Format each header and value with fixed-width spacing
        header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(column_headers, column_widths)) + " |"
        value_row  = "| " + " | ".join(str(v).ljust(w) for v, w in zip(values, column_widths)) + " |"
        
        ttl = f"{header_top}\n{header_row}\n{value_row}"
    else:
        ttl = f"{header_top}\n - Losses - | LR Enc-Dec: {params.learning_rate_enc_dec} | LR Dis: {params.learning_rate_dis}"

    plt.suptitle(ttl, fontsize = fontsize)
    plt.tight_layout()
    if save_fig:    
        plt.savefig(f'{paths.path_results}/history_{paths.suffix}.png', bbox_inches='tight')
        # plt.savefig(f'{paths.path_results_local}/history.png', bbox_inches='tight')
    if destroy_fig:
        plt.close(fig)
    plt.show()
##########################################################################################
def plot_losses(loss_history,params,paths,kl_usuage=False, save_fig=True, destroy_fig=True,verbose_print=False, plot_long_header=True):
        if kl_usuage:
            kl_text = f'USING KL loss'
        else:
            kl_text = f'WITHOUT KL loss'
        fig = plt.figure(figsize=(12, 6))
        
        plt.plot([l["recon_loss"] for l in loss_history], label='Recon Loss',
                linestyle='dashed', color='blue', alpha=0.7)
        plt.plot([l["kl_loss"] for l in loss_history], label='KL Loss',
                linestyle='dashed', color='orange', alpha=0.7)
        plt.plot([l["gan_loss"] for l in loss_history], label='GAN Loss',
                linestyle='dashed', color='green', alpha=0.7)
        plt.plot([l["disc_loss"] for l in loss_history], label='Disc Loss',
                linestyle='dashed',marker='*', color='red', alpha=0.7)


        plt.yscale('log')  # Apply log scale to the y-axis
        if plot_long_header:
                # Define column headers and widths
            column_headers = ["Epochs",
                            "Beta KL", "Beta GAN", "Recon Loss", "Adv Loss"]
            column_widths = [10, 15, 12, 15, 15, 10, 10, 15, 15]

            # Extract values from params
            values = [params.epochs,  
                    params.beta_kl, params.beta_gan, 
                    params.reconstruction_loss_fn, params.adversarial_loss_fn]

            # Format each header and value with fixed-width spacing
            header_row = "| " + " | ".join(h.ljust(w) for h, w in zip(column_headers, column_widths)) + " |"
            value_row  = "| " + " | ".join(str(v).ljust(w) for v, w in zip(values, column_widths)) + " |"

            ttl = f"{header_row}\n{value_row}"
        else:
            ttl = f"{kl_text} - Losses - | LR Enc-Dec: {params.learning_rate_enc_dec} | LR Dis: {params.learning_rate_dis}"
        
        plt.title(ttl)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Log Scale)')
        plt.legend()
        plt.grid(which='both', linestyle='--', linewidth=0.5)  # Improve visibility of grid

        if save_fig:
            plt.savefig(f'{paths.path_results}/history_combined.png', bbox_inches='tight')
        if verbose_print:
            print(f'{paths.path_results}/history.png')
        if destroy_fig:
            plt.close(fig)
        
        plt.show()
##########################################################################################
def plot_img_gt(img,gt,file_n,lbl,groundtruth_classes):
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(np.transpose(img.numpy() *0.5 + 0.5,(1,2,0)))
    axs[0].set_title('Test Image')
    axs[1].imshow(np.transpose(gt.numpy()*0.5 + 0.5,(1,2,0)))
    axs[1].set_title('Ground Truth Image')
    plt.suptitle(f"{os.path.basename(file_n).split('.')[0]} : {lbl} - {groundtruth_classes[lbl]}")
    axs[0].set_xticks([]) ; axs[0].set_yticks([])
    axs[1].set_xticks([]) ; axs[1].set_yticks([])
    
    plt.show()

##############################################################################################################
##                     SSIM            ########
##############################################################################################################
# def get_ssim_full_params():
#     ssim_params = {}
#     # gaussian_weights : bool, optional
#     # If True, each patch has its mean and variance spatially weighted by a
#     # normalized Gaussian kernel of width sigma=1.5.
#     ####################################
#     ssim_params[1] = {'name':'w1-s1.5-max','win_size':1,'gaussian_weights':False,'sigma':1.5,'aggregation_function':'max'}
#     ssim_params[2] = {'name':'w1-s1.5-q-99','win_size':1,'gaussian_weights':False,'sigma':1.5,'aggregation_function':'q-99'}
#     ssim_params[3] = {'name':'w1-s1.5-q-999','win_size':1,'gaussian_weights':False,'sigma':1.5,'aggregation_function':'q-999'}
#     ####################################
#     ssim_params[4] = {'name':'w3-s1.5-max','win_size':3,'gaussian_weights':False,'sigma':1.5,'aggregation_function':'max'}
#     ssim_params[5] = {'name':'w3-s1.5-q-99','win_size':3,'gaussian_weights':False,'sigma':1.5,'aggregation_function':'q-99'}
#     ssim_params[6] = {'name':'w3-s1.5-q-999','win_size':3,'gaussian_weights':False,'sigma':1.5,'aggregation_function':'q-999'}
#     ####################################
#     # ssim_params[7] = {'win_size':3,'gaussian_weights':False,'aggregation_function':'max'}
#     # ssim_params[8] = {'win_size':3,'gaussian_weights':False,'aggregation_function':'q-99'}
#     # ssim_params[9] = {'win_size':3,'gaussian_weights':False,'aggregation_function':'q-999'}
#     ####################################
#     ssim_params[7] = {'name':'w1-s1.0-gaussian-max','win_size':1,'gaussian_weights':True, 'sigma':1.0,'aggregation_function':'max'}
#     ssim_params[8] = {'name':'w1-s1.0-gaussian-q-99','win_size':1,'gaussian_weights':True, 'sigma':1.0,'aggregation_function':'q-99'}
#     ssim_params[9] = {'name':'w1-s1.0-gaussian-q-999','win_size':1,'gaussian_weights':True, 'sigma':1.0,'aggregation_function':'q-999'}

#     ssim_params[10] = {'name':'w1-s1.5-gaussian-max','win_size':1,'gaussian_weights':True, 'sigma':1.5,'aggregation_function':'max'}
#     ssim_params[11] = {'name':'w1-s1.5-gaussian-q-99','win_size':1,'gaussian_weights':True, 'sigma':1.5,'aggregation_function':'q-99'}
#     ssim_params[12] = {'name':'w1-s1.5-gaussian-q-999','win_size':1,'gaussian_weights':True, 'sigma':1.5,'aggregation_function':'q-999'}

#     ssim_params[13] = {'name':'w1-s2.0-gaussian-max','win_size':1,'gaussian_weights':True, 'sigma':2.0,'aggregation_function':'max'}
#     ssim_params[14] = {'name':'w1-s2.0-gaussian-q-99','win_size':1,'gaussian_weights':True, 'sigma':2.0,'aggregation_function':'q-99'}
#     ssim_params[15] = {'name':'w1-s2.0-gaussian-q-999','win_size':1,'gaussian_weights':True, 'sigma':2.0,'aggregation_function':'q-999'}
    
#     return ssim_params
##############################################################################################################
##                     SSIM            ########
##############################################################################################################
def get_ssim_full_params(win_size_ls = [1, 3], 
                         gaussian_weights_ls = [False, True],
                         sigma_ls = [1.0, 1.5, 2.0],
                         aggregation_function_ls = ['max', 'q-99', 'q-999', 'q-9999', 'q-99999']
                         ):
    ssim_params = {}
    
    # offset_ls = [1, 3]
    # gaussian_kernel_ls = [False, True]
    # sigma_ls = [1.0, 1.5, 2.0]
    # aggregation_function_ls = ['max', 'q-99', 'q-999', 'q-9999', 'q-99999']
    
    idx = 1  # Start indexing from 1
    for win_size in win_size_ls:
        for gaussian_weights in gaussian_weights_ls:
            for sigma in sigma_ls:
                for agg_func in aggregation_function_ls:
                    name = f"w{win_size}-s{sigma}-f-{agg_func}" if gaussian_weights else f"w{win_size}-s{sigma}-{agg_func}"
                    ssim_params[idx] = {
                        'name': name,
                        'win_size': win_size,
                        'gaussian_weights': gaussian_weights,
                        'sigma': sigma,
                        'aggregation_function': agg_func
                    }
                    idx += 1
    
    return ssim_params
##############################################################################################################
##                     ComputeDifferences            ########
##############################################################################################################



class ComputeDifferences():
    def __init__(self, data_batch, recon_batch, AS_SIGMA = 1, AS_OFFSET = 1, AS_QUANT = 1.0):
        """
        Initialize the difference computation class.

        Args:
            data_batch (torch.Tensor): Batch of input images (B, C, H, W)
            recon_batch (torch.Tensor): Batch of reconstructed images (B, C, H, W)
        """
        self.data_batch = data_batch.detach()
        self.recon_batch = recon_batch.detach()
        self.AS_OFFSET = AS_OFFSET
        self.AS_SIGMA  = AS_SIGMA
        self.AS_QUANT  = AS_QUANT 

    def compute(self, type='l1', params=None):
        """Computes both the difference map and the score based on the selected metric."""
        if type == 'l1':
            return self.get_l1_difference(supress_output=True)
        elif type == 'l2':
            return self.get_l2_difference(supress_output=True)
        elif type == 'ravi':
            return self.get_ravi_difference()
        elif type == 'gaussian':
            return self.get_gaussian_difference()
        elif type.startswith('ssim'):
            return self.get_ssim_dissimilarity_batch(ssim_type=type, params=params,supress_output=True)
        else:
            raise ValueError(f"Unsupported type: {type}. Use 'l1', 'l2', or 'ssim'.")

    def get_l1_difference(self,supress_output=False):
        """Computes L1 (absolute) difference between original and reconstructed images."""
        diff_map = torch.abs(self.data_batch - self.recon_batch)  # (B, C, H, W)
        diff_score = diff_map.mean(dim=(1, 2, 3)).cpu().numpy()  # (B,)
        if supress_output:
            diff_map = diff_map.cpu().numpy()
            diff_map = np.moveaxis(diff_map, 1, 3)  # (B, C, H, W) → (B, H, W, C)
            diff_map = np.mean(diff_map, axis=-1)  # Convert to grayscale
            diff_map = diff_map[0]  # Select first image in batch
        return diff_map, diff_score

    def get_l2_difference(self,supress_output = False):
        """Computes L2 (Euclidean) difference per pixel and overall score."""
        diff_map = (self.data_batch - self.recon_batch) ** 2  # Pixel-wise squared difference
        diff_score = torch.norm(self.data_batch - self.recon_batch, p=2, dim=(1, 2, 3)).cpu().numpy()  # (B,)
        if supress_output:
            diff_map = diff_map.cpu().numpy()
            diff_map = np.moveaxis(diff_map, 1, 3)  # (B, C, H, W) → (B, H, W, C)
            diff_map = np.mean(diff_map, axis=-1)  # Convert to grayscale
            diff_map = diff_map[0]  # Select first image in batch

        return diff_map, diff_score
    ####################################################################
    ####################################################################
    def get_ravi_difference(self):

        # abs_diff = torch.abs(self.recon_batch - self.data_batch)
        abs_diff = torch.abs(self.data_batch - self.recon_batch)  # (B, C, H, W)
        max_score = torch.max(abs_diff)
        # print(abs_diff.shape)
        abs_diff_nu = abs_diff.detach().cpu().numpy()
        # print(abs_diff.shape)

        diff_img = np.mean(np.moveaxis(abs_diff_nu, 1, 3), axis = -1)
        # print(diff_img.shape)
        diff_score = float(max_score.detach().cpu().numpy())

        return diff_img[0],diff_score
    ####################################################################
    ####################################################################
    def get_gaussian_difference(self,params=None):
        if params is not None:
            offset,sigma,quant = params['offset'],params['sigma'],params['quant']
        else:
            offset,sigma,quant = self.AS_OFFSET,self.AS_SIGMA,self.AS_QUANT
        # Compute Gaussian difference anomaly map
        gaus_score,gaus_image = get_anomaly_score(self.data_batch,self.recon_batch,
                          offset=offset,sigma=sigma,quant=quant, debug = False)
        return gaus_image[0],gaus_score[0]
    ####################################################################
    def get_ssim_dissimilarity_batch(self, ssim_type='ssim', params=None, supress_output=False):
        """
        Computes SSIM-based dissimilarity maps (1 - SSIM) and scores for a batch of images.

        Returns:
            np.ndarray: Batch of SSIM maps (B, H, W)
            np.ndarray: Batch of SSIM scores (B,)
        -------   MAIN SSIM RETURNS
        Returns
        -------
        mssim : float
            The mean structural similarity index over the image.
        grad : ndarray
            The gradient of the structural similarity between im1 and im2 [2]_.
            This is only returned if `gradient` is set to True.
        S : ndarray
            The full SSIM image.  This is only returned if `full` is set to True.
        """
        #########################################################################
        if params is not None:
            win_size, gaussian_weights,sigma = params['win_size'], params['gaussian_weights'],params['sigma']
        else:
            win_size,gaussian_weights,sigma = 3,True,1.5
        use_sample_covariance = False

        batch_size = self.data_batch.shape[0]
        maps, scores = [], []

        for i in range(batch_size):

            data_img = self.data_batch[i].cpu().numpy()
            recon_img = self.recon_batch[i].cpu().numpy()

            data_img = np.moveaxis(data_img, 0, 2)  # (C, H, W) → (H, W, C)
            recon_img = np.moveaxis(recon_img, 0, 2)  

            # Normalize images to [0,1] range
            data_img = (data_img - data_img.min()) / (data_img.max() - data_img.min() + 1e-8)
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)

            # Compute SSIM similarity score and similarity map
            ssim_score, ssim_map = ssim(
                data_img, recon_img, 
                data_range=1.0, 
                win_size=win_size,
                sigma = sigma,
                full=True, 
                gaussian_weights=gaussian_weights,
                use_sample_covariance=use_sample_covariance
            )
            
            if ssim_type == 'ssim-d':
                ssim_result_map = 1 - ssim_map  
                ssim_result_score = 1 - ssim_score  # Convert SSIM score to dissimilarity
            else:
                ssim_result_map = ssim_map
                ssim_result_score = ssim_score

            maps.append(ssim_result_map)
            scores.append(ssim_result_score)
        maps = np.array(maps)
        scores = np.array(scores)
        # print('ssim==+==++==++ ',maps.shape, scores.shape)

        # maps = np.moveaxis(maps, 3, 1)  # (B, C, H, W) → (B, H, W, C)
        # print('maps first',maps.shape)
        if supress_output:
            # diff_map = diff_map.cpu().numpy()
            # diff_map = np.moveaxis(diff_map, 1, 3)  # (B, C, H, W) → (B, H, W, C)
            maps = np.mean(maps, axis=-1)  # Convert to grayscale
            # print('maps',maps.shape)
            maps = maps[0]  # Select first image in batch
            # print('maps',maps.shape)
        # print('ssim==+==++==++ ',maps.shape, scores.shape)
        
        return maps, scores
    ###########################################################################################################
    ###########################################################################################################
    def plot_differences(self, diff_img,diff_scores,type=None,title=None,
                          id=0, cmap='inferno', figsize=(5, 2), sharex=True, sharey=True, title_direct='top'):    
        """
        Plots original, reconstructed, and difference images.

        Args:
            diff_img (torch.Tensor or np.ndarray): Difference image (B, C, H, W) or (B, H, W)
            id (int): Index of image to visualize
            cmap (str): Colormap for difference visualization
            figsize (tuple): Figure size
            sharex, sharey (bool): Whether to share axes
        """
        # print(diff_img.shape)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize, sharex=sharex, sharey=sharey)
        ax = axes.ravel()

        # Convert tensors to numpy before plotting
        original_img = np.moveaxis(self.data_batch[id].cpu().numpy(), 0, 2) * 0.5 + 0.5
        recon_img = np.moveaxis(self.recon_batch[id].cpu().numpy(), 0, 2) * 0.5 + 0.5

        # Handle different formats of diff_img (either PyTorch tensor or NumPy array)
        if isinstance(diff_img, torch.Tensor):
            diff_img = diff_img.cpu().numpy()

        # If the diff image has 3 channels (B, C, H, W), we move axis; otherwise, we use it directly
        # if diff_img.ndim == 4:
        #     diff_img = np.moveaxis(diff_img[id], 0, 2)  # Convert (C, H, W) → (H, W, C)
        # else:
        #     diff_img = diff_img[id]  # Directly use (H, W)

        # Plot images
        ax[0].imshow(original_img)
        ax[1].imshow(recon_img)
        if type is None:
            ax[2].imshow(diff_img, cmap=cmap)
            score_text = f'Score: {diff_scores:.6f}'
        elif type == 'ssim' or type =='ssim_all' or type =='ssim_d' or type =='l1' or type =='l2':
            if diff_img.ndim == 4:
                diff_img = np.moveaxis(diff_img[id], 0, 2)  # Convert (C, H, W) → (H, W, C)
            ax[2].imshow(np.mean(diff_img, axis = -1),cmap=cmap)
            score_text = f'Score: {diff_scores[id]:.6f}'
        else:
            # print('else   ----- ',diff_img.shape)
            # ax[2].imshow(diff_img, cmap=cmap)
            # 
            # print('in plot',diff_img.shape)
            ax[2].imshow(diff_img,cmap=cmap)
            score_text = f'Score: {diff_scores:.6f}'
            
        # print(diff_scores[id])
        
        text_x, text_y = 15, 15  # Adjust position

        # Add a rectangle (background for text)
        rect = patches.Rectangle(
            (text_x - 5, text_y - 10), 100, 20,  # Position and size
            linewidth=0, edgecolor=None, facecolor='white', alpha=0.8  # White background with transparency
        )
        ax[2].add_patch(rect)
        ax[2].text(text_x, text_y, score_text, color='black', fontsize=10, weight='bold')

        if title is not None:
            if title_direct=='top':
                ax[2].set_title(title)
            else:
                ax[0].set_ylabel(title)

        for axis in ax:
            axis.set_xticks([])
            axis.set_yticks([])

        plt.show()
    def plot_differences_all(self, diff_img_ls,type=None,title=None,
                          id=0, cmap='inferno', figsize=(5, 2), sharex=True, sharey=True, title_direct='top'):    
        """
        Plots original, reconstructed, and difference images.

        Args:
            diff_img (torch.Tensor or np.ndarray): Difference image (B, C, H, W) or (B, H, W)
            id (int): Index of image to visualize
            cmap (str): Colormap for difference visualization
            figsize (tuple): Figure size
            sharex, sharey (bool): Whether to share axes
        """
        # print(diff_img.shape)
        fig, axes = plt.subplots(nrows=1, ncols=len(diff_img_ls)+2, figsize=figsize, sharex=sharex, sharey=sharey)
        ax = axes.ravel()

        # Convert tensors to numpy before plotting
        original_img = np.moveaxis(self.data_batch[id].cpu().numpy(), 0, 2) * 0.5 + 0.5
        recon_img = np.moveaxis(self.recon_batch[id].cpu().numpy(), 0, 2) * 0.5 + 0.5

        # Handle different formats of diff_img (either PyTorch tensor or NumPy array)
        for idd,(diff_img_keys_,diff_img_,diff_scores_) in enumerate(diff_img_ls):
            diff_img = diff_img_['image']
            diff_img_key = diff_img_keys_['key']
            diff_scores = diff_scores_['score']
            print(idd, diff_img_keys, diff_img.shape, diff_scores)
            if isinstance(diff_img, torch.Tensor):
                diff_img = diff_img.cpu().numpy()

            # If the diff image has 3 channels (B, C, H, W), we move axis; otherwise, we use it directly
            # if diff_img.ndim == 4:
            #     diff_img = np.moveaxis(diff_img[id], 0, 2)  # Convert (C, H, W) → (H, W, C)
            # else:
            #     diff_img = diff_img[id]  # Directly use (H, W)

            # Plot images
            ax[0].imshow(original_img)
            ax[1].imshow(recon_img)
            if type is None:
                ax[2].imshow(diff_img, cmap=cmap)
                score_text = f'Score: {diff_scores:.6f}'
            elif type == 'ssim' or type =='ssim_all' or type =='ssim_d' or type =='l1' or type =='l2':
                if diff_img.ndim == 4:
                    diff_img = np.moveaxis(diff_img[id], 0, 2)  # Convert (C, H, W) → (H, W, C)
                ax[2].imshow(np.mean(diff_img, axis = -1),cmap=cmap)
                score_text = f'Score: {diff_scores[id]:.6f}'
            else:
                # print('else   ----- ',diff_img.shape)
                # ax[2].imshow(diff_img, cmap=cmap)
                # 
                # print('in plot',diff_img.shape)
                ax[2].imshow(diff_img,cmap=cmap)
                score_text = f'Score: {diff_scores:.6f}'
                
            # print(diff_scores[id])
            
            text_x, text_y = 15, 15  # Adjust position

            # Add a rectangle (background for text)
            rect = patches.Rectangle(
                (text_x - 5, text_y - 10), 100, 20,  # Position and size
                linewidth=0, edgecolor=None, facecolor='white', alpha=0.8  # White background with transparency
            )
            ax[2].add_patch(rect)
            ax[2].text(text_x, text_y, score_text, color='black', fontsize=10, weight='bold')

        if title is not None:
            if title_direct=='top':
                ax[2].set_title(title)
            else:
                ax[0].set_ylabel(title)

        for axis in ax:
            axis.set_xticks([])
            axis.set_yticks([])

        plt.show()
################################################################################################################################################
################################################################################################################################################
def get_time(suff='',verbose=True):
    if verbose:
        print(f'Exp {suff} time : ',format(datetime.now()))
    return datetime.now()

################################################################################################################################################
################################################################################################################################################

def train_pca_on_latent_space(data=None,n_components=2):
    if data is not None:
        data = data.detach().cpu().numpy()
        pca = PCA(n_components=2)
        pca.fit(data)
        data_transformed = pca.transform(data)
        return pca,data_transformed
def get_data_latent_inspection(dataloader,data_type='full',data_part_type='train', verbose=True):
    if data_type=='full':
        data_list = []
        lbls_list = []
        
        if data_part_type=='test_demo':
            filenames = []
            for dt, lbl, filename in dataloader:
                data_list.append(dt)
                lbls_list.append(lbl)
                filenames.extend(filename)  # Extend since filenames are typically a list of strings
        else:
            if verbose:
                dataloader = tqdm(dataloader)
            for dt, lbl in dataloader:
                data_list.append(dt)
                lbls_list.append(lbl)

        # Concatenate all batches along the first dimension (batch axis)
        data = torch.cat(data_list, dim=0)
        labels = torch.cat(lbls_list, dim=0)
        # filenames = torch.cat(filenames_list,  dim=0)
    else:
        data, labels,filenames = next(iter(dataloader))
    if data_part_type=='test_demo':
        del data_list,lbls_list
        return data,labels,filenames
    else:
        return data,labels,None
def get_explore_latent_space(train_loader,Enc,Dec,data_type='full',data_part_type='train', device='cpu'):
    if device=='cpu':
        Enc = Enc.cpu()
        Dec = Dec.cpu()
    else:
        Enc = Enc.cuda()
        Dec = Dec.cuda()

    torch.cuda.empty_cache()
    data,labels,filenames_test = get_data_latent_inspection(train_loader,data_type=data_type,data_part_type=data_part_type)
    if device=='cuda':
        data = data.to(device)
    print('passed data to Encoder')

    mu, logvar = Enc(data)
    print('passed data to Reparameterization')
    z = utmC.reparameterize(mu, logvar)
    
    print(data.shape, mu.shape, logvar.shape, z.shape)
    print('passed data to PCA')
    pca,z_transformed = train_pca_on_latent_space(data=mu,n_components=2)
    print('PCA transformation done')
    print(z.shape, z_transformed.shape)
    
    del data,filenames_test,mu,logvar,z,pca
    torch.cuda.empty_cache()
    
    return z_transformed,labels


#========================================================

# #---------------------------------------------------------------------------------------------------



@torch.no_grad()
def get_explore_latent_space_batched(
    loader,
    Enc,
    device="cuda",
    n_components=2,
    method="pca",                  # "pca" or "tsne"
    ipca_batch_size=4096,
    use_autocast=True,

    # t-SNE params
    tsne_perplexity=30,
    tsne_learning_rate="auto",
    tsne_init="pca",
    tsne_random_state=42,
    tsne_n_iter=1000,

    # optional: for standard PCA instead of IncrementalPCA
    use_incremental_pca=True,
):
    """
    Computes encoder means (mu) over the dataset in batches and returns
    low-dimensional embeddings using PCA or t-SNE.

    Args:
        loader: dataloader
        Enc: encoder model returning (mu, logvar)
        device: torch device
        n_components: number of output dimensions
        method: "pca" or "tsne"
        ipca_batch_size: batch size used in IncrementalPCA
        use_autocast: use mixed precision on CUDA

        tsne_perplexity: t-SNE perplexity
        tsne_learning_rate: t-SNE learning rate
        tsne_init: initialization for t-SNE
        tsne_random_state: random seed
        tsne_n_iter: number of t-SNE iterations

        use_incremental_pca: if True, use IncrementalPCA; else standard PCA

    Returns:
        z_transformed: (N, n_components) np.ndarray
        labels_all:    (N,) np.ndarray or None
    """
    Enc.eval()
    method = method.lower().strip()

    if method not in ["pca", "tsne"]:
        raise ValueError(f"Unsupported method='{method}'. Use 'pca' or 'tsne'.")

    # ------------------------------------------------------------
    # Helper to extract mu and labels
    # ------------------------------------------------------------
    def _extract_mu_and_labels(batch):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            y = None

        x = x.to(device, non_blocking=True)

        if use_autocast and torch.cuda.is_available() and "cuda" in str(device):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                mu, logvar = Enc(x)
        else:
            mu, logvar = Enc(x)

        mu_flat = mu.detach().float().reshape(mu.size(0), -1).cpu().numpy()

        if y is not None:
            if torch.is_tensor(y):
                y_np = y.detach().cpu().numpy()
            elif isinstance(y, (list, np.ndarray)):
                y_np = np.array(y)
            else:
                y_np = None
        else:
            y_np = None

        del x, mu, logvar
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return mu_flat, y_np

    # ============================================================
    # PCA branch
    # ============================================================
    if method == "pca":
        if use_incremental_pca:
            reducer = IncrementalPCA(
                n_components=n_components,
                batch_size=ipca_batch_size
            )

            have_labels = True

            # ---------- Pass 1: fit ----------
            for batch in tqdm(loader, desc="Fitting IncrementalPCA"):
                mu_flat, y_np = _extract_mu_and_labels(batch)
                reducer.partial_fit(mu_flat)
                if y_np is None:
                    have_labels = False

            # ---------- Pass 2: transform ----------
            z_list = []
            labels_all = [] if have_labels else None

            for batch in tqdm(loader, desc="Transforming with IncrementalPCA"):
                mu_flat, y_np = _extract_mu_and_labels(batch)
                z_chunk = reducer.transform(mu_flat)
                z_list.append(z_chunk)

                if have_labels and labels_all is not None:
                    if y_np is not None:
                        labels_all.append(y_np)
                    else:
                        have_labels = False
                        labels_all = None

            z_transformed = np.concatenate(z_list, axis=0)

            if have_labels and labels_all is not None:
                labels_all = np.concatenate(labels_all, axis=0)
            else:
                labels_all = None

            return z_transformed, labels_all

        else:
            # collect all mu first, then standard PCA
            mu_all = []
            labels_all = []
            have_labels = True

            for batch in tqdm(loader, desc="Collecting latent vectors for PCA"):
                mu_flat, y_np = _extract_mu_and_labels(batch)
                mu_all.append(mu_flat)

                if y_np is not None:
                    labels_all.append(y_np)
                else:
                    have_labels = False

            mu_all = np.concatenate(mu_all, axis=0)

            reducer = PCA(n_components=n_components)
            z_transformed = reducer.fit_transform(mu_all)

            if have_labels and len(labels_all) > 0:
                labels_all = np.concatenate(labels_all, axis=0)
            else:
                labels_all = None

            return z_transformed, labels_all

    # ============================================================
    # t-SNE branch
    # ============================================================
    elif method == "tsne":
        mu_all = []
        labels_all = []
        have_labels = True

        # collect all latent vectors first
        for batch in tqdm(loader, desc="Collecting latent vectors for t-SNE"):
            mu_flat, y_np = _extract_mu_and_labels(batch)
            mu_all.append(mu_flat)

            if y_np is not None:
                labels_all.append(y_np)
            else:
                have_labels = False

        mu_all = np.concatenate(mu_all, axis=0)

        reducer = TSNE(
            n_components=n_components,
            perplexity=tsne_perplexity,
            learning_rate=tsne_learning_rate,
            init=tsne_init,
            random_state=tsne_random_state,
            n_iter=tsne_n_iter,
        )
        z_transformed = reducer.fit_transform(mu_all)

        if have_labels and len(labels_all) > 0:
            labels_all = np.concatenate(labels_all, axis=0)
        else:
            labels_all = None

        return z_transformed, labels_all

#---------------------------------------------------------------------------------------------------

def plot_latent(latent_space=None, latent_space_test=None, labels_train=None, labels_test=None, labels=['train','test'],paths=None, normal_class_name='normal', ttl=''):
    classes = np.unique(np.concatenate((labels_train, labels_test)))  # Get all unique classes
    plt.figure(figsize=(8, 6))

    # Define markers for train and test
    # markers = { 'train': '*', 'test': 's' }  # Circles for train, squares for test

    # Plot train latent space
    for cls in np.unique(labels_train):
        mask_train = labels_train == cls
        count_test = np.sum(mask_train)
        # if paths.train_classes[cls] == normal_class_name:
            # color = "#0088FFFF"
        plt.scatter(latent_space[mask_train, 0], latent_space[mask_train, 1], 
                    label=f'{labels[0]} - {paths.test_classes[cls]} (n={count_test})',
                    alpha=0.7, marker='*',
                    color = 'blue' if paths.test_classes[cls]==normal_class_name else 'red')

    # Plot test latent space
    for cls in np.unique(labels_test):
        
        mask_test = labels_test == cls
        count_test = np.sum(mask_test)

        # if paths.test_classes[cls] == normal_class_name:
            # color = '#1104C0'
        # else:
            # color = '#FC2A00'
        colors = 'green' if paths.test_classes[cls]==normal_class_name else 'red'
        markers = '+' if paths.test_classes[cls]==normal_class_name else 'o'
        plt.scatter(latent_space_test[mask_test, 0], latent_space_test[mask_test, 1], 
                    label=f'{labels[1]} - {paths.test_classes[cls]} (n={count_test})', alpha=0.7, marker=markers,
                    color = colors)

    plt.legend()
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title(f'Train vs Test Latent Space Visualization {ttl}')
    plt.show()

    del latent_space, latent_space_test
################################################################################################################################################
################################################################################################################################################


def plot_latent_3d(latent_space_train=None, 
                   latent_space_test=None, 
                   labels_train=None, 
                   labels_test=None, 
                   paths=None, 
                   ttl = '',
                   labels=['train','Val'],
                   normal_class_name='normal'):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # -------------------------
    # Plot train data
    # -------------------------
    for cls in np.unique(labels_train):
        mask_train = labels_train == cls
        count_train = np.sum(mask_train)

        color = 'blue' if paths.test_classes[cls] == normal_class_name else 'red'

        ax.scatter(
            latent_space_train[mask_train, 0],
            latent_space_train[mask_train, 1],
            latent_space_train[mask_train, 2],
            label=f'{labels[0]} - {paths.test_classes[cls]} (n={count_train})',
            alpha=0.6,
            marker='*',
            color=color
        )

    # -------------------------
    # Plot test data
    # -------------------------
    for cls in np.unique(labels_test):
        mask_test = labels_test == cls
        count_test = np.sum(mask_test)

        color = 'green' if paths.test_classes[cls] == normal_class_name else 'red'
        marker = '+' if paths.test_classes[cls] == normal_class_name else 'o'

        ax.scatter(
            latent_space_test[mask_test, 0],
            latent_space_test[mask_test, 1],
            latent_space_test[mask_test, 2],
            label=f'{labels[1]} - {paths.test_classes[cls]} (n={count_test})',
            alpha=0.6,
            marker=marker,
            color=color
        )

    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Latent Dim 3')
    ax.set_title(f'3D Scatter of Latent Space  {ttl}')
    ax.legend()

    plt.show()


def plot_latent_3d_plotly_custom(
        latent_train=None,
        latent_test=None,
        labels_train=None,
        labels_test=None,
        class_names=None,
        normal_class_name="normal",
        labels=['train','Val'],
        title="Latent Space (Train + Test)"
    ):

    fig = go.Figure()

    # -------------------------
    # TRAIN
    # -------------------------
    if latent_train is not None:
        for cls in np.unique(labels_train):
            mask = labels_train == cls
            class_name = class_names[int(cls)] if class_names is not None else str(cls)

            color = "blue" if class_name == normal_class_name else "red"
            marker_symbol = "cross"  # same as your function

            fig.add_trace(go.Scatter3d(
                x=latent_train[mask, 0],
                y=latent_train[mask, 1],
                z=latent_train[mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    symbol=marker_symbol,
                    opacity=0.7
                ),
                name=f"{labels[0]} - {class_name} (n={np.sum(mask)})"
            ))

    # -------------------------
    # TEST
    # -------------------------
    if latent_test is not None:
        for cls in np.unique(labels_test):
            mask = labels_test == cls
            class_name = class_names[int(cls)] if class_names is not None else str(cls)

            color = "green" if class_name == normal_class_name else "red"
            marker_symbol = "cross" if class_name == normal_class_name else "circle"

            fig.add_trace(go.Scatter3d(
                x=latent_test[mask, 0],
                y=latent_test[mask, 1],
                z=latent_test[mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color,
                    symbol=marker_symbol,
                    opacity=0.8
                ),
                name=f"{labels[1]} - {class_name} (n={np.sum(mask)})"
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Latent Dim 1",
            yaxis_title="Latent Dim 2",
            zaxis_title="Latent Dim 3"
        ),
        legend_title="Split / Class"
    )

    fig.show()

#==========================================================================
def plot_latent_surface(latent_space, labels):
    # labels = labels.detach().cpu().numpy()
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y coordinates from latent space
    x = latent_space[:, 0]
    y = latent_space[:, 1]
    z = labels.astype(float)  # Convert labels to float for visualization

    # Create a grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic')  # Interpolation

    # Plot trisurf with color mapping based on labels
    surf = ax.plot_trisurf(x, y, z, cmap='tab10', alpha=0.9, edgecolor='none')

    # Add colorbar for reference
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Class Labels')

    # Labels and title
    ax.set_xlabel('Latent Dim 1')
    ax.set_ylabel('Latent Dim 2')
    ax.set_zlabel('Interpolated Feature')
    ax.set_title('3D Surface of Latent Space')

    plt.show()
################################################################################################################################################
def get_l1_difference(real_images,recon_batch,supress_output=False):
        """Computes L1 (absolute) difference between original and reconstructed images."""
        diff_map = torch.abs(real_images - recon_batch)  # (B, C, H, W)
        diff_score = diff_map.mean(dim=(1, 2, 3)).cpu().numpy()  # (B,)
        if supress_output:
            diff_map = diff_map.cpu().numpy()
            diff_map = np.moveaxis(diff_map, 1, 3)  # (B, C, H, W) → (B, H, W, C)
            diff_map = np.mean(diff_map, axis=-1)  # Convert to grayscale
            diff_map = diff_map[0]  # Select first image in batch
        del diff_map
        return diff_score

################################################################################################################################################

def get_anomaly_score_ravi(real_images, reconstructed_images, quantile=1.0):
    abs_diffs = torch.abs(real_images - reconstructed_images)
    mean_diffs = abs_diffs.mean(dim=1).cpu().detach().numpy()
    mean_diffs = mean_diffs.reshape(mean_diffs.shape[0], -1)
    scores = np.quantile(mean_diffs, q=quantile, axis=1)
    return scores

################################################################################################################################################





#####################
# PLOTTING CON
##############################
# PLOT CONTOUR FROM MASK ON IMAGE
########################

def get_contoured_image(image,paths,subgroup,subgroup_mask='mask',mask_image_name = 3015, thickness=2, contour_color=(255, 0, 0), verbose=False):
    # print('TEST ---------> ', params.subgroup)


    if paths.dataset_type == 'fronttop':
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{subgroup_mask}_ext.png"
    else:
        mask_path = f"{paths.dataset_type}_{mask_image_name}_{subgroup}_{subgroup_mask}.png"

    component_mask_path = os.path.join(paths.mask_dir, mask_path)
    if not os.path.exists(component_mask_path):
        print(f"Component mask not found: {component_mask_path}")
    component_mask = np.array(Image.open(component_mask_path).convert("L"))#.resize(image.size, Image.LANCZOS)
    # binary_mask = component_mask#[:,:,0].astype(np.uint8)
    contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, contours, -1, contour_color, thickness)  # (0, 255, 255) corresponds to yellow, 2 is the thickness
    if verbose:
        print(mask_path)
    return contoured_image,component_mask

##########################################################################################################################################
##########################################################################################################################################
# get_parameters_by_experiment
##########################################################################################################################################
##########################################################################################################################################
def get_parameters_by_experiment(params, verbose=False):
    if params.exp_type =='E1':
        #    SLOW and NOT WORKING
        params.learning_rate_enc_dec    = 0.001  # 
        params.learning_rate_dis        = 0.0001 # should be 10x smaller that enc_dec to improve convergence
        #-----------------------------------------
        params.beta_kl                  = 0.0001
        params.beta_gan                 = 0.001
        #-----------------------------------------
        params.reconstruction_loss_fn   = 'Perceptual'    # CHANGED
        params.adversarial_loss_fn      = 'BCWithLogits'   
    elif params.exp_type =='E2':
        params.learning_rate_enc_dec    = 0.001  # 
        params.learning_rate_dis        = 0.0001 # should be 10x smaller that enc_dec to improve convergence
        #-----------------------------------------
        params.beta_kl                  = 0.0001
        params.beta_gan                 = 0.001
        #-----------------------------------------
        params.reconstruction_loss_fn   = 'MSE'
        params.adversarial_loss_fn      = 'BCWithLogits'
    elif params.exp_type =='E3':
        params.learning_rate_enc_dec    = 0.001  # 
        params.learning_rate_dis        = 0.0001 # should be 10x smaller that enc_dec to improve convergence
        #-----------------------------------------
        params.beta_kl                  = 0.0001
        params.beta_gan                 = 0.0001      # CHANGED
        #-----------------------------------------
        params.reconstruction_loss_fn   = 'MSE'
        params.adversarial_loss_fn      = 'BCWithLogits'
        #-----------------------------------------
    if verbose:
        print('Parameters for experiment', params.exp_type, 'are set:')
        print('Learning rate enc_dec:', params.learning_rate_enc_dec)
        print('Learning rate dis:', params.learning_rate_dis)
        print('Beta KL:', params.beta_kl)
        print('Beta GAN:', params.beta_gan)
        print('Reconstruction loss function:', params.reconstruction_loss_fn)
        print('Adversarial loss function:', params.adversarial_loss_fn)
    return params
########################################################################################################
def get_dataset_version(paths,params, dataset_version='V6',dataset_type='fronttop',subgroup='RoboArm',mask_image_name=3015, verbose=False):

    paths.dataset_version      = dataset_version #'V6'
    paths.path_datasets        =  os.path.join(paths.path_datasets_main,paths.dataset_version)
    paths.dataset_type         =  dataset_type #'fronttop'
    paths.train_dirn           = 'train'
    paths.mask_dirn            = 'masks'
    paths.test_sel_dirn        = 'test'
    paths.mask_image_name      = mask_image_name #3015

    # params.subgroup          = 'PLeft'
    # params.subgroup          = 'PRight'
    # params.subgroup          = 'ConvBelt'
    # params.subgroup            = subgroup #'RoboArm'

    # params.subgroup_mask = 'box'
    params.subgroup_mask = 'mask'
    # params.map_coor = map_coor
    params.aug_type = f'auto-{params.subgroup_mask}'

    ##################################################
    paths.path_dataset_selected = os.path.join(paths.path_datasets,paths.dataset_type)
    paths.train_dir             = os.path.join(paths.path_dataset_selected,paths.train_dirn)
    paths.test_dir              = os.path.join(paths.path_dataset_selected,paths.test_sel_dirn)
    paths.mask_dir              = os.path.join(paths.path_dataset_selected,paths.mask_dirn)

    ##################################################
    if params.subgroup =='full':
        paths.train_dirn_processed    = 'train_processed_full'
        paths.train_dir = os.path.join(paths.path_dataset_selected, paths.train_dirn)

        paths.train_dir_processed_subgroup = os.path.join(paths.path_dataset_selected, paths.train_dirn_processed)
    else:


        paths.train_dirn_processed    = 'train_processed'
        paths.test_sel_dirn_processed = 'test_processed'

        paths.train_dir = os.path.join(paths.path_dataset_selected, paths.train_dirn)
        paths.test_dir = os.path.join(paths.path_dataset_selected, paths.test_sel_dirn)

        paths.train_dir_processed = os.path.join(paths.path_dataset_selected, paths.train_dirn_processed)
        paths.test_dir_processed = os.path.join(paths.path_dataset_selected, paths.test_sel_dirn_processed)

        paths.train_dir_subgroup = os.path.join(paths.train_dir, params.subgroup)
        paths.test_dir_subgroup = os.path.join(paths.test_dir, params.subgroup)

        paths.train_dir_processed_subgroup = os.path.join(paths.train_dir_processed, params.subgroup)
        paths.test_dir_processed_subgroup = os.path.join(paths.test_dir_processed, params.subgroup)
    if verbose:
        print('-'*120)
        print(f'Paths SETUP COMPLETED for {params.subgroup}')
    return paths,params

########################################################################################################
def get_GPU_device(verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f'{"Using device:":<20} {device}')
        print(f'{"GPU Count:":<20} {torch.cuda.device_count()}')
        print(f'{"GPU NAME:":<20} {torch.cuda.get_device_name(0)}')
        print('-'*120)
    return device



########################################################################################################
########################################################################################################
## FUNCTIONS FOR TESTING VIDEO DATASET LOADING
########################################################################################################
########################################################################################################

class StreamVideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform

        # Open video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames

    def __len__(self):
        return self.total_frames  # Number of frames in video

    def __getitem__(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set frame position
        ret, frame = self.cap.read()  # Read frame

        if not ret:
            raise ValueError(f"Error reading frame {idx} from {self.video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Apply transformations if needed
        if self.transform:
            frame = self.transform(frame)
        
        return frame  # Shape: (C, H, W)

    def __del__(self):
        self.cap.release()  # Release video capture on deletion
##################################################################################
class StreamVideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform

        # Open video file to get total frame count
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {video_path}")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  
        cap.release()  # Close it here, and reopen in __getitem__()

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.video_path)  # Open a new instance
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()  # Close it immediately after reading

        if not ret:
            raise ValueError(f"Error reading frame {idx} from {self.video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Apply transformations if needed
        if self.transform:
            frame = self.transform(frame)

        return frame,idx


##################################################################################
##################################################################################
## MaskedCrop --> Custom masking based on subgroup
##################################################################################
##################################################################################
class MaskedCrop:
    """Custom masking based on subgroup"""
    def __init__(self, subgroup=None,mask=None, fill_color=(0, 0, 0),verbose=False):
        self.subgroup = subgroup  # Store subgroup for cropping
        self.fill_color = fill_color  # Color to fill the masked region
        self.mask = mask
        self.verbose = verbose
        
        # print('mask',mask.dtype, mask.shape)
        assert len(mask.shape)==2 and mask.dtype==bool
        # compute the bounding box of the mask
        rows, cols = np.where(mask)
        self.x1, self.x2 = np.min(cols), np.max(cols)
        self.y1, self.y2 = np.min(rows), np.max(rows)
        # create the 3-channel mask for boolean operations
        cropped_mask_1ch = mask[self.y1:self.y2+1, self.x1:self.x2+1]  # Shape: (h, w)
        self.cropped_mask = np.stack([cropped_mask_1ch]*3, axis=-1)  # Shape: (h, w, 3)

    def __call__(self, image):

        # self.map_coor   
        # print(f'self.map_coor[self.subgroup :::: {self.map_coor[self.subgroup]}')
        # return
        if self.subgroup is None:
            return image
        else:
            if self.verbose:
                print('computed ',self.subgroup, self.x1,self.x2,self.y1,self.y2)
            image_np = np.array(image)  # Convert to NumPy array for masking
            image_np = image_np[self.y1:self.y2+1, self.x1:self.x2+1, :]  # Crop the image
            image_np *= self.cropped_mask  # Element-wise multiplication with the mask
            image = Image.fromarray(image_np.astype('uint8'))
            
        return image
    
    def uncrop(self, image_np):
        assert image_np.shape[0] == self.cropped_mask.shape[0] and image_np.shape[1] == self.cropped_mask.shape[1], "Image shape does not match mask shape"
        # reapply the boolean mask to image_np
        image_np = image_np * self.cropped_mask
        # rescale back to the original image size
        no_channels = image_np.shape[2]
        orig_image_np = np.zeros((self.mask.shape[0], self.mask.shape[1], no_channels), dtype=image_np.dtype)
        orig_image_np[self.y1:self.y2+1, self.x1:self.x2+1, :] = image_np
        return orig_image_np
##################################################################################
##################################################################################
## CustomCrop --> Custom masking based on subgroup
##################################################################################
##################################################################################
class CustomCrop:
    """Custom masking based on subgroup"""
    def __init__(self, subgroup=None,map_coor=None, fill_color=(0, 0, 0)):
        self.subgroup = subgroup  # Store subgroup for cropping
        self.fill_color = fill_color  # Color to fill the masked region
        self.map_coor = map_coor

    def __call__(self, image):

        self.map_coor   
        # print(f'self.map_coor[self.subgroup :::: {self.map_coor[self.subgroup]}')
        # return
        if self.subgroup is None:
            return image
        else:
            x, y, w, h = self.map_coor[self.subgroup]
            image = image.crop((x, y, x+w, y+h))
        return image
    

##################################################################################
##################################################################################
## load_tuned_threshold --> Load tuned threshold from a CSV file for anomaly detection
## This function reads a CSV file containing anomaly metrics, sorts them by binormal AUC,
## and returns the best anomaly score, its index, and the corresponding threshold.
## If the CSV file does not exist, it raises a FileNotFoundError.
##################################################################################
##################################################################################

def load_tuned_threshold(paths, params,component=None, VERBOSE=False):
    if component is None:
        return None, None, None, None
    
    save_dir = f'{paths.path_codes_main}/test/threshold/plots_anomaly_results-{component}'
    csv_path = os.path.join(save_dir, "anomaly_metrics.csv")
    if os.path.exists(csv_path):
        # os.remove(csv_path)
        df = pd.read_csv(csv_path, index_col=0)
        df['original_index'] = df.index
        df['original_position'] = range(len(df))

        sorted_df = df.sort_values(by='binormal_AUC', ascending=False)
        if VERBOSE:
            print('subgroup :', component)
        # df.head()
        best_anom_score, \
            best_anomaly_score_index,\
                best_threshold = sorted_df.iloc[0].name, sorted_df.iloc[0].original_position,sorted_df.iloc[0].Threshold

        return best_anom_score, best_anomaly_score_index, best_threshold, sorted_df
    else:
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
##################################################################################
##################################################################################
## plot_intermedidate_figs --> Plot intermediate figures for anomaly detection
## This function visualizes the input image, reconstructed image, and difference image 
## with anomaly scores and thresholds. It saves the figure to a specified path.
## The function takes various parameters including input image, reconstructed image,
## difference image, component name, index, video save path, normalization score,
## color score, normalized score text, threshold text, and batch scores.

##################################################################################
##################################################################################
def plot_intermedidate_figs(inpu,recon,diff_img,component,idx,video_savepath_part,
                            norm_score,color_score,normscore_text,thres_text,batch_scores,
                            destroy_figs = False):
    fig, axxes = plt.subplots(1, 3, figsize=(12, 6))
    ax1,ax2,ax3 = axxes[0],axxes[1],axxes[2]
    ax1.imshow(inpu)
    ax1.set_title(f'Frame {idx}')
    ax2.imshow((recon* 0.5 + 0.5).squeeze(0).cpu().numpy().transpose(1, 2, 0))
    ax2.set_title('Reconstructed')
    

    ax3.imshow(diff_img, cmap=threshold_cmap if norm_score<1 else threshold_cmap_unexpected,
                vmin=0, vmax=2*selected_threshold)
    
    # ax3.set_title(f'{component} AM')
    
    # print(component,'[',np.min(comp_map_list[component]),'~',np.max(comp_map_list[component]),']')
    ax3.set_title(f"{component} - {thres_text}\nN-Score: {normscore_text}× \n score: {batch_scores:.4f}", 
                    color=color_score, 
                    fontsize=10)
    for ax in axxes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(f'{video_savepath_part}/{component}_inter_anomaly_map_{idx}.png', dpi=300, bbox_inches='tight')
    if destroy_figs:
        plt.close(fig)
    plt.show()

#################################################################################################################################################
def plot_anomalymaps(diff_img,component,idx,video_savepath_part,color_score,normscore_text,norm_score,threshold_cmap_ls,fontsize=10,destroy_figs = False):
    threshold_cmap,threshold_cmap_unexpected = threshold_cmap_ls
    fig = plt.figure(figsize=(4, 4))

    plt.imshow(diff_img, cmap=threshold_cmap if norm_score<1 else threshold_cmap_unexpected, vmin=0, vmax=2*selected_threshold)
    plt.title(f"{component} - N-Score: {normscore_text}", 
              color=color_score, 
              fontsize=fontsize)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(f'{video_savepath_part}/{component}_anommap_{idx}.png', dpi=150, bbox_inches='tight')
    if destroy_figs:
        plt.close(fig)
    plt.show()
# plot_anomalymaps(diff_img_np,component,idx,video_savepath_part,
#                  color_score,normscore_text,thres_text,
#                  fontsize=15,
#                  destroy_figs = False)

#################################################################################################################################################

def plot_component_boundary(full_anomaly_map_np,base_mask,norm_score,ax,threshold_cmap, verbose=False):
    boundary_color = 'red' if norm_score > 0.5 else 'black'
    # 2. Overlay the anomaly map with transparency
    ax.imshow(full_anomaly_map_np, cmap=threshold_cmap, vmin=0, vmax=2)
    boundaries = find_boundaries(np.array(base_mask), mode='thick')  # Try 'inner'/'outer'
    # 4. Thicken boundaries (optional)
    thick_boundaries = dilation(boundaries, square(1))  # Adjust '3' for thickness

    # 5. Create a colored contour overlay (RGBA)
    contour_overlay = np.zeros((*np.array(base_mask).shape, 4))  # RGBA image

        # Handle both named colors and RGB/RGBA tuples
    if isinstance(boundary_color, str):
        # Convert named color to RGBA (matplotlib colors format)
        from matplotlib.colors import to_rgba
        rgba_color = to_rgba(boundary_color)
    else:
        # Assume it's already an RGBA tuple
        rgba_color = boundary_color
    
    contour_overlay[thick_boundaries] = rgba_color

    # contour_overlay[thick_boundaries] = [1, 0, 0, 1]  # Red, fully opaque

    # 6. Overlay contours only
    ax.imshow(contour_overlay)
    if verbose:
        print('norm_score',norm_score)
    return ax
###################################################################################################################################################

def plot_comparison(full_anomaly_map, components_masks, comp_map_list, components_ls, 
                   batch_scores_component, component_threshold, video_savepath, idx,
                   threshold_cmap,
                   VERBOSE=True):
    fig, axes = plt.subplots(1, len(components_ls)+1, figsize=(12, 6))
    ax = axes[0]
    
    # First show the base anomaly map
    ax.imshow(full_anomaly_map, cmap=threshold_cmap, vmin=0, vmax=2)
    
    # Create a single overlay for all boundaries
    contour_overlay = np.zeros((*full_anomaly_map.shape[:2], 4))  # RGBA image
    
    for component in components_ls:
        norm_score = batch_scores_component[component] / component_threshold[component]
        
        # Get boundary
        boundaries = find_boundaries(np.array(components_masks[component]), mode='thick')
        # 4. Thicken boundaries (optional)
        thickness = 3  if norm_score > 1 else 1
        thick_boundaries = dilation(boundaries, square(thickness)) 
        
        # Set color based on norm_score
        boundary_color = (1, 0, 0, 1) if norm_score > 1 else (0, 0, 0, 1)  # Red or Black
        
        # Add to overlay
        contour_overlay[thick_boundaries] = boundary_color
        
        if VERBOSE:
            print(f"{component} boundary added with {'red' if norm_score > 1 else 'black'} color")
    
    # Draw all boundaries at once
    ax.imshow(contour_overlay)
    ax.set_title(f"Combined AM - Frame{idx}", pad=20)
    ax.axis('off')
    
    # Plot individual components
    for i, component in enumerate(components_ls):
        ax = axes[i+1]
        norm_score = batch_scores_component[component] / component_threshold[component]
        color_score = 'red' if norm_score > 1 else 'black'
        normscore_text = f'\\textbf{{{norm_score:.4f}}}' if norm_score > 1 else f'{norm_score:.4f}'
        thres_text = fr"$\mathbf{{\tau={component_threshold[component]:.4f}}}$"

        ax.imshow(comp_map_list[component], cmap=threshold_cmap, alpha=0.7, vmin=0, vmax=2*selected_threshold)
        
        if VERBOSE:
            print(component,'[',np.min(comp_map_list[component]),'~',np.max(comp_map_list[component]),']')
        ax.set_title(f"{component} - {thres_text}\nN-Score: {normscore_text}×\n score: {batch_scores_component[component]:.4f}", 
                    color=color_score, 
                    fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{video_savepath}/compared_{idx}.png', dpi=300, bbox_inches='tight')
    plt.show()

#################################################################################################################################################
def plot_final_results_v2(image, full_anomaly_map, components_masks, comp_map_list, components_ls, 
                      batch_scores_component, component_threshold, video_savepath, idx,threshold_cmap,
                      fontsize=10, plot_title=True, destroy_figs=False, VERBOSE=True, dpi=300):
    """
    Displays anomaly detection results with component status text.
    - Left-aligned for most components
    - Center-aligned specifically for ConvBelt
    """
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Prepare boundary overlay (will be used for both subplots)
    contour_overlay = np.zeros((*full_anomaly_map.shape[:2], 4))
    
    for component in components_ls:
        norm_score = batch_scores_component[component] / component_threshold[component]
        is_anomalous = norm_score > 1
        
        # Create boundaries
        boundaries = find_boundaries(np.array(components_masks[component]), mode='thick')
        thickness = 4 if is_anomalous else 1
        thick_boundaries = dilation(boundaries, square(thickness))
        boundary_color = (1, 0, 0, 1) if is_anomalous else (0, 0, 0, 1)
        contour_overlay[thick_boundaries] = boundary_color
        
        # Get the bounding box coordinates
        y, x = np.where(components_masks[component])
        if len(y) > 0 and len(x) > 0:
            # Special handling for ConvBelt
            if component == 'ConvBelt':
                center_x = np.mean(x)  # Center of the component
                bottom_y = np.max(y) - 37  # Below the component
                text_align = 'center'
            elif component == 'PLeft':
                center_x = np.min(x) + (35 if is_anomalous else 30)
                bottom_y = np.max(y) + 8
                text_align = 'left'
            elif component == 'PRight':
                center_x = np.min(x) + (38 if is_anomalous else 40)
                bottom_y = np.max(y) + 8
                text_align = 'left'

            elif component == 'RoboArm':
                center_x = np.max(x) + (305 if is_anomalous else 180)
                bottom_y = np.min(y) + 15
                text_align = 'right'
            
            # Format component info
            status = "UNEXPECTED" if is_anomalous else "normal"
            text = f"{status}: {norm_score:.2f}"
            
            # Common text properties
            text_props = {
                'x': center_x,
                'y': bottom_y,
                's': text,
                'color': 'red' if is_anomalous else 'black',
                'fontsize': fontsize,
                'ha': text_align,  # Dynamic alignment
                'va': 'top',
                'bbox': {
                    'facecolor': 'white',
                    'alpha': 0.7,
                    'edgecolor': 'red' if is_anomalous else 'black',
                    'boxstyle': 'round,pad=0.2'
                }
            }
            
            # Add text to BOTH subplots
            axes[0].text(**text_props)
            axes[1].text(**text_props)
        
        if VERBOSE and idx == 0:
            print(f"{component:<10}: (tau: {component_threshold[component]:<10.3f})(Score: {norm_score:<10.3f} {'UNEXPECTED' if is_anomalous else 'normal'})")
    
    # ===== Left Plot: Original Image =====
    axes[0].imshow(image)
    axes[0].imshow(contour_overlay, alpha=0.7)
    if plot_title:
        axes[0].set_title("Original Image", pad=20)
    
    # ===== Right Plot: Anomaly Map =====
    axes[1].imshow(full_anomaly_map, cmap=threshold_cmap, vmin=0, vmax=2)
    axes[1].imshow(contour_overlay , alpha=0.7)
    if plot_title:
        axes[1].set_title(f"Anomaly Map - Frame {idx}", pad=20)
    
    # Remove ticks from both subplots
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{video_savepath}/full_{idx}.png', dpi=dpi, bbox_inches='tight')
    
    if destroy_figs and not idx == 0:
        plt.close(fig)
    plt.show()




#######################
################################################
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)  # Same as sklearn's accuracy_score
def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)
###############################################################################################################################################################
###  -----------------------------------------------------------------------------------------------------------------------------------------------------  ###
###  SEGMENTATINO RELATED PARTS                                                                                                                             ###
###  -----------------------------------------------------------------------------------------------------------------------------------------------------  ###
###############################################################################################################################################################
# -------------------------------------------------
# Function --> get_allowed_segment_in_safety_area
# -------------------------------------------------

def get_allowed_segment_in_safety_area():
    allowed_segment_in_safety_area = {
        'PLeft': [
            ('Boxes','#4DA43C', (77,164,60), 'Boxes'),
            ('PLeft','#3C66A4', (60, 102, 164),'PLeft'),
            ('RoboArm','#A43C43', (164 , 60 ,67), 'Robot (UR20)'),
            ('Palletizer','#4DA43C', (120,60,164),'Palletizer'),
        ],
        'PRight': [
            ('Boxes','#4DA43C', (77,164,60), 'Boxes'),
            ('PRight','#3C66A4', (60,102,164)),
            ('RoboArm','#A43C43', (164 , 60 ,67), 'Robot (UR20)'),
            ('Palletizer','#4DA43C', (120,60,164),'Palletizer'),
            
        ],
        'RoboArm': [
            ('RoboArm','#A43C43', (164 , 60 ,67), 'Robot (UR20)'),
            ('Boxes','#4DA43C', (77,164,60), 'Boxes'),
            ('Palletizer','#4DA43C', (120,60,164),'Palletizer'),
            ('ConvBelt','#E1DB31', (224,219,49),'Conveyor Belt'),
        ],
        'ConvBelt': [
            ('ConvBelt','#E1DB31', (224,219,49),'Conveyor Belt'),
            ('Boxes','#4DA43C', (77,164,60), 'Boxes'),
            ('Palletizer','#4DA43C', (120,60,164),'Palletizer'),

        ]
    }
    return allowed_segment_in_safety_area
# -------------------------------------------------
# Function --> component_to_colormap
# -------------------------------------------------
def get_component_to_colormap():
    component_to_colormap = {'bg':('#000000',(0,0,0),'Background' ),
                            'PLeft':('#3C66A4',(60,102,164),'Pallet Left' ),
                        'PRight':('#3C66A4',(60,102,164),'Pallet Right' ),
                        'Boxes':('#4DA43C',(77,164,60),'Boxes' ),
                        'RoboArm':('#A43C43',(164 , 60 ,67),'Robot (UR20)'), 
                        'ConvBelt':('#E1DB31',(224,219,49),'Conveyor Belt'),
                        'Palletizer':('#783CA4',(120,60,164),'Palletizer'),
                        'Unexpected_person':('#EF4DDF',(239,77,223),'Unexpected_person'),
                        }
    return component_to_colormap

# -------------------------------------------------
# Function --> get_disallowed_color_for_object
# -------------------------------------------------

def get_disallowed_color_for_object(anomalous_type=None, verbose=True, mode='old'):
    if anomalous_type is None:
        assert 'None'
    if anomalous_type=='unexpected_person':
        if mode == 'old':
            disallowed_color = [239, 77, 223]
            disallowed_color_meta = {'code_rgb':(239, 77, 223),
                                 'code_hex':('#FF00FF'),
                                 'description':'Unexpected_person'}
        else:
            disallowed_color = [248, 150, 241]

            disallowed_color_meta = {'code_rgb':(248, 150, 241),
                                    'code_hex':('#F896F1'),
                                    'description':'Unexpected_person'}
    elif anomalous_type=='unexpected_object_fall':
        
        disallowed_color = [239, 77, 223]
        disallowed_color_meta = {'code_rgb':(239,77,223),
                                 'code_hex':('#FF00FF'),
                                 'description':'Unexpected_person'}
    if verbose:
        print(f'anomalous_type : {anomalous_type}')
        print(f'disallowed_color : {disallowed_color}')
        print(f'disallowed_color : {disallowed_color_meta["code_hex"]}')
    return disallowed_color, disallowed_color_meta
# -------------------------------------------------
# Function --> fast_unique_rows_with_counts
# -------------------------------------------------
def fast_unique_rows_with_counts(arr):
    arr_view = np.ascontiguousarray(arr).view(
        np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    )
    _, idx, counts = np.unique(arr_view, return_index=True, return_counts=True)
    unique_colors = arr[idx]
    return unique_colors, counts

# -------------------------------------------------
# Function --> segments_to_color
# -------------------------------------------------

def segments_to_color(real_batch_segments, verbose = True):
    rounded_colors = real_batch_segments.reshape((-1, 3)).astype(np.uint8)

    # Find unique colors and their counts
    # colors_found, counts = np.unique(rounded_colors, axis=0, return_counts=True)
    colors_found, counts = fast_unique_rows_with_counts(rounded_colors)
    # Sort by count (descending order)
    sorted_indices = np.argsort(counts)[::-1]
    colors_found = colors_found[sorted_indices]
    counts = counts[sorted_indices]
    colors_count_dict = {tuple(color): int(count) 
                        for color, count in zip(colors_found, counts)}
    if verbose:
        print(f'Colors found --> {len(counts)}')
    return colors_found, counts,rounded_colors,colors_count_dict
# -------------------------------------------------
# Function --> print_segments_to_bins
# -------------------------------------------------

def print_segments_to_bins(rounded_colors,component_to_colormap,colors_count_dict):
    # plt.imshow(real_batch_segments, cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # fig, axes = plt.subplots(1, len(component_to_colormap), figsize=(15, 1.5))


    for cid,(component, (color_hex, color_rgb, label)) in enumerate(component_to_colormap.items()):
        
        # Example of how to use the color in a plot
        # plt.figure(figsize=(2, 2))
        
        if color_rgb not in colors_count_dict:
            print(color_rgb, ' COLOR NOT FOUND')
            continue;
        # axes[cid].fill_between([0, 1], 0, 1, color=np.array(color_rgb)/255.0)
        # axes[cid].set_title(component)
        count = colors_count_dict[color_rgb]
        ttl = f'{label} \n {count} \n {f"Perntg : {100*count/rounded_colors.shape[0]:.2f} %"}'
        print(ttl)

# -------------------------------------------------
# Function --> segments_to_bins
# -------------------------------------------------
def segments_to_bins(rounded_colors, component_to_colormap, colors_count_dict, verbose=True):
    total_pixels = rounded_colors.shape[0]
    n_components = len(component_to_colormap)
    if verbose:
        print(f'Total Pixels: {total_pixels}, Components: {n_components}')
    
    fig, axes = plt.subplots(1, n_components, figsize=(max(6, n_components * 2), 1.5))

    if n_components == 1:
        axes = [axes]  # Ensure it's iterable

    for cid, (component, (color_hex, color_rgb, label)) in enumerate(component_to_colormap.items()):
        color_rgb_tuple = tuple(color_rgb)

        count = colors_count_dict.get(color_rgb_tuple)
        if count is None:
            # Color not found, leave the subplot blank or gray
            axes[cid].fill_between([0, 1], 0, 1, color='gray')
            axes[cid].set_title(f"{component}\nNot Found", fontsize=10)
        else:
            axes[cid].fill_between([0, 1], 0, 1, color=np.array(color_rgb) / 255.0)
            axes[cid].set_title(component, fontsize=10)
            text_color = 'white' if np.mean(color_rgb) < 128 else 'black'
            ttl = f'{label}\n{count}\nPerntg: {100 * count / total_pixels:.2f}%'
            axes[cid].text(0.5, 0.5, ttl, ha='center', va='center', fontsize=11, color=text_color)

        axes[cid].set_xticks([])
        axes[cid].set_yticks([])

    plt.subplots_adjust(wspace=0.3, hspace=0.1)
    plt.tight_layout()
    plt.show()



# -------------------------------------------------
# Function --> rgb_to_hex
# -------------------------------------------------
def rgb_to_hex(rgb):
    return '{:02X}{:02X}{:02X}'.format(*rgb)

# -------------------------------------------------
# Function --> plot_found_color
# -------------------------------------------------
def plot_found_color(colors_found,counts,rounded_colors, verbose=True):
    if verbose:
        print('Colors found in segments (sorted by pixel count) --> ', len(colors_found))
        for color, count in zip(colors_found, counts):
            print(f'RGB: ({color[0]:3d}, {color[1]:3d}, {color[2]:3d}) | Pixels: {count:7d} | '
                f'Percentage: {100*count/rounded_colors.shape[0]:.2f}%')

    # Visualize the colors (sorted)
    plt.figure(figsize=(20, 1.5))
    for i, (color, count) in enumerate(zip(colors_found, counts)):
        plt.fill_between([i, i+1], 0, 1, color=color/255.0, edgecolor='black')
        ttl = f'{color[0]},{color[1]},{color[2]}\n{rgb_to_hex(color)}\n{count} \n {f"{100*count/rounded_colors.shape[0]:.2f} %"}'
        plt.text(i+0.5, 0.5, f"{ttl}", ha='center', va='center', color='white' if np.mean(color) < 128 else 'black')
        # plt.text(i+0.5, 0.5, f"{count:,}", ha='center', va='center', color='white' if np.mean(color) < 128 else 'black')
    plt.xlim(0, len(colors_found))
    # for i in range(1, len(colors_found)):
    #     plt.axvline(x=i, color='black', linewidth=0.5, linestyle='--')

    plt.axis('off')
    plt.title(f'Colors sorted by prevalence (with pixel counts) -- {len(colors_found)}')
    plt.show()

# -------------------------------------------------
# Function --> get_component_segmentation         |
# -------------------------------------------------
def get_component_segmentation(seg_np,component_mask):
    component_mask3 = np.stack([component_mask != 0] * 3, axis=-1, dtype=bool)
    # print(component_mask3.shape, component_mask3.dtype)
    return seg_np * component_mask3


# -------------------------------------------------
# Function --> get_combined_mask         |
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import cv2
# import importlib
# importlib.reload(utc)

def get_combined_mask(image_np,seg_np, paths, mask_image_name=None,disallowed_color=None, components_ls=None,
                       combine_types=['PLeft', 'PRight'],plot_=True,plot_found_color_=False,plot_title = True,fixed_contour_color_val=None,
                        fixed_contour_color=(0,0,255),
                       check_anomalous=False,verbose=False):
    if check_anomalous:
        colors_found, counts,rounded_colors,colors_count_dict = segments_to_color(seg_np, verbose=False)
        # print('colors_found',colors_found   )
        # is_anomalous = True if disallowed_color in colors_found else False
        is_anomalous = True if np.any(np.all(colors_found == disallowed_color, axis=1)) else False

        level_anomaly = "ANOMALOUS" if is_anomalous else "NORMAL"
        if plot_found_color_:
            plot_found_color(colors_found,counts,rounded_colors, verbose=False)

    # Make a copy of the input image to draw on
    overlay_image = image_np.copy()

    # Initialize mask accumulator
    component_mask_combined = None

    for component in components_ls:
        if component in combine_types:
            subgroup = component
            
            # This draws the contour directly on a copy of the image
            contoured_image, component_mask = get_contoured_image(overlay_image,
                                                                  paths,
                                                                  subgroup,
                                                                  mask_image_name=mask_image_name,
                                                                  contour_color=(0, 255, 0), thickness=4,  # Bright green
                                                                  )
            cropped_segmentation = get_component_segmentation(seg_np,component_mask)
                
            # cropped_segmentation = get_component_segmentation(real_batch_segments,component_mask)
            colors_found_component_level, counts,rounded_colors,colors_count_dict = segments_to_color(cropped_segmentation, verbose=verbose)
            # is_anomalous = True if disallowed_color in colors_found_component_level else False 
            is_anomalous = True if np.any(np.all(colors_found_component_level == disallowed_color, axis=1)) else False

            component_level_anomaly = "ANOMALOUS" if is_anomalous else "NORMAL"
            if fixed_contour_color_val is None:
                contour_color = fixed_contour_color
            else:
                contour_color = (255,0,0) if is_anomalous else (0,255,0)
            thickness = 5 if is_anomalous else 3

            contoured_image, component_mask = get_contoured_image(overlay_image,
                                                                  paths,
                                                                  subgroup,
                                                                  mask_image_name=mask_image_name,
                                                                  contour_color=contour_color,
                                                                    thickness=thickness,  # Bright green
                                                                  )
            

            # Update the overlay image (accumulate contours on same image)
            overlay_image = contoured_image

            # Combine component masks
            if component_mask_combined is None:
                component_mask_combined = component_mask.copy()
            else:
                if component_mask.shape != component_mask_combined.shape:
                    component_mask = cv2.resize(component_mask, (component_mask_combined.shape[1], component_mask_combined.shape[0]))
                component_mask_combined = np.maximum(component_mask_combined, component_mask)

            del component_mask

    # Show the final combined image with all contours drawn
    if plot_:
        plt.imshow(overlay_image)
        if plot_title:
            plt.title(f"Combined Contours on Input Image - {combine_types}")
        else:
            plt.title("Combined Contours on Input Image")

        plt.xticks([])
        plt.yticks([])

        plt.show()

    return component_mask_combined, overlay_image


# Function to interpolate between two latent vectors
def interpolate_vectors(z1, z2, num_steps=10):
    return [z1 * (1 - alpha) + z2 * alpha for alpha in np.linspace(0, 1, num_steps)]

# Function to plot the interpolated images
def plot_interpolations(encoder, decoder, train_loader, num_pairs=5, num_steps=10):
    fig, axes = plt.subplots(num_pairs, num_steps + 2, figsize=(15, 1.5 * num_pairs))
    
    for pair_idx in range(num_pairs):
        # Select two random samples from the training dataset
        img1, _ = random.choice(train_loader.dataset)
        img2, _ = random.choice(train_loader.dataset)
        
        # Move images to device
        img1, img2 = img1.to(device).unsqueeze(0), img2.to(device).unsqueeze(0)
        
        # Encode the images to latent space
        z1, _ = encoder(img1)
        z2, _ = encoder(img2)
        
        # Interpolate between the two latent vectors
        interpolated_z = interpolate_vectors(z1, z2, num_steps=num_steps)
        
        # Decode the interpolated latent vectors
        interpolated_imgs = [decoder(z).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5 for z in interpolated_z]
        
        # Plot the original images
        axes[pair_idx, 0].imshow(img1.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
        axes[pair_idx, 0].set_title("Original 1")
        axes[pair_idx, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in axes[pair_idx, 0].spines.values():
            spine.set_edgecolor('cyan')
            spine.set_linewidth(2)
        
        axes[pair_idx, -1].imshow(img2.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
        axes[pair_idx, -1].set_title("Original 2")
        axes[pair_idx, -1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[pair_idx, -1].set_title("Original 1")
        for spine in axes[pair_idx, -1].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
        
        # Plot the interpolated images
        for step_idx, img in enumerate(interpolated_imgs):
            axes[pair_idx, step_idx + 1].imshow(img)
            axes[pair_idx, step_idx + 1].axis('off')
            axes[pair_idx, step_idx + 1].set_title(f"Step {step_idx + 1}")
            color = plt.cm.viridis(step_idx / num_steps)
            for spine in axes[pair_idx, step_idx + 1].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2)
    
    plt.tight_layout()
    plt.show()


#######################################################################


def count_images_per_class(dataset, class_names=None):
    # Many torchvision datasets have 'targets' or 'labels'
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    else:
        raise AttributeError("Dataset has no 'targets' or 'labels' attribute.")

    counts = Counter(labels)

    if class_names:
        counts = {class_names[i]: counts[i] for i in range(len(class_names))}
    
    return counts





####################################################################################################################################
### VISULIZATION FUNCTIONS
###
def plot_intermedidate_figs(inpu,recon,diff_img,component,idx,video_savepath_part,
                            norm_score,color_score,normscore_text,thres_text,batch_scores,
                            threshold_values=None,
                            destroy_figs = False):
    threshold_cmap, threshold_cmap_unexpected, selected_threshold =  threshold_values
    fig, axxes = plt.subplots(1, 3, figsize=(12, 6))
    ax1,ax2,ax3 = axxes[0],axxes[1],axxes[2]
    ax1.imshow(inpu)
    ax1.set_title(f'Frame {idx}')
    ax2.imshow((recon* 0.5 + 0.5).squeeze(0).cpu().numpy().transpose(1, 2, 0))
    ax2.set_title('Reconstructed')
    

    ax3.imshow(diff_img, cmap=threshold_cmap if norm_score<1 else threshold_cmap_unexpected,
                vmin=0, vmax=2*selected_threshold)
    
    # ax3.set_title(f'{component} AM')
    
    # print(component,'[',np.min(comp_map_list[component]),'~',np.max(comp_map_list[component]),']')
    ax3.set_title(f"{component} - {thres_text}\nN-Score: {normscore_text}× \n score: {batch_scores:.4f}", 
                    color=color_score, 
                    fontsize=10)
    for ax in axxes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(f'{video_savepath_part}/{component}_inter_anomaly_map_{idx}.png', dpi=300, bbox_inches='tight')
    if destroy_figs:
        plt.close(fig)
    plt.show()

####################################################################################################################################




def plot_label_timeline_dual(loader_or_dataset, fps=5, title="Label Timeline (Frames + Time)"):

    # --------------------------------------------------------------
    # 1) Resolve dataset and iterate through it
    # --------------------------------------------------------------
    if hasattr(loader_or_dataset, "__len__") and hasattr(loader_or_dataset, "dataset"):
        loader = loader_or_dataset
        dataset = loader_or_dataset.dataset
        batch_mode = True
    else:
        dataset = loader_or_dataset
        loader = None
        batch_mode = False

    all_labels = []

    # --------------------------------------------------------------
    # 2) Extract ordered labels
    # --------------------------------------------------------------
    if batch_mode:
        for batch in loader:
            if len(batch) == 3:
                _, labels, _ = batch
            else:
                _, labels = batch
            all_labels.extend(labels.cpu().numpy().tolist())
    else:
        for i in range(len(dataset)):
            sample = dataset[i]
            if len(sample) == 3:
                _, label, _ = sample
            else:
                _, label = sample
            all_labels.append(int(label))

    all_labels = np.array(all_labels)
    n = len(all_labels)

    # binary GT: 0 = normal, 1 = anomaly
    gt = np.array([0 if lbl == 0 else 1 for lbl in all_labels])

    # --------------------------------------------------------------
    # 3) Define axes time scale
    # --------------------------------------------------------------
    frame_indices = np.arange(n)
    times = frame_indices / fps

    # --------------------------------------------------------------
    # 4) Prepare plot
    # --------------------------------------------------------------
    cmap = ListedColormap(["green", "red"])

    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharey=True)

    # ---- Plot A: Frame Index ----
    gt_img = gt[np.newaxis, :]

    axes[0].imshow(
        gt_img,
        aspect="auto",
        cmap=cmap,
        extent=[0, n-1 if n > 1 else 1, 0, 1]
    )
    axes[0].set_title("Anomaly Timeline (Frame Index)", fontsize=13)
    axes[0].set_ylabel("GT label", fontsize=11)
    axes[0].set_yticks([])
    axes[0].set_xlabel("Frame Number", fontsize=11)

    # ---- Plot B: Time in Seconds ----
    axes[1].imshow(
        gt_img,
        aspect="auto",
        cmap=cmap,
        extent=[times[0], times[-1] if n > 1 else 1/fps, 0, 1]
    )
    axes[1].set_title("Anomaly Timeline (Time in Seconds)", fontsize=13)
    axes[1].set_ylabel("GT label", fontsize=11)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time [s]", fontsize=11)

    # ---- Legend ----
    legend_elements = [
        Patch(facecolor="green", edgecolor="none", label="Normal (0)"),
        Patch(facecolor="red", edgecolor="none", label="Anomalous (1)")
    ]

    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    return gt, frame_indices, times
#------------------------- PLOTTING ANOMALIES FROM DATA



def plot_label_timeline_dual_axis(loader_or_dataset, fps=5, title="Anomaly Timeline (Frame + Time)"):

    # --------------------------------------------------------------
    # 1) Resolve loader/dataset and extract ordered labels
    # --------------------------------------------------------------
    if hasattr(loader_or_dataset, "dataset"):
        loader = loader_or_dataset
        dataset = loader.dataset
        batched = True
    else:
        dataset = loader_or_dataset
        loader = None
        batched = False

    labels = []

    if batched:
        for batch in loader:
            if len(batch) == 3:
                _, lbls, _ = batch
            else:
                _, lbls = batch
            labels.extend(lbls.cpu().numpy().tolist())
    else:
        for i in range(len(dataset)):
            sample = dataset[i]
            if len(sample) == 3:
                _, lbl, _ = sample
            else:
                _, lbl = sample
            labels.append(int(lbl))

    labels = np.array(labels)
    n = len(labels)

    # Convert to binary GT: 0=normal, 1=anomaly
    gt = np.where(labels == 0, 0, 1)

    # --------------------------------------------------------------
    # 2) Frame index scale + Time scale
    # --------------------------------------------------------------
    frame_idx = np.arange(n)
    time_sec = frame_idx / fps

    gt_strip = gt[np.newaxis, :]  # shape (1, N)
    cmap = ListedColormap(["green", "red"])

    # --------------------------------------------------------------
    # 3) Create figure with dual x-axes
    # --------------------------------------------------------------
    fig, ax_bottom = plt.subplots(figsize=(14, 2.5))

    # Bottom axis = frame number
    ax_bottom.imshow(
        gt_strip,
        aspect="auto",
        cmap=cmap,
        extent=[frame_idx[0], frame_idx[-1] if n > 1 else 1, 0, 1],
    )
    ax_bottom.set_yticks([])
    ax_bottom.set_ylabel("GT Label", fontsize=11)
    ax_bottom.set_xlabel("Frame Number", fontsize=12)

    # --------------------------------------------------------------
    # 4) Add TOP axis for time
    # --------------------------------------------------------------
    ax_top = ax_bottom.twiny()

    # Map frame index → time
    ax_top.set_xlim(ax_bottom.get_xlim())
    ax_top.set_xticks(np.linspace(frame_idx[0], frame_idx[-1], 8))
    ax_top.set_xticklabels([f"{x/fps:.1f}" for x in np.linspace(frame_idx[0], frame_idx[-1], 8)])
    ax_top.set_xlabel("Time [s]", fontsize=12)

    # --------------------------------------------------------------
    # 5) Legend
    # --------------------------------------------------------------
    legend_elements = [
        Patch(facecolor="green", edgecolor="none", label="Normal"),
        Patch(facecolor="red", edgecolor="none", label="Anomalous")
    ]
    ax_bottom.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.show()

    return gt, frame_idx, time_sec





#--------------------------



def get_summary_json_path(path, params, method_name):
    """
    Build a deterministic JSON filename for this method and subgroup.
    """
    safe_name = method_name.replace(" ", "_")
    return os.path.join(
        path,  # or e.g. paths.path_results_fix / a custom folder
        f"summary_{params.subgroup}_{safe_name}.json"
    )


def summary_to_jsonable(summary):
    """
    Convert a summary dict (with numpy arrays) to a JSON-serializable dict.
    """
    json_summary = {}

    for k, v in summary.items():
        # Metrics (nested)
        if k == "metrics" or k == "roc" or k == "params":
            json_summary[k] = {}
            for mk, mv in v.items():
                if isinstance(mv, np.ndarray):
                    json_summary[k][mk] = mv.tolist()
                else:
                    # cast numpy scalars to Python scalars if needed
                    if isinstance(mv, (np.generic,)):
                        mv = mv.item()
                    json_summary[k][mk] = mv
        # Arrays / lists at top level
        elif k in ("scores", "labels", "preds", "tn_scores", "tp_scores"):
            if isinstance(v, np.ndarray):
                json_summary[k] = v.tolist()
            else:
                json_summary[k] = list(v)
        else:
            # Scalars or other small values
            if isinstance(v, np.ndarray):
                json_summary[k] = v.tolist()
            elif isinstance(v, (np.generic,)):
                json_summary[k] = v.item()
            else:
                json_summary[k] = v

    return json_summary


def json_to_summary(json_summary):
    """
    Convert JSON-loaded dict back to a summary dict with numpy arrays where useful.
    """
    summary = dict(json_summary)

    # Convert lists back to numpy arrays for numeric fields
    for k in ("scores", "labels", "preds", "tn_scores", "tp_scores"):
        if k in summary and summary[k] is not None:
            summary[k] = np.asarray(summary[k], dtype=float if k != "labels" else int)

    if "roc" in summary:
        roc = summary["roc"]
        if "fpr" in roc:
            roc["fpr"] = np.asarray(roc["fpr"], dtype=float)
        if "tpr" in roc:
            roc["tpr"] = np.asarray(roc["tpr"], dtype=float)
        summary["roc"] = roc

    # metrics can stay as plain floats/dicts
    return summary


# %%
