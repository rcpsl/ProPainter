# -*- coding: utf-8 -*-
import os
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import math
import torch
import torchvision

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

def get_frame_info(frame_root):
    """Get basic video information without loading all frames."""
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):
        video_name = os.path.basename(frame_root)[:-4]
        # Read first frame to get size
        vframes, _, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec', num_frames=1)
        size = (vframes.shape[2], vframes.shape[1])  # width, height
        fps = info['video_fps']
        total_frames = None  # Will need to count frames separately for video
        return video_name, fps, size, total_frames, None
    else:
        video_name = os.path.basename(frame_root)
        fr_lst = sorted(os.listdir(frame_root), key=lambda x: int(x.split('.')[0]))
        # Read first frame to get size
        first_frame = cv2.imread(os.path.join(frame_root, fr_lst[0]))
        first_frame = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        size = first_frame.size
        fps = None
        total_frames = len(fr_lst)
        return video_name, fps, size, total_frames, fr_lst

def read_frame_batch(frame_root, frame_list, start_idx, batch_size):
    """Read a batch of frames."""
    frames = []
    end_idx = min(start_idx + batch_size, len(frame_list))
    
    for idx in range(start_idx, end_idx):
        frame = cv2.imread(os.path.join(frame_root, frame_list[idx]))
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(frame)
        
    return frames, start_idx, end_idx

def combine_videos(video_paths, output_path, fps):
    """Combine multiple videos into one."""
    video_clips = []
    for path in video_paths:
        if os.path.exists(path):
            video = imageio.get_reader(path)
            video_clips.extend(list(video))
    
    imageio.mimwrite(output_path, video_clips, fps=fps, quality=7)
    
    # Clean up intermediate files
    for path in video_paths:
        if os.path.exists(path):
            os.remove(path)



# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
        
    return frames, process_size, out_size


#  read frames from video
def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting.
    """
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    # Defines new FOV.
    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    # Extrapolates the FOV for video.
    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    # Generates the mask for missing region.
    masks_dilated = []
    flow_masks = []
    
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    
    mask[H_start+dilate_h: H_start+imgH-dilate_h, 
         W_start+dilate_w: W_start+imgW-dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))

    mask[H_start: H_start+imgH, W_start: W_start+imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))
  
    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

def read_mask_batch(mpath, start_idx, batch_size, size, flow_mask_dilates=8, mask_dilates=5):
    """Read a batch of masks and process them.
    
    Args:
        mpath: Path to mask file or directory
        start_idx: Starting index for batch
        batch_size: Number of masks to process in batch
        size: Target size for masks
        flow_mask_dilates: Dilation iterations for flow mask
        mask_dilates: Dilation iterations for regular mask
        
    Returns:
        flow_masks: List of processed flow masks
        masks_dilated: List of processed dilated masks
        start_idx: Start index of batch
        end_idx: End index of batch
    """
    flow_masks = []
    masks_dilated = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
        # Single mask case - replicate for batch
        mask_img = Image.open(mpath)
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_arr = np.array(mask_img.convert('L'))
        
        # Process single mask for flow and dilation
        flow_mask = process_flow_mask(mask_arr, flow_mask_dilates)
        dilated_mask = process_dilated_mask(mask_arr, mask_dilates)
        
        # Replicate for batch size
        flow_masks = [Image.fromarray(flow_mask * 255)] * batch_size
        masks_dilated = [Image.fromarray(dilated_mask * 255)] * batch_size
        end_idx = start_idx + batch_size
        
    else:
        # Multiple masks case - process batch
        mask_files = sorted(os.listdir(mpath))
        end_idx = min(start_idx + batch_size, len(mask_files))
        
        for idx in range(start_idx, end_idx):
            mask_img = Image.open(os.path.join(mpath, mask_files[idx]))
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            mask_arr = np.array(mask_img.convert('L'))
            
            # Process each mask
            flow_mask = process_flow_mask(mask_arr, flow_mask_dilates)
            dilated_mask = process_dilated_mask(mask_arr, mask_dilates)
            
            flow_masks.append(Image.fromarray(flow_mask * 255))
            masks_dilated.append(Image.fromarray(dilated_mask * 255))
    
    return flow_masks, masks_dilated, start_idx, end_idx

def process_flow_mask(mask_arr, flow_mask_dilates):
    """Process mask for flow computation."""
    if flow_mask_dilates > 0:
        flow_mask = scipy.ndimage.binary_dilation(
            mask_arr, iterations=flow_mask_dilates).astype(np.uint8)
    else:
        flow_mask = binary_mask(mask_arr).astype(np.uint8)
    return flow_mask

def process_dilated_mask(mask_arr, mask_dilates):
    """Process mask for dilation."""
    if mask_dilates > 0:
        dilated = scipy.ndimage.binary_dilation(
            mask_arr, iterations=mask_dilates).astype(np.uint8)
    else:
        dilated = binary_mask(mask_arr).astype(np.uint8)
    return dilated



def combine_videos(input_folder, output_file):
    # Get all mp4 files and sort them numerically
    video_files = [f for f in os.listdir(input_folder) if f.startswith('batch_') and f.endswith('.mp4')]
    video_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by batch number
    
    # Load all video clips
    clips = []
    for video_file in video_files:
        file_path = os.path.join(input_folder, video_file)
        print(f"Loading {video_file}...")
        clip = VideoFileClip(file_path)
        clips.append(clip)
    
    # Concatenate all clips
    print("Combining videos...")
    final_clip = concatenate_videoclips(clips)
    
    # Write the combined video to file
    print("Writing combined video to file...")
    final_clip.write_videofile(output_file, codec='libx264')
    
    # Close all clips to free up resources
    for clip in clips:
        clip.close()
    final_clip.close()
    
    print("Video combination complete!")

def process_batch(frames, flow_masks, masks_dilated, device, fix_raft, fix_flow_complete, model, 
                 args, h, w, ori_frames, out_size, use_half, start_idx):
    video_length = len(frames)
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)
    print("frames shape: ", frames.shape)
    print("flow_masks shape: ", flow_masks.shape)
    with torch.no_grad():
        # Compute flow
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        # Rest of the processing remains the same as in original code
        # [Original flow computation code]
        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f], iters=args.raft_iter)
                
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
            torch.cuda.empty_cache()

        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        # Complete flow
        flow_length = gt_flows_bi[0].size(1)
        pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
        pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
        torch.cuda.empty_cache()

        # Image propagation
        masked_frames = frames * (1 - masks_dilated)
        b, t, _, _, _ = masks_dilated.size()
        prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
        updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
        updated_masks = updated_local_masks.view(b, t, 1, h, w)
        torch.cuda.empty_cache()

        # Process frames
        comp_frames = [None] * video_length
        neighbor_stride = args.neighbor_length // 2
        ref_num = -1

        for f in tqdm(range(0, video_length, neighbor_stride)):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], 
                                    pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, 
                           selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(
                0, 2, 3, 1).numpy().astype(np.uint8)
            
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                    + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)

    return comp_frames



if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        "--ref_stride", type=int, default=10, help='Stride of global reference frames.')
    parser.add_argument(
        "--neighbor_length", type=int, default=10, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=80, help='Length of sub-video for long video inference.')
    parser.add_argument(
        "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'], help="Modes: video_inpainting / video_outpainting")
    parser.add_argument(
        '--scale_h', type=float, default=1.0, help='Outpainting scale of height for video_outpainting mode.')
    parser.add_argument(
        '--scale_w', type=float, default=1.2, help='Outpainting scale of width for video_outpainting mode.')
    parser.add_argument(
        '--save_fps', type=int, default=24, help='Frame per second. Default: 24')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--fp16', action='store_true', help='Use fp16 (half precision) during inference. Default: fp32 (single precision).')

    args = parser.parse_args()
    
    # Use fp16 precision during inference to reduce running memory cost
    use_half = True if args.fp16 else False 
    if device == torch.device('cpu'):
        use_half = False

    # Get video information without loading all frames
    video_name, fps, size, total_frames, frame_list = get_frame_info(args.video)
    
    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))
    
    # Calculate process size once
    out_size = size
    process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
    w, h = process_size
    
    fps = args.save_fps if fps is None else fps
    save_root = os.path.join(args.output, video_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    # Set up models
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                  model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)
    
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                  model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
                                  model_dir='weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    ##############################################
    # ProPainter inference with batch processing
    ##############################################
    # Initialize batch processing
    batch_size = 40
    num_batches = math.ceil(total_frames / batch_size)
    video_paths = []
    print(f'\nProcessing: {video_name} [{total_frames} frames] in {num_batches} batches...')

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        
        # Read batch of frames
        batch_frames, batch_start, batch_end = read_frame_batch(args.video, frame_list, start_idx, batch_size)
        batch_frames = [f.resize(process_size) for f in batch_frames]
        
        # Read corresponding masks in batch
        batch_flow_masks, batch_masks_dilated, _, _ = read_mask_batch(
            args.mask, 
            start_idx, 
            len(batch_frames),  # Use actual number of frames read
            process_size,
            flow_mask_dilates=args.mask_dilation,
            mask_dilates=args.mask_dilation
        )
        
        print(f'\nProcessing batch {batch_idx + 1}/{num_batches} (frames {batch_start}-{batch_end-1})...')
        
        # Convert frames for processing
        frames_inp = [np.array(f).astype(np.uint8) for f in batch_frames]
        
        # Process the batch
        comp_frames = process_batch(
            batch_frames, 
            batch_flow_masks, 
            batch_masks_dilated,
            device, 
            fix_raft, 
            fix_flow_complete, 
            model, 
            args, 
            h, w, 
            frames_inp, 
            out_size, 
            use_half,
            start_idx
        )
        
        # Save batch result
        batch_output_path = os.path.join(save_root, f'batch_{batch_idx}.mp4')
        comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
        imageio.mimwrite(batch_output_path, comp_frames, fps=fps, quality=7)
        video_paths.append(batch_output_path)
        
        # Clear GPU memory after each batch
        torch.cuda.empty_cache()

    # Combine all batches into final video
    final_output_path = os.path.join(save_root, 'inpaint_out.mp4')
    #combine_videos(video_paths, final_output_path, fps)
    combine_videos(save_root, final_output_path)
    print(f'\nAll results are saved in {save_root}')