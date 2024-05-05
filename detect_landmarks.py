import os
import torch
import tqdm
import glob
import numpy as np
import cv2
import face_alignment
import argparse
import glob
from skimage.io import imread
from PIL import Image

from config.config import config


def load_im(path):
    img = imread(path)
    img = img.astype(np.float32) / 255.0
    mask = img[:,:,3:]
    img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
    img_np = np.uint8(img[:, :, :3] * 255.)
    return img_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/NeRSemble_031.yaml')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    device = torch.device('cuda:0')
    torch.cuda.set_device(cfg.gpu_id)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, face_detector='blazeface', device='cuda:%d' % cfg.gpu_id)

    source_folder = cfg.image_folder
    output_folder = '/cluster/scratch/xiychen/data/facescape_landmarks/'

    # frames = sorted(os.listdir(source_folder))
    frames = sorted(glob.glob('/cluster/scratch/xiychen/data/facescape_color_calibrated/001/*/'))
    for frame in tqdm.tqdm(frames):
        if 'background' in frame:
            continue
        # source_frame_folder = os.path.join(source_folder, frame.split(''))
        output_frame_folder = os.path.join(output_folder, frame.split('/')[-3], frame.split('/')[-2])
        os.makedirs(output_frame_folder, exist_ok=True)

        # if len(cfg.camera_ids) > 0:
        #     image_paths = [source_frame_folder + '/image_%s.jpg' % camera_id for camera_id in cfg.camera_ids]
        # else:
        image_paths = sorted(glob.glob(os.path.join(frame, 'view_*', 'rgba_colorcalib_v2.png')))

        images = np.stack([cv2.resize(load_im(image_path)[:, :, ::-1], (cfg.image_size, cfg.image_size)) for image_path in image_paths])
        images = torch.from_numpy(images).float().permute(0, 3, 1, 2).to(device)

        results = fa.get_landmarks_from_batch(images, return_landmark_score=True)
        for i in range(len(results[0])):
            if results[1][i] is None:
                results[0][i] = np.zeros([68, 3], dtype=np.float32)
                results[1][i] = [np.zeros([68], dtype=np.float32)]
            if len(results[1][i]) > 1:
                total_score = 0.0
                for j in range(len(results[1][i])):
                    if np.sum(results[1][i][j]) > total_score:
                        total_score = np.sum(results[1][i][j])
                        landmarks_i = results[0][i][j*68:(j+1)*68]
                        scores_i = results[1][i][j:j+1]
                results[0][i] = landmarks_i
                results[1][i] = scores_i
                
        landmarks = np.concatenate([np.stack(results[0])[:, :, :2], np.stack(results[1]).transpose(0, 2, 1)], -1)
        for i, image_path in enumerate(image_paths):
            landmarks_path = os.path.join(output_frame_folder, image_path.split('/')[-2] + '.npy')
            np.save(landmarks_path, landmarks[i])