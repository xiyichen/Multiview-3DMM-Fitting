import torch
import numpy as np
import glob
import os
import random
import cv2
from skimage import io
import glob, json

class LandmarkDataset():

    def __init__(self, landmark_folder, camera_folder):

        # self.frames = sorted(os.listdir(landmark_folder))
        # self.landmark_folder = landmark_folder
        # self.camera_folder = camera_folder
        # self.frames = [sorted(glob.glob('/cluster/scratch/xiychen/data/facescape_color_calibrated/212/*/'))[17]]
        self.frames = sorted(glob.glob('/cluster/scratch/xiychen/pck/gt/*/*.json'))[:20]
        print(len(self.frames))
        # print(self.frames)
    
    def get_item(self):
        landmarks = []
        extrinsics = []
        intrinsics = []
        vids = []
        for frame_path in self.frames:
            with open(frame_path) as f:
                kpts_all = json.load(f)
            subject_id = frame_path.split('/')[-2]
            exp_id = frame_path.split('/')[-1].split('.')[0]
            landmarks_ = []
            extrinsics_ = []
            intrinsics_ = []
            vids_ = []
            # camera_ids = [item.split('_')[-1][:-4] for item in sorted(os.listdir(os.path.join(self.landmark_folder, frame)))]
            # camera_ids = os.path
            # for v in range(len(camera_ids)):
            #     if os.path.exists(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v])):
            #         landmark = np.load(os.path.join(self.landmark_folder, frame, 'lmk_%s.npy' % camera_ids[v]))
            #         landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
            #         extrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['extrinsic']
            #         intrinsic = np.load(os.path.join(self.camera_folder, frame, 'camera_%s.npz' % camera_ids[v]))['intrinsic']
            #     else:
            #         landmark = np.zeros([66, 3], dtype=np.float32)
            #         extrinsic = np.ones([3, 4], dtype=np.float32)
            #         intrinsic = np.ones([3, 3], dtype=np.float32)
            #     landmarks_.append(landmark)
            #     extrinsics_.append(extrinsic)
            #     intrinsics_.append(intrinsic)
            # camera_paths = sorted(glob.glob(os.path.join(frame_path, 'view_*')))
            with open(f'/cluster/scratch/xiychen/data/facescape_color_calibrated/{subject_id}/{exp_id}/cameras.json', 'r') as f:
                camera_dict = json.load(f)

            
            for camera_id in camera_dict.keys():
                # camera_id = int(camera_path.split('_')[-1])
                if camera_dict[str(camera_id)]['angles']['azimuth'] > 60 or camera_dict[str(camera_id)]['angles']['elevation'] > 30:
                    continue
                # subject_id = camera_path.split('/')[-3]
                # exp_id = camera_path.split('/')[-2]
                # try:
                #     # landmark = np.load(os.path.join('/cluster/scratch/xiychen/pck/gt', subject_id, exp_id, f'view_{str(camera_id).zfill(5)}.npy'))
                #     # landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
                #     # extrinsic = np.array(camera_dict[str(camera_id)]['extrinsics'])
                #     # intrinsic = np.array(camera_dict[str(camera_id)]['intrinsics'])
                # except:
                #     landmark = np.zeros((66, 3))
                #     extrinsic = np.array(camera_dict[str(camera_id)]['extrinsics'])
                #     intrinsic = np.array(camera_dict[str(camera_id)]['intrinsics'])
                if camera_id not in kpts_all.keys():
                    continue
                landmark = np.array(kpts_all[camera_id])
                landmark = np.vstack([landmark[0:48], landmark[49:54], landmark[55:68]])
                extrinsic = np.array(camera_dict[str(camera_id)]['extrinsics'])
                intrinsic = np.array(camera_dict[str(camera_id)]['intrinsics'])
                landmarks_.append(landmark)
                extrinsics_.append(extrinsic)
                intrinsics_.append(intrinsic)
                vids_.append(camera_id)
            
            while len(landmarks_) < 40:
                landmarks_.append(np.zeros((66, 3)))
                extrinsics_.append(extrinsics_[-1])
                intrinsics_.append(intrinsics_[-1])
                vids_.append(vids_[-1])
            landmarks_ = np.stack(landmarks_)
            extrinsics_ = np.stack(extrinsics_)
            intrinsics_ = np.stack(intrinsics_)
            vids_ = np.stack(vids_)
            landmarks.append(landmarks_)
            extrinsics.append(extrinsics_)
            intrinsics.append(intrinsics_)
            vids.append(vids_)
        landmarks = np.stack(landmarks)
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)
        vids = np.stack(vids)

        return landmarks, extrinsics, intrinsics, self.frames, vids
    
    def __len__(self):
        return len(self.frames)
    
    