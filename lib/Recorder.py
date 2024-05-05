import torch
import numpy as np
import os
import cv2
import trimesh
import pyrender
import trimesh
import json
from skimage.io import imread

def load_im(path):
    img = imread(path)
    img = img.astype(np.float32) / 255.0
    mask = img[:,:,3:]
    img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
    img_np = np.uint8(img[:, :, :3] * 255.)
    return img_np

class Recorder():
    def __init__(self, save_folder, camera, visualize=False, save_vertices=False):

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        self.camera = camera

        self.visualize = visualize
        self.save_vertices = save_vertices

    def log(self, log_data):
        frames = log_data['frames']
        face_model = log_data['face_model'] 
        intrinsics = log_data['intrinsics'] # b, nv, 3, 3
        extrinsics = log_data['extrinsics'] # b, nv, 3, 4
        vids = log_data['vids'] # b, 40
        
        identity_params = []
        for frame in frames:
            subject_id = frame.split('/')[-2].zfill(3)
            with open(f'/cluster/scratch/xiychen/data/fitted_shapes/{str(subject_id).zfill(3)}.json') as f:
                identity_param = np.array(json.load(f))[:300]
            identity_params.append(identity_param)
        identity_params = np.stack(identity_params)
        
        identity_params=None
        
        with torch.no_grad():
            vertices, landmarks = log_data['face_model'](identity_params) # b, 5023, 3; b, 66, 3
            faces = log_data['face_model'].faces.detach().cpu().numpy()
        
        intrinsics0 = intrinsics[:,1,:,:]
        extrinsics0 = extrinsics[:,1,:,:]
        
        # Expand dimensions for batch matrix multiplication
        landmarks_homogeneous = torch.cat([landmarks, torch.ones_like(landmarks[..., :1])], dim=-1)

        # Apply extrinsic transformation
        landmarks_cam = torch.bmm(extrinsics0, landmarks_homogeneous.permute(0,2,1))
        # landmarks_cam = landmarks_cam.squeeze(-1)

        # Apply intrinsic transformation
        projected_points = torch.matmul(intrinsics0, landmarks_cam).permute(0,2,1)
        projected_points = (projected_points[:, :, :2] / projected_points[:, :, 2:])
        for n, frame in enumerate(frames):
            subject_id = frame.split('/')[-2]
            exp_id = frame.split('/')[-1].split('.')[0]
            os.makedirs(os.path.join(self.save_folder, subject_id, exp_id), exist_ok=True)
            # face_model.save('%s/params.npz' % (os.path.join(self.save_folder, frame)), batch_id=n)
            # np.save('%s/lmk_3d.npy' % (os.path.join(self.save_folder, frame)), landmarks[n].cpu().numpy())
            
            if self.save_vertices:
                # print('%s/vertices.npy' % (os.path.join(self.save_folder, subject_id, exp_id)))
                # np.save('%s/vertices.npy' % (os.path.join(self.save_folder, subject_id, exp_id)), vertices[n].cpu().numpy())
                mesh = trimesh.Trimesh(vertices[n].cpu().numpy(), faces, process=False)
                mesh.export('%s/mesh.obj' % (os.path.join(self.save_folder, subject_id, exp_id)))
                print('%s/mesh.obj' % (os.path.join(self.save_folder, subject_id, exp_id)))
            
            
            lmk_2d = projected_points[n]
            vid = vids[n][1]
            img = load_im(f'/cluster/scratch/xiychen/data/facescape_color_calibrated/{subject_id}/{exp_id}/view_{str(vid).zfill(5)}/rgba_colorcalib_v2.png')
            for idx_, loc in enumerate(lmk_2d):
                x = int(loc[0])
                y = int(loc[1])
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)
            cv2.imwrite('%s/proj.jpg' % (os.path.join(self.save_folder, subject_id, exp_id)), img[:,:,::-1])
            
            if self.visualize:
                # for v in range(intrinsics.shape[1]):
                for v in range(2):
                    # print('%s/vis_%d.jpg' % (os.path.join(self.save_folder, subject_id, exp_id), v))
                    faces = log_data['face_model'].faces.cpu().numpy()
                    mesh_trimesh = trimesh.Trimesh(vertices=vertices[n].cpu().numpy(), faces=faces)
                    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

                    self.camera.init_renderer(intrinsic=intrinsics[n, v], extrinsic=extrinsics[n, v])
                    render_image = self.camera.render(mesh)
                    cv2.imwrite('%s/vis_%d.jpg' % (os.path.join(self.save_folder, subject_id, exp_id), v), render_image[:,:,::-1])
                