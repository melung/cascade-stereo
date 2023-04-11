import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from struct import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from gipuma import gipuma_filter

from multiprocessing import Pool, Process, Queue
from multiprocessing.pool import ThreadPool
import subprocess

from functools import partial
import signal
import pymeshlab
from scipy import ndimage
import mediapipe as mp
import pytorch3d
import glob
import numpngw
import png
cudnn.benchmark = True
#os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"

#os.environ['CUDA_VISIBLE_DEVICES'] ='0,1'

#device = torch.device("cuda:1")

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')



parser.add_argument('--testpath', default="/root/lhs/cascade-stereo/CasMVSNet/MVSSTEREOTH_1",help='testing data dir for some scenes')


parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='mdi_melungl', help='select dataset')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', default="lists/our_list.txt", help='testing scene list')
parser.add_argument('--landmark', default=True, help='extract landmark')
parser.add_argument('--faceseg', default=False, help='Face segmentation')
parser.add_argument('--facealign', default=False, help='Face alignment')
parser.add_argument('--face_opt', default=False, help='Face optimization')



parser.add_argument('--local_rank',help='testing data dir for some scenes')


parser.add_argument('--batch_size', type=int, default=4, help='testing batch size')#7 GPUs can use maximum 18 batches +- 2
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')#192 no significant performance improvement even if increased

parser.add_argument('--loadckpt', default="/root/lhs/cascade-stereo/casmvsnet.ckpt", help='load a specific checkpoint')


parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')#48 32 8 no significant performance improvement even if increased
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')


parser.add_argument('--interval_scale', type=float, default=4.0, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')#2 is blur, 10 is too much
parser.add_argument('--max_h', type=int, default=1024, help='testing max h')#864
parser.add_argument('--max_w', type=int, default=1024, help='testing max w')#1152


parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=16, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=60, help='save freq of local pcd')
parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")

#filter

#filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='../../fusibile/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.9')
parser.add_argument('--disp_threshold', type=float, default='1.0')
parser.add_argument('--num_consistent', type=float, default='4')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

Interval_Scale = args.interval_scale

##############################################################################################

def write_2d_landmark(landmark_filename, lmk_array, landmark_image):
    np.savez(landmark_filename, lmk=lmk_array)
    cv2.imwrite(landmark_filename[:-3]+'.png', landmark_image)
    
    

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics

# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5

# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()


# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(model, test_dataset):
    # dataset, dataloader  
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    #

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            sample_cuda = tocuda(sample)

            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            end_time = time.time()
            print('only inference: ' +str(end_time-start_time))
            outputs = tensor2numpy(outputs)

            del sample_cuda
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(num_stage)].numpy()
            imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            # save depth maps and confidence maps
            if args.landmark:
                face_mesh = mp.solutions.face_mesh.FaceMesh(
                            static_image_mode=True,
                            max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.5) 
            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, \
                                                            outputs["depth"], outputs["photometric_confidence"]):

                img = img[0]  #ref view
                cam = cam[0]  #ref cam
                depth_filename = os.path.join(args.testpath + '_output', (filename[:-11] + filename[-10:]).format('2333__', '/disp.dmb'))
                normal_filename = os.path.join(args.testpath + '_output', (filename[:-11] + filename[-10:]).format('2333__', '/normals.dmb'))
                landmark_filename = os.path.join(args.testpath + '_output', filename.format('2d_landmark', '.npz'))
                
                cam_filename = os.path.join(args.testpath + '_output', filename.format('cams', '.jpg.P'))
                img_filename = os.path.join(args.testpath + '_output', filename.format('images', '.jpg'))
                
                os.makedirs(depth_filename[:-8], exist_ok=True)
                os.makedirs(cam_filename[:-14], exist_ok=True)
                os.makedirs(img_filename[:-12], exist_ok=True)
                if args.landmark:
                    os.makedirs(landmark_filename[:-12], exist_ok=True)
                
                #save depth maps
                save_pfm(depth_filename+'.pfm', depth_est)
                #save confidence maps
                #save_pfm(confidence_filename, photometric_confidence)
                #save cams, img
                depth_est[photometric_confidence < args.prob_threshold] = 0
                write_gipuma_dmb(depth_filename, depth_est)
                
                fake_gipuma_normal(cam, normal_filename, depth_est)
                
                write_gipuma_cam(cam, cam_filename)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)
                
                if args.landmark:
                    image_height, image_width, _ = img.shape
                    #print(img.shape)
                    img_landmark = img
                    results = face_mesh.process(img_landmark)
                    img_landmark = cv2.cvtColor(img_landmark, cv2.COLOR_RGB2BGR)
                    if not results.multi_face_landmarks:
                        # print("no detected data - ", file)
                        continue
                    face_landmarks = results.multi_face_landmarks[0].landmark
                    lmk_array = np.zeros((len(face_landmarks), 3))
                    for f_idx in range(len(face_landmarks)):
                        lmk_array[f_idx, 0] = face_landmarks[f_idx].x * image_width
                        lmk_array[f_idx, 1] = face_landmarks[f_idx].y * image_height
                        lmk_array[f_idx, 2] = face_landmarks[f_idx].z
                        landmark_image = cv2.circle(img_landmark,(int(lmk_array[f_idx,0]),int(lmk_array[f_idx,1])),1,(255,255,255),-1)
                    write_2d_landmark(landmark_filename, lmk_array, landmark_image)
                
                
                
                

    gc.collect()
    torch.cuda.empty_cache()
    
def write_gipuma_cam(cam, out_path):
    '''convert mvsnet camera to gipuma camera format'''
    #cam[0] extrinsic cam[1] intrinsic

    projection_matrix = np.matmul(cam[1], cam[0])
    projection_matrix = projection_matrix[0:3][:]

    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    f = open(out_path+'camera-intrinsics.txt', "w")
    for i in range(0, 3):
        for j in range(0, 3):
            if j ==2:
                f.write(str(cam[1][i][j]))
            else:
                f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()
    
    f = open(out_path+'frame-000000.pose.txt', "w")
    for i in range(0, 4):
        for j in range(0, 4):
            if j ==3:
                f.write(str(cam[0][i][j]/1000))
            else:
                f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()


    return


def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''

    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = image.squeeze()
        #print(image)

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    np.save(path+".npy", image)
    
    if np.shape(image)[-1] !=3:
        image = image.astype(np.uint16)
        #print(image)
        with open(path+'.png', 'wb') as f:
            writer = png.Writer(width=image.shape[1], height=image.shape[0], bitdepth=16,
                                greyscale=True)
            zgray2list = image.tolist()
            writer.write(f, zgray2list)

    return



def gradient(im_smooth):

    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.arange(-1,2).astype(float)
    kernel = - kernel / 2

    gradient_x = ndimage.convolve(gradient_x, kernel[np.newaxis])
    gradient_y = ndimage.convolve(gradient_y, kernel[np.newaxis].T)

    return gradient_x,gradient_y


def sobel(im_smooth):
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    gradient_x = ndimage.convolve(gradient_x, kernel)
    gradient_y = ndimage.convolve(gradient_y, kernel.T)

    return gradient_x,gradient_y


def compute_normal_map(gradient_x, gradient_y, intensity=1):

    width = gradient_x.shape[1]
    height = gradient_x.shape[0]
    max_x = np.max(gradient_x)
    max_y = np.max(gradient_y)

    max_value = max_x

    if max_y > max_x:
        max_value = max_y

    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    intensity = 1 / intensity

    #print(max_value)
    strength = max_value / (max_value * intensity)

    normal_map[..., 0] = gradient_x / max_value
    normal_map[..., 1] = gradient_y / max_value

    normal_map[..., 2] = 1 / strength

    norm = np.sqrt(np.power(normal_map[..., 0], 2) + np.power(normal_map[..., 1], 2) + np.power(normal_map[..., 2], 2))

    normal_map[..., 0] /= norm
    normal_map[..., 1] /= norm
    normal_map[..., 2] /= norm


    return normal_map



def Convert(im,intensity):

    sobel_x, sobel_y = sobel(im)

    normal_map = compute_normal_map(sobel_x, sobel_y, intensity)

    return normal_map


def fake_gipuma_normal(cam, path, depth_image):
    image_shape = np.shape(depth_image)
    h, w = np.shape(depth_image)
    extrinsic = cam[0]
    
    a = True
    
    if a == True:
        #tt = time.time()
        normals = Convert(depth_image,1000)
        #ee = time.time()
        #print("noraml time : ", str(ee-tt))

        normal_image = normals
        
        # for x in range(h):
        #     for y in range(w):
        #         normal_image[x,y,:] = -np.matmul(extrinsic[:3,:3].T,normals[x,y,:])
        # eee = time.time()
        # print("noraml r time : ", str(eee-ee))        
    else:
        fake_normal = np.array(-extrinsic[2,:3])
        
        normal_image = np.ones_like(depth_image) #depth
        normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1)) #one channel depth
        normal_image = np.tile(normal_image, [1, 1, 3])

        normal_image[:, :, 0] = fake_normal[0]
        normal_image[:, :, 1] = fake_normal[1]
        normal_image[:, :, 2] = fake_normal[2]


    normal_image = np.float32(normal_image)
    write_gipuma_dmb(path, normal_image)
    
    #########################################
    # normal_image *= 0.5
    # normal_image += 0.5
    # normal_image *= 255
    # normals *= 0.5
    # normals += 0.5
    # normals *= 255
    
    # normal_image.astype('uint8')
    # normals.astype('uint8')
    # tt = time.time()
    # normal_image = cv2.cvtColor(normal_image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("/root/lhs/normal/normal_g"+ str(tt) +".png", normal_image)
    # normals = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("/root/lhs/normal/normal_o"+ str(tt) +".png", normals)
    ##########################################
    
    return

def obj_read_w_color(obj_path):
    vs = []
    fs = []
    cs = []
    with open(obj_path, 'r') as obj_f:
        lines = obj_f.readlines()
        for line in lines:
            line_ = line.split(' ')
            if len(line_) < 4:
                line_ = line.split('\t')
            if line_[0] == 'v':
                v1 = float(line_[1])
                v2 = float(line_[2])
                v3 = float(line_[3])
                vs.append([v1, v2, v3])
                c1 = float(line_[4])
                c2 = float(line_[5])
                c3 = float(line_[6])
                cs.append([c1, c2, c3])
            if line_[0] == 'f':
                f1 = int(line_[1].split('/')[0]) - 1
                f2 = int(line_[2].split('/')[0]) - 1
                f3 = int(line_[3].split('/')[0]) - 1
                fs.append([f1, f2, f3])
    vs = np.array(vs)
    fs = np.array(fs)
    cs = np.array(cs)

    return vs, fs, cs  


def Extract_3d_landmark(lmk_2d_folder, used_view_list):
    full_lmk_2d_list = sorted(glob.glob(lmk_2d_folder+'/*.npz'))
    lmk_2d_list = [full_lmk_2d_list[i] for i in used_view_list]
    lmks_2d = np.zeros((len(lmk_2d_list), 478, 2))
     
    for i, lmk_name in enumerate(lmk_2d_list):
        lmk2d = np.load(lmk_name)
        lmks_2d[i,:,:] = lmk2d['lmk'][...,:2]
    
    
    A = proj_matricies[None, :, 2:3].repeat(n_lmk, 1, 2, 1) * lmk_2d[None].transpose(0, 2, 1, 3).reshape(
            n_lmk, n_view, 2, 1)
    A -= proj_matricies[None, :, :2].tile(n_lmk, 1, 1, 1)

    # A = A.detach().cpu().numpy()
    u, s, vh = np.linalg.svd(A.reshape(n_lmk, n_view * 2, 4))
    point_3d_homo = -vh[:, 3, :]

    # lmk_3d : n_lmk x 3
    lmk_3d = homogeneous_to_euclidean(point_3d_homo)
    #print(np.shape(lmk_3d))
    # np.savez(os.path.join(args.lmk_dir, '3D_lmk/lmk_3D_%03d.npz' % i), lmk=lmk_3d)

    if i == 0:
        lmk_3d_tr = tr.tensor(lmk_3d).float().to(args.device)
        knn = knn_points(lmk_3d_tr[None], v.float())

        lmk_idx = knn[1].squeeze(0).squeeze(-1).detach().cpu().numpy()
        lmk_3d2 = v[0, lmk_idx, :].detach().cpu().numpy()

        np.savez(os.path.join(args.lmk_dir, 'lmk_ref_with_idx.npz'), lmk_idx=lmk_idx, lmk=lmk_3d)
        np.savez(os.path.join(args.lmk_dir, 'lmk_3D_%03d.npz' % i), lmk=lmk_3d)

        if args.visualize:
            lmk_3d = tr.tensor(lmk_3d).to(args.device)
            lmk_3d2 = tr.tensor(lmk_3d2).to(args.device)
            # print(lmk_3d.shape,lmk_3d2.shape)
            point_cloud1 = Pointclouds(points=[lmk_3d],
                                        features=[tr.ones_like(lmk_3d) * tr.tensor([[1, 0, 0]]).to(args.device)])
            point_cloud2 = Pointclouds(points=v, features=tr.ones_like(v) * tr.tensor([[0, 1, 0]]).to(args.device))
            point_cloud3 = Pointclouds(points=[lmk_3d2], features=[tr.ones_like(lmk_3d2) * tr.tensor([[0, 0, 1]]).to(args.device)])
            # render both in the same plot in different traces
            fig = plot_scene({
                "Pointcloud": {
                    "lmk_3d": point_cloud1,
                        "v": point_cloud2,
                    "lmk_3d_mapping": point_cloud3
                }
            })
            fig.show()
            #exit()    
    
    
    
    
    
    
    return lmk_3d



def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent, testpath, scene):
    cam_folder = os.path.join(point_folder, 'cams')
    image_folder = os.path.join(point_folder, 'images')
    
    scan_folder = os.path.join(point_folder, '3D_Scan')
    
    depth_min = 0.0001
    depth_max = 100000000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + point_folder + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    #print(cmd)
    os.system(cmd)
    cmd = "PoissonRecon --in " + scan_folder + "/point_clouds.ply --out " + scan_folder + "/mesh9.ply" + " --depth 9 --threads 10"
    t = time.time()
    os.system(cmd)
    e = time.time()
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(scan_folder + '/mesh9.ply')
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag = pymeshlab.Percentage(20))
    ms.save_current_mesh(scan_folder + '/filtered_mesh_9.obj')
    
    ##for Poisson8
    cmd = "PoissonRecon --in " + scan_folder + "/point_clouds.ply --out " + scan_folder + "/mesh8.ply" + " --depth 8 --threads 10"
    t = time.time()
    os.system(cmd)
    e = time.time()
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(scan_folder + '/mesh8.ply')
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag = pymeshlab.Percentage(20))
    ms.meshing_surface_subdivision_loop(iterations = 1, threshold = pymeshlab.Percentage(0))
    
    ms.save_current_mesh(scan_folder + '/filtered_mesh_8_sub.obj')
    
    

    if args.face_opt:
        thf = Process(target=face_optim, args=(testpath, scene, "filtered_mesh_8_sub.obj"))
        thf.start()    
        
        # thf2 = Process(target=face_optim, args=(testpath, scene, "filtered_mesh_9.obj"))
        # thf2.start() 
        
    
    
    #cmd = "python /root/lhs/face-parsing.PyTorch/test.py -p " + point_folder 
    
    if args.faceseg:
        cmd = "python /root/lhs/eye_mask/face_mask.py -p " + point_folder 
        os.system(cmd)
        
        image_folder = os.path.join(point_folder, 'seg_img')
        cmd = fusibile_exe_path
        cmd = cmd + ' -input_folder ' + point_folder + '/'
        cmd = cmd + ' -p_folder ' + cam_folder + '/'
        cmd = cmd + ' -images_folder ' + image_folder + '/'
        cmd = cmd + ' --depth_min=' + str(depth_min)
        cmd = cmd + ' --depth_max=' + str(depth_max)
        cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
        cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
        cmd = cmd + ' --num_consistent=' + str(num_consistent)
        #print(cmd)
        os.system(cmd)
        cmd = "PoissonRecon --in " + scan_folder + "/point_clouds.ply --out " + scan_folder + "/mesh_seg.ply" + " --depth 9 --threads 10"
        t = time.time()
        os.system(cmd)
        e = time.time()
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(scan_folder + '/mesh_seg.ply')
        ms.meshing_remove_connected_component_by_diameter()
        ms.save_current_mesh(scan_folder + '/filtered_mesh_seg.obj')
    print("PoissonRecon time: ", e - t)
    print("Writing ply file ", scan_folder + "/mesh.ply")
    print("store 3D mesh to ply file" )
    
    if args.facealign:
        lmk_2d_folder = os.path.join(point_folder, '2d_landmark')
        lmk_3d = Extract_3d_landmark(lmk_2d_folder,[1,2,3,4])
        
        
        vs, fs, cs = obj_read_w_color(scan_folder + '/filtered_mesh.obj')
        print(np.shape(vs))
        vertex_mean = np.mean(vs, 0)
        print(vertex_mean)
        #print(point_folder[:-16]+'/cams/*_cam.txt')
        cam_list = sorted(glob.glob(point_folder[:-16]+'/cams/*_cam.txt'))
        R = np.zeros((len(cam_list),3,3))
        T = np.zeros((len(cam_list),3,1))
        K = np.zeros((len(cam_list),3,3))
        RTT = np.zeros((len(cam_list),4,4))
        for i, filename in enumerate(cam_list):
            Ki, RT = read_camera_parameters(filename)
            R[i,:,:] = RT[:3,:3]
            T[i,:,0] = RT[:3,3]
            K[i,:,:] = Ki[:,:]
            RTT[i,:,:] = RT[:,:]
        # print(R)
        # print(T)   
        # print(K)      
        
        #R_world = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        theta_x = np.radians(0)
        theta_y = np.radians(0)

        # Create rotation matrices for x and y axes
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]])

        R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]])

        # Compute combined rotation matrix
        R_world = np.dot(R_x, R_y)
        
        
        
        r_mesh = torch.matmul(torch.Tensor(vs), torch.Tensor(R_world).T)
        #np.mean(r_mesh.to('cpu').numpy(),0)
        new_origin = np.mean(r_mesh.numpy(),0).T
        #new_origin = vertex_mean.T
        #new_origin = np.array([0,0,0])
        print(new_origin)
        #new_origin = -np.array([-1100,200,-500]).astype(np.float64)
        #print(new_origin)
        
        
        R_world_inv = np.linalg.inv(R_world)
        T_world_inv = -np.matmul(R_world_inv, np.array([0,0,0])).astype(np.float64)
        T_world_inv2 = -np.matmul(np.identity(3), new_origin).astype(np.float64)
        for i, filename in enumerate(cam_list): 
            
            
            focal = K[i, 0, 0]

            # Define the transformation matrix that maps points from the new world coordinate system to the camera coordinate system
            M_world_to_cam = np.array([
                [R[i, 0, 0], R[i, 0, 1], R[i, 0, 2], T[i, 0, 0]],
                [R[i, 1, 0], R[i, 1, 1], R[i, 1, 2], T[i, 1, 0]],
                [R[i, 2, 0], R[i, 2, 1], R[i, 2, 2], T[i, 2, 0]],
                [0, 0, 0, 1]
            ])
            M_new_world_to_cam = np.array([
                [R_world[0, 0], R_world[0, 1], R_world[0, 2], T_world_inv[0]],
                [R_world[1, 0], R_world[1, 1], R_world[1, 2], T_world_inv[1]],
                [R_world[2, 0], R_world[2, 1], R_world[2, 2], T_world_inv[2]],
                [0, 0, 0, 1]
            ])
            
            M_new_cam_to_world = np.linalg.inv(M_new_world_to_cam.astype(np.float64))
            M_new_world_to_cam = np.matmul(M_world_to_cam, M_new_cam_to_world)
            # Extract the rotation matrix R_new, translation vector T_new, and intrinsic camera matrix K_new from the new transformation matrix
            R_new = M_new_world_to_cam[:3, :3]
            T_new = M_new_world_to_cam[:3, 3:]
            
            
            M_world_to_cam = np.array([
                [R_new[0, 0], R_new[0, 1], R_new[0, 2], T_new[0, 0]],
                [R_new[1, 0], R_new[1, 1], R_new[1, 2], T_new[1, 0]],
                [R_new[2, 0], R_new[2, 1], R_new[2, 2], T_new[2, 0]],
                [0, 0, 0, 1]
            ])
            M_new_world_to_cam = np.array([
                [1, 0, 0, T_world_inv2[0]],
                [0, 1, 0, T_world_inv2[1]],
                [0, 0, 1, T_world_inv2[2]],
                [0, 0, 0, 1]
            ])
            
            M_new_cam_to_world = np.linalg.inv(M_new_world_to_cam.astype(np.float64))
            M_new_world_to_cam = np.matmul(M_world_to_cam, M_new_cam_to_world)
            # Extract the rotation matrix R_new, translation vector T_new, and intrinsic camera matrix K_new from the new transformation matrix
            R_new = M_new_world_to_cam[:3, :3]
            T_new = M_new_world_to_cam[:3, 3:]
            
            
            
            K_new = K[i,:,:].copy()
            #K_new[0, 2] = -T_new[0, 0] / T_new[2, 0] * focal + K[i, 0, 2]
            #K_new[1, 2] = -T_new[1, 0] / T_new[2, 0] * focal + K[i, 1, 2]
            
            cam = np.zeros((2,4,4))
            cam[0,:3,:3] = R_new
            cam[1,:3,:3] = K_new
            cam[0,:3, 3] = T_new.T
            cam[0, 3, 3] = 1
            #print(filename[:-17] +"_align")
            os.makedirs(filename[:-17]+"_align", exist_ok= True)
            newfile = filename.replace("cams", 'cams_align')
            f = open(newfile, "w")
            f.write('extrinsic\n')
            for i in range(0, 4):
                for j in range(0, 4):
                    f.write(str(cam[0][i][j]) + ' ')
                f.write('\n')
            f.write('\n')

            f.write('intrinsic\n')
            for i in range(0, 3):
                for j in range(0, 3):
                    f.write(str(cam[1][i][j]) + ' ')
                f.write('\n')

            f.write('\n' + str(900) + ' ' + str(1.5))

            f.close()        
                #meshing_remove_connected_component_by_diameter
    
    return
   
def face_optim(data_path, data_name, meshname):

    cmd = "python /root/lhs/optimization/main_face.py --data_path " + data_path + " --data_name " + data_name + " --obj_filename " + meshname + " --iters 30" 
    print(cmd)
    os.system(cmd)



def data_loader_thread(testpath, scene, num_view, numdepth, max_h, max_w, fix_res, dataset):
    MVSDataset = find_dataset_def(dataset)
    
    tmp_test_dataset = MVSDataset(testpath, [scene], "test", num_view, numdepth, Interval_Scale,max_h=max_h, max_w=max_w, fix_res=fix_res)
    
    return tmp_test_dataset        


def save_depth(testlist):
    model = CascadeMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                        share_cr=args.share_cr,
                        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                        grad_method=args.grad_method)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    start_time = time.time()
    state_dict = torch.load(args.loadckpt, map_location='cpu')
    model.load_state_dict(state_dict['model'], strict=True)

    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    end_time = time.time()
    print('Model_loaded Parallel Time: ' + str(end_time - start_time))
    MVSDataset = find_dataset_def(args.dataset)
    
    
    pool = ThreadPool(processes=1)
    for i, scene in enumerate(testlist):
        print(scene)
        if i == 0:
            test_dataset = MVSDataset(args.testpath, [scene], "test", args.num_view, args.numdepth, Interval_Scale,max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
        
        if i+1 < len(testlist):
            tmp_test_dataset = pool.apply_async(data_loader_thread, (args.testpath, testlist[i+1], args.num_view, args.numdepth, args.max_h, args.max_w, args.fix_res, args.dataset))

        
        t = time.time()
        save_scene_depth(model, test_dataset)
        t1 = time.time()

        print("iter : ", t1 - t)
        locals()['th_{}'.format(i)] = Process(target=depth_map_fusion, args=(args.testpath+'_output' + "/" +scene, args.fusibile_exe_path, args.disp_threshold, args.num_consistent, args.testpath, scene))
        locals()['th_{}'.format(i)].start()

        
        #pc2mesh('./outputs/scan1444')
        
        if i+1 < len(testlist):
            test_dataset = tmp_test_dataset.get()
        
    for i, scene in enumerate(testlist):    
        locals()['th_{}'.format(i)].join()
        

if __name__ == '__main__':

    start_time = time.time()
    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        #for tanks & temples or eth3d or colmap
        testlist = []
        for i in range(82,138):
            print(i)
            if i != 86:
                testlist.append("scan"+format(i,"04")) 
        # testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
        #     if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    
    
    save_depth(testlist)
    end_time = time.time()  


    print("Total time", str(end_time - start_time))


