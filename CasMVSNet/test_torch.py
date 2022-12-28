import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn

import torch.nn.parallel

#torch.distributed.init_process_group(backend="nccl",world,rank = 0)

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from gipuma import gipuma_filter

from multiprocessing import Pool
from functools import partial
import signal

cudnn.benchmark = True
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2,3'
#device = torch.device("cuda:1")

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='general_eval', help='select dataset')
parser.add_argument('--testpath', default="data/",help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', default="lists/our_list.txt", help='testing scene list')

parser.add_argument('--local_rank',help='testing data dir for some scenes')

parser.add_argument('--batch_size', type=int, default=4, help='testing batch size')#7 GPUs can use maximum 18 batches +- 2
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')#192 no significant performance improvement even if increased

parser.add_argument('--loadckpt', default="/hdd1/lhs/dev/code/cascade-stereo/casmvsnet.ckpt", help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')#48 32 8 no significant performance improvement even if increased
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--interval_scale', type=float, default=3.0, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')#2 is blur, 10 is too much
parser.add_argument('--max_h', type=int, default=1024, help='testing max h')#864
parser.add_argument('--max_w', type=int, default=1024, help='testing max w')#1152
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=10, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=60, help='save freq of local pcd')


parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")

#filter
parser.add_argument('--conf', type=float, default=0.9, help='prob confidence')
parser.add_argument('--thres_view', type=int, default=0, help='threshold of num view')

#filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='../../fusibile/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.9')
parser.add_argument('--disp_threshold', type=float, default='0.2')
parser.add_argument('--num_consistent', type=float, default='3')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])

Interval_Scale = args.interval_scale
print("***********Interval_Scale**********\n", Interval_Scale)


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    extrinsics = torch.from_numpy(extrinsics)
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    intrinsics =torch.from_numpy(intrinsics)

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

def save_depth(testlist):

    for scene in testlist:
        save_scene_depth([scene])

# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist):
    # dataset, dataloader

    MVSDataset = find_dataset_def(args.dataset)


    test_dataset = MVSDataset(args.testpath, testlist, "test", args.num_view, args.numdepth, Interval_Scale,
                              max_h=args.max_h, max_w=args.max_w, fix_res=args.fix_res)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=10, drop_last=False)


    # model

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
    #model = nn.parallel.DistributedDataParallel(model)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()
    end_time = time.time()
    print('Model_loaded Parallel Time: ' + str(end_time - start_time))

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
            for filename, cam, img, depth_est, photometric_confidence in zip(filenames, cams, imgs, \
                                                            outputs["depth"], outputs["photometric_confidence"]):
                img = img[0]  #ref view
                cam = cam[0]  #ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                #save depth maps
                save_pfm(depth_filename, depth_est)
                #save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                #save cams, img
                write_cam(cam_filename, cam)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_filename, img_bgr)

    gc.collect()
    torch.cuda.empty_cache()



# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    print(width)
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = torch.meshgrid(torch.arange(0, height), torch.arange(0,width))
    print(x_ref)
    print("step00")
    x_ref, y_ref = torch.reshape(x_ref,(-1,)), torch.reshape(y_ref,(-1,))
    # reference 3D space
    print("step0")
    xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref),
                        torch.vstack((x_ref.cuda(), y_ref.cuda(), torch.ones_like(x_ref).cuda())) * depth_ref.reshape([-1]))
    # source 3D space (from xyz_ref)
    print("step1")
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.linalg.inv(extrinsics_ref)),
                        torch.vstack((xyz_ref, torch.ones_like(x_ref).cuda())))[:3]

    print("step2")
    # source view x, y (projected reference 3d point to source view)
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width])
    y_src = xy_src[1].reshape([height, width])

    print(y_src)

    # index = torch.bucketize(depth_src.ravel(), x_src)
    # sampled_depth_src = y_src[index].reshape(depth_src.shape)
    #

    #sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    sampled_depth_src = depth_src

    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src),
                        torch.vstack((xy_src, torch.ones_like(x_ref).cuda())) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)),
                                torch.vstack((xyz_src, torch.ones_like(x_ref).cuda())))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width])
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width])
    y_reprojected = xy_reprojected[1].reshape([height, width])

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src



#for gpu
def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    print(width)

    x_ref, y_ref = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
    x_ref = x_ref.cuda()
    y_ref = y_ref.cuda()
    print(x_ref)

    depth_ref = depth_ref.to("cuda:0")
    intrinsics_ref = intrinsics_ref.to("cuda:0")
    extrinsics_ref = extrinsics_ref.to("cuda:0")
    depth_src = depth_src.to("cuda:0")
    intrinsics_src = intrinsics_src.to("cuda:0")
    extrinsics_src = extrinsics_src.to("cuda:0")
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)

    # print(x2d_reprojected)
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = torch.logical_and(dist < 1, relative_depth_diff < 0.01)
    #mask = np.logical_and(1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0
    return mask, depth_reprojected, x2d_src, y2d_src



def filter_depth(pair_folder, scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views

    for ref_view, src_views in pair_data:
        # src_views = src_views[:args.num_view]
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)))
        # load the reference image
        img_filename = os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view))
        if not os.path.exists(img_filename):
            img_filename = os.path.join(scan_folder, 'images/{:0>8}.png'.format(ref_view))
        ref_img = read_img(img_filename)
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]

        # use gt mask
        #gt_mask = read_img(os.path.join(out_folder, 'mask/{:0>8}.png'.format(ref_view))).astype(np.uint8)

        photo_mask = confidence > args.conf



        # use gt mask
        #photo_mask = np.logical_and(photo_mask, gt_mask)



        photo_mask = torch.from_numpy(photo_mask.copy())

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        ref_depth_est = torch.from_numpy(ref_depth_est.copy())
        ref_intrinsics = ref_intrinsics.to("cuda:0")
        ref_extrinsics = ref_extrinsics.to("cuda:0")

        # for src_view in src_views:
        #     # camera parameters of the source view
        #     src_intrinsics, src_extrinsics = read_camera_parameters(
        #         os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)))
        #     # the estimated depth of the source view
        #
        #     src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
        #
        #
        #
        #
        #     src_depth_est = torch.from_numpy(src_depth_est.copy())
        #
        #
        #     geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics,
        #                                                               src_depth_est,
        #                                                               src_intrinsics, src_extrinsics)
        #
        #
        #     print("ahh")
        #
        #     geo_mask_sum += geo_mask
        #
        #     all_srcview_depth_ests.append(depth_reprojected)
        #
        #
        #
        #     all_srcview_x.append(x2d_src)
        #     all_srcview_y.append(y2d_src)
        #     all_srcview_geomask.append(geo_mask)
        #

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)




        # at least 3 source views matched
        #geo_mask = geo_mask_sum >= args.thres_view

        #final_mask = torch.logical_and(photo_mask, geo_mask)
        final_mask = photo_mask

        print("processing")

        #use no filter
        #depth_est_averaged = ref_depth_est

        height, width = depth_est_averaged.shape[:2]
        x, y = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))

        # print(x.size())
        # print(y.size())
        # print(depth_est_averaged.size())

        #use only gt mask
        # valid_points = gt_mask.astype(np.bool)

        valid_points = final_mask

        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]


        x = x.cuda()
        y = y.cuda()
        depth = depth.cuda()

        #color = ref_img[1:-16:4, 1::4, :][valid_points]  # hardcoded for DTU dataset

        if num_stage == 1:
            color = ref_img[1::4, 1::4, :][valid_points]
        elif num_stage == 2:
            color = ref_img[1::2, 1::2, :][valid_points]
        elif num_stage == 3:
            color = ref_img[valid_points]

        xyz_ref = torch.matmul(torch.linalg.inv(ref_intrinsics),
                            torch.vstack((x, y, torch.ones_like(x).cuda())) * depth)

        xyz_world = torch.matmul(torch.linalg.inv(ref_extrinsics),
                              torch.vstack((xyz_ref, torch.ones_like(x).cuda())))[:3]


        #print(np.shape(xyz_world))
        vertexs.append(torch.transpose(xyz_world,1, 0).cpu().detach().numpy())
        vertex_colors.append((color * 255).astype(np.uint8))




#################

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)



    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)



    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def pcd_filter_worker(scan):
    print(scan)
    if args.testlist != "all":
        scan_id = int(scan[4:])
        save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)
    else:
        save_name = '{}.ply'.format(scan)
    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))


def pcd_filter(testlist, number_worker):

    partial_func = partial(pcd_filter_worker)

    p = Pool(number_worker, init_worker)
    try:
        p.map(partial_func, testlist)
    except KeyboardInterrupt:
        print("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()

if __name__ == '__main__':
    start_time0 = time.time()

    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        #for tanks & temples or eth3d or colmap
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    save_depth(testlist)

    # step2. filter saved depth maps with photometric confidence maps and geometric constraints
    start_time = time.time()
    #if args.filter_method != "gipuma":
        #support multi-processing, the default number of worker is 4
    #pcd_filter(testlist, args.num_worker)


    scan = "scan14404"
    scan_id = 14404
    save_name = 'mvsnet{:0>3}_l3.ply'.format(scan_id)

    pair_folder = os.path.join(args.testpath, scan)
    scan_folder = os.path.join(args.outdir, scan)
    out_folder = os.path.join(args.outdir, scan)
    filter_depth(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))

    #else:
    # gipuma_filter(testlist, args.outdir, args.prob_threshold, args.disp_threshold, args.num_consistent,
    #                    args.fusibile_exe_path)

    end_time = time.time()
    print("Depth Total time", str(start_time - start_time0))

    print("Fusion Total time", str(end_time-start_time))

    print("Total time", str(end_time - start_time0))