import pymeshlab
import time
import argparse
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./', help='path')
parser.add_argument('--method', default='meshlab', help='path')
args = parser.parse_args()

def pc2mesh(scan_folder, rec_method):
    if rec_method == 'meshlab':
        ms = pymeshlab.MeshSet()
        s0 = time.time()
        ms.load_new_mesh(scan_folder + "/point_clouds.ply")
        s1 = time.time()
        #ms.apply_filter("compute_normal_for_point_clouds",k = 50)
        s2 = time.time()
        #ms.apply_filter("apply_normal_point_cloud_smoothing",k = 50)
        s3 = time.time()
        ms.apply_filter("generate_surface_reconstruction_screened_poisson",depth = 9)
        s4 = time.time()
        ms.save_current_mesh(scan_folder + "/mesh.obj")
        s5 = time.time()
    else:
        pcd = o3d.geometry.PointCloud()
        s0 = time.time()
        pcd = o3d.io.read_point_cloud(scan_folder + "/point_clouds.ply")
        s1 = time.time()
        #pcd.estimate_normals()
        s2 = time.time()
        
        s3 = time.time()
        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, width=0, scale=1.1, linear_fit=True)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        p_mesh_crop = poisson_mesh.crop(bbox)
        s4 = time.time()
        o3d.io.write_triangle_mesh(scan_folder + "/mesh.ply", p_mesh_crop)
        s5 = time.time()

    print("load time: ", s1 - s0)
    print("compute normals: ", s2 - s1)
    print("smoothing normals: ", s3 - s2)
    print("poisson recon time: ", s4 - s3)
    print("save time: ",s5 - s4 )



if __name__ == '__main__':
    scan_folder = args.path 
    rec_method = args.method 
    pc2mesh(scan_folder, rec_method)