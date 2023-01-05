import pymeshlab
import time

def ddd():
    ms = pymeshlab.MeshSet()
    s0 = time.time()
    ms.load_new_mesh("./outputs/scan1444/3D_Scan/point_clouds.ply")
    s1 = time.time()

    #ms.apply_filter("compute_normal_for_point_clouds",k = 50)
    s2 = time.time()

    #ms.apply_filter("apply_normal_point_cloud_smoothing",k = 50)
    s3 = time.time()

    ms.apply_filter("generate_surface_reconstruction_screened_poisson",depth = 9)
    s4 = time.time()
    ms.save_current_mesh('./result.obj')
    s5 = time.time()

    print("load time: ", s1 - s0)
    print("compute normals: ", s2 - s1)
    print("smoothin normals: ", s3 - s2)

    print("poisson recon time: ", s4 - s3)
    print("save time: ",s5 - s4 )

if __name__ == '__main__':
    ddd()