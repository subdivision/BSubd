import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from DCELCore import DCtrlMesh
#from BSplineAvg import bspline_average_export_3D

INP_PATH_PREFIX = 'C:/TAU/InputMeshes/'
RES_PATH_PREFIX = 'C:/TAU/DebugMeshes/'

#-----------------------------------------------------------------------------
def blend_meshes(a_cm, b_cm):
    ''' a_cm get modified '''
    min_d, max_d, avg_d = a_cm.get_corresp_mesh_dist(b_cm)
    a_cm.morph_to_corresp_mesh(b_cm, max_d)

#-----------------------------------------------------------------------------
def plot_results(orig_ctrl_mesh, circ_avg_ctrl_mesh, lin_ctrl_mesh):
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d', aspect='equal')
    ax.view_init(azim=0, elev=0)

    if orig_ctrl_mesh:
        orig_ctrl_mesh.plot(ax, True, 'k')
    if circ_avg_ctrl_mesh:
        circ_avg_ctrl_mesh.plot(ax, True, 'r')
    if lin_ctrl_mesh:
        lin_ctrl_mesh.plot(ax, False, 'b')

    plt.show()

#-----------------------------------------------------------------------------
def avg_fn_to_str(avg_fn):
    #return '_circ_' if avg_fn == circle_avg_3D else '_bez_'
    return '_bez_'
#-----------------------------------------------------------------------------
def create_crystal3_mesh(id, avg_func):
    file_prefix = 'crystal_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    offset = 10.
    fktr = 3.
    orig_ctrl_mesh.init_as_tetrahedron(offset)
    orig_ctrl_mesh.extrude_3triang_face(5, fktr * offset)
    #orig_ctrl_mesh.extrude_3triang_face(6, fktr * offset)
    orig_ctrl_mesh.extrude_3triang_face(7, fktr * offset)
    #orig_ctrl_mesh.extrude_3triang_face(8, fktr * offset)
    fids = [f.eid for f in orig_ctrl_mesh.f]
    for i in fids:
        orig_ctrl_mesh.extrude_3triang_face(i, 0)
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tetrahedron3_mesh(id, avg_func):
    file_prefix = 'tetrahedron_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_tetrahedron()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tetrahedron4_mesh(id, avg_func):
    file_prefix = 'tetrahedron_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_tetrahedron()
    orig_ctrl_mesh = orig_ctrl_mesh.refine_as_catmull_clark(\
        get_edge_vertex_func = DCtrlMesh.get_edge_vertex_as_mid,
        get_vrtx_vertex_func = DCtrlMesh.get_vrtx_vertex_as_copy)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tower4_mesh(id, avg_func):
    file_prefix = 'tower_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_quad_cube(10.)
    orig_ctrl_mesh.extrude_face(9, 20.)
    orig_ctrl_mesh.extrude_face(30, 20.)
    orig_ctrl_mesh.extrude_face(34, 20.)
    orig_ctrl_mesh.set_naive_normals()
    #orig_ctrl_mesh.print_ctrl_mesh()
    return orig_ctrl_mesh, file_prefix

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection = '3d', aspect='equal')
    ax.view_init(azim=30, elev=25)
    orig_ctrl_mesh.plot(ax, True, 'k')
    plt.show()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tube4_mesh(id, avg_func):
    file_prefix = 'tube_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_quad_cube(10.)
    orig_ctrl_mesh.extrude_face(9, 20.)
    dux = 60.
    duy = 60.
    duz = 240.
    orig_ctrl_mesh.id2obj[27].set_pt(np.array([ dux,  duy, duz]))
    orig_ctrl_mesh.id2obj[31].set_pt(np.array([-dux,  duy, duz]))
    orig_ctrl_mesh.id2obj[35].set_pt(np.array([-dux, -duy, duz]))
    orig_ctrl_mesh.id2obj[39].set_pt(np.array([ dux, -duy, duz]))
    dux = 60.
    duy = 60.
    duz = 0.4
    orig_ctrl_mesh.id2obj[5].set_pt(np.array([ dux,  duy, -duz]))
    orig_ctrl_mesh.id2obj[6].set_pt(np.array([ dux, -duy, -duz]))
    orig_ctrl_mesh.id2obj[7].set_pt(np.array([-dux, -duy, -duz]))
    orig_ctrl_mesh.id2obj[8].set_pt(np.array([-dux,  duy, -duz]))

    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tube3_mesh(id, avg_func):
    file_prefix = 'tube_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh, _ = create_tube4_mesh(id, avg_func)
    orig_ctrl_mesh = orig_ctrl_mesh.triangulize_quad_mesh()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_tower3_mesh(id, avg_func):
    file_prefix = 'tower_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh, _ = create_tower4_mesh(id - 1, avg_func)
    orig_ctrl_mesh = orig_ctrl_mesh.triangulize_quad_mesh()
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_cube4_mesh(id, avg_func):
    file_prefix = 'cube_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_quad_cube(10.)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_cube3_mesh(id, avg_func):
    file_prefix = 'cube_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_triang_cube(10.)
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_torus4_mesh(id, avg_func):
    global n_of_verts_in_init_torus
    file_prefix =   'torus'+ str(n_of_verts_in_init_torus) \
                  + '_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_torus(False, n_of_verts = n_of_verts_in_init_torus)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_torus3_mesh(id, avg_func):
    global n_of_verts_in_init_torus
    file_prefix =   'torus'+ str(n_of_verts_in_init_torus) \
                  + '_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    orig_ctrl_mesh.init_as_torus(True, n_of_verts = n_of_verts_in_init_torus)
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_mesh3_stl_file(id, avg_func):
    global stl_file_name
    file_prefix = stl_file_name + '_3' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    inp_file = INP_PATH_PREFIX + stl_file_name
    orig_ctrl_mesh.init_as_triang_mesh_stl_file(inp_file)
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def create_mesh4_stl_file(id, avg_func):
    global stl_file_name
    file_prefix = stl_file_name + '_4' + avg_fn_to_str(avg_func)
    orig_ctrl_mesh = DCtrlMesh(id, avg_func)
    inp_file = INP_PATH_PREFIX + stl_file_name
    orig_ctrl_mesh.init_as_triang_mesh_stl_file(inp_file)
    orig_ctrl_mesh = orig_ctrl_mesh.refine_as_catmull_clark(\
        get_edge_vertex_func = DCtrlMesh.get_edge_vertex_as_mid,
        get_vrtx_vertex_func = DCtrlMesh.get_vrtx_vertex_as_copy)
    orig_ctrl_mesh.set_naive_normals()
    return orig_ctrl_mesh, file_prefix

#-----------------------------------------------------------------------------
def get_initial_mesh(demo_mesh, b_quadr = True):
    ''' 'tower', 'cube', 'torus', 'tube', 'mesh', 'tetra'
    '''
    global stl_file_name, n_of_verts_in_init_torus
    n_of_verts_in_init_torus = 6
    #stl_file_name = 'fox.stl'
    stl_file_name = 'tiger.stl'
    #stl_file_name = 'bunny.stl'
    #stl_file_name = 'cube.stl'

    demos = {('tower', True)  : create_tower4_mesh,
             ('tower', False) : create_tower3_mesh,
             ('cube',  True)  : create_cube4_mesh,
             ('cube',  False) : create_cube3_mesh,
             ('torus', True)  : create_torus4_mesh,
             ('torus', False) : create_torus3_mesh,
             ('tube',  True)  : create_tube4_mesh,
             ('tube',  False) : create_tube3_mesh,
             ('mesh',  True)  : create_mesh4_stl_file,
             ('mesh',  False) : create_mesh3_stl_file,
             ('tetra', True)  : create_tetrahedron4_mesh,
             ('tetra', False) : create_tetrahedron3_mesh,
             ('cryst', True)  : None,
             ('cryst', False) : create_crystal3_mesh}

    bez_avg_ctrl_mesh, bez_res_name = \
        demos[(demo_mesh, b_quadr)](2, None)
    bez_avg_ctrl_mesh.init_tangent_dirs()

    return bez_avg_ctrl_mesh, bez_res_name


#-----------------------------------------------------------------------------
def srf_main():
    n_of_iterations = 3
    b_quad = True
    #example_name = 'tower'
    example_name = 'cube'
    #example_name = 'torus'
    #example_name = 'tube'
    #example_name = 'mesh'
    #example_name = 'tetra'

    res_file_suffix = 'bez_' + str(n_of_iterations) + 'iters.off'
    bez_avg_ctrl_mesh, bez_res_name = get_initial_mesh(example_name, b_quad)

    orig_ctrl_mesh, _ = get_initial_mesh(example_name, b_quad)
    orig_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + example_name + '_orig.off')

    bez_res_name += res_file_suffix

    for i in range(n_of_iterations):
        bez_avg_ctrl_mesh = bez_avg_ctrl_mesh.refine_by_bspl_interpolation()

    #cmda, ccad, cmmd =  circ_avg_ctrl_mesh.get_dehidral_angle_stats()[1],\
    #                    circ_avg_ctrl_mesh.get_gaus_curvature_abs_delta(),\
    #                    circ_avg_ctrl_mesh.get_mesh_to_mesh_dist(orig_ctrl_mesh)
    #lmda, lcad, lmmd =  lin_ctrl_mesh.get_dehidral_angle_stats()[1],\
    #                    lin_ctrl_mesh.get_gaus_curvature_abs_delta(),\
    #                    lin_ctrl_mesh.get_mesh_to_mesh_dist(orig_ctrl_mesh)
    #print 'Linear {:1.5f} & {:1.5f} & {:1.5f}'.format(lmda, lcad, lmmd)
    #print 'Circle {:1.5f} & {:1.5f} & {:1.5f}'.format(cmda, ccad, cmmd)

    plot_results(orig_ctrl_mesh, bez_avg_ctrl_mesh)   

    #blend_meshes(circ_avg_ctrl_mesh, lin_ctrl_mesh)

    bez_avg_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + bez_res_name)
    #lin_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + lin_res_name)
    #d = lin_ctrl_mesh.get_corresp_mesh_dist(circ_avg_ctrl_mesh)
    #print d
    
    #Tube3 butterfly, 3iters, v1
    #Linear    2.94997 & 0.58621 & 56.81865
    #Circle v1 1.08278 & 0.09594 & 55.79320
    #Circle v2 1.10755 & 0.07216 & 55.76661
    #Circle v3 1.16634 & 0.07225 & 55.83580

#-----------------------------------------------------------------------------
def rotate_normals():
    n_of_iterations = 4
    ref_method, ref_name = DCtrlMesh.refine_as_catmull_clark, 'cc_'
    #ref_method, ref_name = DCtrlMesh.refine_as_kob4pt, 'kob4pt_'
    #ref_method, ref_name = DCtrlMesh.refine_as_butterfly, 'butterfly_'
    #ref_method, ref_name = DCtrlMesh.refine_as_loop, 'loop_'

    for w in np.linspace(0, 1, 11):
        circ_avg_ctrl_mesh, circ_res_name, _, _ = \
            get_initial_mesh('cube', True)
        #circ_avg_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + 'cube4.off')
        circ_avg_ctrl_mesh.init_normals(np.array([1., 1., 1.]))
        circ_avg_ctrl_mesh.set_naive_normals(w)

        res_file_suffix = ref_name + 'wei_' + str(w) + '_'\
                          + str(n_of_iterations)\
                          + 'iters.off'
        circ_curr_res_name = circ_res_name + res_file_suffix

        for i in range(n_of_iterations):
            circ_avg_ctrl_mesh = ref_method(circ_avg_ctrl_mesh)
       
        circ_avg_ctrl_mesh.dump_obj_file(RES_PATH_PREFIX + circ_curr_res_name)
    
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    srf_main()
    #rotate_normals()
#============================= END OF FILE ===================================