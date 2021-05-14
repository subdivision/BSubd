import numpy as np

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.path as Path
import matplotlib.patches as patches

from BSubdAvg import BezierCrv
from CircAvg2D import *

#-----------------------------------------------------------------------------
def get_angle_two_vectors_3D(v1, v2):
    ''' assumption: v1, v2 normalized'''
    cos_gamma = np.min(np.dot(v1, v2), 1.)
    gamma = np.arccos(cos_gamma)
    #gamma = gamma * 180./np.pi
    return gamma

#-----------------------------------------------------------------------------
def geodesic_avg_two_vectors_3D(v1, v2, w):
    ''' assumption: v1, v2 normalized'''
    if eeq(w, 0.):
        return v0
    elif eeq(w, 1.):
        return v1

    gamma = get_angle_two_vectors_3D(v1, v2)
    gamma *= (1.- w)

    nr = np.cross(v1, v2)
    nr /= np.linalg.norm(nr)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)
    res_vec = v1 * cos_gamma \
                    + np.cross(nr, v1) * sin_gamma\
                    + np.dot(nr, v1) * nr * (1. - cos_gamma)
    res_vec /= np.linalg.norm(res_vec)
    return res_vec


#-----------------------------------------------------------------------------
class PntTng(object):
    def __init__(self, pt, nr = None):
        self.pt = pt.copy()
        self.nr = nr.copy() if nr is not None else None
        self.left_tg = None
        self.right_tg = None

    def setNormal(self, nr):
        self.nr = nr.copy()

    def getDistanceTo(self, other):
        return np.linalg.norm(self.pt - other.pt)

    def setTangents(self, prev_pt, next_pt):
        v_left = prev_pt.pt - self.pt
        v_left /= np.linalg.norm(v_left)
        v_right = next_pt.pt - self.pt
        v_right /= np.linalg.norm(v_right)
        bnorm = np.cross(v_left, v_right)
        self.right_tg = np.cross(self.nr, bnorm)
        self.left_tg = -(self.right_tg.copy())

    def setNaiveNorm(self, prev_pt, next_pt):
        v_prev = prev_pt.pt - self.pt
        prev_len = np.linalg.norm(v_prev)
        v_prev /= prev_len
        v_next = next_pt.pt - self.pt
        next_len = np.linalg.norm(v_next) 
        v_next /= next_len
        bnorm = np.cross(v_prev, v_next)
        prev_perp = np.cross(v_prev, bnorm)
        next_perp = np.cross(bnorm, v_next)
        w = prev_len / (prev_len + next_len)
        self.nr = geodesic_avg_two_vectors_3D(prev_perp, next_perp, w)

#-----------------------------------------------------------------------------
def init_tangents(pnts, b_open):
    i = 0
    N = len(pnts)

    for i in range(N):
        pnts[i].setTangents(pnts[(i-1+N)%N], pnts[(i+1)%N])


#-----------------------------------------------------------------------------
def create_helix_polyline(mesh_size = 10, low_bnd = -np.pi, upr_bnd = np.pi):
    step = (upr_bnd - low_bnd) / (mesh_size - 1);
    pnts = []
    b_open = True
    for i in range(mesh_size):
        t = low_bnd + i * step;
        x = M.cos( t )
        y = M.sin( t )
        z = t
        pt_obj = PntTng(np.array([x,y,z]))
        dx = -1.0 * M.sin(t)
        dy =  M.cos(t)
        dz =  1
        first_der = np.array([dx, dy, dz])
        ddx = -1.0 * M.cos(t)
        ddy = -1.0 * M.sin(t)
        ddz = 0
        second_der = np.array([ddx, ddy, ddz])
        bi_norm = np.cross(first_der, second_der)
        res_norm = np.cross(bi_norm, first_der)
        res_norm /= np.linalg.norm(res_norm)
        pt_obj.setNormal(-res_norm)
        pnts.append(pt_obj)

    return pnts, b_open

#-----------------------------------------------------------------------------
def create_input_on_a_polygon5():
    b_open = True #False
    pnts = []
    c30 = (3.0**0.5)/2.0
    pts = [np.array([-2., -2.,  -1.]),
           np.array([ 4., -3.,  1.]),
           np.array([ 2.,  0.,  0.]),
           np.array([ 5.,  3.5, -1.]),
           np.array([-1.,  3.,  1.])]

    nrm = [np.array([-1., -1., 0.]),
           np.array([ 1., -1., 0.1]),
           #np.array([  c30, -0.5, 0.5]), #rd
           #np.array([ c30, 0.5]), #ru
           np.array([ -c30, 0.5, 0.5]), #lu
           #np.array([ -c30, -0.5]), #ld
           np.array([ 1.,  1., -0.1]),
           np.array([-1.,  1., 0])]

    for p, n in zip(pts,nrm):
        pnts.append(PntTng(p, n/np.linalg.norm(n)))

    return pnts, b_open 

#-----------------------------------------------------------------------------
def plot_pts_and_norms_3D(pts, b_open, draw_norms, clr, label='a curve',
                          linestyle='', linewidth=1.0, cnvs = plt ):
    n = len(pts)
    xs = [pts[i].pt[0] for i in range(n) ]
    ys = [pts[i].pt[1] for i in range(n) ]
    zs = [pts[i].pt[2] for i in range(n) ]
    if not b_open:
        xs.append(pts[0].pt[0])
        ys.append(pts[0].pt[1])
        zs.append(pts[0].pt[2])
    cnvs.plot(xs, ys, zs, label=label, 
            color=clr, linestyle=linestyle, 
            linewidth=linewidth)
    if draw_norms:
        us = [pts[i].nr[0] for i in range(n) ]
        vs = [pts[i].nr[1] for i in range(n) ]
        ws = [pts[i].nr[2] for i in range(n) ]
        us_lt = [pts[i].left_tg[0] for i in range(n) ]
        vs_lt = [pts[i].left_tg[1] for i in range(n) ]
        ws_lt = [pts[i].left_tg[2] for i in range(n) ]
        vs_rt = [pts[i].right_tg[1] for i in range(n) ]
        us_rt = [pts[i].right_tg[0] for i in range(n) ]
        ws_rt = [pts[i].right_tg[2] for i in range(n) ]
        if not b_open:
            xs = xs[:-1]
            ys = ys[:-1]
            zs = zs[:-1]
        cnvs.quiver(xs,ys,zs,us,vs,ws, color=clr, pivot='tail')
        cnvs.quiver(xs,ys,zs,us_lt,vs_lt,ws_lt, color="green", pivot='tail')
        cnvs.quiver(xs,ys,zs,us_rt,vs_rt,ws_rt, color="cyan", pivot='tail')


#-----------------------------------------------------------------------------
def bqa_3D(t0, p0, p1):
    bez_crv = BezierCrv.make_bezier_crv(p0.pt, p0.nr, p0.right_tg, 
                                        p1.pt, p1.nr, p1.left_tg)
    res_pos = bez_crv.eval(t0)
    res_norm = bez_crv.norm(t0)
    res_obj = PntTng(res_pos, res_norm)
    res_obj.setTangents(p0, p1)
    return res_obj


#----------------------------------------------------------------------------
def double_polygon_bqa(pnts, b_preserve, b_opened):	
    i = 0
    res = []
    N = len(pnts)
    NN = N-1 if b_opened else N

    for i in range(NN):
        r = bqa_3D(0.5, pnts[i], pnts[(i+1)%N])
        if b_preserve:
            res.append(pnts[i])
        res.append(r)

    if b_preserve and b_opened:
        res.append(pnts[-1])

    return res

#-----------------------------------------------------------------------------
def curves_main_3D():
    n_of_iterations = 5
    #orig_pts, b_open = create_helix_polyline()
    orig_pts, b_open = create_input_on_a_polygon5()
    init_tangents(orig_pts, b_open)

    bsubd_INS_pts = orig_pts[:]
    for _ in range(n_of_iterations):
        bsubd_INS_pts   = double_polygon_bqa(bsubd_INS_pts, True, b_open)

    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.view_init(azim=0, elev=90)
    plot_pts_and_norms_3D(orig_pts, b_open, True, label = 'Input',
                       clr='k', linewidth=0.4, linestyle='dotted', cnvs=ax)
    plot_pts_and_norms_3D(bsubd_INS_pts, b_open, False, label = 'BQA Result',
                       clr='b', linewidth=0.6, linestyle='solid',cnvs=ax)
    ax.legend()
    plt.show()


#-----------------------------------------------------------------------------
if __name__ == "__main__":
    curves_main_3D()

#============================ END OF FILE ====================================
