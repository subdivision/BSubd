import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as Path
import matplotlib.patches as patches


#-----------------------------------------------------------------------------
class BezierCrv():
    def __init__(self, a, b, c, d):
        self.a = a.copy()
        self.b = b.copy()
        self.c = c.copy()
        self.d = d.copy()

    @classmethod
    def make_bezier_crv(cls, p0, n0, d0, p1, n1, d1):
        a = p0
        d = p1
        theta = np.arccos( np.dot(n0, n1) )
        p0p1_dist = np.linalg.norm(p0 - p1)
        tang_len = p0p1_dist / (3. * ( np.cos(theta/4.) ** 2. ) )
        b = p0 + d0 * tang_len
        c = p1 + d1 * tang_len
        return cls(a, b, c, d)

    @classmethod
    def make_flipped(cls, other):
        return cls(other.d, other.c, other.b, other.a)

    def der(self, t):
        t2 = t * t
        mt = 1 - t
        mt2 = mt * mt
        a = 3.*(self.b - self.a)
        b = 3.*(self.c - self.b)
        c = 3.*(self.d - self.c)
        return a*mt2 + b*2*mt*t + c*t2

    def eval(self, t):
        t2 = t * t
        t3 = t2 * t
        mt = 1-t
        mt2 = mt * mt
        mt3 = mt2 * mt
        return self.a * mt3 + self.b * 3. * mt2 * t \
               + self.c * 3. * mt * t2 + self.d * t3

#-----------------------------------------------------------------------------
class CoonsPatch():
    def __init__(self, c0, c1, d0, d1):
        self.c0 = c0
        self.c1 = c1
        self.d0 = d0
        self.d1 = d1

    def _eval_Lc(self, u, v):
        return (1. - v) * self.c0.eval(u) + v * self.c1.eval(u)

    def _eval_dLc_du(self, u, v):
        return (1. - v) * self.c0.der(u) + v * self.c1.der(u)

    def _eval_dLc_dv(self, u, v):
        return self.c1.eval(u) - self.c0.eval(u)


    def _eval_Ld(self, u, v):
        return (1. - u) * self.d0.eval(v) + u * self.d1.eval(v)

    def _eval_dLd_du(self, u, v):
        return self.d1.eval(v) - self.d0.eval(v)

    def _eval_dLd_dv(self, u, v):
        return (1. - u) * self.d0.der(v) + u * self.d1.der(v)


    def _eval_B(self, u, v):
        return   (1. - u) * (1. - v) * self.c0.eval( 0. ) \
               + u * (1. - v) *  \
               + (1. - u) * v * self.c1.eval( 0. ) \
               + u * v * self.c1.eval( 1. )

    def _eval_dB_du(self, u, v):
        return    (1. - v) * (self.c0.eval( 1. ) - self.c0.eval( 0. ))\
                + v * (self.c1.eval( 1. ) - self.c1.eval( 0. ) )

    def _eval_dB_dv(self, u, v):
        return    (1. - u) * ( self.c1.eval( 0. ) - self.c0.eval( 0. ) ) \
                  + u * ( self.c1.eval( 1. ) - self.c0.eval( 1. ) ) 


    def _eval_pt(self, u, v):
        pt = self._eval_Lc( u, v ) + self._eval_Ld( u, v ) \
              - self._eval_B( u, v )    
        return pt

    def _eval_norm(self, u, v):
        dS_du = self._eval_dLc_du( u, v ) + self._eval_dLd_du( u, v ) \
              - self._eval_dB_du( u, v )    
        dS_dv = self._eval_dLc_dv( u, v ) + self._eval_dLd_dv( u, v ) \
              - self._eval_dB_dv( u, v )
        nr = np.cross( dS_du, dS_dv )
        nr /= np.linalg.norm( nr )
        return nr

    def eval(self, u, v):
        pt = self._eval_pt(u, v)
        nr = self._eval_norm(u, v)
        return pt, nr

#-----------------------------------------------------------------------------
#def cheb_nodes(N, a, b):
#    N = N - 2
#    jj = 2.*np.arange(N) + 1
#    c = np.cos(np.pi * jj / (2 * N) )[::-1]
#    x = 0.5*(b-a)*c + 0.5 * (a+b)
#    x = np.append(np.insert(x, 0, a), b)
#    return x

#-----------------------------------------------------------------------------
def extrapolate_bezier(a, b, c, d, t):
    v = 1. / (1. + 1./t)
    u = 1. - v
    A = a
    factor = 1. / u
    b_a_len = get_dist(a, b)
    b_a_uvec = (b - a)/b_a_len
    P = a + b_a_uvec * (b_a_len * factor)
    c_b_len = get_dist(b, c)
    c_b_uvec = (c - b)/c_b_len
    H = b + c_b_uvec * (c_b_len * factor)
    d_c_len = get_dist(d, c)
    d_c_uvec = (d - c)/d_c_len
    E = c + d_c_uvec * (d_c_len * factor)
    H_P_len = get_dist(H, P)
    H_P_uvec = (H - P)/H_P_len
    Q = P + H_P_uvec * (H_P_len * factor)
    E_H_len = get_dist(E, H)
    E_H_uvec = (E - H)/E_H_len
    F = H + E_H_uvec * (E_H_len * factor)
    F_Q_len = get_dist(F, Q)
    F_Q_uvec = (F-Q)/F_Q_len
    G = Q + F_Q_uvec * (F_Q_len * factor)
    return A, P, Q, G

#-----------------------------------------------------------------------------
def correct_derivative_2D(anchor_pt, norm_dir, der_length, b_ccw_rot):
    if b_ccw_rot:
        der_dir = np.array([ -norm_dir[1],  norm_dir[0] ])
    else:
        der_dir = np.array([  norm_dir[1], -norm_dir[0] ])

    res_ctrl_pt = anchor_pt + der_dir * der_length
    return res_ctrl_pt

#-----------------------------------------------------------------------------
def get_tangent_dir_3D(norm, p0, p1):
    vec = p1 - p0
    vec /= np.linalg.norm(vec)
    bnorm = np.cross(norm, vec)
    deriv = np.cross(norm, bnorm)
    deriv /= np.linalg.norm(deriv)
    return deriv
#-----------------------------------------------------------------------------
def bspline_average_3D(t0, p0, p1, n0, n1):
    if vec_eeq(p0, p1):
        return p0, n0
    x_axis = np.array([1.,0.,0.])
    y_axis = np.array([0.,1.,0.])
    new_z = np.cross(n0, n1)
    if vec_eeq(new_z, np.array([0.,0.,0.])) and not vec_eeq(n0, n1):
        print 'Num inst.'
        return linear_avg(t0, t1, b_open, p0, p1, n0, n1)
    new_z /= np.linalg.norm(new_z)
    new_y = np.cross(new_z, x_axis)
    new_y /= np.linalg.norm(new_y)
    new_x = np.cross(new_y, new_z)
    new_x /= np.linalg.norm(new_x)

    direct_transf = np.array([[new_x[0], new_x[1], new_x[2]],
                              [new_y[0], new_y[1], new_y[2]],
                              [new_z[0], new_z[1], new_z[2]]])
    p1_moved = p1 - p0
    orig_seg_length = np.linalg.norm(p1_moved)
    p0_tag = np.array([.0,.0,.0])
    p1_tag = project_point_to_plane(p0_tag, new_z, p1_moved)
    new_p1 = get_vect_in_coord_sys(p1_tag, direct_transf)
    #new_p1 = get_vect_in_coord_sys(p1_moved, direct_transf)
    new_n0 = get_vect_in_coord_sys(n0, direct_transf)
    new_n0 /= np.linalg.norm(new_n0)
    new_n1 = get_vect_in_coord_sys(n1, direct_transf)
    new_n1 /= np.linalg.norm(new_n1)

    p0_2D = np.array([.0,.0])
    p1_2D = np.array([new_p1[0],new_p1[1]])
    n0_2D = np.array([new_n0[0],new_n0[1]])
    n1_2D = np.array([new_n1[0],new_n1[1]])

    res_pt_2D, res_norm_2D = bspline_average_v3(t0, p0_2D, p1_2D, 
                                              n0_2D, n1_2D)
    z_plane_offset = p1_moved.dot(new_z)
    z_plane_offset *= (1. - t0)
    res_pt_pln = np.array([res_pt_2D[0], res_pt_2D[1], z_plane_offset ]) 
    res_norm_pln = np.array([res_norm_2D[0], res_norm_2D[1], 0. ]) 
    #cntr_pt_pln = np.array([cntr_2D[0], cntr_2D[1], 0. ]) 
    back_transf = np.linalg.inv(direct_transf)
    res_pt = get_vect_in_coord_sys(res_pt_pln, back_transf)
    res_pt += p0 
    #cntr_pt = get_vect_in_coord_sys(cntr_pt_pln, back_transf)
    #cntr_pt += p0 
    res_norm = get_vect_in_coord_sys(res_norm_pln, back_transf)

    return res_pt, res_norm

#-----------------------------------------------------------------------------
def bspline_average_2D(t0, p0, p1, n0, n1):
    p0_p1_dist = get_dist( p0, p1 )
    if p0_p1_dist < 0.001:
        return p1, n1

    theta = get_angle_between(n0, n1)
    der_length = p0_p1_dist/(3. * (np.cos(theta/4.) ** 2.))
    a = p0 
    b = correct_derivative_v3(p0, n0, der_length, b_ccw_rot = False)
    c = correct_derivative_v3(p1, n1, der_length, b_ccw_rot = True)
    d = p1

    res_t = t0
    #if t0 < 0.:
    #    d, c, b, a = extrapolate_bezier(d, c, b, a, np.abs(t0))
    #    #a, b, c, d = extrapolate_bezier(d, c, b, a, np.abs(t0))
    #    res_t = 0.0
    #elif t0 > 1.:
    #    a, b, c, d = extrapolate_bezier(a, b, c, d, t0-1.)
    #    res_t = 1.

    res_pt, res_der = eval_cubic_bezier( a, b, c, d, 1. - res_t )
    res_norm = np.array([-res_der[1], res_der[0]])
    res_norm /= np.linalg.norm(res_norm)
    #if np.linalg.norm(res_norm - ca_norm) > \
    #   np.linalg.norm(res_norm + ca_norm):
    #    res_norm = -res_norm    

    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        verts = [
            (a[0], a[1]), # P0
            (b[0], b[1]), # P1
            (c[0], c[1]), # P2
            (d[0], d[1]), # P3
            ]
        
        codes = [Path.Path.MOVETO,
                 Path.Path.CURVE4,
                 Path.Path.CURVE4,
                 Path.Path.CURVE4,
                 ]
        
        crv = Path.Path(verts, codes)
        
        ca_pt, ca_norm, ca_center, ca_radius, ca_beta0, ca_beta1 = \
          circle_avg(0.5, 0.5, True, p0, p1, n0, n1)
        fig = plt.figure() 
        plt.xlim([(ca_center[0] - ca_radius)*1.2, (ca_center[0] + ca_radius)*1.2])
        plt.ylim([(ca_center[1] - ca_radius)*1.2, (ca_center[1] + ca_radius)*1.2])
        cr1 = plt.Circle( (ca_center[0], ca_center[1]), radius=ca_radius, fc='y', ec='none')
        norm_len_factor = 0.2
        nr000 = plt.Arrow(p0[0], p0[1], 
                          n0[0]*norm_len_factor, n0[1]*norm_len_factor, 
                          width=0.03, fc='b', ec='none' )
        nr_res = plt.Arrow(res_pt[0], res_pt[1], 
                          res_norm[0]*norm_len_factor, res_norm[1]*norm_len_factor, 
                          width=0.03, fc='r', ec='none' )
        nr_ca = plt.Arrow(ca_pt[0], ca_pt[1], 
                          ca_norm[0]*norm_len_factor, ca_norm[1]*norm_len_factor, 
                          width=0.03, fc='g', ec='none' )
        nr100 = plt.Arrow(p1[0], p1[1], 
                          n1[0]*norm_len_factor, n1[1]*norm_len_factor, 
                          width=0.03, fc='b', ec='none' )
        plt.plot([a[0], b[0], c[0], d[0]], 
                 [a[1], b[1], c[1], d[1]], '-c' )
        plt.gca().add_patch(cr1)
        plt.gca().add_patch(nr000)
        plt.gca().add_patch(nr_res)
        plt.gca().add_patch(nr_ca)
        plt.gca().add_patch(nr100)
        patch = patches.PathPatch(crv, facecolor='none', lw=1)
        plt.gca().add_patch(patch)
        plt.axis('equal')
        #plt.axis([-1.0, 2., -1.0, 2])
        plt.show()
    return res_pt, res_norm


#-----------------------------------------------------------------------------
def create_input_on_a_polygon6():
    angs = np.linspace(0.0, 2*np.pi, 6,endpoint = False)
    pts = []
    nrm = []
    radius = 2.
    for a in angs:
        curr_norm = np.array([np.cos(a), np.sin(a)])
        nrm.append(curr_norm)
        curr_pt = curr_norm.copy()
        pts.append(curr_pt*radius)
    c45 = np.cos(2.**0.5/2.0)
    nrm[1] = np.array([-c45, c45])
    #nrm[1] = np.array([0., 1.])
    #nrm[2] = np.array([c45, c45])
    nrm[4] = np.array([-1., 0.])
    return pts, nrm

#-----------------------------------------------------------------------------
def create_input_on_a_polygon5():
    c30 = 3.0**0.3/2.0
    pts = [np.array([-2., -2.] ),
           np.array([-1.,  3.] ),
           np.array([ 5.,  3.5]),
           np.array([ 2.,  0.] ),
           np.array([ 4., -3.] )
           ]
    nrm = [np.array([-1., -1.]),
           np.array([-1.,  1.]),
           np.array([ 1.,  1.]),

           #np.array([ 1.,  0.]),
           np.array([ -1,  0]),

           np.array([ 1., -1.])
           #np.array([ 0., -1.]),
           #np.array([  0,  1]),
           #np.array([  0,  -1]),
           #np.array([  c30, -0.5]), #rd
           #np.array([ c30, 0.5]), #ru
           #np.array([ -c30, 0.5]), #lu
           #np.array([ -c30, -0.5]), #ld
           ]
    nnrm = []
    for n in nrm:
        nnrm.append( n/np.linalg.norm(n))
    return pts, nnrm
#-----------------------------------------------------------------------------
def create_input_on_a_square():
    pts = [np.array([ 0., 0.]),
           np.array([ 10., 0.]),
           np.array([ 10., 10.]),
           np.array([ 0., 10.])]
    nrm = [np.array([ -1., -1.]),
           np.array([  1., -1.]),
           np.array([  1.,  0.1]),
           np.array([  -0.1,  1.])]
    return pts, nrm

#-----------------------------------------------------------------------------
def create_input_on_a_square_diff_norms():
    pts = [np.array([  0.,  0.]),
           np.array([ 10.,  0.]),
           np.array([ 10., 10.]),
           np.array([  0., 10.])]
    s30 = 0.5
    c30 = (3.0**0.5) / 2.0
    nrm = [np.array([ -1.0, -0.1]),
           np.array([ -1.0, -1.0]),
           np.array([  1.0,  0.9]),
           np.array([  0.1,  1.0])]
    #nrm = [np.array([ 0., -1.]),
    #       np.array([ 1.,  0.]),
    #       np.array([ 0.,  1.]),
    #       np.array([-1.,  0.])]
    nnrm = []
    for n in nrm:
        nnrm.append( n/np.linalg.norm(n))
    return pts, nnrm

#-----------------------------------------------------------------------------
def plot_pts_and_norms(pts, nrm, b_open, draw_norms, clr, 
                       linestyle='', linewidth=1.0, cnvs = plt ):
    n = len(pts)
    nn = n-1 if b_open else n
    for i in range(nn):
        curr_pt = pts[i]
        next_pt = pts[(i+1)%n]
        if linestyle.startswith('da'):
            cnvs.plot([curr_pt[0], next_pt[0]], 
                        [curr_pt[1], next_pt[1]], 
                        color=clr, linestyle=linestyle,
                        linewidth=linewidth, dashes=(1,15))
        else:
            cnvs.plot([curr_pt[0], next_pt[0]], 
                        [curr_pt[1], next_pt[1]], 
                        color=clr, linestyle=linestyle, 
                        linewidth=linewidth)

        if draw_norms:
            curr_norm = nrm[i]
            #wdth = 0.15 if i != 2 else 0.3
            wdth = 0.3
            #colr = clr if i != 2 else 'r'
            colr = clr
            #curr_norm /= 2.0 #if i == 2 else 1.0
            gnr = cnvs.Arrow(curr_pt[0], curr_pt[1], 
                               curr_norm[0], curr_norm[1], 
                               width=wdth, fc=colr, ec='none' )
            cnvs.gca().add_patch(gnr)
    if draw_norms and b_open:
        curr_norm = nrm[-1]
        curr_pt = pts[-1]
        gnr = cnvs.Arrow(curr_pt[0], curr_pt[1], 
                           curr_norm[0], curr_norm[1], 
                           width=0.05, fc=clr, ec='none' )
        cnvs.gca().add_patch(gnr)
#-----------------------------------------------------------------------------
def build_curves():
    n_of_iterations = 5
    bspline_average_export = bspline_average_export_v3
    b_open = False
    subd_pts, subd_nrm = create_input_on_a_polygon5()
    #subd_pts, subd_nrm = create_input_on_a_polygon6()
    #subd_pts, subd_nrm = create_input_on_a_square()
    #subd_pts, subd_nrm = create_input_on_a_square_diff_norms()
    #subd_nrm = init_normals(subd_pts, b_open)
    orig_pts = subd_pts[:]

    bsubd_INS_pts,  bsubd_INS_nrm  = subd_pts[:], subd_nrm[:]
    bsubd_MLR2_pts, bsubd_MLR2_nrm = subd_pts[:], subd_nrm[:]
    bsubd_MLR3_pts, bsubd_MLR3_nrm = subd_pts[:], subd_nrm[:]
    bsubd_MLR5_pts, bsubd_MLR5_nrm = subd_pts[:], subd_nrm[:]
    bsubd_4pt_pts,  bsubd_4pt_nrm  = subd_pts[:], subd_nrm[:]

    csubd_INS_pts,  csubd_INS_nrm  = subd_pts[:], subd_nrm[:]
    csubd_MLR3_pts, csubd_MLR3_nrm = subd_pts[:], subd_nrm[:]
    csubd_MLR5_pts, csubd_MLR5_nrm = subd_pts[:], subd_nrm[:]
    csubd_4pt_pts,  csubd_4pt_nrm  = subd_pts[:], subd_nrm[:]

    lsubd_INS_pts,  lsubd_INS_nrm  = subd_pts[:], subd_nrm[:]
    lsubd_MLR3_pts, lsubd_MLR3_nrm = subd_pts[:], subd_nrm[:]
    lsubd_MLR5_pts, lsubd_MLR5_nrm = subd_pts[:], subd_nrm[:]
    lsubd_4pt_pts,  lsubd_4pt_nrm  = subd_pts[:], subd_nrm[:]

    opt_INS_pts,    opt_INS_nrm    = subd_pts[:], subd_nrm[:]
    opt_MLR3_pts,   opt_MLR3_nrm   = subd_pts[:], subd_nrm[:]
    opt_MLR5_pts,   opt_MLR5_nrm   = subd_pts[:], subd_nrm[:]
    opt_4pt_pts,    opt_4pt_nrm    = subd_pts[:], subd_nrm[:]

    corn_cut_pts,  corn_cut_nrm  = subd_pts[:], subd_nrm[:]

    for k in range(n_of_iterations):
        #--- Bezier Average
        bsubd_INS_pts, bsubd_INS_nrm   = double_polygon(bsubd_INS_pts, bsubd_INS_nrm,
                                                       True, b_open,
                                                      bspline_average_export)
        bsubd_MLR2_pts, bsubd_MLR2_nrm = subd_LR_one_step(bsubd_MLR2_pts, bsubd_MLR2_nrm, 
                                                          b_open, bspline_average_export, n_deg = 2)
        bsubd_MLR3_pts, bsubd_MLR3_nrm = subd_LR_one_step(bsubd_MLR3_pts, bsubd_MLR3_nrm, 
                                                          b_open, bspline_average_export)
        #bsubd_MLR5_pts, bsubd_MLR5_nrm = subd_LR_one_step(bsubd_MLR5_pts, bsubd_MLR5_nrm, 
        #                                                 b_open, bspline_average_export, n_deg = 5)
        #bsubd_4pt_pts, bsubd_4pt_nrm = subd_4PT_one_step(bsubd_4pt_pts, bsubd_4pt_nrm, 
        #                                                 b_open, bspline_average_export)
        
        #--- Circle Average
        #csubd_INS_pts, csubd_INS_nrm    = double_polygon(csubd_INS_pts, csubd_INS_nrm,
        #                                                 True, b_open,
        #                                                 circle_avg)
        #csubd_MLR3_pts, csubd_MLR3_nrm  = subd_LR_one_step(csubd_MLR3_pts, csubd_MLR3_nrm, 
        #                                                   b_open, circle_avg)
        #csubd_MLR5_pts, csubd_MLR5_nrm  = subd_LR_one_step(csubd_MLR5_pts, csubd_MLR5_nrm, 
        #                                                   b_open, circle_avg, n_deg = 5)
        #csubd_4pt_pts, csubd_4pt_nrm = subd_4PT_one_step(csubd_4pt_pts, csubd_4pt_nrm, 
        #                                                 b_open, circle_avg)

        #--- Linear Average
        #lsubd_INS_pts, lsubd_INS_nrm    = double_polygon(lsubd_INS_pts, lsubd_INS_nrm,
        #                                                 True, b_open,
        #                                                 linear_avg)
        #lsubd_MLR3_pts, lsubd_MLR3_nrm  = subd_LR_one_step(lsubd_MLR3_pts, lsubd_MLR3_nrm, 
        #                                                   b_open, linear_avg)
        #lsubd_MLR5_pts, lsubd_MLR5_nrm  = subd_LR_one_step(lsubd_MLR5_pts, lsubd_MLR5_nrm, 
        #                                                   b_open, linear_avg, n_deg = 5)
        #lsubd_4pt_pts, lsubd_4pt_nrm = subd_4PT_one_step(lsubd_4pt_pts, lsubd_4pt_nrm, 
        #                                                 b_open, linear_avg)

        #--- Optimizer Average
        #opt_INS_pts, opt_INS_nrm    = double_polygon(opt_INS_pts, opt_INS_nrm,
        #                                              True, b_open,
        #                                              bopt_average_export)
        #opt_MLR3_pts, opt_MLR3_nrm  = subd_LR_one_step(opt_MLR3_pts, opt_MLR3_nrm, 
        #                                               b_open, bopt_average_export)
        #opt_MLR5_pts, opt_MLR5_nrm   = subd_LR_one_step(opt_MLR5_pts, opt_MLR5_nrm, 
        #                                                b_open, bopt_average_export, n_deg = 5)
        #opt_4pt_pts, opt_4pt_nrm    = subd_4PT_one_step(opt_4pt_pts, opt_4pt_nrm, 
        #                                                b_open, bopt_average_export)

        #--- Corner cutting
        #corn_cut_pts, corn_cut_nrm  = subd_CornerCutting_one_step( corn_cut_pts, corn_cut_nrm, 
        #                                                           b_open, bspline_average_export, 1./3.)

    fig = plt.figure()#figsize=(8,8), dpi=100, frameon = False)
    #frame1 = plt.gca()
    #frame1.axes.get_xaxis().set_visible(False)
    #frame1.axes.get_yaxis().set_visible(False)

    plot_pts_and_norms(bsubd_INS_pts, bsubd_INS_nrm, b_open, True, clr='c', linewidth=1.0, linestyle='solid')
    plot_pts_and_norms(bsubd_MLR2_pts, bsubd_MLR2_nrm, b_open, True, clr='#9d68e6', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(bsubd_MLR3_pts, bsubd_MLR3_nrm, b_open, True, clr='#8f9cde', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(bsubd_MLR5_pts, bsubd_MLR5_nrm, b_open, False, clr='#9d9bd9', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(bsubd_4pt_pts, bsubd_4pt_nrm, b_open, False, clr='#4441a9', linewidth=1.0, linestyle='solid')

    #plot_pts_and_norms(csubd_INS_pts, csubd_INS_nrm, b_open, True, clr='#9dee80', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(csubd_MLR3_pts, csubd_MLR3_nrm, b_open, True, clr='#90d876', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(csubd_MLR5_pts, csubd_MLR5_nrm, b_open, True, clr='#43a941', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(csubd_4pt_pts, csubd_4pt_nrm, b_open, False, clr='g', linewidth=1.0, linestyle='solid')

    #plot_pts_and_norms(lsubd_INS_pts,  lsubd_INS_nrm, b_open, False, clr='#ff93de', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(lsubd_MLR3_pts, lsubd_MLR3_nrm, b_open, False, clr='g', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(lsubd_MLR5_pts, lsubd_MLR5_nrm, b_open, False, clr='#d286bb', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(lsubd_4pt_pts,  lsubd_4pt_nrm, b_open, False, clr='#ff93de', linewidth=1.0, linestyle='solid')

    #plot_pts_and_norms(opt_ins_pts, opt_ins_nrm, b_open, True, clr='y', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(opt_MLR3_pts, opt_MLR3_nrm, b_open, True, clr='g', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(opt_MLR5_pts, opt_MLR5_nrm, b_open, True, clr='#cfcf30', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(opt_4pt_pts, opt_4pt_nrm, b_open, True, clr='y', linewidth=1.0, linestyle='solid')

    #plot_pts_and_norms(corn_cut_pts, corn_cut_nrm, b_open, True, clr='#45ff02', linewidth=1.0, linestyle='solid')

    plot_pts_and_norms(orig_pts, subd_nrm, b_open, True, clr='k', linewidth=1.0, linestyle='dotted')

    plt.axis('equal')
    plt.xlim([-4, 6])
    plt.ylim([-5, 6])
    plt.axis('off')
    plt.show()


    #xs1 = compute_x_axes_values(bsubd_MLR2_pts, b_open, n_of_iterations)
    #slopes_ins = compute_slopes(bsubd_INS_pts, b_open, n_of_iterations)
    #curv_ins = compute_curvature(bsubd_INS_pts, b_open, n_of_iterations)
    #slopes_mlr2 = compute_slopes(bsubd_MLR2_pts, b_open, n_of_iterations)
    #curv_mlr2 = compute_curvature(bsubd_MLR2_pts, b_open, n_of_iterations)
    #slopes_mlr3 = compute_slopes(bsubd_MLR3_pts, b_open, n_of_iterations)
    #curv_mlr3 = compute_curvature(bsubd_MLR3_pts, b_open, n_of_iterations)
    #f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

    #ax1.plot(xs1, slopes_ins, 'c')
    #ax1.set_xlabel('Slope')
    #ax1.xaxis.set_label_position('top') 
    #ax1.set_ylabel('MLR1')

    #ax2.plot(xs1, curv_ins, 'r')
    #ax2.xaxis.set_label_position('top') 
    #ax2.set_xlabel('Curvature')

    #ax3.plot(xs1, slopes_mlr2, 'c', label = 'MLR2')
    #ax3.set_ylabel('MLR2')

    #ax4.plot(xs1, curv_mlr2, 'r')
    
    ##ax5.plot(xs1, slopes_mlr3, 'c', label = 'MLR3')
    ##ax5.set_ylabel('MLR3')
    ##ax6.plot(xs1, curv_mlr3, 'r')
    #plt.tight_layout()
    #plt.show()

#-----------------------------------------------------------------------------
def bezier_test():
    a = np.array([0.0,  0.0])
    b = np.array([1.0,  1.0])
    c = np.array([2.0, -1.0])
    d = np.array([3.0,  0.0])
    print 'Orig half-length = ', estimate_bezier_length_v2(a,b,c,d)/2.
    lb = split_cubic_bezier_get_left(a,b,c,d, 0.5)
    print 'Left length      = ', estimate_bezier_length_v2(lb[0],lb[1],lb[2],lb[3])

    res000_pt, res000_norm = eval_cubic_bezier(a, b, c, d, 0.0 )
    res025_pt, res025_norm = eval_cubic_bezier(a, b, c, d, 0.25)
    res050_pt, res050_norm = eval_cubic_bezier(a, b, c, d, 0.5 )
    res075_pt, res075_norm = eval_cubic_bezier(a, b, c, d, 0.75)
    res100_pt, res100_norm = eval_cubic_bezier(a, b, c, d, 1.0 )
    res000_norm = np.array([res000_norm[1], -res000_norm[0]])
    res025_norm = np.array([res025_norm[1], -res025_norm[0]])
    res050_norm = np.array([res050_norm[1], -res050_norm[0]])
    res075_norm = np.array([res075_norm[1], -res075_norm[0]])
    res100_norm = np.array([res100_norm[1], -res100_norm[0]])
    print get_angle(res000_norm[0], res000_norm[1]) * 180. / np.pi
    print get_angle(res025_norm[0], res025_norm[1])* 180. / np.pi
    print get_angle(res050_norm[0], res050_norm[1])* 180. / np.pi
    print get_angle(res075_norm[0], res075_norm[1])* 180. / np.pi
    print get_angle(res100_norm[0], res100_norm[1])* 180. / np.pi
    #DEBUG = True
    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        verts = [
            (a[0], a[1]), # P0
            (b[0], b[1]), # P1
            (c[0], c[1]), # P2
            (d[0], d[1]), # P3
            ]
        
        codes = [Path.Path.MOVETO,
                 Path.Path.CURVE4,
                 Path.Path.CURVE4,
                 Path.Path.CURVE4,
                 ]
        
        crv = Path.Path(verts, codes)
        
        fig = plt.figure() 
        nr000 = plt.Arrow(res000_pt[0], res000_pt[1], 
                          res000_norm[0]*.1, res000_norm[1]*.1, width=0.03, fc='g', ec='none' )
        nr025 = plt.Arrow(res025_pt[0], res025_pt[1], 
                          res025_norm[0]*.1, res025_norm[1]*.1, width=0.03, fc='g', ec='none' )
        nr050 = plt.Arrow(res050_pt[0], res050_pt[1], 
                          res050_norm[0]*.1, res050_norm[1]*.1, width=0.03, fc='g', ec='none' )
        nr075 = plt.Arrow(res075_pt[0], res075_pt[1], 
                          res075_norm[0]*.1, res075_norm[1]*.1, width=0.03, fc='g', ec='none' )
        nr100 = plt.Arrow(res100_pt[0], res100_pt[1], 
                          res100_norm[0]*.1, res100_norm[1]*.1, width=0.03, fc='g', ec='none' )
        plt.gca().add_patch(nr000)
        plt.gca().add_patch(nr025)
        plt.gca().add_patch(nr050)
        plt.gca().add_patch(nr075)
        plt.gca().add_patch(nr100)
        patch = patches.PathPatch(crv, facecolor='none', lw=2)
        plt.gca().add_patch(patch)
        plt.axis('equal')
        plt.axis([-1.0, 2., -1.0, 2])
        plt.show()

#-----------------------------------------------------------------------------
def one_pair_test():
    points  = [(  0.0, 0.0),
               (  0.0, 1.0)]
    normals = [(  1.0, 1.0),
               (  1.0, 1.0)]
    normals[0] /= np.linalg.norm(normals[0])
    normals[1] /= np.linalg.norm(normals[1])
    p0 = np.array([points[0][0], points[0][1]])
    p1 = np.array([points[1][0], points[1][1]])
    n0 = np.array([normals[0][0], normals[0][1]])
    n1 = np.array([normals[1][0], normals[1][1]])
    bspline_average_v3(0.5, p0, p1, n0, n1)
    #bspline_average_v1(1.125, p0, p1, n0, n1)
    #bspline_average_v2(-0.125, p0, p1, n0, n1)

#-----------------------------------------------------------------------------
def one_line_test():
    n_of_pnps  = 5
    orig_pts   = [np.array([float(i), 0.]) for i in range(n_of_pnps)]
    orig_pts.append(np.array([float(n_of_pnps-1)/2., -3.]))
    orig_norms = [np.array([0., 1. if i%2==0 else -1.]) for i in range(n_of_pnps)]
    orig_norms.append(np.array([0., -1.]))
    n_of_iterations = 5
    b_open = False
    bsubd_INS_pts, bsubd_INS_nrm = orig_pts[:], orig_norms[:]
    bsubd_MLR2_pts, bsubd_MLR2_nrm = orig_pts[:], orig_norms[:]
    for k in range(n_of_iterations):
        bsubd_INS_pts, bsubd_INS_nrm   = double_polygon(bsubd_INS_pts, 
                                                        bsubd_INS_nrm,
                                                        True, 
                                                        b_open,
                                                  bspline_average_export_v3)
        bsubd_MLR2_pts, bsubd_MLR2_nrm = subd_LR_one_step(bsubd_MLR2_pts, 
                                                          bsubd_MLR2_nrm, 
                                                          b_open, 
                                                  bspline_average_export_v3, 
                                                     n_deg = 2)
    plot_pts_and_norms(bsubd_INS_pts, bsubd_INS_nrm, b_open, 
                       True, clr='c', linewidth=1.0, linestyle='solid')
    plot_pts_and_norms(bsubd_MLR2_pts, bsubd_MLR2_nrm, b_open, 
                       True, clr='b', linewidth=1.0, linestyle='solid')
    plot_pts_and_norms(orig_pts, orig_norms, b_open, 
                       True, clr='k', linewidth=1.0, linestyle='dotted')
    plt.axis('equal')
    plt.xlim([-5, 6])
    plt.ylim([-5, 6])
    #plt.axis('off')
    plt.show()
#-----------------------------------------------------------------------------
def max_dist_test():
    global IN_DEBUG
    n0_angs = np.arange(0.0, 2*np.pi, np.pi/180., float)
    n1_angs = np.arange(0.0, 2*np.pi, np.pi/180., float)
    p0 = np.array([0., 0.])
    p1 = np.array([1., 0.])
    p0_r_max = (0.0, -1, -1)
    p1_r_max = (0.0, -1, -1)
    b_inter = False
    j = b_two_segments_intersect(p0, np.array([0., 1.]), p1, np.array([-1.1, 0.1]))
    for n0_ang_idx in range(len(n0_angs)):
        for n1_ang_idx in range(len(n1_angs)):
            n0_ang = n0_angs[n0_ang_idx]
            n1_ang = n1_angs[n1_ang_idx]
            n0 = np.array([np.cos(n0_ang), np.sin(n0_ang)])
            n1 = np.array([np.cos(n1_ang), np.sin(n1_ang)])
            
            ca_pt, ca_norm, ca_center, ca_radius, ca_beta0, ca_beta1 = \
                        circle_avg(0.5, 0.5, True, p0, p1, n0, n1)
            theta = get_angle_between(n0, n1)
            der_length = 4./3. * np.tan(theta/4.) * ca_radius
            t0 = np.array([n0[1], -n0[0]]) * der_length
            t1 = np.array([-n1[1], n1[0]]) * der_length
            b_inter = b_two_segments_intersect(p0, t0, p1, t1)
            if b_inter:
                print 'Intersection. Der len = ', der_length, 't0 = (', t0[0], t0[1], ') t1 = (', t1[0], t1[1], ')'
    if not b_inter:
        print 'No intersectons found'
    #        r_pt, r_norm = bspline_average_v3(0.5, p0, p1, n0, n1)
    #        dist_p0_r = get_dist(p0, r_pt)
    #        if dist_p0_r > p0_r_max[0]:
    #            p0_r_max = (dist_p0_r, n0_ang_idx, n1_ang_idx)
    #            #print 'Updating P0 max, with ', dist_p0_r, 'at', n0_ang_idx, ',', n1_ang_idx
    #        dist_p1_r = get_dist(p1, r_pt)
    #        if dist_p1_r > p1_r_max[0]:
    #            p1_r_max = (dist_p1_r, n0_ang_idx, n1_ang_idx)
    #            #print 'Updating P1 max, with ', dist_p1_r, 'at', n0_ang_idx, ',', n1_ang_idx
    #print 'P0, max dist ', p0_r_max[0], ' n0 ', n0_angs[p0_r_max[1]], ' n1 ', n1_angs[p0_r_max[2]] 
    #print 'P1, max dist ', p1_r_max[0], ' n0 ', n0_angs[p1_r_max[1]], ' n1 ', n1_angs[p1_r_max[2]] 
    #IN_DEBUG = True
    #n0_ang = n0_angs[p0_r_max[1]]
    #n1_ang = n1_angs[p0_r_max[2]]
    #n0 = np.array([np.cos(n0_ang), np.sin(n0_ang)])
    #n1 = np.array([np.cos(n1_ang), np.sin(n1_ang)])
    #bspline_average_v3(0.5, p0, p1, n0, n1)
#-----------------------------------------------------------------------------
def derivative_length_graph():
    x_s = np.linspace(0.0, np.pi-0.1, 100)
    y_s = [4./3. * np.tan(x/4.) * 1. / (2*np.cos(x/2.)) for x in x_s]
    y1_s = [np.tan(x/4.) for x in x_s]
    fig = plt.figure()
    plt.plot(x_s, y_s, 'c')
    plt.plot(x_s, y1_s, 'g')
    plt.axis('equal')
    plt.show()#block=False)

#-----------------------------------------------------------------------------
if __name__ == "__main__":
    global IN_DEBUG
    IN_DEBUG = True
    IN_DEBUG = False
    #max_dist_test()
    #bezier_test()
    #build_curves()
    #one_pair_test()
    one_line_test()
    #derivative_length_graph()
#============================ END OF FILE ====================================
