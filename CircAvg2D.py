import math as M
import numpy as np
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------
def eeq( d1, d2 ):
    if M.fabs(d1 - d2) < 0.0001:
        return True
    else:
        return False

#----------------------------------------------------------------------------
def vec_eeq( v1, v2 ):
    b_result = True
    for i in range(len(v1)):
        b_result = b_result and eeq(v1[i], v2[i])
    return b_result

#----------------------------------------------------------------------------
def double_polygon( Poly, Norms, bPreserve, b_opened, fn_avg ):	
    i = 0
    j = 0
    ResPoly = []
    ResNorm = []
    N = len(Poly)
    NN = N-1 if b_opened else N

    for i in range(NN):
        r = fn_avg( 0.5,
                        Poly[i],  Poly[ (i+1)%N], 
					    Norms[i], Norms[(i+1)%N] )
        j = i
        if bPreserve:
            ResPoly.append(Poly[i])
            ResNorm.append(Norms[i])
        ResPoly.append(r[0])
        ResNorm.append(r[1])

    if bPreserve and b_opened:
        ResPoly.append(Poly[-1])
        ResNorm.append(Norms[-1])
    return ResPoly, ResNorm

#----------------------------------------------------------------------------
def get_angle_between(v1, v2):
    a= np.dot(v1, v2) / np.linalg.norm(v1)*np.linalg.norm(v2)
    a = a if -1<=a<=1 else (-1 if a < -1 else 1)
    return M.acos(a)

#----------------------------------------------------------------------------
def get_delta_vector(p1, p2, t):
    # @@TODO: rewrite this
    # No more assumption that we are working with w = 1/2
    inter_pts = p2 - p1
    middle_perp = np.array([-inter_pts[1], inter_pts[0]])
    middle_perp = middle_perp / np.linalg.norm(middle_perp)
    l = np.linalg.norm(inter_pts) / 2.0
    l *= M.tan(t/4)
    middle_perp = middle_perp * l
    return middle_perp

#----------------------------------------------------------------------------
def get_halfplane(a,b,c):
    s = a[0]*(b[1] - c[1]) + b[0]*(c[1] - a[1]) +  c[0]*(a[1] - b[1])
    #z = x1 (y2 - y3) + x2 (y3 - y1) + x3 (y1 - y2)

    #a1 = plt.Arrow( p1[0], p1[1], n1[0], n1[1], width=0.03, fc='b', ec='none' )
    #a2 = plt.Arrow( p2[0], p2[1], n2[0], n2[1], width=0.03, fc='g', ec='none' )
    #pt1 = plt.Circle( (a[0], a[1]), radius=0.02, fc='r', ec='none')
    #pt2 = plt.Circle( (b[0], b[1]), radius=0.02, fc='k', ec='none')
    #pt3 = plt.Circle( (c[0], c[1]), radius=0.02, fc='k', ec='none')
    #pt4 = plt.Circle( (d[0], d[1]), radius=0.02, fc='r', ec='none')
    #plt.gca().add_patch(a1)
    #plt.gca().add_patch(a2)
    #plt.gca().add_patch(pt1)
    #plt.gca().add_patch(pt2)
    #plt.gca().add_patch(pt3)
    #plt.gca().add_patch(pt4)
    #plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-' )
    #plt.axis('scaled')
    #plt.show()

    n_res = 1 
    if eeq(s, 0.0):
        n_res = 0
    elif s < 0.0:
        n_res = -1
    return n_res

#----------------------------------------------------------------------------
def get_intersection_point(p1, p2, n1, n2):
    p3 = p1 + n1
    p4 = p2 + n2
    x1 = p1[0]
    x2 = p3[0]
    x3 = p2[0]
    x4 = p4[0]
    y1 = p1[1]
    y2 = p3[1]
    y3 = p2[1]
    y4 = p4[1]
    d = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    l = (x1*y2 - y1*x2)
    r = (x3*y4 - y3*x4)
    x = ( l*(x3-x4) - (x1 - x2)*r ) / d
    y = ( l*(y3-y4) - (y1 - y2)*r ) / d
    return np.array([x, y])

#----------------------------------------------------------------------------
def get_dist( p1, p2 ):
    return  np.linalg.norm( p1 - p2 )

#----------------------------------------------------------------------------
def get_weighted_angle(t1, t2, n1, n2):
    a1 = get_angle(n1[0], n1[1])
    a2 = get_angle(n2[0], n2[1])
    b_passes_zero = M.fabs(a1-a2) > M.pi
    a2 += 2*M.pi if b_passes_zero else 0
    a2 = 2*M.pi if M.fabs(a2) < 0.0001 and a1 > M.pi else a2
    res_t = t1* a1 + t2 * a2
    return res_t

#----------------------------------------------------------------------------
def decide_halfplane(p1, p2, n1, n2, 
                     n1_halfplane, n2_halfplane):
    q_halfplane = 0
    #regular_norm_halfplane = n2_halfplane \
    #                            if 0 == n1_halfplane else n1_halfplane

    if 0 == n1_halfplane:
        p_vec = p2-p1
        p_vec /= np.linalg.norm(p_vec)
        q_halfplane = -n2_halfplane if vec_eeq(p_vec,n1) else n2_halfplane
    elif 0 == n2_halfplane:
        p_vec = p1-p2
        p_vec /= np.linalg.norm(p_vec)
        q_halfplane = -n1_halfplane if vec_eeq(p_vec,n2) else n1_halfplane

    return q_halfplane

#----------------------------------------------------------------------------
def linear_avg( t1, t2, b_open, p1, p2, n1, n2 ):
    #@@TODO: normals shall be computed on the unit circle
    return p1*t1 + p2*t2, n1*t1 + n2*t2, p1, 0.0, 0.0, 0.0

#----------------------------------------------------------------------------
def circle_avg_v2( t1, t2, b_open, p1, p2, n1, n2 ):
    '''
    @assumption: t1 == t2 == 0.5
    '''
    n1_halfplane = get_halfplane(p1+n1, p1, p2)
    n2_halfplane = get_halfplane(p1, p2, p2+n2)
    if n1_halfplane != n2_halfplane:
        print 'Normals are not in the same halfplane'
        return p1, n1, None, -1, 0, 0

    p1_p2_dist = get_dist( p1, p2 )
    if vec_eeq(n1, n2):
        res_norm = n1
    else:
        res_t = get_weighted_angle(t1, t2, n1, n2)
        res_norm = np.array([M.cos(res_t), M.sin(res_t)])
    if p1_p2_dist<0.0001:
        return p1, res_norm, p1, 0.0, 0.0, 0.0

    q_pt = get_intersection_point(p1, p2, n1, n2)
    q_halfplane = get_halfplane(p1,p2,q_pt)
    r1 = np.linalg.norm(q_pt - p1)
    r2 = np.linalg.norm(q_pt - p2)
    res_radius = r1*(1. - t1) + r2*t1
    if q_halfplane != n1_halfplane:
        res_pt = q_pt + res_norm * res_radius
    else:
        res_pt = q_pt - res_norm * res_radius
        
    return res_pt, res_norm, q_pt, 0.0, 0.0, 0.0
       
    
#----------------------------------------------------------------------------
def circle_avg( t1, t2, b_open, p1, p2, n1, n2 ):
    P1P2Dist = get_dist( p1, p2 )
    if vec_eeq(n1, n2):
        ResNorm = n1
    else:
        res_t = get_weighted_angle(t1, t2, n1, n2)
        ResNorm = np.array([M.cos(res_t), M.sin(res_t)])
    if P1P2Dist<0.0001:
        return p1, ResNorm, p1, 0.0, 0.0, 0.0

    MiddlePt = ( p1 + p2 )/2.0

    Theta = get_angle_between( n1, n2 )
    if eeq(Theta, 0.0) or eeq(Theta, M.pi):
        return linear_avg(t1, t2, b_open, p1, p2, n1, n2) 
    DeltaVec = get_delta_vector( p1, p2, Theta )
    Cand = [MiddlePt + DeltaVec, MiddlePt - DeltaVec]
    n1_halfplane = get_halfplane(p1+n1, p1, p2)
    n2_halfplane = get_halfplane(p1, p2, p2+n2)
    q_pt = get_intersection_point(p1, p2, n1, n2)
    if 0 == n1_halfplane or 0 == n2_halfplane:
        q_halfplane = decide_halfplane(p1, p2, n1, n2, 
                                       n1_halfplane, n2_halfplane)
    else:
        q_halfplane = get_halfplane(p1,p2,q_pt)

    Cand1Dist = get_dist(Cand[0], q_pt)
    Cand2Dist = get_dist(Cand[1], q_pt)
    bFirstIsCloser = Cand1Dist < Cand2Dist
    bDistAreEq = False
    CandDistDiff = M.fabs( Cand1Dist - Cand2Dist )
    ResIdx = 0
    if CandDistDiff / (Cand1Dist + Cand2Dist) < 0.00001:
        bDistAreEq = True
    if bDistAreEq:
        regular_norm = n2 if 0 == n1_halfplane else n1
        regular_norm_end_pt = p2+n2 if 0 == n1_halfplane else p1+n1
        regular_norm_halfplane = n2_halfplane \
                                 if 0 == n1_halfplane else n1_halfplane
        take_closer_candidate = regular_norm_halfplane == q_halfplane
        Cand1Dist = get_dist(Cand[0], regular_norm_end_pt)
        Cand2Dist = get_dist(Cand[1], regular_norm_end_pt)
        bFirstIsCloser = Cand1Dist < Cand2Dist
        if take_closer_candidate:
            ResIdx = 0 if bFirstIsCloser else 1
        else:
            ResIdx = 1 if bFirstIsCloser else 0
    else:
        norms_in_same_halfplane = n1_halfplane == n2_halfplane
        if norms_in_same_halfplane:
            ResIdx = 1 if bFirstIsCloser else 0
        else:
            ResIdx = 0 if bFirstIsCloser else 1

    VecToCenter = MiddlePt - Cand[ResIdx]
    VecToCenter /= np.linalg.norm(VecToCenter)
    Center = MiddlePt + VecToCenter * P1P2Dist/(2*M.tan(Theta/2.0))
    #----- Experiment #2
    #nv1 = p1 - Center
    #nv1 /= np.linalg.norm(nv1)
    #dlt1 = getAngleBetween(nv1, n1)
    #nv2 = p2 - Center
    #nv2 /= np.linalg.norm(nv2)
    #dlt2 = getAngleBetween(nv1, n2)
    #t1 = t1*dlt1/(dlt1+dlt2)
    #t2 = t2*dlt2/(dlt1+dlt2)
    #-------------------
    r1 = (p1 - Center)
    Radius = np.linalg.norm(r1)
    r1 /= Radius
    r2 = (p2 - Center) / Radius
    if t1 == 0.5 and t2 == 0.5:
        ResPt = Cand[ResIdx]
    else:
        res_t = get_weighted_angle(t1, t2, r1, r2 )
        ResNorm2 = np.array([M.cos(res_t), M.sin(res_t)])
        ResPt = Radius * ResNorm2
        ResPt += Center

    beta1 = get_angle( r1[0], r1[1] ) * 180.0 / M.pi
    beta2 = get_angle( r2[0], r2[1] ) * 180.0 / M.pi
 
    #----Experiment #1
    #if not areNormsOnSameSide(p1, ResPt, n1, ResNorm2):
    #    ResNorm2 = -ResNorm2
    #ResNorm = ResNorm2 #+ ResNorm
    #ResNorm /= np.linalg.norm(ResNorm)
    #-------------------


    #crl = plt.Circle( (Center[0], Center[1]), radius=Radius, fc='y', ec='none')
    #pt1 = plt.Circle( (p1[0], p1[1]), radius=0.02, fc='b', ec='none')
    #pt2 = plt.Circle( (p2[0], p2[1]), radius=0.02, fc='g', ec='none')
    #pt3 = plt.Circle( (ResPt[0], ResPt[1]), radius=0.02, fc='r', ec='none')
    #pt4 = plt.Circle( (Center[0], Center[1]), radius=0.02, fc='k', ec='none')
    #nr1 = plt.Arrow( p1[0], p1[1], n1[0], n1[1], width=0.03, fc='b', ec='none' )
    #nr2 = plt.Arrow( p2[0], p2[1], n2[0], n2[1], width=0.03, fc='g', ec='none' )
    #nr3 = plt.Arrow( ResPt[0], ResPt[1], ResNorm[0], ResNorm[1], 
    #                width=0.05, fc='r', ec='none' )
    #plt.gca().add_patch(crl)
    #plt.gca().add_patch(pt1)
    #plt.gca().add_patch(pt2)
    #plt.gca().add_patch(pt3)
    #plt.gca().add_patch(pt4)
    #plt.gca().add_patch(nr1)
    #plt.gca().add_patch(nr2)
    #plt.gca().add_patch(nr3)
    #plt.axis('scaled')
    #plt.show()

    return ResPt, ResNorm, Center, Radius, beta1, beta2

#----------------------------------------------------------------------------
def get_quater(a):
    if eeq(a,0):
        return 1
    elif eeq(a, M.pi/2.0):
        return 2
    elif eeq(a, M.pi):
        return 3
    elif eeq(a, 1.5*M.pi):
        return 4
    if 0 <= a < M.pi/2.0:
        return 1
    elif M.pi/2.0 <= a < M.pi:
        return 2
    elif M.pi <= a < 1.5*M.pi:
        return 3
    else:
        return 4
#----------------------------------------------------------------------------
def get_angle( dx, dy ):
    Res = -1
    if eeq( dx, 0 ):
        if dy > 0 :
            Res = M.pi/2
        else:
            Res = 3*M.pi/2
    elif eeq( dy, 0 ):
        if dx > 0:
            Res = 0
        else:
            Res = M.pi
    else:
        tan_alpha = M.fabs( dy / dx )
        alpha = M.atan( tan_alpha )
        if dx > 0 :
            if dy > 0:
                Res = alpha
            else:
                Res = 2*M.pi - alpha
        else:
            if dy > 0:
                Res = M.pi - alpha
            else:
                Res = M.pi + alpha
    return Res

#-----------------------------------------------------------------------------
def subd_4PT_one_step(pnts, nrms, b_open = True, fn_avg = circle_avg, 
                      n_ignore=0):
    smoothed_pts = [] 
    smoothed_norms = []
    N = len(pnts)
    if N < 4:
        return pnts, nrms

    #if b_open:
    #    smoothed_pts.append(pnts[0])
    #    smoothed_norms.append(nrms[0])
        #smoothed_pts.append(pnts[1])
        #smoothed_norms.append(nrms[1])

    NN = N-3 if b_open else N
    for i in range( NN ):
        p0, n0 = pnts[i],       nrms[i]
        p1, n1 = pnts[(i+1)%N], nrms[(i+1)%N]
        p2, n2 = pnts[(i+2)%N], nrms[(i+2)%N]
        p3, n3 = pnts[(i+3)%N], nrms[(i+3)%N]
        smoothed_pts.append(p1)
        smoothed_norms.append(n1)
        ##implementation assumes CCW-sequence
        lp, ln = fn_avg(-1/8.0, p0, p1, n0, n1) 
        rp, rn = fn_avg(9/8.0, p2, p3, n2, n3) 

        res_pt, res_nr = fn_avg(0.5, lp, rp, ln, rn)
        smoothed_pts.append(res_pt)
        smoothed_norms.append(res_nr)

    if b_open:
        smoothed_pts.append(pnts[-2])
        smoothed_norms.append(nrms[-2])
        #smoothed_pts.append(pnts[-1])
        #smoothed_norms.append(nrms[-1])
        
    return smoothed_pts, smoothed_norms

#-----------------------------------------------------------------------------
def subd_LR_one_step(pnts, nrms, b_open = True, fn_avg = circle_avg, n_deg = 3):
    SmoothedPts, SmoothedNorms = double_polygon(pnts,nrms, True, b_open, fn_avg)
    for i in range(n_deg - 1):
        SmoothedPts, SmoothedNorms = double_polygon(SmoothedPts, SmoothedNorms, 
                                                   False, b_open, fn_avg)
    return SmoothedPts, SmoothedNorms


#-----------------------------------------------------------------------------
def eval_circle( x, center, radius, beta1, beta2 ):
    x0 = center[0]
    y0 = center[1]
    d = ( radius**2.0 - (x0 - x)**2.0 ) ** 0.5
    y1 = y0 + d
    y2 = y0 - d
    p1 = np.array( [x0+radius*M.cos(beta1*M.pi/180.0),
                    y0+radius*M.sin(beta1*M.pi/180.0)])
    p2 = np.array( [x0+radius*M.cos(beta2*M.pi/180.0),
                    y0+radius*M.sin(beta2*M.pi/180.0)])
    p3a = np.array([x,y1])
    p3b = np.array([x,y2])
    dista = np.linalg.norm(p1-p3a) + np.linalg.norm(p2-p3a)
    distb = np.linalg.norm(p1-p3b) + np.linalg.norm(p2-p3b)
    y = y1 if dista <= distb else y2
    return y

#-----------------------------------------------------------------------------
def eval_circle2( x, anchor_y, x0, y0, radius ):
    d = ( radius**2.0 - (x0 - x)**2.0 ) ** 0.5
    y1 = y0 + d
    y2 = y0 - d
    b_1st_closer = M.fabs( y1 - anchor_y ) < M.fabs( y2 - anchor_y )
    y = y1 if b_1st_closer else y2
    return y

#-----------------------------------------------------------------------------
def get_nearest_pt_on_circle( x, y, x0, y0, radius ):
    cntr_to_pt_vec = np.array( [x - x0, y - y0])
    cntr_to_pt_vec /= np.linalg.norm(cntr_to_pt_vec)
    cntr_to_pt_vec  *= radius
    res_pt = np.array( [x0 + cntr_to_pt_vec[0], y0 + cntr_to_pt_vec[1]])
    return res_pt

#-----------------------------------------------------------------------------
def init_normals(poly, b_open):
    n = len(poly)
    norms = []
    s = 0
    e = n
    b_flip = -1.0
    if b_open:
        v = poly[1] - poly[0]
        nr = b_flip * np.array([v[1], -v[0]])
        nr /= np.linalg.norm(nr)
        norms.append(nr)
        s = 1
        e -= 1
    for i in range(s, e):
        p1 = poly[(i-1+n)%n]
        p2 = poly[i]
        p3 = poly[(i+1)%n]
        v1 = p2 - p1
        v2 = p3 - p2
        v1n = np.array([v1[1], -v1[0]])
        v2n = np.array([v2[1], -v2[0]])
        v1len = np.linalg.norm(v1)
        v2len = np.linalg.norm(v2)
        v1n /= v1len
        v2n /= v2len
        #v3res = v1n * v2len/(v1len+v2len) + v2n * v1len/(v1len+v2len)
        #v3res = v1n * 0.5 + v2n * 0.5
        #v3res /= np.linalg.norm(v3res)
        t1 = v1len/(v1len+v2len)
        t2 = v2len/(v1len+v2len)
        res_t = get_weighted_angle(t1, t2, v1n, v2n )
        v3res = b_flip * np.array([M.cos(res_t), M.sin(res_t)])
        norms.append(v3res)
    if b_open:
        v = poly[-1] - poly[-2]
        nr = b_flip * np.array([v[1], -v[0]])
        nr /= np.linalg.norm(nr)
        norms.append(nr)
    return norms

#============================== END OF FILE ==================================