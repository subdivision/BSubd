import numpy as np
from sympy import Point, Line, Circle
import matplotlib.pyplot as plt
import matplotlib.patches as mpt
from CircAvg2D import get_halfplane, eeq, get_angle, vec_eeq

#=============================================================================
#============================ DATA GENERATOR =================================
#=============================================================================
def create_input_on_a_square():
    pts = [np.array([ 0., 0.]),
           np.array([ 0., 10.]),
           np.array([ 10., 10.]),
           np.array([ 10., 0.])]
    tng = [np.array([  -1., 1.]),
           np.array([   1., 1.]),
           np.array([  1.,  -1.]),
           np.array([  -1., -1.])]
    rdi = [5.]*4
    return pts, tng, rdi

#-----------------------------------------------------------------------------
def create_input_keggle():
    pts = [np.array([  0.,  0.]),
           np.array([  0., 10.]),
           np.array([  0., 20.]),
           np.array([  0., 30.]),
           np.array([ 10., 30.]),
           np.array([ 10., 20.]),
           np.array([ 10., 10.]),
           np.array([ 10.,  0.])]

    tgs = [np.array([  -1.,   1.]),
           np.array([   1.,   1.]),
           np.array([  -1.,   1.]),
           np.array([   1.,   1.]),
           np.array([   1.,  -1.]),
           np.array([  -1.,  -1.]),
           np.array([   1.,  -1.]),
           np.array([  -1.,  -1.])]

    rdi = [5.]*8
    return pts, tgs, rdi

#-----------------------------------------------------------------------------
def create_input_on_a_polygon5():
    c30 = 3.0**0.3/2.0
    pts = [np.array([-2., -2.] ),
           np.array([-1.,  3.] ),
           np.array([ 5.,  3.5]),
           np.array([ 2.,  0.] ),
           np.array([ 4., -3.] )
           ]
    tgs = [np.array([-1., 1.]),
           np.array([ 1.,  1.]),
           np.array([ 1.,  -1.]),

           #np.array([ -1.,  -1.]),
           #np.array([ -1.,  0.]),
           np.array([ 0,  -1]),

           np.array([ -1., -1.])
           #np.array([ 0., -1.]),
           #np.array([  0,  1]),
           #np.array([  0,  -1]),
           #np.array([  c30, -0.5]), #rd
           #np.array([ c30, 0.5]), #ru
           #np.array([ -c30, 0.5]), #lu
           #np.array([ -c30, -0.5]), #ld
           ]

    #rdi = [1.]*5
    #rdi = [1.5]*5
    #rdi = [0.5, 1., 1.5, 1.3, 0.7]
    #rdi = [0.5, 1, 2., 1.5, 1.]
    return pts, tgs, rdi

#-----------------------------------------------------------------------------
def create_input_konsole():
    pts = [np.array([3., 0.] ),
           np.array([3., 6.] ),
           np.array([9., 6.] ),
           np.array([9., 0.] ),
           np.array([7.5, 1.5] ),
           np.array([4.5, 1.5] )
           ]
    tgs = [np.array([-1.,  0.]),
           np.array([ 1.,  0.]),
           np.array([ 1.,  0.]),
           np.array([-1.,  0.]),
           np.array([ 0.,  1.]),
           np.array([ 0., -1.])
           ]
    rdi = [3., 3., 3., 1.5, 1.5, 1.5]
    return pts, tgs, rdi



#=============================================================================
#============================ ANALYSIS TOOLS =================================
#=============================================================================
def get_radius(a,b,c):
    d1 = np.array([b[1] - a[1], a[0] - b[0]])
    d2 = np.array([c[1] - a[1], a[0] - c[0]])
    k = d2[0] * d1[1] - d2[1] * d1[0]
    if -0.00001 <= k <= 0.00001:
        return None

    s1 = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2])
    s2 = np.array([(a[0] + c[0]) / 2, (a[1] + c[1]) / 2])
    l  = d1[0] * (s2[1] - s1[1]) - d1[1] * (s2[0] - s1[0])
    m  = l / k
    center = np.array([s2[0] + m * d2[0], s2[1] + m * d2[1]])
    dx = center[0] - a[0]
    dy = center[1] - a[1]
    radius = (dx * dx + dy * dy)**0.5
    return 1 / radius

#----------------------------------------------------------------------------
def get_slope(a,b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    if eeq(dx, 0.):
        return 0.0
    return dy/dx

#----------------------------------------------------------------------------
def get_radii(poly):
    n = len(poly) - 1
    radii = []
    for i in range(1,n):
        a = poly[i-1]
        b = poly[i]
        c = poly[i+1]
        r = get_radius(a,b,c)
        if r != None and not eeq(r, 0.):
            radii.append(1./r)
        else:
            radii.append(0.)#None)
    res = np.array(radii)
    res = res * (res < 20) + (res > 20) * 20
    return res


#----------------------------------------------------------------------------
def get_angle_diffs(poly, iter):
    n = len(poly) - 1
    ang_diffs = []
    for i in range(1,n):
        a = poly[i-1]
        b = poly[i]
        c = poly[i+1]
        slope_prev = get_slope(a, b)
        slope_curr = get_slope(b, c)
        ang_prev = np.arctan(slope_prev)
        ang_curr = np.arctan(slope_curr)
        ang_diffs.append((ang_curr - ang_prev)/(2**iter))
    return ang_diffs

#=============================================================================
#============================ VISUALIZATION ==================================
#=============================================================================
def plot_pts_and_norms(pts, nrm, b_open, draw_norms, clr, bold_norms = False, 
                       nr_clr = None,
                       linestyle='', linewidth=1.0, cnvs = plt ):
    n = len(pts)
    nn = n-1 if b_open else n
    if nr_clr is None:
        nr_clr = clr

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
            if bold_norms:
                curr_norm = curr_norm * 1.2
            #wdth = 0.15 if i != 2 else 0.3
            wdth = 0.3 if not bold_norms else 0.6
            #colr = clr if i != 2 else 'r'
            colr = nr_clr
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
                           width=wdth, fc = nr_clr, ec='none' )
        cnvs.gca().add_patch(gnr)

#-----------------------------------------------------------------------------
def draw_line(line, color):
    uvec = np.array([float(line.direction.unit.x), float(line.direction.unit.y)])
    anch = np.array([float(line.points[0].x), float(line.points[0].y)])
    p0 = -20. * uvec + anch
    p1 =  20. * uvec + anch
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color= color, linewidth=0.5)

#=============================================================================
#============================= BIARC AVERAGE  ================================
#=============================================================================
class BiArc:
    def __init__(self, p0, p1, center):
        self.p0 = p0
        self.p1 = p1
        self.center = center
        self.radius = self.center.distance(self.p0)
        line1 = Line(self.center, self.p1)
        line2 = Line(self.center, self.p0)
        self.central_angle = line1.angle_between(line2)

    def get_length(self):
        res = self.radius * self.central_angle
        return res

    def eval_pt_tang(self, arc_len):
        full_len = self.get_length()
        weight = arc_len / full_len
        res_cntr_angle = float(weight * self.central_angle)
        halfpl = get_halfplane(self.p0, self.p1, self.center)
        if -1 == halfpl:
            res_cntr_angle = -res_cntr_angle
        res_pt = self.p0.rotate(res_cntr_angle, self.center)
        res_norm = Line(self.center, res_pt).direction.unit
        res_tang = res_norm.rotate(halfpl* np.pi/2.)
        return res_pt, res_tang
    
    def eval_end_pt_tang(self, first):
        trgt_pt = self.p0 if first else self.p1
        res_norm = Line(self.center, trgt_pt).direction.unit
        res_tang = res_norm.rotate(np.pi/2.)
        halfpl = get_halfplane(self.p0, self.p1, self.center)
        if -1 == halfpl:
            res_tang = -res_tang
        return res_tang


    def draw(self, cnvs, color):
        if eeq(self.p0.x - self.center.x, 0.0):
            p0_angle = 90 if self.p0.y > self.center.y else 270
        else:
            p0_angle = get_angle(float(self.p0.x - self.center.x),
                                 float(self.p0.y - self.center.y)) * 180./np.pi
        if eeq(self.p1.x - self.center.x, 0.0):
            p1_angle = 90 if self.p1.y > self.center.y else 270
        else:
            p1_angle = get_angle(float(self.p1.x - self.center.x),
                                 float(self.p1.y - self.center.y)) * 180./np.pi
        halfpl = get_halfplane(self.p0, self.p1, self.center)
        if -1 == halfpl:
            p0_angle, p1_angle = p1_angle, p0_angle
        arc = mpt.Arc((float(self.center.x), float(self.center.y)), 
                      2.*float(self.radius), 2.*float(self.radius), angle = 0, 
                      theta1 = p0_angle, theta2 = p1_angle, ec = color)
        crc = mpt.Circle((float(self.center.x), float(self.center.y)), 
                         radius = float(0.2), color = color, fill=False)

        cnvs.add_patch(arc)
        cnvs.add_patch(crc)


#-----------------------------------------------------------------------------
def get_bisector(p0, p1):
    #orig_line = Line(p0, p1)
    #mid_point = (p0+p1)/2.
    #bisec = orig_line.perpendicular_line(mid_point)
    #return bisec
    mid_pt = (float(p0.x + p1.x)/2., float(p0.y + p1.y)/2.)
    bisec_vec = (-float(p1.y - p0.y), float(p1.x - p0.x))
    other_pt = (mid_pt[0] + bisec_vec[0], mid_pt[1] + bisec_vec[1])
    return Line(mid_pt, other_pt, evaluate = False)

#-----------------------------------------------------------------------------
def get_junction_circle(p0, p1, t0, t1):
    mid_pt = (p0+p1)/2.

    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        plt.plot([float(p0.x), float(p1.x)], [float(p0.y), float(p1.y)], 
                 color= "#66FFB2", linewidth=0.5)
        p0tan = mpt.Arrow(float(p0.x), float(p0.y), 
                            float(t0.x), float(t0.y), 
                            width=0.6, fc="b", ec='none' )
        p1tan = mpt.Arrow(float(p1.x), float(p1.y), 
                            float(t1.x), float(t1.y), 
                            width=0.6, fc="#2E86C1", ec='none' )
        plt.gca().add_patch(p0tan)
        plt.gca().add_patch(p1tan)
        crc = mpt.Circle([float(mid_pt.x), float(mid_pt.y)], 
                         radius = float(0.2), ec = "k", fill = False)
        plt.gca().add_patch(crc)

    if t1.distance(t0) < 0.0001:
        # The tangents are equal. We have the "parallelogram" case
        return mid_pt, -1.

    bisec1 = get_bisector(p0, p1)
    bisec2 = get_bisector(p0 + t0, p1 + t1)

    intr = bisec1.intersection(bisec2)
    if 0 == len(intr) and t0 == t1:
        raise ValueError("No intersection for adjacent circle center")
        #return mid_pt

    if isinstance(intr[0],Line):
        if t0 != t1:
            line0 = Line(p0, p0 + t0)
            line1 = Line(p1, p1 + t1)
            circ_cntr = line0.intersection(line1)[0]
        else:
            return mid_pt, -1.
    else:
        circ_cntr = intr[0]
    radius = p0.distance(circ_cntr)
    #if mid_pt.distance(circ_cntr) < 0.0001:
    #    # Half a circle case
    #    bisec_radius_vec = t0
    #else:
    #    bisec_radius_vec = Line(circ_cntr, mid_pt).direction.unit


    #anchor_pt = circ_cntr + bisec_radius_vec * radius

    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        draw_line(bisec1, "#F5CBA7")
        draw_line(bisec2, "#F5CBA7")

        crc = mpt.Circle([float(circ_cntr.x), float(circ_cntr.y)], 
                         radius = float(radius), ec = "#FFCC99", fill=False)
        plt.gca().add_patch(crc)
        crc = mpt.Circle([float(circ_cntr.x), float(circ_cntr.y)], 
                         radius = float(0.1), color = "#FF9933")
        plt.gca().add_patch(crc)
        
        #plt.axis('equal')
        #plt.xlim([-5, 20])
        #plt.ylim([-15, 10])
        #plt.show()


    return Point(float(circ_cntr.x), float(circ_cntr.y), evaluate=False), \
           float(radius) #anchor_pt

#-----------------------------------------------------------------------------
def get_closest_pt(p0, p1, anchor_pt):
    d0 = p0.distance(anchor_pt)
    d1 = p1.distance(anchor_pt)
    res = p0 if d0 <= d1 else p1
    res = Point(float(res.x), float(res.y), evaluate = False)
    return res

#-----------------------------------------------------------------------------
def circles_are_equal(cntr0, radius0, cntr1, radius1):
    x0 = float(cntr0.x)
    y0 = float(cntr0.y)
    r0 = float(radius0)
    x1 = float(cntr1.x)
    y1 = float(cntr1.y)
    r1 = float(radius1)
    return (eeq(x0, x1) and eeq(y0, y1) and eeq(r0, r1))
#-----------------------------------------------------------------------------
def pt_is_between(a, q, b):
    return a[0] <= q[0] <= b[0] and a[1] <= q[1] <= b[1]

#-----------------------------------------------------------------------------
def check_equal_points(pts):
    if vec_eeq(pts[0], pts[2]):
        return (0, 2)
    elif vec_eeq(pts[0], pts[3]):
        return (0, 3)
    elif vec_eeq(pts[1], pts[2]):
        return (1, 2)
    elif vec_eeq(pts[1], pts[3]):
        return (1, 3)
    else:
        return None

#-----------------------------------------------------------------------------
def get_circles_intersection(c0, r0, c1, r1):

    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        crc = mpt.Circle([float(c0.x), float(c0.y)], 
                         radius = float(r0), ec = "#7DCEA0", fill=False)
        plt.gca().add_patch(crc)
        crc = mpt.Circle([float(c1.x), float(c1.y)], 
                         radius = float(r1), color = "#FFCC99", fill=False)
        plt.gca().add_patch(crc)

    if circles_are_equal(c0, r0, c1, r1):
        intr = []
    else:
        c2c_uvec = Line(c0, c1, evaluate=False).direction.unit
        pts = [ center + c2c_uvec * sign * radius 
               for center, radius in ((c0, r0), (c1, r1)) for sign in (1, -1)]
        pts_coords = [(float(p.x), float(p.y)) for p in pts]
        eq_pts_idxs = check_equal_points(pts_coords)
        if eq_pts_idxs:
            # we have one point touch
            intr = [pts[eq_pts_idxs[0]]]
        else:
            circ1_is_in_circ0 = pt_is_between(pts_coords[0], pts_coords[2], pts_coords[1]) \
                                and pt_is_between(pts_coords[0], pts_coords[3], pts_coords[1])
            circ0_is_in_circ1 = pt_is_between(pts_coords[2], pts_coords[0], pts_coords[3]) \
                                and pt_is_between(pts_coords[2], pts_coords[1], pts_coords[3])
            complete_inclusion =  circ1_is_in_circ0 or circ0_is_in_circ1
                                 
            if complete_inclusion:
                # one circle is completely inside the other
                intr = []
            else:
                # generic case - 2 points intersection
                c2c_dist = float(c0.distance(c1))
                cos_alpha = (r0**2 + c2c_dist**2 - r1**2) / (2. * r0 * c2c_dist)
                intr_proj_len = r0 * cos_alpha
                intr_offset = 0. if eeq(cos_alpha, 1.) else r0 * ((1. - cos_alpha**2.) ** 0.5)
                intr_proj_pt = c0 + c2c_uvec * intr_proj_len
                intr_perp = Point(-c2c_uvec.y, c2c_uvec.x, evaluate=False)
                intr = [intr_proj_pt + intr_perp * sign * intr_offset for sign in (1.,-1.)] 
    #intr = junc_circle.intersection(Circle(aux_cntr, radius, evaluate=False))
    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
            if 0 < len(intr):
                crc = mpt.Circle([float(intr[0].x), float(intr[0].y)], 
                                 radius = float(0.1), color = "#FFCC99", fill=True)
                plt.gca().add_patch(crc)
            if 1 < len(intr):
                crc = mpt.Circle([float(intr[1].x), float(intr[1].y)], 
                                 radius = float(0.1), color = "#FFCC99", fill=True)
                plt.gca().add_patch(crc)

    return intr

#-----------------------------------------------------------------------------
def get_auxiliary_pt(pt, tang, radius, junc_circle, other_pt):
    init_line = Line(pt, pt + tang)
    tang_perp = init_line.perpendicular_line(pt).direction.unit
    aux_cntr1 = pt + tang_perp * radius
    aux_cntr2 = pt - tang_perp * radius
    aux_cntr = get_closest_pt(aux_cntr1, aux_cntr2, junc_circle.center)
    intr = get_circles_intersection(aux_cntr, radius, 
                                    junc_circle.center, junc_circle.radius)

    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        crc = mpt.Circle([float(aux_cntr.x), float(aux_cntr.y)], 
                         radius = float(radius), ec = "#7DCEA0", fill=False)
        plt.gca().add_patch(crc)
        crc = mpt.Circle([float(aux_cntr.x), float(aux_cntr.y)], 
                         radius = float(0.1), color = "#7DCEA0", fill=True)
        plt.gca().add_patch(crc)


    if isinstance(intr, Circle) or 0 == len(intr):
        aux_pt = pt
        #raise ValueError("No intersection for secondary biarc center")
    elif 1 == len(intr):
        aux_pt = intr[0]
    elif 2 == len(intr):
        aux_pt = get_closest_pt(intr[0], intr[1], other_pt)
        DEBUG = 'IN_DEBUG' in globals()
        if DEBUG and IN_DEBUG:
            crc = mpt.Circle([float(intr[0].x), float(intr[0].y)], 
                             radius = float(0.1), color = "#FFCC99", fill=True)
            plt.gca().add_patch(crc)
            crc = mpt.Circle([float(intr[1].x), float(intr[1].y)], 
                             radius = float(0.1), color = "#FFCC99", fill=True)
            plt.gca().add_patch(crc)

            #plt.gca().add_patch(crc)
            #plt.axis('equal')
            #plt.xlim([-5, 20])
            #plt.ylim([-15, 10])
            #plt.show()
    return aux_pt

#-----------------------------------------------------------------------------
def get_anchor_pt(p0, p1, t0, t1, r0, r1):
    jnc_cntr, jnc_radius = get_junction_circle(p0, p1, t0, t1)
    if -1 == jnc_radius:
        # Straight line - "infinite" circle
        anchor_pt = (p0 + p1)/2.
    else:
        jnc_circ = Circle(jnc_cntr, jnc_radius, evaluate=False)
        aux_start = get_auxiliary_pt(p0, t0, r0, jnc_circ, p1)
        aux_end = get_auxiliary_pt(p1, t1, r1, jnc_circ, p0)
        aux_mid_pt = (aux_start + aux_end)/2.
        if aux_mid_pt.distance(jnc_cntr) < 0.0001:
            # half a circle. Give the preference to p0
            assert abs(aux_mid_pt.distance(aux_start) - jnc_radius) < 0.0001
            diameter_line = Line(aux_start, aux_end)
            bisec_radius_vec = diameter_line.perpendicular_line(jnc_cntr).direction.unit
            anch_pt_1 = jnc_cntr + bisec_radius_vec * jnc_radius
            anch_pt_2 = jnc_cntr - bisec_radius_vec * jnc_radius
            test_pt = p0 + t0
            closest_cand = get_closest_pt(anch_pt_1, anch_pt_2, test_pt)
            if closest_cand == anch_pt_2:
                bisec_radius_vec = -bisec_radius_vec
        else:
            bisec_radius_vec = Line(jnc_cntr, aux_mid_pt).direction.unit
        anchor_pt = jnc_cntr + bisec_radius_vec * jnc_radius

        DEBUG = 'IN_DEBUG' in globals()
        if DEBUG and IN_DEBUG:
            crc = mpt.Circle([float(aux_start.x), float(aux_start.y)], 
                             radius = float(0.1), color = "#D4AC0D", fill=True)
            plt.gca().add_patch(crc)
            crc = mpt.Circle([float(aux_end.x), float(aux_end.y)], 
                             radius = float(0.1), color = "#D4AC0D", fill=True)
            plt.gca().add_patch(crc)
            crc = mpt.Circle([float(anchor_pt.x), float(anchor_pt.y)], 
                             radius = float(0.1), color = "#85929E", fill=True)
            plt.gca().add_patch(crc)


    return anchor_pt

#-----------------------------------------------------------------------------
def get_secondary_biarc(pt, tng, anchor, is_left):
    init_line = Line(pt, pt + tng)
    tng_perp = init_line.perpendicular_line(pt)
    pt_anchor_bisec = get_bisector(pt, anchor)
    intr = tng_perp.intersection(pt_anchor_bisec)
    if 0 == len(intr):
        raise ValueError("No intersection for secondary biarc center")
    if isinstance(intr[0],Line):
        raise ValueError("Line for secondary biarc center")
    circ_cntr = intr[0]
    radius = pt.distance(circ_cntr)

    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        color = "#FFCCFF" if is_left else "#B266FF"
        draw_line(tng_perp, color)
        draw_line(pt_anchor_bisec, color)

    if is_left:
        res = BiArc(pt, anchor, circ_cntr)
    else:
        res = BiArc(anchor, pt, circ_cntr)
    return res


#-----------------------------------------------------------------------------
def biarc_avg(w0, p0, p1, t0, t1, c0radius = 1., c1radius = 1.):
    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        fig = plt.figure()
    
    numpy_io = False
    if isinstance(p0, np.ndarray):
        numpy_io = True
        p0 = Point(p0[0],  p0[1], evaluate = False)
        p1 = Point(p1[0],  p1[1], evaluate = False)
        t0 = Point(t0[0],  t0[1], evaluate = False) 
        t1 = Point(t1[0],  t1[1], evaluate = False) 

    t0 = Point(float(t0.unit.x), float(t0.unit.y), evaluate = False)
    t1 = Point(float(t1.unit.x), float(t1.unit.y), evaluate = False)
    p0p1_segm_vec = Line(p0, p1).direction.unit
    tangents_equal = t1.distance(t0) < 0.0001
    segm_tang_collinear = t1.distance(p0p1_segm_vec) < 0.0001 or \
                          t1.distance(-p0p1_segm_vec) < 0.0001
    if tangents_equal and segm_tang_collinear:
        res_pt = (1. - w0) * p0 + w0 * p1
        res_tg = Point(t0[0], t0[1]) if numpy_io else t0
    else:
        anchor_pt = get_anchor_pt(p0, p1, t0, t1, c0radius, c1radius)
        left_biarc = get_secondary_biarc(p0, t0, anchor_pt, True)
        rght_biarc = get_secondary_biarc(p1, t1, anchor_pt, False)
        left_len = float(left_biarc.get_length())
        rght_len = float(rght_biarc.get_length())
        trgt_len = float(w0*(left_len + rght_len))
        if np.abs(trgt_len - left_len) < 0.0001:
            res_pt = anchor_pt
            res_tg = left_biarc.eval_end_pt_tang(False)
        else:
            if float(trgt_len) < float(left_len):
                trgt_biarc = left_biarc
            else:
                trgt_biarc = rght_biarc
                trgt_len -= left_len
            res_pt, res_tg = trgt_biarc.eval_pt_tang(trgt_len)

    res_radius = (1.-w0)*c0radius + w0*c1radius

    DEBUG = 'IN_DEBUG' in globals()
    if DEBUG and IN_DEBUG:
        left_biarc.draw(plt.gca(), "#FF99FF")
        rght_biarc.draw(plt.gca(), "#7F00FF")
        rnr = mpt.Arrow(float(res_pt.x), float(res_pt.y), 
                            float(res_tg.x), float(res_tg.y), 
                            width=0.6, fc="r", ec='none' )
        plt.gca().add_patch(rnr)
        crc = mpt.Circle([float(res_pt.x), float(res_pt.y)], 
                         radius = float(0.4), ec = "r", fill = False)
        plt.gca().add_patch(crc)


        plt.axis('equal')
        plt.xlim([-5, 20])
        plt.ylim([-15, 10])
        plt.show()

    if numpy_io:
        res_pt = np.array([float(res_pt.x), float(res_pt.y)])
        res_tg = np.array([float(res_tg.x), float(res_tg.y)])
        if np.linalg.norm(res_tg) < 0.0001:
            a = 5
        res_tg /= np.linalg.norm(res_tg)
    return res_pt, res_tg, res_radius

#=============================================================================
#============================= ALGORITHMS ====================================
#=============================================================================
def double_polygon( Pts, Tangs, Radii, bPreserve, bOpened, fnAvg = biarc_avg):	
    i = 0
    ResPts = []
    ResTangs = []
    ResRadii = []
    N = len(Pts)
    NN = N-1 if bOpened else N

    for i in range(NN):
        effective_radius = (Radii[i]+Radii[(i+1)%N])/2.
        r = fnAvg(  0.5,
                    Pts[i], Pts[ (i+1)%N], 
				    Tangs[i], Tangs[(i+1)%N],
                    Radii[i], Radii[(i+1)%N] )
                    #effective_radius, effective_radius)
        if bPreserve:
            ResPts.append(Pts[i])
            ResTangs.append(Tangs[i])
            ResRadii.append(Radii[i])
        ResPts.append(r[0])
        ResTangs.append(r[1])
        ResRadii.append(r[2])

    if bPreserve and bOpened:
        ResPts.append(Pts[-1])
        ResTangs.append(Tangs[-1])
        ResRadii.append(Radii[-1])

    return ResPts, ResTangs, ResRadii

#-----------------------------------------------------------------------------
def subd_LR_one_step(Pts, Tangs, Radii, bOpen = True, nDeg = 3, fnAvg = biarc_avg):
    SmoothedPts, SmoothedTangs, SmoothedRadii = double_polygon(
                                        Pts, Tangs, Radii, True, bOpen, fnAvg)
    for i in range(nDeg - 1):
        SmoothedPts, SmoothedTangs, SmoothedRadii = double_polygon(
            SmoothedPts, SmoothedTangs, SmoothedRadii, False, bOpen, fnAvg)
    return SmoothedPts, SmoothedTangs, SmoothedRadii

#-----------------------------------------------------------------------------
def algo_main():
    #global IN_DEBUG
    #IN_DEBUG = True

    n_of_iterations = 5
    b_open = False
    #src_pts, src_tgs, src_rdi = create_input_on_a_square()
    src_pts, src_tgs, src_rdi = create_input_keggle()
    #src_pts, src_tgs, src_rdi = create_input_on_a_polygon5()
    #src_pts, src_tgs, src_rdi = create_input_konsole()

    #mlr1_pts, mlr1_tgs, mlr1_rdi = src_pts[:], src_tgs[:], src_rdi[:]
    mlr2_pts, mlr2_tgs, mlr2_rdi = src_pts[:], src_tgs[:], src_rdi[:]
    #mlr3_pts, mlr3_tgs, mlr3_rdi = src_pts[:], src_tgs[:], src_rdi[:]
    
    for k in range(n_of_iterations):
        #mlr1_pts, mlr1_tgs, mlr1_rdi = double_polygon(
        #                            mlr1_pts, mlr1_tgs, mlr1_rdi, True, b_open)
        mlr2_pts, mlr2_tgs, mlr2_rdi = subd_LR_one_step(
                            mlr2_pts, mlr2_tgs, mlr2_rdi, b_open, 2)
        #mlr3_pts, mlr3_tgs, mlr3_rdi = subd_LR_one_step(
        #                    mlr3_pts, mlr3_tgs, mlr3_rdi, b_open, 3)

    fig = plt.figure(1)
    plt.subplot(121)

    #=========================== Graphs ======================================
    #r = get_angle_diffs(mlr1_pts, n_of_iterations)
    #t = np.arange(0.0, len(r))
    #plt.plot(t, r, color='#76D7C4')

    r = get_angle_diffs(mlr2_pts, n_of_iterations)
    t = np.arange(0.0, len(r))
    plt.plot(t, r, color='#7F00FF')

    #r = get_angle_diffs(mlr3_pts, n_of_iterations)
    #t = np.arange(0.0, len(r))
    #plt.plot(t, r, color='#7F00FF')

    #r = get_radii(mlr1_pts)
    #t = np.arange(0.0, len(r))
    #plt.plot(t, r, color='#76D7C4')

    r = get_radii(mlr2_pts)
    t = np.arange(0.0, len(r))
    plt.plot(t, r, color='#7F00FF')

    #r = get_radii(mlr3_pts)
    #t = np.arange(0.0, len(r))
    #plt.plot(t, r, color='#7F00FF')

    #=========================== Curve plots =================================

    plt.subplot(122)
    #plot_pts_and_norms(mlr1_pts, mlr1_tgs, b_open, True, clr='#76D7C4', linewidth=1.0, linestyle='solid')
    plot_pts_and_norms(mlr2_pts, mlr2_tgs, b_open, True, clr='#7F00FF', linewidth=1.0, linestyle='solid')
    #plot_pts_and_norms(mlr3_pts, mlr3_tgs, b_open, False, clr='#7F00FF', linewidth=1.0, linestyle='solid')
    plot_pts_and_norms(src_pts, src_tgs, b_open, True, clr='k', bold_norms = True, linewidth=1.0, linestyle='dotted')

    plt.axis('equal')
    plt.xlim([-4, 5.5])
    plt.ylim([-5, 5.5])
    plt.axis('off')
    plt.show()

#=============================================================================
#=============================== TESTS =======================================
#=============================================================================
def test_get_anchor_pt():
    p0 = Point( 0., 0.)
    p1 = Point( 1., 0.)
    t0 = Point(-1., 1.)
    t1 = Point( 1., 1.)
    anchor_pt = get_anchor_pt(p0, p1, t0, t1)

def test_biarc_get_length():
    p0 = Point(  0.,  0. )
    p1 = Point(  1.,  0. )
    cn = Point( 0.5, -0.5)
    biarc = BiArc(p0, p1, cn)
    len = biarc.get_length()

def test_get_secondary_biarc():
    p0 = Point( 0., 0.)
    p1 = Point( 1., 0.)
    t0 = Point(-1., 1.)
    t1 = Point( 1., 1.)
    anchor_pt = get_anchor_pt(p0, p1, t0, t1)
    get_secondary_biarc(p0, t0, anchor_pt)

def test_biarc_eval_pt_norm():
    p0 = Point(  0.,  0. )
    p1 = Point(  1.,  0. )
    cn = Point( 0.5, -0.5)
    biarc = BiArc(p0, p1, cn)
    len = biarc.get_length()
    biarc.eval_pt_norm(0.5*len)

def test_biarc_avg_horiz():
    p0 = Point( 0., 0.)
    p1 = Point( 10., 0.)
    t0 = Point( 1., 1.)
    t1 = Point( 1., -1.)
    res_pt, res_tg, res_radius = biarc_avg(0.5, p0, p1, t0, t1, 3., 2.)

def test_biarc_avg_generic():
    p0 = Point( 0., 0.)
    p1 = Point( 10., 0.)
    t0 = Point( 1., 1.)
    t1 = Point( 2., -1.)
    res_pt, res_tg, res_radius = biarc_avg(0.5, p0, p1, t0, t1, 3., 2.)

def test_get_circles_intersection_generic():
    c0 = Point(-1.29289321881345, -1.29289321881345, evaluate=False)
    r0 = 1.0
    c1 = Point(1.0, 0.0, evaluate=False)
    r1 = 3.60555127546400
    intr = get_circles_intersection(c0, r0, c1, r1)
    plt.axis('equal')
    plt.xlim([-4, 5.5])
    plt.ylim([-5, 5.5])
    plt.axis('off')
    plt.show()

def test_get_circles_intersection_generic_v2():
    c0 = Point(2., 0., evaluate=False)
    r0 = 2.0
    c1 = Point(-0.5, 0., evaluate=False)
    r1 = 1.
    intr = get_circles_intersection(c0, r0, c1, r1)
    plt.axis('equal')
    plt.xlim([-4, 5.5])
    plt.ylim([-5, 5.5])
    plt.axis('off')
    plt.show()


def test_biarc_avg_vert():
    p0 = np.array([ 0., 10.])
    p1 = np.array([ 0., 20.])
    t0 = np.array([ 1.,  1.])
    t1 = np.array([-1.,  1.])
    res_pt, res_norm, res_radius = biarc_avg(0.3, p0, p1, t0, t1)

def test_biarc_avg_vert_perp_tang():
    p1 = np.array([ 0., 0.])
    p0 = np.array([ 0., 6.])
    t1 = np.array([-1.,  0.])
    t0 = np.array([1.,  0.])
    res_pt, res_norm, res_radius = biarc_avg(0.5, p0, p1, t0, t1)

def test_biarc_avg_tng_parallel():
    p0 = np.array([1.53281482438188, 17.7059805007310])
    p1 = np.array([-1.53281482438188, 22.2940194992690])
    t0 = np.array([-0.38268343,  0.92387953])
    t1 = np.array([-0.38268343,  0.92387953])
    res_pt, res_norm, res_radius = biarc_avg(0.3, p0, p1, t0, t1)

def test_biarc_avg_tng_parallel_v2():
    p0 = np.array([0., 0.])
    p1 = np.array([10., 0.])
    t0 = np.array([ 1., 1.])
    t1 = np.array([ 1., 1.])
    res_pt, res_norm, res_radius = biarc_avg(0.5, p0, p1, t0, t1)

def test_biarc_avg_tng_perp_par():
    p0 = np.array([ 0., 0.])
    p1 = np.array([ 20., 0.])
    t0 = np.array([ 0., 1.])
    t1 = np.array([ 0., 1.])
    res_pt, res_norm, res_radius = biarc_avg(0.5, p0, p1, t0, t1)

def test_biarc_avg_tng_perp_antipar():
    p0 = np.array([ 0., 0.])
    p1 = np.array([ 20., 0.])
    t0 = np.array([ 0., 1.])
    t1 = np.array([ 0., -1.])
    res_pt, res_norm, res_radius = biarc_avg(0.5, p0, p1, t0, t1)

def test_biarc_avg_tng_perp_antipar_v2():
    p0 = np.array([ 3., 0.])
    p1 = np.array([ 0., 0.])
    t0 = np.array([ 0., 1.])
    t1 = np.array([ 0., -1.])
    res_pt, res_norm, res_radius = biarc_avg(0.5, p0, p1, t0, t1, 1.5, 1.5)

def test_biarc_avg_test_case_diag_tang45():
    p1 = Point( 3.0, 0.0, evaluate=False)
    p0 = Point( 6.0, 3.0, evaluate=False)
    t1 = Point(-1.0, 0.0, evaluate=False)
    t0 = Point( 0.0, 1.0, evaluate=False)
    res_pt, res_norm, res_radius = biarc_avg(0.5, p0, p1, t0, t1, 1., 1.)

def test_main():
    global IN_DEBUG
    IN_DEBUG = True
    #IN_DEBUG = False

    np.seterr(all='raise')

    #test_get_anchor_pt()
    #test_biarc_get_length()
    #test_get_secondary_biarc()
    #test_biarc_eval_pt_norm()
    #test_biarc_avg_horiz()
    test_biarc_avg_generic()
    #test_biarc_avg_vert()
    #test_biarc_avg_vert_perp_tang()
    #test_biarc_avg_tng_parallel()
    #test_biarc_avg_tng_parallel_v2()
    #test_biarc_avg_tng_perp_par()
    #test_biarc_avg_tng_perp_antipar()
    #test_biarc_avg_tng_perp_antipar_v2()
    #test_biarc_avg_test_case_diag_tang45()
    #test_get_circles_intersection_generic()
    #test_get_circles_intersection_generic_v2()

#=============================================================================
if __name__ == "__main__":
    #test_main()
    algo_main()

#============================ END OF FILE ====================================
