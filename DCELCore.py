import numpy as np
import math as m
from stl import mesh
from CircAvg3D import *

#-----------------------------------------------------------------------------
def vec_almost_zero(v):
    EPS = 0.00001
    return np.abs(v[0]) <= EPS and  np.abs(v[1]) <= EPS and np.abs(v[2]) <= EPS

#-----------------------------------------------------------------------------
def sort_by_per_first(vec_norm_idx):
    v = vec_norm_idx[0]
    n = vec_norm_idx[1]
    return (np.dot(v,n), vec_norm_idx[2])

#=============================================================================
class IDGenerator(object):
    def __init__(self, start = 0, step = 1):
        self.curr = start
        self.step = step 

    #-------------------------------------------------------------------------
    def get_id(self):
        self.curr += self.step
        return self.curr       

#=============================================================================
class DElement(object):
    def __init__(self, eid = -1):
        self.eid = eid
        self.he = None

    #-------------------------------------------------------------------------
    def set_he(self, he):
        self.he = he

#=============================================================================
class DVertex(DElement):
    def __init__(self, eid = -1):
        DElement.__init__(self, eid)
        self.pt = None
        self.nr = None

    #-------------------------------------------------------------------------
    def set_pt(self, pt):
        self.pt = np.copy(pt)

    #-------------------------------------------------------------------------
    def move_pt(self, offset_vec):
        self.pt += offset_vec

    #-------------------------------------------------------------------------
    def set_nr(self, nr):
        self.nr = np.copy(nr)
        self.nr /= np.linalg.norm(self.nr)

    #-------------------------------------------------------------------------
    def set_norm_as_pt(self):
        self.nr = np.copy(self.pt)
        self.nr /= np.linalg.norm(self.nr)
    
    #-------------------------------------------------------------------------
    def get_faces(self):
        curr_he = self.he
        result = []
        while True:
            assert curr_he
            result.append(curr_he.face)
            curr_he = curr_he.twin.next
            if curr_he == self.he:
                break
        return result

    #-------------------------------------------------------------------------
    def get_edges(self):
        curr_he = self.he
        result = []
        while True:
            assert curr_he
            result.append(curr_he.edge)
            curr_he = curr_he.twin.next
            if curr_he == self.he:
                break
        return result

    #-------------------------------------------------------------------------
    def get_neighbor_vertices(self, anchor_edge = None):
        curr_he = self.he
        if anchor_edge:
            anchor_eid = anchor_edge.eid
            while curr_he.edge.eid != anchor_eid:
                curr_he = curr_he.twin.next
                if curr_he == self.he:
                    break
        result = []
        start_he = curr_he
        while True:
            assert curr_he
            result.append(curr_he.dest())
            #curr_he = curr_he.twin.next
            curr_he = curr_he.prev.twin
            if curr_he == start_he:
                break
        return result

    #-------------------------------------------------------------------------
    def get_face_hedge(self, face):
        curr_he = self.he
        result = None
        while result == None:
            assert curr_he
            result = curr_he if curr_he.face == face else None
            curr_he = curr_he.twin.next
            if curr_he == self.he:
                break
        return result

    #-------------------------------------------------------------------------
    def init_tangent_dirs(self):
        self.tang_dirs = {}
        nei_edges = self.get_edges()
        n = len(nei_edges)
        rec_idx = -1
        rec_ratio = 0.
        i = 0
        edges_dirs = []
        tangs_step = 2. * np.pi / n
        for idx in range(n):
            curr_vec = nei_edges[index].get_as_unit_vec_from(self)
            edges_dirs.append( (curr_vec, self.nr, idx) )
        edges_dirs.sort( key = sort_by_per_first )
        prim_tang = np.cross(self.nr, edges_dirs[0][0])
        prim_tang = np.cross(self.nr, prim_tang)
        for idx in range(n):
            curr_2d_ang = idx * tangs_step
            cos2d = np.cos(curr_2d_ang)
            curr_3d_tang = prim_tang * cos2d \
                           + np.cross(self.nr, prim_tang) * np.sin(curr_2d_ang)\
                           + np.dot(self.nr, prim_tang) * self.nr * (1. - cos2d)
            curr_3d_tang /= np.linalg.norm(curr_3d_tang)
            self.tang_dirs[nei_edges[idx]] = curr_3d_tang

    #-------------------------------------------------------------------------
    def get_gaussian_curvature(self):
        nei_faces = self.get_faces()
        corner_angles = 0.
        cell_area = 0.
        for f in nei_faces:
            ar, an = f.get_triang_area_and_corner_angle(self)
            corner_angles += an
            cell_area += ar
        gaus_curv = (2. * np.pi - corner_angles)/cell_area
        return gaus_curv

    #-------------------------------------------------------------------------
    def dist_to_face(self, face):
        ''' Not implemented!!!
        '''
        vrtx = face.get_vertices()
        n = len(vrtx)
        dist = 0.0
        if 3 == n:
            dist = dist_to_plane(vrtx[0].pt, vrtx[1].pt, vrtx[2].pt, self.pt)
        elif 4 == n:
            dist = max(dist_to_plane(vrtx[0].pt, vrtx[1].pt, vrtx[2].pt, self.pt),
                       dist_to_plane(vrtx[2].pt, vrtx[3].pt, vrtx[0].pt, self.pt))
        return dist


#=============================================================================
class DEdge(DElement):
    def __init__(self, eid = -1):
        DElement.__init__(self, eid)
    
    def get_face_hedge(self, face):
        return self.he if self.he.face.eid == face.eid else self.he.twin

    def get_src_vertex(self, face):
        return self.get_face_hedge(face).vert

    def get_faces(self):
        return self.he.face, self.he.twin.face

    def get_vertices(self):
        return self.he.vert, self.he.twin.vert

    def get_other_face(self, neighb_face):
        assert self.he.face == neighb_face or self.he.twin.face == neighb_face
        return self.he.twin.face if self.he.face == neighb_face  \
                                 else self.he.face

    def get_other_vertex(self, end_vrtx):
        assert self.he.vert == end_vrtx or self.he.twin.vert == end_vrtx
        return self.he.twin.vert if self.he.vert == end_vrtx  \
                                 else self.he.vert

    def nullify_face_hedge(self, face):
        victim_hedge = self.get_face_hedge(face)
        if victim_hedge.vert.he == victim_hedge:
            victim_hedge.vert.he = victim_hedge.twin.next
        #if victim_hedge.face.he == victim_hedge:
        #    victim_hedge.face.he = victim_hedge.next
        if self.he == victim_hedge:
            self.he = victim_hedge.twin
        victim_hedge.twin.twin = None
        del victim_hedge

    def get_as_vec_from(self, vert):
        other = self.get_other_vertex(vert)
        return other.pt -  vert.pt

    def get_as_unit_vec_from(self, vert):
        other = self.get_other_vertex(vert)
        edge_vec = other.pt -  vert.pt
        edge_vec /= np.linalg.norm(edge_vec)
        return edge_vec

    def get_dehidral_angle(self):
        np.seterr(all='raise')
        left_face = self.he.face
        right_face = self.he.twin.face 
        norm_left = left_face.get_face_normal(self.he)
        norm_right = right_face.get_face_normal(self.he.twin)
        cos_gamma = np.dot(norm_left, norm_right)
        if cos_gamma > 1.0: #numerical stability
            cos_gamma = 1.0
        gamma = np.arccos(cos_gamma)
        return np.abs(gamma)

#=============================================================================
class DFace(DElement):
    def __init__(self, eid = -1):
        DElement.__init__(self, eid)

    #-------------------------------------------------------------------------
    def get_vertices(self):
        curr_he = self.he
        result = []
        while True:
            result.append(curr_he.vert)
            curr_he = curr_he.next
            if curr_he == self.he:
                break
        return result

    #-------------------------------------------------------------------------
    def get_edges(self):
        curr_he = self.he
        result = []
        while True:
            result.append(curr_he.edge)
            curr_he = curr_he.next
            if curr_he == self.he:
                break
        return result

    #-------------------------------------------------------------------------
    def get_neighbor_faces(self):
        curr_he = self.he
        result = []
        while True:
            result.append(curr_he.get_other_face(self))
            curr_he = curr_he.next
            if curr_he == self.he:
                break
        return result

    #-------------------------------------------------------------------------
    def get_common_edge(self, other):
        curr_he = self.he
        other_verts = other.get_vertices()
        while True:
            if curr_he.vert in other_verts and curr_he.dest() in other_verts:
                return curr_he.edge
            curr_he = curr_he.next
            if curr_he == self.he:
                break
        return None

    #-------------------------------------------------------------------------
    def get_vertex_hedge(self, vrtx):
        curr_he = self.he
        while True:
            if curr_he.vert == vrtx:
                return curr_he
            curr_he = curr_he.next
            if curr_he == self.he:
                break
        return None

    #-------------------------------------------------------------------------
    def flip_orientation_in_triang_face(self, anchor_hedge):
        v0 = anchor_hedge.next.vert
        v1 = anchor_hedge.vert
        v2 = anchor_hedge.twin.prev.vert
        e0 = anchor_hedge.edge
        e1 = anchor_hedge.twin.prev.edge
        e2 = anchor_hedge.twin.next.edge
        self.compile_face([v0, v1, v2], [e0, e1, e2], True)

    #-------------------------------------------------------------------------
    def compile_face(self, verts, edges, force_new_hedges = False):
        n = len(verts)
        hedges = []
        for i in range(n):
            hedges.append(DHalfEdge())
        for i in range(n):
            hedges[i].vert = verts[i]
            hedges[i].edge = edges[i]
            hedges[i].face = self
            hedges[i].next = hedges[(i+1)%n]
            hedges[i].prev = hedges[(i-1+n)%n]
            if not verts[i].he or force_new_hedges:
                verts[i].he = hedges[i]
            if not edges[i].he:
                edges[i].he = hedges[i]
            else:
                hedges[i].twin = edges[i].he
                edges[i].he.twin = hedges[i]
        self.he = hedges[0]
        curr_edges = self.get_edges()
        assert len(curr_edges) == n, 'Different number of edges' 

    #-------------------------------------------------------------------------
    def get_face_normal(self, he):
        n = len(self.get_vertices())
        edge_der = he.edge.get_as_vec_from(he.vert)
        edge_der /= np.linalg.norm(edge_der)
        if 4 == n:
            v0 = he.vert.pt
            v1 = he.next.vert.pt
            v2 = he.next.next.vert.pt
            v3 = he.prev.vert.pt
            cbd = (v3+v2)/2. - (v0+v1)/2.
        elif 3 == n:
            v1 = he.next.vert.pt
            v2 = he.next.next.vert.pt
            cbd = v2-v1
        cbd /= np.linalg.norm(cbd)
        res_norm = np.cross(edge_der, cbd)
        res_norm /= np.linalg.norm(res_norm)
        return res_norm

    #-------------------------------------------------------------------------
    def get_face_center(self):
        vrts = self.get_vertices()
        cntr = np.array([0., 0., 0.])
        for v in vrts:
            cntr += v.pt
        cntr /= len(vrts)
        return cntr

    #-------------------------------------------------------------------------
    def get_triang_area_and_corner_angle(self, vrtx):
        he = self.get_vertex_hedge(vrtx)
        v1 = he.dest()
        v2 = he.prev.vert
        a = np.linalg.norm(vrtx.pt - v1.pt)
        b = np.linalg.norm(vrtx.pt - v2.pt)
        c = np.linalg.norm(v2.pt - v1.pt)
        p  = (a + b + c)/2.
        ar = (p * (p-a)*(p-b)*(p-c) )**0.5
        dir1 = (vrtx.pt - v1.pt) / a
        dir2 = (vrtx.pt - v2.pt) / b
        cos_ang = np.dot(dir1, dir2)
        an = np.arccos(cos_ang)
        return ar/3., an

#=============================================================================
class DHalfEdge(object):
    def __init__(self):
        self.vert = None
        self.edge = None
        self.face = None
        self.twin = None
        self.next = None
        self.prev = None
    
    #-------------------------------------------------------------------------
    def dest(self):
        return self.twin.vert

    #-------------------------------------------------------------------------
    def get_other_face(self, face):
        assert self.twin.face == face or face == self.face
        return self.twin.face if face == self.face else self.face

#=============================================================================
class DCtrlMesh(object):
    def __init__(self, eid, averaging_fn):
        self.eid = eid
        self.descr = 'DCEL Mesh'
        self.v = []
        self.e = []
        self.f = []
        self.id2obj = {}
        self.idgen = None
        self.averaging_fn = averaging_fn

    #-------------------------------------------------------------------------
    def dump_obj_file(self, filename):
        id2idx = {}
        for i in range(len(self.v)):
            id2idx[self.v[i].eid] = i
        with open(filename, 'w') as of:
            #of.write('# NSubd\n\n')
            #of.write('g ' + self.descr + '\n')
            #of.write('\n')
            of.write('OFF\n')
            of.write(str(len(self.v)) + ' ' + str(len(self.f)) + ' 0\n')
            for v in self.v:
                of.write('' + str(v.pt[0]) + ' ' \
                              + str(v.pt[1]) + ' ' \
                              + str(v.pt[2]) + '\n')
            #of.write('\n')
            for f in self.f:
                curr_verts = f.get_vertices()
                curr_idxs = []
                for v in curr_verts:
                    curr_idxs.append(id2idx[v.eid])
                #of.write('')
                of.write(str(len(curr_idxs)))
                for i in curr_idxs:
                    of.write( ' ' + str(i))
                of.write('\n')

    #-------------------------------------------------------------------------
    def print_ctrl_mesh(self):
        for v in self.v:
            print '[{0:3d}] pt = ({1}, {2}, {3}), nr = ({4}, {5}, {6})'.\
                format(v.eid, v.pt[0], v.pt[1], v.pt[2], 
                       v.nr[0], v.nr[1], v.nr[2])
            edges = v.get_edges()
            print '     Edges:',
            for e in edges:
                print e.eid,
            print ''

        for f in self.f:
            print '[{0:3d}] '.format(f.eid),
            edges = f.get_edges()
            for e in edges:
                print e.eid,
            print ''

        for e in self.e:
            assert e.he.twin.twin == e.he
            print '[{0:3d}] {1}->{2}\n Faces: [{3}, {4}]'.format(e.eid, 
                    e.he.vert.eid, e.he.twin.vert.eid,
                    e.he.face.eid, e.he.twin.face.eid)

    #-------------------------------------------------------------------------
    def get_dehidral_angle_stats(self):
        all_angles = []
        for e in self.e:
            all_angles.append(e.get_dehidral_angle())
        max_ang = max(all_angles)
        min_ang = min(all_angles)
        mean_ang = np.mean(all_angles)
        return min_ang, max_ang, mean_ang

    #-------------------------------------------------------------------------
    def get_gaus_curvature_stats(self):
        all_curvs = []
        for v in self.v:
            all_curvs.append(v.get_gaussian_curvature())
        max_crv = max(all_curvs)
        min_crv = min(all_curvs)
        mean_crv = np.mean(all_curvs)
        return min_crv, max_crv, mean_crv

    #-------------------------------------------------------------------------
    def get_gaus_curvature_abs_delta(self):
        mn, mx, _ = self.get_gaus_curvature_stats()
        return np.abs(mn - mx)

    #-------------------------------------------------------------------------
    def get_mesh_to_mesh_dist(self, other):
        self_to_other_dist = []
        for v1 in self.v:
            curr_self_to_other_dist = []
            for v2 in other.v:
                curr_self_to_other_dist.append(np.linalg.norm(v1.pt - v2.pt))
            self_to_other_dist.append(min(curr_self_to_other_dist))

        other_to_self_dist = []
        for v1 in other.v:
            curr_other_to_self_dist = []
            for v2 in self.v:
                curr_other_to_self_dist.append(np.linalg.norm(v1.pt - v2.pt))
            other_to_self_dist.append(min(curr_other_to_self_dist))

        the_dist = max(max(self_to_other_dist), max(other_to_self_dist))
        return the_dist

    #-------------------------------------------------------------------------
    def get_corresp_mesh_dist(self, other):
        '''Compute distances between corresponding vertices of two meshes
           Assumption: meshes have the same structure'''
        dists = []
        for i in range(len(self.v)):
            self_curr_vrtx = self.v[i]
            othr_curr_vrtx = other.v[i]
            dists.append(
                np.linalg.norm(self_curr_vrtx.pt - othr_curr_vrtx.pt))

        return min(dists), max(dists), np.mean(dists)

    #-------------------------------------------------------------------------
    def morph_to_corresp_mesh(self, other, d):
        dists = []
        for i in range(len(self.v)):
            self_curr_vrtx = self.v[i]
            othr_curr_vrtx = other.v[i]
            move_vec = othr_curr_vrtx.pt - self_curr_vrtx.pt
            move_vec /= np.linalg.norm(move_vec)
            move_vec *= d
            self_curr_vrtx.move_pt(move_vec)

    #-------------------------------------------------------------------------
    def create_vertex(self, coords):
        curr_vrt = DVertex(self.idgen.get_id())
        curr_pos = np.asarray(coords)
        curr_vrt.set_pt(curr_pos)
        curr_vrt.set_norm_as_pt()
        self.id2obj[curr_vrt.eid] = curr_vrt
        self.v.append(curr_vrt)
        return curr_vrt
        
    #-------------------------------------------------------------------------
    def create_edge(self):
        curr_edge = DEdge(self.idgen.get_id())
        self.id2obj[curr_edge.eid] = curr_edge
        self.e.append(curr_edge)
        return curr_edge

    #-------------------------------------------------------------------------
    def create_face(self):
        curr_face = DFace(self.idgen.get_id())
        self.id2obj[curr_face.eid] = curr_face
        self.f.append(curr_face)
        return curr_face
    
    #-------------------------------------------------------------------------
    def compile_face(self, verts, edges, face):
        face.compile_face(verts, edges)

    #-------------------------------------------------------------------------
    def produce_face(self, vertices, vertices_2_edges):
        n = len(vertices)
        new_edges = []
        for i in range(n):
            curr_vrtx = vertices[i]
            next_vrtx = vertices[(i+1)%n]
            new_edges.append(
                self.get_or_create_edge(vertices_2_edges, 
                                        curr_vrtx, 
                                        next_vrtx))
        new_face = self.create_face()
        self.compile_face(vertices, new_edges, new_face)

    #-------------------------------------------------------------------------
    def init_normals(self, unit_vector):
        for v in self.v:
            v.set_nr(unit_vector)

    #-------------------------------------------------------------------------
    def set_naive_normals(self, weight = 1.0):
        ''' For every vertex
            Determine a naive normal with discrete Laplace-Beltrami operator.
            Set the normal as a weighted average between the current and 
            the naive.
        '''
        for v in self.v:
            neighb = v.get_neighbor_vertices()
            n = len(neighb)
            vert_sum = 0.0
            norms = []
            wghts = []
            res_norm = np.array([0.,0.,0.])
            for i in range(n):
                v0 = neighb[i].pt - v.pt
                v0 /= np.linalg.norm(v0)
                v1 = neighb[(i+1)%n].pt - v.pt
                v1 /= np.linalg.norm(v1)
                curr_norm = np.cross(v0, v1)
                curr_norm /= np.linalg.norm(curr_norm)
                cos_a = np.dot(v0, v1)
                curr_weight = np.arccos(cos_a)
                vert_sum += curr_weight
                norms.append(curr_norm)
                wghts.append(curr_weight)
            for i in range(n):
                res_norm += norms[i] * (wghts[i] / vert_sum)
            res_norm /= np.linalg.norm(res_norm)
            #@@TODO: why do we need to flip that?
            res_norm = -res_norm
            #@@TODO: it's not geodesic! It's linear.
            res_norm = weight*res_norm + (1. - weight)*v.nr
            v.set_nr(res_norm)
 
    #-------------------------------------------------------------------------
    def init_as_quad_cube(self, offset = 10.0):
        self.idgen = IDGenerator()
        v1 = self.create_vertex([ offset, offset, offset])
        #v1.set_nr(np.array([0., 0., 1.]))
        v2 = self.create_vertex([-offset, offset, offset])
        v3 = self.create_vertex([-offset,-offset, offset])
        v4 = self.create_vertex([ offset,-offset, offset])
        v5 = self.create_vertex([ offset, offset,-offset])
        v6 = self.create_vertex([ offset,-offset,-offset])
        v7 = self.create_vertex([-offset,-offset,-offset])
        v8 = self.create_vertex([-offset, offset,-offset])

        f1234 = self.create_face()
        f5678 = self.create_face()
        f1582 = self.create_face()
        f4651 = self.create_face()
        f3764 = self.create_face()
        f2873 = self.create_face()

        e12 = self.create_edge()
        e23 = self.create_edge()
        e34 = self.create_edge()
        e41 = self.create_edge()

        e56 = self.create_edge()
        e67 = self.create_edge()
        e78 = self.create_edge()
        e85 = self.create_edge()

        e15 = self.create_edge()
        e46 = self.create_edge()
        e37 = self.create_edge()
        e28 = self.create_edge()

        self.compile_face([v1, v2, v3, v4], [e12, e23, e34, e41], f1234)
        self.compile_face([v5, v6, v7, v8], [e56, e67, e78, e85], f5678)
        self.compile_face([v1, v5, v8, v2], [e15, e85, e28, e12], f1582)
        self.compile_face([v4, v6, v5, v1], [e46, e56, e15, e41], f4651)
        self.compile_face([v3, v7, v6, v4], [e37, e67, e46, e34], f3764)
        self.compile_face([v2, v8, v7, v3], [e28, e78, e37, e23], f2873)

    #-------------------------------------------------------------------------
    def init_as_triang_cube(self, offset = 10.0):
        self.idgen = IDGenerator()
        v1 = self.create_vertex([ offset, offset, offset])
        v2 = self.create_vertex([-offset, offset, offset])
        v3 = self.create_vertex([-offset,-offset, offset])
        v4 = self.create_vertex([ offset,-offset, offset])
        v5 = self.create_vertex([ offset, offset,-offset])
        v6 = self.create_vertex([ offset,-offset,-offset])
        v7 = self.create_vertex([-offset,-offset,-offset])
        v8 = self.create_vertex([-offset, offset,-offset])

        f123 = self.create_face()
        f134 = self.create_face()
        f568 = self.create_face()
        f678 = self.create_face()
        f465 = self.create_face()
        f145 = self.create_face()
        f215 = self.create_face()
        f258 = self.create_face()
        f287 = self.create_face()
        f273 = self.create_face()
        f374 = self.create_face()
        f476 = self.create_face()

        e13 = self.create_edge()
        e45 = self.create_edge()
        e47 = self.create_edge()
        e27 = self.create_edge()
        e25 = self.create_edge()
        e68 = self.create_edge()
        e12 = self.create_edge()
        e23 = self.create_edge()
        e34 = self.create_edge()
        e14 = self.create_edge()
        e56 = self.create_edge()
        e67 = self.create_edge()
        e78 = self.create_edge()
        e58 = self.create_edge()
        e15 = self.create_edge()
        e46 = self.create_edge()
        e37 = self.create_edge()
        e28 = self.create_edge()

        self.compile_face([v1, v2, v3], [e12, e23, e13], f123)
        self.compile_face([v1, v3, v4], [e13, e34, e14], f134)
        self.compile_face([v5, v6, v8], [e56, e68, e58], f568)
        self.compile_face([v6, v7, v8], [e67, e78, e68], f678)
        self.compile_face([v4, v6, v5], [e46, e56, e45], f465)
        self.compile_face([v1, v4, v5], [e14, e45, e15], f145)
        self.compile_face([v2, v1, v5], [e12, e15, e25], f215)
        self.compile_face([v2, v5, v8], [e25, e58, e28], f258)
        self.compile_face([v2, v8, v7], [e28, e78, e27], f287)
        self.compile_face([v2, v7, v3], [e27, e37, e23], f273)
        self.compile_face([v3, v7, v4], [e37, e47, e34], f374)
        self.compile_face([v4, v7, v6], [e47, e67, e46], f476)

    #-------------------------------------------------------------------------
    def init_as_triang_cube_v6_v3(self, offset = 10.0):
        self.idgen = IDGenerator()
        v1 = self.create_vertex([ offset, offset, offset])
        v2 = self.create_vertex([-offset, offset, offset])
        v3 = self.create_vertex([-offset,-offset, offset])
        v4 = self.create_vertex([ offset,-offset, offset])
        v5 = self.create_vertex([ offset, offset,-offset])
        v6 = self.create_vertex([ offset,-offset,-offset])
        v7 = self.create_vertex([-offset,-offset,-offset])
        v8 = self.create_vertex([-offset, offset,-offset])

        f123 = self.create_face()
        f134 = self.create_face()
        f568 = self.create_face()
        f678 = self.create_face()
        f643 = self.create_face()
        f637 = self.create_face()
        f873 = self.create_face()
        f832 = self.create_face()
        f821 = self.create_face()
        f158 = self.create_face()
        f516 = self.create_face()
        f146 = self.create_face()

        e12 = self.create_edge()
        e13 = self.create_edge()
        e14 = self.create_edge()
        e15 = self.create_edge()
        e16 = self.create_edge()
        e18 = self.create_edge()
        e23 = self.create_edge()
        e28 = self.create_edge()
        e34 = self.create_edge()
        e36 = self.create_edge()
        e37 = self.create_edge()
        e38 = self.create_edge()
        e46 = self.create_edge()
        e56 = self.create_edge()
        e58 = self.create_edge()
        e67 = self.create_edge()
        e68 = self.create_edge()
        e78 = self.create_edge()

        self.compile_face([v1, v2, v3], [e12, e23, e13], f123)
        self.compile_face([v1, v3, v4], [e13, e34, e14], f134)
        self.compile_face([v5, v6, v8], [e56, e68, e58], f568)
        self.compile_face([v6, v7, v8], [e67, e78, e68], f678)
        self.compile_face([v6, v4, v3], [e46, e34, e36], f643)
        self.compile_face([v6, v3, v7], [e36, e37, e67], f637)
        self.compile_face([v8, v7, v3], [e78, e37, e38], f873)
        self.compile_face([v8, v3, v2], [e38, e23, e28], f832)
        self.compile_face([v8, v2, v1], [e28, e12, e18], f821)
        self.compile_face([v1, v5, v8], [e15, e58, e18], f158)
        self.compile_face([v5, v1, v6], [e15, e16, e56], f516)
        self.compile_face([v1, v4, v6], [e14, e46, e16], f146)

    #-------------------------------------------------------------------------
    def compute_angles(self, idxs, vrts):
        n = len(vrts)
        res = {}
        for i in range(n):
            curr = i
            prev = (i-1+n)%n
            next = (i+1)%n
            v0 = vrts[next].pt - vrts[curr].pt
            v1 = vrts[prev].pt - vrts[curr].pt
            v0 /= np.linalg.norm(v0)
            v1 /= np.linalg.norm(v1)
            cos_angle = np.dot(v0, v1)
            curr_angle = np.arccos(cos_angle)
            res[idxs[i]] = curr_angle
        return res

    #-------------------------------------------------------------------------
    def init_as_triang_mesh_stl_file(self, stl_file_name):
        src_mesh = mesh.Mesh.from_file(stl_file_name)
        all_vertices = []
        all_faces = []
        all_normals = []
        all_angles = []
        vrtx_to_faces = {}
        for i in range(len(src_mesh.vectors)):
            face_as_idx = []
            for v in src_mesh.vectors[i]:
                vv = (float(v[0]), float(v[1]), float(v[2]))
                try:
                    idx = all_vertices.index(vv)
                except:
                    all_vertices.append(vv)
                    idx = len(all_vertices) - 1
                face_as_idx.append(idx)
                v2f = vrtx_to_faces.get(idx, [])
                v2f = v2f[:]+[i]
                vrtx_to_faces[idx] = v2f
            all_faces.append(face_as_idx)
            curr_norm = src_mesh.normals[i]
            curr_norm /= np.linalg.norm(curr_norm)
            all_normals.append(curr_norm)

        self.idgen = IDGenerator()
        for i in range(len(all_vertices)):
            all_vertices[i] = self.create_vertex(all_vertices[i])
        for i in range(len(all_faces)):
            all_angles.append(
                self.compute_angles(all_faces[i],
                                    [all_vertices[all_faces[i][0]],
                                        all_vertices[all_faces[i][1]],
                                        all_vertices[all_faces[i][2]]]))
        all_edges = {}
        for f in all_faces:
            n = len(f)
            curr_vrtx = []
            curr_edges = []
            for i in range(n):
                v0_idx = f[i]
                v1_idx = f[(i+1)%n]
                min_idx = min(v0_idx, v1_idx)
                max_idx = max(v0_idx, v1_idx)
                if all_edges.get((min_idx, max_idx)) == None:
                    all_edges[(min_idx, max_idx)] = self.create_edge()
                curr_vrtx.append(all_vertices[v0_idx])
                curr_edges.append(all_edges[(min_idx, max_idx)])
            self.compile_face(curr_vrtx, curr_edges, self.create_face())
        for i in range(len(all_vertices)):
            neigh_faces = vrtx_to_faces[i]
            sum_angles = 0.0
            for neigh_face in neigh_faces:
                sum_angles += all_angles[neigh_face][i]
            sum_norms = np.array([0.,0.,0.])
            for neigh_face in neigh_faces:
                curr_weight = all_angles[neigh_face][i] / sum_angles
                sum_norms += all_normals[neigh_face] * curr_weight
            if not vec_almost_zero(sum_norms):
                sum_norms /= np.linalg.norm(sum_norms)
            else:
                sum_norms =  all_vertices[i].pt
            all_vertices[i].set_nr(sum_norms)
        self.orient_as_first_face()

    #-------------------------------------------------------------------------
    def orient_as_first_face(self):
        checked_faces = {}
        checked_faces[self.f[0]] = 1
        faces_to_check = []
        for f in self.f[0].get_neighbor_faces():
            faces_to_check.append((f,self.f[0]))
            checked_faces[f] = 0
        while False in checked_faces.values():
            curr_face, correct_face = faces_to_check[0]
            if self.set_orientation(correct_face, curr_face):
                checked_faces[curr_face] = 1
                curr_face_neigh = curr_face.get_neighbor_faces()
                for f in curr_face_neigh:
                    if -1 == checked_faces.get(f, -1):
                        faces_to_check.append((f,curr_face))
                        checked_faces[f] = 0
            del faces_to_check[0]
            
    #-------------------------------------------------------------------------
    def set_orientation(self, f0, f1):
        common_edge = f0.get_common_edge(f1)
        f0_hedge = common_edge.get_face_hedge(f0)
        if f0_hedge.vert == f0_hedge.twin.vert:
            #We need to flip f1 orientation
            f1.flip_orientation_in_triang_face(f0_hedge)
        return True

    #-------------------------------------------------------------------------
    def plot(self, cnvs, draw_norms, clr,
             linestyle='', linewidth=1.0):
        for edge in self.e:
            assert edge.he and edge.he.twin
            xs = [edge.he.vert.pt[0], edge.he.twin.vert.pt[0]]
            ys = [edge.he.vert.pt[1], edge.he.twin.vert.pt[1]]
            zs = [edge.he.vert.pt[2], edge.he.twin.vert.pt[2]]
            #cnvs.plot_wireframe(xs, ys, zs, color=clr, linewidth=linewidth, 
            #                    rstride=2, cstride=2)
            cnvs.plot(xs, ys, zs, color=clr, linewidth=linewidth)

        if draw_norms:
            for vert in self.v:
                xs = [vert.pt[0]]
                ys = [vert.pt[1]]
                zs = [vert.pt[2]]
                us = [-vert.nr[0]]
                vs = [-vert.nr[1]]
                ws = [-vert.nr[2]]
                cnvs.quiver(xs,ys,zs,us,vs,ws, color=clr, pivot='tail')

    #-------------------------------------------------------------------------
    def get_extrusion_dir(self, face_vertices):
        n = len(face_vertices)
        res_dir = np.array([0.,0.,0.])
        for i in range(n):
            v0 = face_vertices[(i+1)%n].pt - face_vertices[i].pt
            v1 = face_vertices[(i-1+n)%n].pt - face_vertices[i].pt
            cr = np.cross(v0, v1)
            res_dir += cr
        res_dir /= np.linalg.norm(res_dir)
        return res_dir

    #-------------------------------------------------------------------------
    def extrude_3triang_face(self, face_id, length):
        extr_face = self.id2obj[face_id]
        orig_vrtxs = extr_face.get_vertices()
        orig_edges = extr_face.get_edges()

        face_nr = extr_face.get_face_normal(
                              orig_edges[0].get_face_hedge(extr_face))
        cntr_pt = extr_face.get_face_center()
        trgt_pt = cntr_pt  + face_nr * length
        extr_vrtx = self.create_vertex(trgt_pt)

        extr_edges = []
        extr_faces = []
        n = len(orig_vrtxs)

        for i in range(n):
            extr_edges.append(self.create_edge())
            extr_faces.append(self.create_face())
            orig_edges[i].nullify_face_hedge(extr_face)

        for i in range(n):
            curr_v = [orig_vrtxs[i], 
                      orig_vrtxs[(i+1)%n],
                      extr_vrtx]
            curr_e = [orig_edges[i],
                      extr_edges[(i+1)%n],
                      extr_edges[i%n]]
            extr_faces[i].compile_face(curr_v, curr_e)

        self.id2obj.pop(face_id)
        self.f.remove(extr_face)

    #-------------------------------------------------------------------------
    def extrude_6triang_face(self, face_id, length):
        extr_face = self.id2obj[face_id]
        orig_vrtxs = extr_face.get_vertices()
        orig_edges = extr_face.get_edges()

        face_nr = extr_face.get_face_normal(
                              orig_edges[0].get_face_hedge(extr_face))
        cntr_pt = extr_face.get_face_center()
        trgt_pt = cntr_pt  + face_nr * length
        extr_vrtx = self.create_vertex(trgt_pt)

        extr_edges = []
        extr_faces = []
        n = len(orig_vrtxs)

        for i in range(n):
            extr_edges.append(self.create_edge())
            extr_faces.append(self.create_face())
            orig_edges[i].nullify_face_hedge(extr_face)

        for i in range(n):
            curr_v = [orig_vrtxs[i], 
                      orig_vrtxs[(i+1)%n],
                      extr_vrtx]
            curr_e = [orig_edges[i],
                      extr_edges[(i+1)%n],
                      extr_edges[i%n]]
            extr_faces[i].compile_face(curr_v, curr_e)

        self.id2obj.pop(face_id)
        self.f.remove(extr_face)

    #-------------------------------------------------------------------------
    def extrude_face(self, face_id, length):
        extr_face = self.id2obj[face_id]
        orig_vrtxs = extr_face.get_vertices()
        orig_edges = extr_face.get_edges()
        extr_vrtxs = []
        extr_top_edges = []
        extr_side_edges = []
        extr_faces = []
        n = len(orig_vrtxs)
        old_neigh_vrts = []
        old_neigh_edges = []
        extr_dir = self.get_extrusion_dir(orig_vrtxs)
        for i in range(n):
            extr_pt = orig_vrtxs[i].pt + extr_dir * length
            extr_vrtxs.append(self.create_vertex(extr_pt))
            extr_top_edges.append(self.create_edge())
            extr_side_edges.append(self.create_edge())
            extr_faces.append(self.create_face())
            orig_edges[i].nullify_face_hedge(extr_face)

        for i in range(n):
            curr_v = [orig_vrtxs[i], 
                      extr_vrtxs[i],
                      extr_vrtxs[(i-1+n)%n],
                      orig_vrtxs[(i-1+n)%n]]
            curr_e = [extr_side_edges[i],
                      extr_top_edges[(i-1+n)%n],
                      extr_side_edges[(i-1+n)%n],
                      orig_edges[(i-1+n)%n]]
            extr_faces[(i-1+n)%n].compile_face(curr_v, curr_e)
        extr_face.compile_face(extr_vrtxs, extr_top_edges)
        #self.set_naive_normals()

    #-------------------------------------------------------------------------
    def triangulize_quad_mesh(self):
        tri_mesh = DCtrlMesh(self.eid+1, self.averaging_fn)
        tri_mesh.idgen = IDGenerator(self.idgen.curr + self.idgen.step)
        vertices_2_edges = {}
        old_vrtxs_2_new_vrtxs = {}
        for face in self.f:
            orig_vrtxs = face.get_vertices()
            n = len(orig_vrtxs)
            assert n == 4
            new_vrtxs = []
            for ov in orig_vrtxs:
                new_vrtx = old_vrtxs_2_new_vrtxs.get(ov.eid, None)
                if new_vrtx == None:
                    new_vrtx = tri_mesh.create_vertex(ov.pt)
                    new_vrtx.set_nr(ov.nr)
                    old_vrtxs_2_new_vrtxs[ov.eid] = new_vrtx
                new_vrtxs.append(new_vrtx)
                
            fv1 = [new_vrtxs[0], new_vrtxs[1], new_vrtxs[2]]
            fv2 = [new_vrtxs[0], new_vrtxs[2], new_vrtxs[3]]
            tri_mesh.produce_face(fv1, vertices_2_edges)
            tri_mesh.produce_face(fv2, vertices_2_edges)
        return tri_mesh

    #-------------------------------------------------------------------------
    def refine_as_kob4pt(self):
        refined_mesh = self.refine_by_interpolation(
                          DCtrlMesh.split_edge_as_kob4pt, 
                          DCtrlMesh.split_quad_face_by_center_insert_4neighbs)
        return refined_mesh
    
    #-------------------------------------------------------------------------
    def refine_as_butterfly(self):
        refined_mesh = self.refine_by_interpolation(
                                     DCtrlMesh.split_edge_as_butterfly, 
                                     DCtrlMesh.split_triang_face_as_butterfly)
        return refined_mesh

    #-------------------------------------------------------------------------
    def refine_as_catmull_clark(self, get_edge_vertex_func = None, 
                                get_vrtx_vertex_func = None):
        if get_edge_vertex_func == None:
            get_edge_vertex_func = DCtrlMesh.get_edge_vertex_as_cc
        if get_vrtx_vertex_func == None:
            get_vrtx_vertex_func = DCtrlMesh.get_vrtx_vertex_as_cc

        refined_mesh = DCtrlMesh(self.eid+1, self.averaging_fn)
        refined_mesh.idgen = IDGenerator(self.idgen.curr + self.idgen.step)
        old_faces_2_new_verts = {}
        for face in self.f:
            avg_pt, avg_nr = self.get_face_average_vertex(face)
            avg_vert = refined_mesh.create_vertex(avg_pt)
            avg_vert.set_nr(avg_nr)
            old_faces_2_new_verts[face.eid] = avg_vert

        old_edges_2_new_verts = {}
        for edge in self.e:
            avg_pt, avg_nr = get_edge_vertex_func(self, 
                                                  edge, 
                                                  old_faces_2_new_verts)
            avg_vert = refined_mesh.create_vertex(avg_pt)
            avg_vert.set_nr(avg_nr)
            old_edges_2_new_verts[edge.eid] = avg_vert

        old_verts_2_new_verts = {}
        for vrtx in self.v:
            avg_pt, avg_nr = get_vrtx_vertex_func(self, 
                                                  vrtx, 
                                                  old_faces_2_new_verts,
                                                  old_edges_2_new_verts)
            avg_vert = refined_mesh.create_vertex(avg_pt)
            avg_vert.set_nr(avg_nr)
            old_verts_2_new_verts[vrtx.eid] = avg_vert

        vertices_2_edges = {}
        for face in self.f:
            curr_edges = face.get_edges()
            n = len(curr_edges)
            for i in range(n):
                curr_edge = curr_edges[i]
                prev_edge = curr_edges[(i-1+n)%n]
                old_vrtx = curr_edge.get_src_vertex(face)
                new_vertices = [old_verts_2_new_verts[old_vrtx.eid],
                                old_edges_2_new_verts[curr_edge.eid],
                                old_faces_2_new_verts[face.eid],
                                old_edges_2_new_verts[prev_edge.eid] ]
                refined_mesh.produce_face(new_vertices, vertices_2_edges)

        return refined_mesh
                   
    #-------------------------------------------------------------------------
    def get_edge_vertex_as_cc(self, edge, old_faces_2_new_verts):
        left_face, right_face = edge.get_faces()
        left_face_vrtx = old_faces_2_new_verts[left_face.eid]
        right_face_vrtx = old_faces_2_new_verts[right_face.eid]
        mfv_pt, mfv_nr = self.average_vertices(0.5, 
                                               left_face_vrtx.pt, 
                                               right_face_vrtx.pt,
                                               left_face_vrtx.nr, 
                                               right_face_vrtx.nr)
        src_vrtx, dst_vrtx = edge.get_vertices()
        mid_pt, mid_nr = self.average_vertices(0.5, 
                                               src_vrtx.pt, 
                                               dst_vrtx.pt,
                                               src_vrtx.nr, 
                                               dst_vrtx.nr)
        res_pt, res_nr = self.average_vertices(0.5, mid_pt, mfv_pt,
                                               mid_nr, mfv_nr)
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def get_vrtx_vertex_as_cc(self, vrtx, faces_2_verts, edges_2_verts):
        faces = vrtx.get_faces()
        n = len(faces)
        face_vrtxs = []
        for f in faces:
            face_vrtxs.append(faces_2_verts[f.eid])
        favg_pt, favg_nr = self.compute_sum_as_repeated_averages(face_vrtxs)

        edges = vrtx.get_edges()
        mid_vrtxs = []
        for e in edges:
            src_vrtx, dst_vrtx = e.get_vertices()
            mid_pt, mid_nr = self.average_vertices(0.5, 
                                                   src_vrtx.pt, 
                                                   dst_vrtx.pt,
                                                   src_vrtx.nr, 
                                                   dst_vrtx.nr)
            mv = DVertex()
            mv.set_pt(mid_pt)
            mv.set_nr(mid_nr)
            mid_vrtxs.append(mv)

        eavg_pt, eavg_nr = self.compute_sum_as_repeated_averages(mid_vrtxs)

        f2r_pt, f2r_nr = self.average_vertices(1./3., favg_pt, eavg_pt, 
                                               favg_nr, eavg_nr)
        res_pt, res_nr = self.average_vertices(3./float(n), f2r_pt, vrtx.pt, 
                                               f2r_nr, vrtx.nr)
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def get_vrtx_vertex_as_copy(self, vrtx, 
                                faces_2_verts = None, 
                                edges_2_verts = None):
        return vrtx.pt, vrtx.nr

    #-------------------------------------------------------------------------
    def get_edge_vertex_as_mid(self, edge, old_faces_2_new_verts = None):
        src_vrtx, dst_vrtx = edge.get_vertices()
        res_pt, res_nr = self.average_vertices(0.5, 
                                               src_vrtx.pt, 
                                               dst_vrtx.pt,
                                               src_vrtx.nr, 
                                               dst_vrtx.nr)
        return res_pt, res_nr        

    #-------------------------------------------------------------------------
    def compute_sum_as_repeated_averages(self, pnps, weights=None):
        n = len(pnps)
        if weights == None:
            weights = [1./n]*n
        v0 = pnps[0]
        v1 = pnps[1]
        acc_weight = weights[0] + weights[1]
        avg_pt, avg_nr = self.average_vertices(weights[0]/acc_weight, 
                                               v0.pt, v1.pt, 
                                               v0.nr, v1.nr ) 
        for i in range(2, n):
            acc_weight += weights[i]
            avg_pt, avg_nr = self.average_vertices(weights[i]/acc_weight, 
                                                   pnps[i].pt, avg_pt,
                                                   pnps[i].nr, avg_nr)
        return avg_pt, avg_nr

    #-------------------------------------------------------------------------
    def get_face_average_vertex(self, face):
        v = face.get_vertices()
        res_pt, res_norm = self.compute_sum_as_repeated_averages(v)
        return res_pt, res_norm
        
    #-------------------------------------------------------------------------
    def refine_as_loop(self):
        ''' copy/paste of refine_by_interpolation
            @@TODO: unite these two, change the name '''
        refined_mesh = DCtrlMesh(self.eid+1, self.averaging_fn)
        refined_mesh.idgen = IDGenerator(self.idgen.curr + self.idgen.step)
        old_edges_2_new_verts = {}
        old_edges_2_new_edges = {}
        old_verts_2_new_verts = {}

        for edge in self.e:
            avg_pt, avg_nr = self.split_edge_as_loop(edge)
            avg_vert = refined_mesh.create_vertex(avg_pt)
            avg_vert.set_nr(avg_nr)
            old_edges_2_new_verts[edge.eid] = avg_vert
            new_edge1 = refined_mesh.create_edge()
            new_edge2 = refined_mesh.create_edge()
            old_edges_2_new_edges[(edge.eid, edge.he.vert.eid)] = new_edge1
            old_edges_2_new_edges[(edge.eid, edge.he.twin.vert.eid)] = new_edge2

        for vert in self.v:
            avg_pt, avg_nr = self.refine_vertex_as_loop(vert)
            refined_vert = refined_mesh.create_vertex(avg_pt)
            refined_vert.set_nr(avg_nr)
            old_verts_2_new_verts[vert.eid] = refined_vert

        for face in self.f:
            self.split_triang_face_as_butterfly(face, 
                                                refined_mesh, 
                                                old_edges_2_new_verts,
                                                old_edges_2_new_edges,
                                                old_verts_2_new_verts)
        return refined_mesh

    #-------------------------------------------------------------------------
    def split_edge_as_loop(self, edge):
        s = edge.he.vert
        d = edge.he.dest()
        l = edge.he.prev.vert
        r = edge.he.twin.prev.vert
        w = [3./8., 3./8., 1./8., 1./8.]
        v = [s,d,l,r]
        res_pt, res_nr = self.compute_sum_as_repeated_averages(v, w)
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def refine_vertex_as_loop(self, vert):
        vrts = vert.get_neighbor_vertices()
        n = len(vrts)
        vrts.insert(0, vert)
        #if n > 3:
        #    beta = 3./(8.*n)
        #else:
        #    beta = 3./16.
        #alpha = 1. - n*beta
        alpha = 0.375 + (0.375 + 0.25 * np.cos((2. * np.pi) / n)) ** 2
        beta = (1. - alpha)/n
        weights = [alpha] + [beta]*n
        res_pt, res_nr = self.compute_sum_as_repeated_averages(vrts, weights)
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def refine_by_interpolation(self,
                                edge_split_func, face_split_func):
        refined_mesh = DCtrlMesh(self.eid+1, self.averaging_fn)
        refined_mesh.idgen = IDGenerator(self.idgen.curr + self.idgen.step)
        old_edges_2_new_verts = {}
        old_edges_2_new_edges = {}
        old_verts_2_new_verts = {}

        for edge in self.e:
            avg_pt, avg_nr = edge_split_func(self, edge)
            avg_vert = refined_mesh.create_vertex(avg_pt)
            avg_vert.set_nr(avg_nr)
            old_edges_2_new_verts[edge.eid] = avg_vert
            new_edge1 = refined_mesh.create_edge()
            new_edge2 = refined_mesh.create_edge()
            old_edges_2_new_edges[(edge.eid, edge.he.vert.eid)] = new_edge1
            old_edges_2_new_edges[(edge.eid, edge.he.twin.vert.eid)] = new_edge2

        for vert in self.v:
            preserved_vert = refined_mesh.create_vertex(vert.pt)
            preserved_vert.set_nr(vert.nr)
            old_verts_2_new_verts[vert.eid] = preserved_vert

        for face in self.f:
            face_split_func(self, face, 
                            refined_mesh, 
                            old_edges_2_new_verts,
                            old_edges_2_new_edges,
                            old_verts_2_new_verts)
        return refined_mesh

    #-------------------------------------------------------------------------
    def split_triang_face_as_butterfly(self, face, 
                                       refined_mesh, 
                                       old_edges_2_new_verts,
                                       old_edges_2_new_edges,
                                       old_verts_2_new_verts):
        old_verts = face.get_vertices()
        old_edges = face.get_edges()
        new_inner_edges = []
        inner_face_verts = []
        for i in [0,1,2]:
            curr_old_vert = old_verts[i].eid
            curr_old_edge = old_edges[i].eid
            prev_old_edge = old_edges[(i+2)%3].eid

            new_face = refined_mesh.create_face()
            new_inner_edge = refined_mesh.create_edge()

            new_inner_edges.append(new_inner_edge)
            curr_new_edge_vert = old_edges_2_new_verts[curr_old_edge]
            prev_new_edge_vert = old_edges_2_new_verts[prev_old_edge]
            inner_face_verts.append(prev_new_edge_vert)

            new_face_verts = [old_verts_2_new_verts[curr_old_vert], 
                              curr_new_edge_vert, prev_new_edge_vert]

            new_face_edges = [old_edges_2_new_edges[(curr_old_edge, 
                                                     curr_old_vert)],
                              new_inner_edge,
                              old_edges_2_new_edges[(prev_old_edge, 
                                                     curr_old_vert)]]
            refined_mesh.compile_face(new_face_verts, 
                                      new_face_edges, 
                                      new_face)

        inner_face = refined_mesh.create_face()
        refined_mesh.compile_face(inner_face_verts, new_inner_edges, inner_face)

    #-------------------------------------------------------------------------
    def split_quad_face_by_center_insert_4neighbs(self, face, 
                                                  refined_mesh, 
                                                  old_edges_2_new_verts,
                                                  old_edges_2_new_edges,
                                                  old_verts_2_new_verts):
        '''
        @assumption: all faces have 4 edges
        '''
        old_verts = face.get_vertices()
        old_edges = face.get_edges()
        new_mid_vertices = []
        new_inner_edges = []
        for i in range(len(old_edges)):
            new_mid_vertices.append(old_edges_2_new_verts[old_edges[i].eid])
            new_inner_edges.append(refined_mesh.create_edge())
        vp_im1, vp_i, vp_ip1, vp_ip2 = self.collect_4_edge_midpoints(
                                                        face, old_edges[0],
                                                        old_edges_2_new_verts)
        hp_im1, hp_i, hp_ip1, hp_ip2 = self.collect_4_edge_midpoints(
                                                        face, old_edges[1],
                                                        old_edges_2_new_verts) 
        ub_pt, ub_norm = self.apply_4pt_rule(vp_im1, vp_i, vp_ip1, vp_ip2)
        rl_pt, rl_norm = self.apply_4pt_rule(hp_im1, hp_i, hp_ip1, hp_ip2)
        cntr_pt, cntr_norm = self.average_vertices(0.5, ub_pt, rl_pt, 
                                                   ub_norm, rl_norm)
        cntr_vert = refined_mesh.create_vertex(cntr_pt)
        cntr_vert.set_nr(cntr_norm)
        for i in [0,1,2,3]:
            new_face = refined_mesh.create_face()
            inner_prev_edge = new_inner_edges[(i+3)%4]
            inner_curr_edge = new_inner_edges[i]
            new_face_verts = [old_verts_2_new_verts[old_verts[i].eid], 
                              new_mid_vertices[i], 
                              cntr_vert, 
                              new_mid_vertices[(i+3)%4]]
            new_face_edges = [old_edges_2_new_edges[(old_edges[i].eid, 
                                                     old_verts[i].eid)],
                              inner_curr_edge,
                              inner_prev_edge,
                              old_edges_2_new_edges[(old_edges[(i+3)%4].eid, 
                                                     old_verts[i].eid)]]
            refined_mesh.compile_face(new_face_verts, 
                                      new_face_edges, 
                                      new_face)

    #-------------------------------------------------------------------------
    def collect_4_edge_midpoints(self, face, edge, old_edges_2_new_verts):
        '''
        @assumption: all face have 4 edges
        '''
        i_hedge = edge.get_face_hedge(face)
        im1_hedge = i_hedge.twin.next.next
        ip1_hedge = i_hedge.next.next
        ip2_hedge = i_hedge.next.next.twin.next.next
        return old_edges_2_new_verts[im1_hedge.edge.eid],\
               old_edges_2_new_verts[i_hedge.edge.eid],\
               old_edges_2_new_verts[ip1_hedge.edge.eid],\
               old_edges_2_new_verts[ip2_hedge.edge.eid]
    #-------------------------------------------------------------------------
    def split_edge_as_kob4pt(self, edge):
        '''
        @assumption: both vertices has valence 4
        '''
        p_i = edge.he.vert
        p_i_val = len(p_i.get_edges())
        p_ip1 = edge.he.twin.vert
        p_ip1_val = len(p_ip1.get_edges())
        if p_i_val == 4:
            p_im1 = edge.he.prev.twin.prev.vert
        else:
            p_im1 = self.compute_virt_edge_vrtx_kob4pt(edge.he)
        if p_ip1_val == 4:
            p_ip2 = edge.he.next.twin.next.twin.vert
        else:
            p_ip2 = self.compute_virt_edge_vrtx_kob4pt(edge.he.twin)
        return self.apply_4pt_rule(p_im1, p_i, p_ip1, p_ip2)

    #-------------------------------------------------------------------------
    def compute_virt_edge_vrtx_kob4pt(self, zero_hedge):
        curr_he = zero_hedge
        direct_neighb = []
        diag_neighb = []
        n = 0
        while True:
            assert curr_he
            n += 1
            direct_neighb.append(curr_he.dest())
            diag_neighb.append(curr_he.next.dest())
            curr_he = curr_he.prev.twin
            if curr_he == zero_hedge:
                break
        weights = [4./n]*n + [-1.]*3 + [0.2]*4 + [-4./(5.*n)]*n
        vertices = direct_neighb[:]
        vertices.extend([direct_neighb[-1],
                         direct_neighb[0],
                         direct_neighb[1]])
        vertices.extend([diag_neighb[-2],
                         diag_neighb[-1],
                         diag_neighb[0],
                         diag_neighb[1]])
        vertices.extend(diag_neighb[:])
        res_pt, res_nr = self.compute_sum_as_repeated_averages(\
                          vertices, weights)
        res_vert = DVertex()
        res_vert.set_pt(res_pt)
        res_vert.set_nr(res_nr)
        return res_vert
    #-------------------------------------------------------------------------
    def apply_4pt_rule(self, p_im1, p_i, p_ip1, p_ip2):
        res1_pt, res1_nr = self.average_vertices(-0.125, 
                                                 p_im1.pt, p_i.pt, 
                                                 p_im1.nr, p_i.nr)
        res2_pt, res2_nr = self.average_vertices(1.125, 
                                                 p_ip1.pt, p_ip2.pt, 
                                                 p_ip1.nr, p_ip2.nr)
        res_pt, res_nr = self.average_vertices(0.5, 
                                               res1_pt, res2_pt, 
                                               res1_nr, res2_nr)
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def split_edge_as_butterfly(self, edge):
        
        #res_pt, res_norm = self.split_edge_as_butterfly_v1(edge)
        res_pt, res_norm = self.split_edge_as_mod_butterfly_v1(edge)
        #res_pt, res_norm = self.split_edge_as_butterfly_v2(edge)
        #res_pt, res_norm = self.split_edge_as_butterfly_v3(edge)
        #res_pt, res_norm = self.split_edge_as_butterfly_straight_linear(edge)
        #res_pt, res_norm = self.get_edge_vertex_as_mid(edge)
        return res_pt, res_norm

    #-------------------------------------------------------------------------
    def split_edge_as_butterfly_straight_linear(self, edge):
        s = edge.he.vert
        d = edge.he.twin.vert 
        u = edge.he.prev.vert
        b = edge.he.twin.prev.vert
        ur = edge.he.next.twin.prev.vert
        ul = edge.he.prev.twin.prev.vert
        br = edge.he.twin.prev.twin.prev.vert
        bl = edge.he.twin.next.twin.prev.vert
        res_pt = 0.5*s.pt + 0.5*d.pt + 0.125* u.pt + 0.125*b.pt \
                  - 0.0625 * ur.pt - 0.0625 * ul.pt - 0.0625 * br.pt - 0.0625 * bl.pt
        res_nr  = s.nr.copy()
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def split_edge_as_butterfly_v1(self, edge):
        '''
           UL-------U-------UR
             \     / \     /
              \   /   \   /
               \ /     \ /
                S---R-->D
               / \     / \
              /   \   /   \
             /     \ /     \
           BL-------B-------BR
        R =    1/2( 5/4 (4/5 S + 1/5 U) - 1/4 (1/2 BL + 1/2 UL) )
             + 1/2( 5/4 (4/5 D + 1/5 B) - 1/4 (1/2 UR + 1/2 BR) )
        '''
        s = edge.he.vert
        d = edge.he.twin.vert 
        u = edge.he.prev.vert
        b = edge.he.twin.prev.vert
        ur = edge.he.next.twin.prev.vert
        ul = edge.he.prev.twin.prev.vert
        br = edge.he.twin.prev.twin.prev.vert
        bl = edge.he.twin.next.twin.prev.vert
        s_u_pt, s_u_nr = self.average_vertices(0.8, s.pt, u.pt, s.nr, u.nr )
        d_b_pt, d_b_nr = self.average_vertices(0.8, d.pt, b.pt, d.nr, b.nr )
        bl_ul_pt, bl_ul_nr = self.average_vertices(0.5, bl.pt, ul.pt, 
                                                        bl.nr, ul.nr )
        ur_br_pt, ur_br_nr = self.average_vertices(0.5, ur.pt, br.pt, 
                                                        ur.nr, br.nr )
        lft_pt, lft_nr = self.average_vertices(1.25, s_u_pt, bl_ul_pt, 
                                                     s_u_nr, bl_ul_nr)
        rgh_pt, rgh_nr = self.average_vertices(1.25, d_b_pt, ur_br_pt, 
                                                     d_b_nr, ur_br_nr)
        res_pt, res_nr = self.average_vertices(0.5, lft_pt, rgh_pt, 
                                                    lft_nr, rgh_nr)
        if res_pt[0] == np.nan:
            a = 5
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def get_edge_extraordinary_vertex_as_mod_butterfly(self, n, vrts):
        if n == 3:
            w = [5./12., -1./12., -1./12.]
        elif n == 4:
            w = [3./8., 0., -1./8., 0.]
        else:
            w = []
            for i in range(n):
                curr_w = 1./n * (0.25 +       np.cos( 2.*np.pi*i / n ) + \
                                        0.5 * np.cos( 4.*np.pi*i / n ) )
                w.append( curr_w )
        w.append(3./4.)
        res_pt, res_nr = self.compute_sum_as_repeated_averages( vrts, w )
        return res_pt, res_nr
    
    #-------------------------------------------------------------------------
    def split_edge_as_mod_butterfly_v1(self, edge):
        s = edge.he.vert
        d = edge.he.twin.vert 
        src_vrts = s.get_neighbor_vertices( edge )
        dst_vrts = d.get_neighbor_vertices( edge )
        src_val = len(src_vrts)
        dst_val = len(dst_vrts)
        dst_vrts.append( d )
        src_vrts.append( s )
        if src_val == dst_val == 6:
            return self.split_edge_as_butterfly_v3(edge)
        else:
            if src_val != 6:
                lft_pt, lft_nr = \
                    self.get_edge_extraordinary_vertex_as_mod_butterfly(src_val, src_vrts)
                res_pt, res_nr = lft_pt, lft_nr
            if dst_val != 6:
                rgh_pt, rgh_nr = \
                    self.get_edge_extraordinary_vertex_as_mod_butterfly(dst_val, dst_vrts)
                res_pt, res_nr = rgh_pt, rgh_nr
            if src_val != 6 and dst_val != 6:
                res_pt, res_nr = self.average_vertices(0.5, lft_pt, rgh_pt, 
                                                    lft_nr, rgh_nr)
        if res_pt[0] == np.nan:
            print 'NAN in Butterfly'
        return res_pt, res_nr
           
    #-------------------------------------------------------------------------
    def split_edge_as_butterfly_v2(self, edge):
        '''
           UL-------U-------UR
             \     / \     /
              \   /   \   /
               \ /     \ /
                S---R-->D
               / \     / \
              /   \   /   \
             /     \ /     \
           BL-------B-------BR
        R =  1/2 ( 1/2 ( 5/4 (4/5 S + 1/5 U) - 1/4 UL) ) +
                   1/2 ( 5/4 (4/5 S + 1/5 B) - 1/4 BL) ) ) +
             1/2 ( 1/2 ( 5/4 (4/5 D + 1/5 U) - 1/4 UR) ) + 
                   1/2 ( 5/4 (4/5 D + 1/5 B) - 1/4 BR) ) )
        '''
        s = edge.he.vert
        d = edge.he.twin.vert 
        u = edge.he.prev.vert
        b = edge.he.twin.prev.vert
        ur = edge.he.next.twin.prev.vert
        ul = edge.he.prev.twin.prev.vert
        br = edge.he.twin.prev.twin.prev.vert
        bl = edge.he.twin.next.twin.prev.vert
        s_u_pt, s_u_nr = self.average_vertices(0.8, s.pt, u.pt, s.nr, u.nr )
        s_b_pt, s_b_nr = self.average_vertices(0.8, s.pt, b.pt, s.nr, b.nr )
        d_u_pt, d_u_nr = self.average_vertices(0.8, d.pt, u.pt, d.nr, u.nr )
        d_b_pt, d_b_nr = self.average_vertices(0.8, d.pt, b.pt, d.nr, b.nr )

        s_u_ul_pt, s_u_ul_nr = self.average_vertices(1.25, s_u_pt, ul.pt, 
                                                           s_u_nr, ul.nr )
        s_b_bl_pt, s_b_bl_nr = self.average_vertices(1.25, s_b_pt, bl.pt, 
                                                           s_b_nr, bl.nr )
        d_u_ur_pt, d_u_ur_nr = self.average_vertices(1.25, d_u_pt, ur.pt, 
                                                           d_u_nr, ur.nr )
        d_b_br_pt, d_b_br_nr = self.average_vertices(1.25, d_b_pt, br.pt, 
                                                           d_b_nr, br.nr )

        lft_pt, lft_nr = self.average_vertices(0.5, s_u_ul_pt, s_b_bl_pt, 
                                                    s_u_ul_nr, s_b_bl_nr)
        rgh_pt, rgh_nr = self.average_vertices(0.5, d_u_ur_pt, d_b_br_pt, 
                                                    d_u_ur_nr, d_b_br_nr)

        res_pt, res_nr = self.average_vertices(0.5, lft_pt, rgh_pt, 
                                                    lft_nr, rgh_nr)
        return res_pt, res_nr
           
    #-------------------------------------------------------------------------
    def split_edge_as_butterfly_v3(self, edge):
        '''
          -1/16     1/8    -1/16
            UL-------U-------UR
              \     / \     /
               \   /   \   /
                \ /     \ /
             1/2 S---R-->D 1/2
                / \     / \
               /   \   /   \
              /     \ /     \
            BL-------B-------BR
          -1/16     1/8    -1/16
        '''
        s = edge.he.vert
        d = edge.he.twin.vert 
        u = edge.he.prev.vert
        b = edge.he.twin.prev.vert
        ur = edge.he.next.twin.prev.vert
        ul = edge.he.prev.twin.prev.vert
        br = edge.he.twin.prev.twin.prev.vert
        bl = edge.he.twin.next.twin.prev.vert
        vertices = [s,d, u, b, ur, ul, br, bl]
        w = 0.
        weights = [0.5 - w, 0.5 - w, 1./8. + 2. * w, 1./8. + 2.*w] \
                  + [-1./16 - w]*4
        res_pt, res_nr = self(\
                          vertices, weights)
        return res_pt, res_nr

    #-------------------------------------------------------------------------
    def average_vertices(self, t0, p0, p1, n0, n1):
        res_pt, res_norm, cntr_pt,  radius_2D, beta1, beta2 = \
                                       self.averaging_fn(t0, 1.0 - t0, True, 
                                       p0, p1, n0, n1)
        return res_pt, res_norm

    #-------------------------------------------------------------------------
    def get_or_create_edge(self, vertices_2_edges, v1, v2):
        min_id = min(v1.eid, v2.eid) 
        max_id = max(v1.eid, v2.eid)
        edge = vertices_2_edges.get((min_id, max_id), None)
        if not edge:
            edge = self.create_edge()
            vertices_2_edges[(min_id, max_id)] = edge
        return edge
            
    #-------------------------------------------------------------------------
    def init_as_torus(self, b_triang = True, 
                      orbit = 20., inner_radius = 5., n_of_verts = 6):
        self.idgen = IDGenerator()
        delta_angle = (2.*m.pi)/n_of_verts
        vertices = []
        for i in range(n_of_verts): 
            glob_sin = m.sin(i * delta_angle)
            glob_cos = m.cos(i * delta_angle)
            curr_vert_ring = []
            for j in range(n_of_verts):
                inn_ang = j * delta_angle
                inn_x = orbit + inner_radius * m.cos(inn_ang)
                inn_y = 0.0
                inn_z = inner_radius * m.sin(inn_ang)
                glob_x =  glob_cos * inn_x - glob_sin * inn_y
                glob_y =  glob_sin * inn_x - glob_cos * inn_y
                glob_z =  inn_z
                curr_vert = self.create_vertex([glob_x, glob_y, glob_z])
                glob_cntr_x = glob_cos * orbit
                glob_cntr_y = glob_sin * orbit
                curr_norm = np.array([glob_x - glob_cntr_x, 
                                      glob_y - glob_cntr_y, 
                                      glob_z])
                curr_vert.set_nr(curr_norm)
                curr_vert_ring.append(curr_vert)
            vertices.append(curr_vert_ring)
        
        vertices_2_edges = {}
        for i in range(n_of_verts):
            curr_ring_idx = i
            next_ring_idx = (i + 1)%n_of_verts
            for j in range(n_of_verts):
                curr_inn_idx = j
                next_inn_idx = (j+1)%n_of_verts
                brv = vertices[curr_ring_idx][curr_inn_idx]
                blv = vertices[curr_ring_idx][next_inn_idx]
                urv = vertices[next_ring_idx][curr_inn_idx]
                ulv = vertices[next_ring_idx][next_inn_idx]
                be = self.get_or_create_edge(vertices_2_edges, brv, blv)
                ue = self.get_or_create_edge(vertices_2_edges, urv, ulv)
                re = self.get_or_create_edge(vertices_2_edges, brv, urv)
                le = self.get_or_create_edge(vertices_2_edges, blv, ulv)
                if b_triang:
                    de = self.get_or_create_edge(vertices_2_edges, brv, ulv)
                    rf = self.create_face()
                    lf = self.create_face()
                    self.compile_face([brv, urv, ulv], [re, ue, de], rf)
                    self.compile_face([blv, brv, ulv], [be, de, le], lf)
                else:
                    ff = self.create_face()
                    self.compile_face([brv, urv, ulv, blv], 
                                      [re, ue, le, be], ff)

    #-------------------------------------------------------------------------
    def init_as_tetrahedron(self, offset=10.):
        self.idgen = IDGenerator()
        #v1 = self.create_vertex([0., 0., 10.])
        #v2 = self.create_vertex([10.0, 0., 0.])                     
        #v3 = self.create_vertex([10. * np.cos(2.*m.pi/3.), 
        #                         10. * np.sin(2.*m.pi/3.), 
        #                         0.])
        #v4x = -np.sin(m.pi/6.)*np.cos(m.pi/3.) * 10.
        #v4y = -np.sin(m.pi/6.)*np.sin(m.pi/3.) * 10.
        #v4z = -np.cos(m.pi/6.) * 10.
        #v4 = self.create_vertex([v4x, v4y, v4z])                     

        v1 = self.create_vertex([-offset, -offset,  offset])
        v2 = self.create_vertex([ offset,  offset,  offset])
        v3 = self.create_vertex([-offset,  offset, -offset])
        v4 = self.create_vertex([ offset, -offset, -offset])

        f123 = self.create_face()
        f134 = self.create_face()
        f142 = self.create_face()
        f243 = self.create_face()

        e12 = self.create_edge()
        e13 = self.create_edge()
        e14 = self.create_edge()
        e23 = self.create_edge()
        e34 = self.create_edge()
        e24 = self.create_edge()

        self.compile_face([v1, v2, v3], [e12, e23, e13], f123)
        self.compile_face([v1, v3, v4], [e13, e34, e14], f134)
        self.compile_face([v1, v4, v2], [e14, e24, e12], f142)
        self.compile_face([v2, v4, v3], [e24, e34, e23], f243)

    #-------------------------------------------------------------------------
    def refine_by_bspl_interpolation(self):
        refined_mesh = DCtrlMesh(self.eid+1, None)
        refined_mesh.idgen = IDGenerator(self.idgen.curr + self.idgen.step)
        old_edges_2_new_verts = {}
        old_edges_2_new_edges = {}
        old_verts_2_new_verts = {}

        for vert in self.v:
            vert.init_tangent_dirs()
            preserved_vert = refined_mesh.create_vertex(vert.pt)
            preserved_vert.set_deriv_dirs(vert.tang_dirs)
            preserved_vert.set_nr(vert.nr)
            old_verts_2_new_verts[vert.eid] = preserved_vert

        for edge in self.e:
            edge.init_bspl_crv()
            avg_pt, avg_nr = eval_bspl_crv(0.5, edge.bspl_crv)
            avg_vert = refined_mesh.create_vertex(avg_pt)
            avg_vert.set_nr(avg_nr)
            old_edges_2_new_verts[edge.eid] = avg_vert
            new_edge1 = refined_mesh.create_edge()
            new_edge2 = refined_mesh.create_edge()
            old_edges_2_new_edges[(edge.eid, edge.he.vert.eid)] = new_edge1
            old_edges_2_new_edges[(edge.eid, edge.he.twin.vert.eid)] = new_edge2

        for face in self.f:
            cntr_pr, cntr_norm = face.eval_bspl_srf(0.5, 0.5)
            cntr_vert = refined_mesh.create_vertex(cntr_pt)
            cntr_vert.set_nr(cntr_norm)
            for i in [0,1,2,3]:
                new_face = refined_mesh.create_face()
                inner_prev_edge = new_inner_edges[(i+3)%4]
                inner_curr_edge = new_inner_edges[i]
                new_face_verts = [old_verts_2_new_verts[old_verts[i].eid], 
                                  new_mid_vertices[i], 
                                  cntr_vert, 
                                  new_mid_vertices[(i+3)%4]]
                new_face_edges = [old_edges_2_new_edges[(old_edges[i].eid, 
                                                         old_verts[i].eid)],
                                  inner_curr_edge,
                                  inner_prev_edge,
                                  old_edges_2_new_edges[(old_edges[(i+3)%4].eid, 
                                                         old_verts[i].eid)]]
                refined_mesh.compile_face(new_face_verts, 
                                          new_face_edges, 
                                          new_face)            
            return refined_mesh

    
#============================== END OF FILE ==================================
