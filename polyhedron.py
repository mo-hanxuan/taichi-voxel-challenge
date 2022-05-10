"""
    creat 3 pyramids, with color style of Wasteland literature, 
    where the piramid is created by carving through 4 planes, and orientation of the pyramid can be roteted. 
"""

from matplotlib.pyplot import getp
from scene import Scene
import taichi as ti
from taichi.math import *
import numpy as np
from math_utils import np_rotate_matrix

SIZE = 128

@ti.data_oriented
class Polyhedron:
    def __init__(self, facetNormals, facetPoints, dtype=ti.f32):
        self.facetNormals = ti.Matrix(facetNormals)
        self.facetPoints = ti.Matrix(facetPoints)
    
    @ti.kernel
    def setPolyhedron(self, ):
        ### facetNormals: each line in the matrix is a vector, pointed to the outside of the facet
        for i, j, k in ti.ndrange(SIZE, SIZE, SIZE):
            pos = vec3(i - SIZE//2, j - SIZE//2, k - SIZE//2)
            inside = True
            for ifacet in ti.static(range(self.facetNormals.n)):
                normal = vec3(self.facetNormals[ifacet, 0], self.facetNormals[ifacet, 1], self.facetNormals[ifacet, 2])
                point = vec3(self.facetPoints[ifacet, 0], self.facetPoints[ifacet, 1], self.facetPoints[ifacet, 2])
                if dot_product(pos - point, normal) > 0:
                    inside = False
            if inside:
                scene.set_voxel(ivec3(i - SIZE//2, j - SIZE//2, k - SIZE//2), 1, vec3(0.5, 0.5, 0.))


@ti.data_oriented
class Ellip:
    def __init__(self, center, radiusX, radiusY, radiusZ, axis, theta):
        self.center = vec3(center[0], center[1], center[2])
        rotate = np_rotate_matrix(np.array(axis), theta)[:3, :3]
        self.eigenVecs = rotate @ np.array([
            [1./radiusX**2, 0., 0.],
            [0., 1./radiusY**2, 0.],
            [0., 0., 1./radiusZ**2],
        ]) @ rotate.transpose()
        self.eigenVecs = ti.Matrix(self.eigenVecs)
    
    @ti.kernel
    def setEllip(self, ):
        for i, j, k in ti.ndrange(SIZE, SIZE, SIZE):
            pos = vec3(i - SIZE//2, j - SIZE//2, k - SIZE//2)
            if ((pos - self.center).transpose() @ self.eigenVecs @ (pos - self.center))[0] <= 1.:
                scene.set_voxel(ivec3(i - SIZE//2, j - SIZE//2, k - SIZE//2), 1, vec3(0.5, 0.5, 0.))


@ti.func
def dot_product(vec1, vec2):
    res = 0.
    for i in ti.static(range(vec1.n)):
        res += vec1[i] * vec2[i]
    return res


@ti.kernel
def draw_pyramidal(centerX: ti.f32, centerZ: ti.f32, size: ti.f32, theta: ti.f32):
    center = [centerX, centerZ]
    vertex = vec3(center[0], 3.**0.5 / 2 * size, center[1])
    rotate = ti.Matrix([[ti.cos(theta), 0., -ti.sin(theta)], 
                        [0., 1., 0.], 
                        [ti.sin(theta), 0., ti.cos(theta)]])
    normals = ti.Matrix([[3.**0.5 / 2., 0.5, 0.], [0., 0.5, -3.**0.5 / 2.],  # 4 normals of the triangles of pyramidal
                         [-3.**0.5 / 2., 0.5, 0.], [0., 0.5, 3.**0.5 / 2.]])
    normals = normals @ rotate

    for I in ti.grouped(ti.ndrange((vertex[0] - size//2 * 2.**0.5, vertex[0] + size//2 * 2.**0.5), 
                                   (0, vertex[1]), 
                                   (vertex[2] - size//2 * 2.**0.5, vertex[2] + size//2 * 2.**0.5))):
        flag = 1
        for l in ti.static(range(4)):
            normal = vec3(normals[l, 0], normals[l, 1], normals[l, 2])
            if dot_product(vec3(I[0], I[1], I[2]) - vertex, normal) > 0:
                flag = False
        if flag == 1:
            scene.set_voxel(ivec3(I[0], I[1], I[2]), 1, vec3(0.5, 0.5, 0.))


scene = Scene(exposure=10)
scene.set_floor(-0.05, (1.0, 1.0, 1.0))
scene.set_background_color((0.25, 0.15, 0.05))
scene.set_directional_light(vec3(0.4, 0.4, 0.4), 0.2, vec3(0.25, 0.15, 0.05))

pi = 3.141592653589793
draw_pyramidal(-64. + 32.*2.**0.5, -64. + 32.*2.**0.5, 64., pi/6.)
draw_pyramidal(20., 20., 32., pi/3.)
draw_pyramidal(64. - 13*2.**0.5, 64. - 13*2.**0.5, 26., pi/4.)

facetNormals_ = np.array((
    (1., 0., 0.), 
    (0., 1., 0.), 
    (0., 0., 1.), 
    (-1., 0., 0.), 
    (0., -1., 0.), 
    (0., 0., -1.), 
))
facetPoints_ = np.array((
    (20., 0., 0.),
    (0., 60., 0.),
    (0., 0., 20.),
    (-20., 0., 0.),
    (0., 20., 0.),
    (0., 0., -20.),
))
Polyhedron(facetNormals=facetNormals_, facetPoints=facetPoints_).setPolyhedron()

# Ellip(center=[20., 20., 0.], radiusX=40., radiusY=5., radiusZ=20., axis=[0., 1., 0.], theta=np.pi/6.).setEllip()

@ti.kernel
def xAxis():
    for i in ti.ndrange((-64, 64)):
        scene.set_voxel(ivec3(i, 10, 0), 1, vec3(0.5, 0.5, 0.))
xAxis()

scene.finish()
