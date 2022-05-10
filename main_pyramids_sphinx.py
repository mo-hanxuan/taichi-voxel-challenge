from scene import Scene
import taichi as ti
from taichi.math import *
import numpy as np
from math_utils import np_rotate_matrix

@ti.data_oriented
class Polyhedron:
    def __init__(self, facetNormals, facetPoints, dtype=ti.f32, material=1, color=[0.5, 0.5, 0.]):
        self.material, self.color = material, color
        self.facetNormals, self.facetPoints = ti.Matrix(facetNormals), ti.Matrix(facetPoints)
        vertexes = []  # deduce all vertexes and get xyz ranges correspondingly
        for facet0, facet1, facet2 in ti.ndrange(len(facetNormals), len(facetNormals), len(facetNormals)):
            if facet0 != facet1 and facet0 != facet2 and facet1 != facet2:
                mat = np.array([facetNormals[facet0], facetNormals[facet1], facetNormals[facet2]])
                if abs(np.linalg.det(mat)) > 1.e-6:  # not singular matrix
                    rhsVec = np.array([facetNormals[facet0] @ facetPoints[facet0],
                                       facetNormals[facet1] @ facetPoints[facet1],
                                       facetNormals[facet2] @ facetPoints[facet2],])
                    vertex = np.linalg.inv(mat) @ rhsVec
                    inside = True  # test if the vertex is inside the polyhedron
                    for i in range(len(facetNormals)):
                        if (vertex - facetPoints[i]) @ facetNormals[i] > 1.e-6:
                            inside = False
                    if inside:
                        vertexes.append(vertex)
        vertexes = np.array(vertexes)
        self.xMin, self.xMax, self.yMin, self.yMax, self.zMin, self.zMax = \
            min(vertexes[:, 0]), max(vertexes[:, 0]), min(vertexes[:, 1]), max(vertexes[:, 1]), min(vertexes[:, 2]), max(vertexes[:, 2])
    
    @ti.kernel
    def set_voxels(self, ):
        ### facetNormals: each line in the matrix is a vector, pointed to the outside of the facet
        for i, j, k in ti.ndrange((self.xMin - 1, self.xMax + 1), (self.yMin - 1, self.yMax + 1), (self.zMin - 1, self.zMax + 1)):
            pos = vec3(i, j, k)
            inside = True
            for ifacet in ti.static(range(self.facetNormals.n)):
                normal = vec3(self.facetNormals[ifacet, 0], self.facetNormals[ifacet, 1], self.facetNormals[ifacet, 2])
                point = vec3(self.facetPoints[ifacet, 0], self.facetPoints[ifacet, 1], self.facetPoints[ifacet, 2])
                if dot(pos - point, normal) > 0:
                    inside = False
            if inside:
                scene.set_voxel(ivec3(i, j, k), self.material, vec3(self.color[0], self.color[1], self.color[2]))

@ti.data_oriented
class Ellip:
    def __init__(self, center, radiusX, radiusY, radiusZ, axis, theta, material=1, color=[0.5, 0.5, 0.]):
        self.material, self.color = material, color
        self.center = vec3(center[0], center[1], center[2])
        rotate = np_rotate_matrix(np.array(axis), theta)[:3, :3]
        baseVecs = np.array([radiusX, radiusY, radiusZ])
        self.xMin, self.xMax, self.yMin, self.yMax, self.zMin, self.zMax = \
            center[0] - max(baseVecs), center[0] + max(baseVecs), center[1] - max(baseVecs), center[1] + max(baseVecs), center[2] - max(baseVecs), center[2] + max(baseVecs)
        self.quadMat = ti.Matrix(rotate @ np.diag(baseVecs**(-2)) @ rotate.transpose())  # quadritic matrix for ellip
    
    @ti.kernel
    def set_voxels(self, ):
        for i, j, k in ti.ndrange((self.xMin - 1, self.xMax + 1), (self.yMin - 1, self.yMax + 1), (self.zMin - 1, self.zMax + 1)):
            pos = vec3(i, j, k)
            if ((pos - self.center).transpose() @ self.quadMat @ (pos - self.center))[0] <= 1.:
                scene.set_voxel(ivec3(i, j, k), self.material, vec3(self.color[0], self.color[1], self.color[2]))

scene = Scene(exposure=10)
scene.set_floor(-0.025, (1.0, 1.0, 1.0))
scene.set_background_color((0.25, 0.15, 0.05))
scene.set_directional_light(vec3(0.4, 0.4, 0.4), 0.2, vec3(0.25, 0.15, 0.05))

Polyhedron(facetNormals=np.array([[ 0.75, 0.5, -0.4330127], [-0.4330127, 0.5, -0.75], 
                                  [-0.75, 0.5,  0.4330127], [ 0.4330127, 0.5,  0.75], [0., -1., 0.]]), 
           facetPoints=np.array([[-18.745166, 55.42562584, -18.745166], [-18.745166, 55.42562584, -18.745166], 
                                 [-18.745166, 55.42562584, -18.745166], [-18.745166, 55.42562584, -18.745166], [0., 0., 0.]])).set_voxels()
Polyhedron(facetNormals=np.array([[ 0.4330127, 0.5, -0.75], [-0.75, 0.5, -0.4330127], 
                                  [-0.4330127, 0.5,  0.75], [ 0.75, 0.5,  0.4330127], [0., -1., 0.]]), 
           facetPoints=np.array([[20., 27.71281292, 20.], [20., 27.71281292, 20.], 
                                 [20., 27.71281292, 20.], [20., 27.71281292, 20.], [0., 0., 0.]])).set_voxels()
Polyhedron(facetNormals=np.array([[ 0.61237244, 0.5, -0.61237244], [-0.61237244, 0.5, -0.61237244], 
                                  [-0.61237244, 0.5, 0.61237244], [ 0.61237244, 0.5, 0.61237244], [0., -1., 0.]]), 
           facetPoints=np.array([[45.61522369, 22.5166605, 45.61522369], [45.61522369, 22.5166605, 45.61522369], 
                                 [45.61522369, 22.5166605, 45.61522369], [45.61522369, 22.5166605, 45.61522369], [0., 0., 0.]]), ).set_voxels()
Polyhedron(facetNormals=np.array([[ 0.75, 0.5, -0.4330127], [-0.4330127, 0.5, -0.75], 
                                  [-0.75, 0.5,  0.4330127], [ 0.4330127, 0.5,  0.75], 
                                  [-0.3660254 , -1.41421356,  1.3660254 ], [ 1.3660254 , -1.41421356,  0.3660254 ],
                                  [ 0.3660254 , -1.41421356, -1.3660254 ], [-1.3660254 , -1.41421356, -0.3660254 ]]), 
           facetPoints=np.array([[-18.745166, 56, -18.745166], [-18.745166, 56, -18.745166], 
                                 [-18.745166, 56, -18.745166], [-18.745166, 56, -18.745166], 
                                 [-18.745166, 24., -18.745166], [-18.745166, 24., -18.745166], 
                                 [-18.745166, 24., -18.745166], [-18.745166, 24., -18.745166]]), color=[0.7, 0.7, 0.7]).set_voxels()
### the Sphinx
Ellip(center=[-32., 3., 45.], radiusX=26., radiusY=3., radiusZ=10., axis=[0., 1., 0.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-32., 6., 45.], radiusX=24., radiusY=3., radiusZ=8., axis=[0., 1., 0.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-32., 9., 45.], radiusX=24., radiusY=3., radiusZ=8., axis=[0., 1., 0.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-9., 18., 45.], radiusX=6., radiusY=6., radiusZ=6., axis=[0., 1., 0.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()  # head
Ellip(center=[-9., 10., 45.], radiusX=3., radiusY=3., radiusZ=3., axis=[0., 1., 0.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
### back leges
Ellip(center=[-51., 6., 55.], radiusX=2., radiusY=6., radiusZ=2., axis=[0., 0., 1.], theta=-np.pi/6., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-48., 1., 55.], radiusX=6., radiusY=2., radiusZ=2., axis=[0., 0., 1.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-51., 6., 35.], radiusX=2., radiusY=6., radiusZ=2., axis=[0., 0., 1.], theta=-np.pi/6., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-48., 1., 35.], radiusX=6., radiusY=2., radiusZ=2., axis=[0., 0., 1.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
### front leges
Ellip(center=[-10., 6., 55.], radiusX=2., radiusY=6., radiusZ=2., axis=[0., 0., 1.], theta=-np.pi/6., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-7., 1., 55.], radiusX=12., radiusY=2., radiusZ=2., axis=[0., 0., 1.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-10., 6., 35.], radiusX=2., radiusY=6., radiusZ=2., axis=[0., 0., 1.], theta=-np.pi/6., color=[0.7, 0.7, 0.7]).set_voxels()
Ellip(center=[-7., 1., 35.], radiusX=12., radiusY=2., radiusZ=2., axis=[0., 0., 1.], theta=0., color=[0.7, 0.7, 0.7]).set_voxels()
### hairs 
Polyhedron(facetNormals=np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], 
                                  [0., 1., -3**0.5], [0., -1., 3**0.5], [0., 1., 3**0.5], [0., -1., -3**0.5]]), 
           facetPoints=np.array([[-10., 15., 45.], [-11., 15., 45.], [-9., 24., 45.], [-9., 6., 45.], 
                                 [-9., 15., 33.], [-9., 15., 57.], [-9., 15., 57.], [-9., 15., 33.],]), color=[0.7, 0.7, 0.7]).set_voxels()
scene.finish()