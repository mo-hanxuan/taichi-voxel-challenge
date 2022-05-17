""" this version totally satisfies the rules of voxel-challenge"""
from scene import Scene
import taichi as ti
from taichi.math import *

def rotate(axis=[0., 0., 1.], angle=0.):  # get the rotation matrix by axis and angle (mo-hanxuan)
    eye = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    cosAngle, sinAngle = ti.cos(angle), ti.sin(angle)
    axisLength = sum(ele**2 for ele in axis) ** 0.5
    axis = [ele / (axisLength + 1.e-16) for ele in axis]  # normalize to length = 1
    axisAxis = ti.Matrix([[axis[i] * axis[j] for j in range(len(axis))] for i in range(len(axis))]) # "i, j -> ij"
    axisCross = ti.Matrix([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return axisAxis + (eye - axisAxis) * cosAngle + axisCross * sinAngle


@ti.data_oriented
class Polyhedron:
    def __init__(self, facetNormals, facetPoints, dtype=ti.f32, material=1, color=[0.5, 0.5, 0.]):
        self.material, self.color = material, color
        self.facetNormals, self.facetPoints = ti.Matrix(facetNormals), ti.Matrix(facetPoints)
    
    @ti.kernel
    def set_voxels(self, ):
        ### facetNormals: each line in the matrix is a vector, pointed to the outside of the facet
        for i, j, k in ti.ndrange((-64, 64), (-64, 64), (-64, 64)):
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
        self.material, self.color, self.center = material, color, vec3(center[0], center[1], center[2])
        # rotate = np_rotate_matrix(np.array(axis), theta)[:3, :3]
        rota = rotate(axis, theta)
        baseVecs = ti.Matrix([[radiusX**(-2), 0., 0.], [0., radiusY**(-2), 0.], [0., 0., radiusZ**(-2)]])
        self.quadMat = rota @ baseVecs @ rota.transpose()  # quadritic matrix for ellip
    
    @ti.kernel
    def set_voxels(self, ):
        for i, j, k in ti.ndrange((-64, 64), (-64, 64), (-64, 64)):
            pos = vec3(i, j, k)
            if ((pos - self.center).transpose() @ self.quadMat @ (pos - self.center))[0] <= 1.:
                scene.set_voxel(ivec3(i, j, k), self.material, vec3(self.color[0], self.color[1], self.color[2]))

scene = Scene(exposure=10)
scene.set_floor(-0.025, (1.0, 1.0, 1.0))
scene.set_background_color((0.25, 0.15, 0.05))
scene.set_directional_light(vec3(0.4, 0.4, 0.4), 0.2, vec3(0.25, 0.15, 0.05))

polyhedronData = [  # facet normals and facet points of polyhedron
  {"normals": [[0.75,0.5,-0.433], [-0.433,0.5,-0.75], [-0.75,0.5,0.433], [0.433,0.5,0.75], [0.,-1.,0.]],
   "points": [[-18.75,55.43,-18.75], [-18.75,55.43,-18.75], [-18.75,55.43,-18.75], [-18.75,55.43,-18.75], [0.,0.,0.]]},
  {"normals": [[0.433,0.5,-0.75], [-0.75,0.5,-0.433], [-0.433,0.5,0.75], [0.75,0.5,0.433], [0.,-1.,0.]],
   "points": [[20., 27.71, 20.], [20., 27.71, 20.], [20., 27.71, 20.], [20., 27.71, 20.], [0., 0., 0.]]},
  {"normals": [[0.612,0.5,-0.612], [-0.612,0.5,-0.612], [-0.612,0.5,0.612], [0.612,0.5,0.612], [0.,-1.,0.]],
   "points": [[45.6, 22.5, 45.6], [45.6, 22.5, 45.6], [45.6, 22.5, 45.6], [45.6, 22.5, 45.6], [0., 0., 0.]]},
  {"normals": [[ 0.75, 0.5, -0.433], [-0.433, 0.5, -0.75], [-0.75, 0.5,  0.433], [0.433, 0.5,  0.75], 
               [-0.366, -1.414, 1.37 ], [1.37, -1.414,  0.366], [0.366, -1.414, -1.37], [-1.37, -1.414, -0.366 ]],
   "points": [[-18.7, 56, -18.7], [-18.7, 56, -18.7], [-18.7, 56, -18.7], [-18.7, 56, -18.7], 
               [-18.7, 24., -18.7], [-18.7, 24., -18.7], [-18.7, 24., -18.7], [-18.7, 24., -18.7]]}]

for polyhedron in polyhedronData[:-1]:
    Polyhedron(facetNormals=polyhedron["normals"], facetPoints=polyhedron["points"]).set_voxels()
Polyhedron(facetNormals=polyhedronData[-1]["normals"],  # the snow-top of big pyramid
           facetPoints=polyhedronData[-1]["points"], color=[0.7, 0.7, 0.7]).set_voxels()    

### the Sphinx
PI = 3.141592653589793
Ellip(center=[-32.,3.,45.],radiusX=26.,radiusY=3.,radiusZ=10.,axis=[0.,1.,0.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
Ellip(center=[-32.,6.,45.],radiusX=24.,radiusY=3.,radiusZ=8.,axis=[0.,1.,0.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
Ellip(center=[-32.,9.,45.],radiusX=24.,radiusY=3.,radiusZ=8.,axis=[0.,1.,0.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
Ellip(center=[-9.,18.,45.],radiusX=6.,radiusY=6.,radiusZ=6.,
      axis=[0.,1.,0.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()  # the head of Sphinx
Ellip(center=[-9.,10.,45.],radiusX=3.,radiusY=3.,radiusZ=3.,axis=[0.,1.,0.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
### back leges
Ellip([-51.,6.,55.],radiusX=2.,radiusY=6.,radiusZ=2.,axis=[0.,0.,1.],theta=-PI/6.,color=[0.7,0.7,0.7]).set_voxels()
Ellip(center=[-48.,1.,55.],radiusX=6.,radiusY=2.,radiusZ=2.,axis=[0.,0.,1.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
Ellip([-51.,6.,35.],radiusX=2.,radiusY=6.,radiusZ=2.,axis=[0.,0.,1.],theta=-PI/6.,color=[0.7,0.7,0.7]).set_voxels()
Ellip(center=[-48.,1.,35.],radiusX=6.,radiusY=2.,radiusZ=2.,axis=[0.,0.,1.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
###frontleges
Ellip([-10.,6.,55.],radiusX=2.,radiusY=6.,radiusZ=2.,axis=[0.,0.,1.],theta=-PI/6.,color=[0.7,0.7,0.7]).set_voxels()
Ellip(center=[-7.,1.,55.],radiusX=12.,radiusY=2.,radiusZ=2.,axis=[0.,0.,1.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
Ellip([-10.,6.,35.],radiusX=2.,radiusY=6.,radiusZ=2.,axis=[0.,0.,1.],theta=-PI/6.,color=[0.7,0.7,0.7]).set_voxels()
Ellip(center=[-7.,1.,35.],radiusX=12.,radiusY=2.,radiusZ=2.,axis=[0.,0.,1.],theta=0.,color=[0.7,0.7,0.7]).set_voxels()
### hairs 
Polyhedron(facetNormals=[[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.], 
                         [0., 1., -3**0.5], [0., -1., 3**0.5], [0., 1., 3**0.5], [0., -1., -3**0.5]], 
           facetPoints=[[-10., 15., 45.], [-11., 15., 45.], [-9., 24., 45.], [-9., 6., 45.], 
                        [-9., 15., 33.], [-9., 15., 57.], [-9., 15., 57.], [-9., 15., 33.],], 
           color=[0.7, 0.7, 0.7]).set_voxels()
scene.finish()