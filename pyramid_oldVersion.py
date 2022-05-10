"""
    creat 3 pyramids, with color style of Wasteland literature, 
    where the piramid is created by carving through 4 planes, and orientation of the pyramid can be roteted. 
"""

from scene import Scene
import taichi as ti
from taichi.math import *
import numpy as np


@ti.func
def dot_product(vec1, vec2):
    res = 0.
    for i in ti.static(range(vec1.n)):
        res += vec1[i] * vec2[i]
    return res


def printNormals(centerX: ti.f32, centerZ: ti.f32, size: ti.f32, theta: ti.f32):
    center = [centerX, centerZ]
    vertex = vec3(center[0], 3.**0.5 / 2 * size, center[1])
    print("vertex =", vertex)
    normals = np.array([[3.**0.5 / 2., 0.5, 0.], [0., 0.5, -3.**0.5 / 2.],  # 4 normals of the triangles of pyramidal
                        [-3.**0.5 / 2., 0.5, 0.], [0., 0.5, 3.**0.5 / 2.]])
    rotate = np.array([[ti.cos(theta), 0., -ti.sin(theta)], 
                        [0., 1., 0.], 
                        [ti.sin(theta), 0., ti.cos(theta)]])
    print("normals @ rotate =", normals @ rotate)


@ti.kernel
def draw_pyramidal(centerX: ti.f32, centerZ: ti.f32, size: ti.f32, theta: ti.f32):
    center = [centerX, centerZ]
    vertex = vec3(center[0], 3.**0.5 / 2 * size, center[1])
    print("vertex =", vertex)
    rotate = ti.Matrix([[ti.cos(theta), 0., -ti.sin(theta)], 
                        [0., 1., 0.], 
                        [ti.sin(theta), 0., ti.cos(theta)]])
    normals = ti.Matrix([[3.**0.5 / 2., 0.5, 0.], [0., 0.5, -3.**0.5 / 2.],  # 4 normals of the triangles of pyramidal
                         [-3.**0.5 / 2., 0.5, 0.], [0., 0.5, 3.**0.5 / 2.]])
    print("normals =", normals)
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

printNormals(-64. + 32.*2.**0.5, -64. + 32.*2.**0.5, 64., pi/6.)
printNormals(20., 20., 32., pi/3.)
printNormals(64. - 13*2.**0.5, 64. - 13*2.**0.5, 26., pi/4.)

scene.finish()
