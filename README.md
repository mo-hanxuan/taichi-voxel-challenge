# <a name="title">Pyramids & Sphinx (Taichi Voxel Challenge)</a>

<img src="README.assets/image-20220510163400401.png" alt="image-20220510163400401" style="zoom:70%;" />

<img src="README.assets/image-20220510163430279.png" alt="image-20220510163430279" style="zoom:33%;" /><img src="README.assets/image-20220510163553191.png" alt="image-20220510163553191" style="zoom:50%;" />

> Figure: result of `main.py`. 

We invite you to create your voxel artwork, by putting your [Taichi](https://github.com/taichi-dev/taichi) code in `main.py`!
Refer to [taichi voxel challenge](https://github.com/taichi-dev/community/blob/main/events/voxel-challenge/reference-zh_cn.md) for more info.
<br>

## how to create the pyramids and the Sphinx
We use **polyhedrons** and **ellipses** as basic building blocks. 
> + **polyhedron**: defined by several planes, each plane defined by plane normal (pointed to the outside of polyhedron) and a point on this plane
> + **ellipse**: defined by center, three axises (with lengths) and orientation (where the orientation is defined by rotation axis and rotation angle of counter-clockwise)

## Installation

Make sure your `pip` is up-to-date:

```bash
pip3 install pip --upgrade
```

Assume you have a Python 3 environment, simply run:

```bash
pip3 install -r requirements.txt
```

to install the dependencies of the voxel renderer.

## Quickstart

```sh
python3 main_pyramids_sphinx.py
```

Mouse and keyboard interface:

+ Drag with your left mouse button to rotate the camera.
+ Press `W/A/S/D/Q/E` to move the camera.
+ Press `P` to save a screenshot.

## More examples

<a href="https://github.com/raybobo/taichi-voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/city.jpg" width="45%"></img></a>  <a href="https://github.com/victoriacity/voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/city2.jpg" width="45%"></img></a> 
<a href="https://github.com/yuanming-hu/voxel-art"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/tree2.jpg" width="45%"></img></a> <a href="https://github.com/neozhaoliang/voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/desktop.jpg" width="45%"></img></a> 
<a href="https://github.com/maajor/maajor-voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/earring_girl.jpg" width="45%"></img></a>  <a href="https://github.com/rexwangcc/taichi-voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/pika.jpg" width="45%"></img></a> 
<a href="https://github.com/houkensjtu/qbao_voxel_art"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/yinyang.jpg" width="45%"></img></a>  <a href="https://github.com/ltt1598/voxel-challenge"><img src="https://github.com/taichi-dev/public_files/blob/master/voxel-challenge/lang.jpg" width="45%"></img></a> 
