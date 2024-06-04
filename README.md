# Simple Viser Viewer for 2D Gaussian Splatting for Geometrically Accurate Radiance Fields

[2D GS Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Original Github](https://github.com/hbb1/2d-gaussian-splatting) <br>

This repo contains the *unofficial* viewer for the "2D Gaussian Splatting for Geometrically Accurate Radiance Fields". 
A significant portion of this project is built upon several existing works to provide enhanced visualization and editing capabilities for 2D Gaussian Splatting data.

| Preview | Render |
| --- | --- |
|![visualization](assets/viser_capture.gif)|![visualization](assets/viser_general_opt.gif) | 

|  Edit  | Transform |
| --- | --- |
| ![visualization](assets/viser_edit_opt.gif) | ![visualization](assets/viser_transform_opt.gif) |


## ⭐New Features  
- 2024/06/03
    - General Features
        - Supports various render types including ***Normal / Depth / Depth2Normal***
        - Direct comparison between different render types, *e.g.,* normal vs depth-to-normal
        - Crop Box Region
        - Pointcloud visualization
    - Editing Features
        - Edit, delete and save point clouds (Recommended with: 'Add pointcloud')
    - Transform Features
        - Rigid Transform 
- 2024/05/31
    - code release

## Installation

- Follow the original installation instructions from 2D GS, then add all files from this project to the original project directory. 
(*i.e.,* 'viewer.py' is in '~/2d-gaussian-splatting/viewer.py')
- Tested on Ubuntu 20.04.

```bash
pip install viser==0.1.32
pip install splines  
pip install lightning
```
### ⭐IMPORTANT 

- Also, you have to fix line 135 of '_message_api.py' (or '_scene_api.py') in viser (I don't know why the bug occurs)

- Original
```python
    def cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
        if not isinstance(vector, tuple):
            assert cast(onp.ndarray, vector).shape == (
                length,
        ), f"Expected vector of shape {(length,)}, but got {vector.shape} instead"
    return cast(TVector, tuple(map(float, vector)))
```

- Fix it as follows 
```python 
    def cast_vector(vector: TVector | onp.ndarray, length: int) -> TVector:
        if isinstance(vector, tuple): return cast(TVector, vector)
        else:
            if vector.__class__.__name__ == 'RollPitchYaw':
                x = vector.roll 
                y = vector.pitch 
                z = vector.yaw
            return cast(TVector, (x, y, z))
        else:
            vector = tuple(vector)
            return cast(TVector, vector)

```





## Usage
- common use
```bash
python viewer.py <path to pre-trained model> <or direct path to the ply file>
```
- for transform the scene 
```bash
python viewer.py <path to pre-trained model> <or direct path to the ply file> --enable_transform
```

### Control 
- **'q/e'** for up & down
- **'w/a/s/d'** for moving
- Mouse wheel for zoom in/out


## Acknowledgements
This project is built upon the following works
- [Original 2D GS Github](https://github.com/hbb1/2d-gaussian-splatting)
- [Viser](https://github.com/nerfstudio-project/viser)
- [Gaussian Splatting Pytorch Lightning](https://github.com/yzslab/gaussian-splatting-lightning).

Realted Blog Post: [Review of 2D Gaussian Splatting (Korean)](https://velog.io/@gjghks950/Review-2D-Gaussian-Splatting-for-Geometrically-Accurate-Radiance-Fields-Viewer-%EA%B5%AC%ED%98%84-%EC%86%8C%EA%B0%9C)
