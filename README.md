# RoRD
Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching

> [Project Page](https://uditsinghparihar.github.io/RoRD/) | [Paper link](https://arxiv.org/abs/2103.08573)

## Evaluation and Datasets

- **MMA** : Training on [PhotoTourism](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) and testing on [HPatches](https://github.com/hpatches/hpatches-dataset) and proposed [Rotated HPatches](add_link)
- **Pose Estimation** : Training on same PhotoTourism datasets as used for MMA and testing on proposed [DiverseView](https://drive.google.com/file/d/1BkhcHBKwcjNHgbLZ1XKurpcP7v4hFD_b/view?usp=sharing)  
- **Visual Place Recognition** : Oxford RobotCar [training sequence](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-07-14-14-49-50/) and [testing sequence](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-06-26-09-24-58/)


## Pretrained Models
<!-- - [RoRD](add_link)
- [RoRD-Combined](add_link)
- [D2-Net](https://dsmn.ml/files/d2-net/d2_tf.pth) -->

Download models from [Google Drive](https://drive.google.com/file/d/1-5aLHyZ_qlHFNfRnDpXUh5egtf_XtoiA/view?usp=sharing) (73.9 MB)  


## DiverseView Dataset  

Download dataset from [Google Drive](https://drive.google.com/file/d/1BkhcHBKwcjNHgbLZ1XKurpcP7v4hFD_b/view?usp=sharing) (97.8 MB)  

## Evaluation on DiverseView Dataset  
The DiverseView Dataset is a custom dataset consisting of 4 scenes with images having high-angle camera rotations and viewpoint changes.
- RT estimation (single image pair):  
`cd evaluation/demo/`  
`python MatchIcp.py --rgb1 <path to rgb image 1> --rgb2 <path to rgb image 2> --depth1 <path to depth image 1> --depth2 <path to depth image 2> --model_rord <path to the model file RoRD>`  
Example   
`python MatchIcp.py --rgb1 rgb/rgb2_1.jpg --rgb2 rgb/rgb2_2.jpg --depth1 depth/depth2_1.png --depth2 depth/depth2_2.png --model_rord <path to the model file RoRD>`
- RT estimation (on query-database pairs)  
`cd evaluation/DiverseViewDataset/RT_estimation/`  
`python extractMatchICP.py --rgb_csv <csv file containing query-database rgb image pairs> --depth_csv <csv file containing query-database depth image pairs> --output_dir <path to the output directory> --camera_file <path to the camera intrinsics txt file> --model_rord <path to the model file RoRD>`

## Credits
Our base model is borrowed from [D2-Net](https://github.com/mihaidusmanu/d2-net).  

## BibTex
If you use this code in your project, please cite the following paper:

```bibtex

@misc{rord2021,
      title={RoRD: Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching}, 
      author={Udit Singh Parihar and Aniket Gujarathi and Kinal Mehta and Satyajit Tourani and Sourav Garg and Michael Milford and K. Madhava Krishna},
      year={2021},
      eprint={2103.08573},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

