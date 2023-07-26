# RoRD
Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching

> [Project Page](https://uditsinghparihar.github.io/RoRD/) | [Paper link](https://arxiv.org/abs/2103.08573)

> Accepted to [IROS 2021](https://www.iros2021.org/)  

<img src="assets/teaser2.jpg" alt="pipeline" width="1000" height="220" /> 

## Evaluation and Datasets

- **MMA** : Training on [PhotoTourism](https://www.cs.ubc.ca/~kmyi/imw2020/data.html) and testing on [HPatches](https://github.com/hpatches/hpatches-dataset) and proposed [Rotated HPatches](https://drive.google.com/file/d/1yrxChlq1XdH-fmTPmmiIzRlJlfO_EPW0/view?usp=sharing)
- **Pose Estimation** : Training on same PhotoTourism datasets as used for MMA and testing on proposed [DiverseView](https://drive.google.com/file/d/1yAcwLwSjJ6ammy8-7bRSCsqp8vjGoRM4/view?usp=sharing)  
- **Visual Place Recognition** : Oxford RobotCar [training sequence](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-07-14-14-49-50/) and [testing sequence](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-06-26-09-24-58/)


## Pretrained Models

Download models from [Google Drive](https://drive.google.com/file/d/1-5aLHyZ_qlHFNfRnDpXUh5egtf_XtoiA/view?usp=sharing) (73.9 MB) in the base directory.  

## Evaluating RoRD  
You can evaluate RoRD on demo images or replace it with your custom images.  
1. Dependencies can be installed in a `conda` or `virtualenv` (using `python 3.6`) by running:   
	1. `pip install -r requirements.txt`  
2. `python extractMatch.py <rgb_image1> <rgb_image2> --model_file  <path to the model file RoRD>`
3. Example:  
	`python extractMatch.py demo/rgb/rgb1_1.jpg demo/rgb/rgb1_2.jpg --model_file models/rord.pth`  
4. This should give you output like this:  

#### RoRD  
<img src="assets/rord_extract.jpg" alt="pipeline" width="600" height="220" />   

#### SIFT  
<img src="assets/sift_extract.jpg" alt="pipeline" width="600" height="220" />  



## DiverseView Dataset  

Download dataset from [Google Drive](https://drive.google.com/file/d/1yAcwLwSjJ6ammy8-7bRSCsqp8vjGoRM4/view?usp=sharing) (97.8 MB) in the base directory (only needed if you want to evaluate on DiverseView Dataset).    

## Evaluation on DiverseView Dataset  
The DiverseView Dataset is a custom dataset consisting of 4 scenes with images having high-angle camera rotations and viewpoint changes.  
1. Pose estimation on single image pair of DiverseView dataset:  
	1. `cd demo`  
	2. `python register.py --rgb1 <path to rgb image 1> --rgb2 <path to rgb image 2> --depth1 <path to depth image 1> --depth2 <path to depth image 2> --model_rord <path to the model file RoRD>`  
	3. Example:   
		`python register.py --rgb1 rgb/rgb2_1.jpg --rgb2 rgb/rgb2_2.jpg --depth1 depth/depth2_1.png --depth2 depth/depth2_2.png --model_rord ../models/rord.pth`  
	4. This should give you output like this:  

#### RoRD matches in perspective view  
<img src="assets/register_persp.jpg" alt="pipeline" width="600" height="220" />  

#### RoRD matches in orthographic view  
<img src="assets/register_ortho.jpg" alt="pipeline" width="600" height="220" />  


2. To visualize the registered point cloud, use `--viz3d command`:  
	1. `python register.py --rgb1 rgb/rgb2_1.jpg --rgb2 rgb/rgb2_2.jpg --depth1 depth/depth2_1.png --depth2 depth/depth2_2.png --model_rord ../models/rord.pth --viz3d`  

#### PointCloud registration using correspondences  
<img src="assets/register_pointcloud.jpg" alt="pipeline" width="600" height="400" />  

3. Pose estimation on a sequence of DiverseView dataset:  
	1. `cd evaluation/DiverseView/`  
	2. `python evalRT.py --dataset <path to DiverseView dataset> --sequence <sequence name> --model_rord <path to RoRD model> --output_dir <name of output dir>`  
	3. Example:  
		1. `python evalRT.py --dataset /path/to/preprocessed/ --sequence data1 --model_rord ../../models/rord.pth --output_dir out`  
	4. This would generate `out` folder containing predicted transformations and matching results in `out/vis` folder, containing images like below:
	
	#### RoRD  
	<img src="assets/rord_evalRT.jpg" alt="pipeline" width="600" height="220" />   	
	
	5. SIFT Matching:	
		1. `python evalRT.py --dataset /path/to/preprocessed/ --sequence data1 --sift --output_dir out_sift`  
	6. Matching on perspective view:  
		1. `python evalRT.py --dataset /path/to/preprocessed/ --sequence data1 --model_rord ../../models/rord.pth --output_dir out_persp --persp`  

## Training RoRD on PhotoTourism Images  
1. Training using rotation homographies with initialization from D2Net weights (Download base models as mentioned in [Pretrained Models](#pretrained-models)).  

2. Download branderburg_gate dataset that is used in the `configs/train_scenes_small.txt` from [here](https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/brandenburg_gate.tar.gz)(5.3 Gb) in `phototourism` folder.  

3. Folder stucture should be:  
	```
	phototourism/  
	___ brandenburg_gate  
	___ ___ dense  
	___ ___	___ images  
	___ ___	___ stereo  
	___ ___	___ sparse  
	```  

4. `python trainPT_ipr.py --dataset_path <path_to_phototourism_folder> --init_model models/d2net.pth  --plot`  


## TO-DO
- [ ] Provide VPR code  
- [ ] Provide combine training of RoRD + D2Net  
- [ ] Provide code for calculating error in Diverseview Dataset  


## Credits
Our base model is borrowed from [D2-Net](https://github.com/mihaidusmanu/d2-net).  


## BibTex
If you use this code in your project, please cite the following paper:

```bibtex
@inproceedings{parihar2021rord,
  title={RoRD: Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching},
  author={Parihar, Udit Singh and Gujarathi, Aniket and Mehta, Kinal and Tourani, Satyajit and Garg, Sourav and Milford, Michael and Krishna, K Madhava},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={1593--1600},
  organization={IEEE}
}
```

