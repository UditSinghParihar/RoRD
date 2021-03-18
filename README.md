# RoRD
Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching

> [Project Page](https://uditsinghparihar.github.io/RoRD/)
> [Paper link](https://arxiv.org/abs/2103.08573)

## Dataset Links
- [PhotoTourism](https://www.cs.ubc.ca/~kmyi/imw2020/data.html): Used for training
- [Oxford RobotCar](add_link): Used for training and VPR evaluation
- [Hpatches](https://github.com/hpatches/hpatches-dataset): MMA evaluation
- [Rotated Hpatches](add_link): MMA evaluation
- [DiverseView](add_link): R/t evaluation on high viewpoint changes

## Pretrained Models
- [RoRD](add_link)
- [RoRD-Combined](add_link)
- [D2-Net](https://dsmn.ml/files/d2-net/d2_tf.pth)

## Credits
Our base model is borrowed from [D2-Net](https://github.com/mihaidusmanu/d2-net).  

## BibTex
If you use this code in your project, please cite the following paper:

```bibtex

@misc{parihar2021rord,
      title={RoRD: Rotation-Robust Descriptors and Orthographic Views for Local Feature Matching}, 
      author={Udit Singh Parihar and Aniket Gujarathi and Kinal Mehta and Satyajit Tourani and Sourav Garg and Michael Milford and K. Madhava Krishna},
      year={2021},
      eprint={2103.08573},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

<!-- ```
@InProceedings{Dusmanu2019CVPR,
    author = {Dusmanu, Mihai and Rocco, Ignacio and Pajdla, Tomas and Pollefeys, Marc and Sivic, Josef and Torii, Akihiko and Sattler, Torsten},
    title = {{D2-Net: A Trainable CNN for Joint Detection and Description of Local Features}},
    booktitle = {Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2019},
}
``` -->
