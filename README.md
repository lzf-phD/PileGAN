# PileGAN
Intelligent Generative Structural Design Method for bridge single-pile design based on "image-condition" Pix2PixHD

![graph abstract.png](other/graph%20abstract.png)

# Lib Name-Version
python                    3.6.13  
cudatoolkit               11.0.221  
pytorch                   1.7.1
torchaudio                0.7.2  
torchvision               0.8.2  
nltk                      3.6.7  
numpy                     1.19.2  
opencv-python             4.5.5.64  
scikit-image              0.17.2  
scipy                     1.2.0  
tensorboard               1.15.0  
tensorflow                1.15.0  
tensorflow-estimator      1.15.1  

# Acknowledgments
This code borrows heavily from [pix2pixHD](https://github.com/NVIDIA/pix2pixHD).

# PileGAN1 and PileGAN2 selection
Please modify line 39 in the file path _models/networks_.py as follows:
PileGAN1 selects to GlobalGenerator1.
PileGAN2 selects to GlobalGenerator2.

# dataset
1. The directory datasets/piles_EB stores the partial datasets for end bearing pile samples.
2. The directory datasets/piles_F stores the partial datasets for friction pile samples.

# Training
To train models using either end bearing pile or friction pile datasets, modify opt.dataroot in the _train_ file.
```bash
   python train.py
```

# Inference
To test models using either end bearing pile or friction pile datasets, modify opt.dataroot in the _test_ file.
The trained model weight files (.pth) are stored in the checkpoint directory.
```bash
   python test.py
```
