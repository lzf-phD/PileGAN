# PileGAN
Intelligent Generative Structural Design Method for bridge pile foundation design based on "image-condition" Pix2PixHD

![graph abstract.png](other/graph abstract.png)

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

# PileGAN1 and PileGAN2 selection
Please modify line 39 in the file path _models/networks_.py as follows:
PileGAN1 selects to GlobalGenerator1.
PileGAN2 selects to GlobalGenerator2.

# dataset
1. The directory `datasets/piles_EB` stores the  datasets for end bearing piles.
2. The directory `datasets/piles_F` stores the datasets for friction piles.
3. For training sets:  
   - `datasets/pile_EB/train_A` and `datasets/pile_F/train_A` store the **sources**.  
   - `datasets/pile_EB/train_B` and `datasets/pile_F/train_B` store the **labels**. 
   - `datasets/pile_EB/cond/train` and `datasets/pile_F/cond/train` store the **mask maps**.
   - `datasets/pile_EB/raw_txt/train` and `datasets/pile_F/raw_txt/train` store the **parameters**.
4. For testing sets:  
   - `datasets/pile_EB/test_A` and `datasets/pile_F/test_A` store the **sources**.  
   - `datasets/pile_EB/test_B` and `datasets/pile_F/test_B` store the **labels**. 
   - `datasets/pile_EB/cond/test` and `datasets/pile_F/cond/test` store the **mask maps**.
   - `datasets/pile_EB/raw_txt/test` and `datasets/pile_F/raw_txt/test` store the **parameters**.  
5. Model prediction results (outputs) are stored in:  
   - `results` for end-bearing pile samples and friction pile samples.

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

# Evaluation
To evaluate models using either end bearing pile or friction pile results.
```bash
   python eval/eval_fusion.py
```
