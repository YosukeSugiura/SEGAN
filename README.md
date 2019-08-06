# SEGAN (NNabla)

Implementation of Speech Enhancement GAN (SEGAN) by [NNabla](https://nnabla.readthedocs.io/en/latest/#)

Read me Japanese Ver.  (**日本語バージョンはこちら**) -> [Link](https://github.com/YosukeSugiura/SEGAN/blob/master/README_ja.md)

**Original Paper**  
SEGAN: Speech Enhancement Generative Adversarial Network  
https://arxiv.org/abs/1703.09452

##  Requrement

### Python

  - Python 3.6
  - CUDA 10.0 & CuDNN 7.6
    + Please choose the appropriate CUDA and CuDNN version to match your [NNabla version](https://github.com/sony/nnabla/releases) 

### Packages

Please install the following packages with pip.
(If necessary, install latest pip first.)

  - nnabla  (over v1.0.19)
  - nnabla-ext-cuda  (over v1.0.19)
  - scipy 
  - numba  
  - joblib  
  - pyQT5  
  - pyqtgraph  (after installing pyQT5)
  - pypesq (see ["install with pip"](https://github.com/ludlows/python-pesq#install-with-pip) in offical site)

## Contents

  - **segan.py**  
      This is main source code. Run this.
  
  - **data.py**  
      This is for creating Batch Data. Before runnning, please download wav dataset as seen below.
      
  - **settings.py**  
      This includes setting parameters.
      
  - **display.py**  
      This includes some functions to display results.

## Download & Create Database

   1.   Download ```segan.py```, ```settings.py```, ```data.py``` and save them into the same directory.
   
   2.  In the directory, make three folders  ```data```, ```pkl```, ```params``` .
   
        - ```data```  folder :  save wav data.
        - ```pickle``` folder  :  save pickled database "~.pkl".
        - ```params``` folder  :  save parameters including network models.

   3.   Download  the following 4 dataset, and unzip them.

          - [clean_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_trainset_wav.zip)
          - [noisy_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_trainset_wav.zip)
          - [clean_testset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_testset_wav.zip)
          - [noisy_testset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_testset_wav.zip)

   4. Move those unzipped 4 folders into ```data```  folder.

   5.  Convert the sampling frequency of all the wav data to 16kHz.
         For example, [this site](https://online-audio-converter.com/) is useful.
         After converting, you can delete the original wav data. 
    
## Settings

### settings.py

```settings.py``` is a parameter list including the setting parameters for learning & predicting.
Refer to the below when you want to know how to use the spectial paramters.

- `self.epoch_from` :   
   Number of starting Epoch when retraining. If `self.epoch_from` > 0, restart learing after loading pre-trained models "discriminator_param_xxxx.h5" and "generator_param_xxxx.h5". The value of  `self.epoch_from` should be corresponding to "xxxx".  
   If `self.epoch_from` = 0, retraining does not work.
   
 - `self.model_save_cycle` :  
    Cycle of Epoch for saving network model. If "1", network model is saved for every 1 epoch.

   
### Float 16bit (Half Precision Floating Point Mode)

If you are facing GPU Memory Stack Error, please try **Half Precision Floating Point Mode** which can downsize the calculation precision and thus reduce the memory usage. If you want to use, please run the following commands before defining the network.
```python
ctx = get_extension_context('cudnn', device_id=args.device_id, type_config='half')
nn.set_default_context(ctx)
```
In `segan.py`, this mode is enable by default.
Refer to ["nnabla-ext-cuda"](https://github.com/sony/nnabla-ext-cuda) for more information.
   
##  Run

   1. If training, set ```Train=1``` in main function of ```segan.py```. If predicting, set ```Train=0``` .
   
```python
    Train = 0
    if Train:
        # Training
        nn.set_default_context(ctx)
        train(args)
    else:
        # Test
        #nn.set_default_context(ctx)
        test(args)
        pesq_score('clean.wav','output_segan.wav')
```

   2.  Run ```segan.py```.
   
### During Training

If you run ```train(args)``` function,  the training dataset (xxxx.pkl) is created in ```pkl``` at the beginning (for only the first time). And network model (xxxx.h5) is saved in ```params``` folder by every cycle that you set by ```self.model_save_cycle```.
   
### During Predicting
 
 If you run ```test(args)``` function,  the test dataset (xxxx.pkl) is created in ```pkl``` at the beginning (for only the first time). And the following wav data are generated as the results. PESQ value is also displayed.
   
   - clean.wav  :  clean speech wav file
   - noisy.wav  :  noisy speech wav file
   - output.wav  : reconstructed speech wav file
