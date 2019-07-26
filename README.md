# SEGAN (NNabla)

Implementation of Speech Enhancement GAN (SEGAN) by [NNabla](https://nnabla.readthedocs.io/en/latest/#)

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

  - nnabla  (over v1.0.20)
  - nnabla-ext-cuda  (over v1.0.20)
  - scipy 
  - numba  
  - joblib  
  - pyQT5  
  - pyqtgraph  (after installing pyQT5)
  - pypesq (see ["install with pip"](https://github.com/ludlows/python-pesq#install-with-pip) in offical site)

## Download & Create Database

 **[English]**

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
   
 **[Japanese]**
   
   1.   ```segan.py```, ```settings.py```, ```data.py``` をダウンロードし，同じディレクトリに保存する．
   
   2.   ディレクトリ内に以下の３つのフォルダ "data", "pkl", "params" を作成する．
   
        - 「data」フォルダ： wavデータを保存する．
        - 「pkl」フォルダ  ：pklファイル（圧縮ファイル）を保存する．
        - 「params」フォルダ  ：ネットワークモデルを保存する．
   
   3.   下記のサイトから４つのデータセット(zip)をダウンロードし，解凍する．  
   
          - [clean_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_trainset_wav.zip)
          - [noisy_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_trainset_wav.zip)
          - [clean_testset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_testset_wav.zip)
          - [noisy_testset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_testset_wav.zip)

   4.  "data"フォルダ内に解凍した４つのフォルダを保存する．

   5. すべてのwavファイルのサンプリング周波数を16kHzに変換する．  
         例えば[このサイト](https://online-audio-converter.com/ja/)が便利．
         変換前のwavファイルは削除してよい．
    
## Settings

### settings.py

```settings.py``` is a parameter list including the setting parameters for learning & predicting.
Refer to the below when you want to know how to use the spectial paramters.

 **[English]**

- `self.epoch_from` :   
   Number of starting Epoch when retraining. If `self.epoch_from` > 0, restart learing after loading pre-trained models "discriminator_param_xxxx.h5" and "generator_param_xxxx.h5". The value of  `self.epoch_from` should be corresponding to "xxxx".  
   If `self.epoch_from` = 0, retraining does not work.
   
 - `self.model_save_cycle` :  
    Cycle of Epoch for saving network model. If "1", network model is saved for every 1 epoch.
   
 **[Japanese]**
 
- `self.epoch_from` ：   
   再学習する際の開始エポック数．
   学習済みのモデル「discriminator_param_xxxx.h5」と「generator_param_xxxx.h5」を読み込み，学習を再開する．
   それらの"xxxx"に対応する値を設定しなければならない．再学習させない場合は"0"に設定する．

- `self.model_save_cycle` ：  
   ネットワークモデルを保存するエポック数の周期．"1"なら毎エポックでモデルを保存する．
   
   
### Float 16bit (Half Precision Floating Point Mode)

 **[English]**
 
If you are facing GPU Memory Stack Error, please try **Half Precision Floating Point Mode** which can downsize the calculation precision and thus reduce the memory usage. If you want to use, please run the following commands before defining the network.
```python
ctx = get_extension_context('cudnn', device_id=args.device_id, type_config='half')
nn.set_default_context(ctx)
```
In `segan.py`, this mode is enable by default.
Refer to ["nnabla-ext-cuda"](https://github.com/sony/nnabla-ext-cuda) for more information.

 **[Japanese]**
 
バッチサイズを大きくしたり，ネットワークサイズを大きくすると，GPUで Memory Stack Error が発生する場合がある(CUDAのエラーとしてコンソールに表示されるはず)．
このエラーはGPUでの計算精度を16bit(半精度浮動小数)にすることで抑えられる場合がある．半精度浮動小数で計算したい場合，nnablaのコンテキストを以下のように設定すれば良い(ソースコードではデフォルトでこれに設定している)．
```python
ctx = get_extension_context('cudnn', device_id=args.device_id, type_config='half')
nn.set_default_context(ctx)
```
これでもだめなら，CUDAで使用するワークスペースのメモリを制限する必要がある．
詳しい説明は [nnabla-ext-cudaのページ](https://github.com/sony/nnabla-ext-cuda)を参照のこと．
   
##  Run

   1. If training, enable ```train(arg)``` and comment out ```test(arg)``` in main function of ```segan.py```. If predicting, enable ```test(arg)``` and comment out ```train(arg)```.
   
```python
# Training
train(args)

# Test
#test(args)
```

   2.  Run ```segan.py```.
   
### During Training
   
 **[English]**

If you run ```train(args)``` function,  the training dataset (xxxx.pkl) is created in ```pkl``` at the beginning (for only the first time). And network model (xxxx.h5) is saved in ```params``` folder by every cycle that you set by ```self.model_save_cycle```.
 

 **[Japanese]**
 
main関数内のtrain(args)関数を実行すると，"pkl"フォルダに学習用データセット（xxxx.pkl）が生成される．また self.model_save_cycle で設定されたエポックごとに"params"フォルダにネットワークモデル（xxxx.h5）が保存される．
   
### During Predicting

 **[English]**
 
 If you run ```test(args)``` function,  the test dataset (xxxx.pkl) is created in ```pkl``` at the beginning (for only the first time). And the following wav data are generated as the results. PESQ value is also displayed.
   
   - clean.wav  :  clean speech wav file
   - noisy.wav  :  noisy speech wav file
   - output.wav  : reconstructed speech wav file

 **[Japanese]**
 
main関数内のtest(args)関数を実行すると，"pkl"フォルダに推論用データセット（xxxx.pkl）が生成される．また，処理結果として以下のwavファイルが生成される．PESQ値も表示される．
   
   - clean.wav ：原音声 wavファイル
   - noisy.wav ：雑音重畳 wavファイル
   - output.wav ：処理結果 wavファイル
   
