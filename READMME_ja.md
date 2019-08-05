# SEGAN (NNabla)

Speech Enhancement GAN (SEGAN)の実装.  
英語の Read Me は[こちら](https://github.com/YosukeSugiura/SEGAN)．

**原論文**  
SEGAN: Speech Enhancement Generative Adversarial Network  
https://arxiv.org/abs/1703.09452

##  Requrement

### Python

  - Python 3.6
  - CUDA 10.0 & CuDNN 7.6
    + [NNablaのバージョン](https://github.com/sony/nnabla/releases) に適合するものを選択してください．

### Packages

以下のパッケージを```pip```でインストールすること．  
コマンドプロンプトで以下のように入力すればよい．
```
 pip install (package名) --update
```

  - nnabla  (v1.0.20以上)
  - nnabla-ext-cuda  (v1.0.20以上)
  - scipy 
  - numba  
  - joblib  
  - pyQT5  
  - pyqtgraph  (pyQT5を先にインストール)
  - pypesq (["install with pip"](https://github.com/ludlows/python-pesq#install-with-pip)を参照のこと)

## Download & Create Database
   
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

```settings.py```は設定パラメータ群が収められている．
ここでは，特殊なパラメータをピックアップして説明する．
 
- `self.epoch_from` ：   
   再学習する際の開始エポック数．
   学習済みのモデル「discriminator_param_xxxx.h5」と「generator_param_xxxx.h5」を読み込み，学習を再開する．
   それらの"xxxx"に対応する値を設定しなければならない．再学習させない場合は"0"に設定する．

- `self.model_save_cycle` ：  
   ネットワークモデルを保存するエポック数の周期．"1"なら毎エポックでモデルを保存する．
   
   
### 半精度少数点16bitモード
 
バッチサイズを大きくしたり，ネットワークサイズを大きくすると，GPUで Memory Stack Error が発生する場合がある(CUDAのエラーとしてコンソールに表示されるはず)．
このエラーはGPUでの計算精度を16bit(半精度浮動小数)にすることで抑えられる場合がある．半精度浮動小数で計算したい場合，nnablaのコンテキストを以下のように設定すれば良い(ソースコードではデフォルトでこれに設定している)．
```python
ctx = get_extension_context('cudnn', device_id=args.device_id, type_config='half')
nn.set_default_context(ctx)
```
これでもだめなら，CUDAで使用するワークスペースのメモリを制限する必要がある．
詳しい説明は [nnabla-ext-cudaのページ](https://github.com/sony/nnabla-ext-cuda)を参照のこと．
   
##  実行

   1. 学習の場合, ```Train```の値を0にする．推論の場合，```Train```の値を1にする．
   
```python
# Training
train(args)

# Test
#test(args)
```

   2.  Run ```segan.py```.
   
### 学習時
 
main関数内のtrain(args)関数を実行すると，"pkl"フォルダに学習用データセット（xxxx.pkl）が生成される．また self.model_save_cycle で設定されたエポックごとに"params"フォルダにネットワークモデル（xxxx.h5）が保存される．
   
### 推論時

main関数内のtest(args)関数を実行すると，"pkl"フォルダに推論用データセット（xxxx.pkl）が生成される．また，処理結果として以下のwavファイルが生成される．PESQ値も表示される．
   
   - clean.wav ：原音声 wavファイル
   - noisy.wav ：雑音重畳 wavファイル
   - output.wav ：処理結果 wavファイル
   
