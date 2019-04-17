# SEGAN (NNabla)

ソースコードの使い方

##  動作保証環境

### Python 関連

  - Python 3.6 (3.5以上なら動くはず)
  - CUDA 9.2 ([NNablaのversion](https://github.com/sony/nnabla/releases)に適合するもの)
  - CuDNN 7.3

### Python パッケージ

以下のパッケージをpipでインストールしてください．  
（pip を最新版にする必要があるかもしれません．）

  - nnabla v1.0.12  (v1.0.12以上)
  - nnabla-ext-cuda v1.0.12 (v1.0.12以上)
  - scipy 
  - numba
  - joblib


## ダウンロードと準備

   1.   "segan.py", "settings.py", "data.py" をダウンロードし，同じディレクトリに保存する．
   
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
    
## 設定

"settings.py"ファイルには学習・推論の設定パラメータが保存されている．  
ここでは特殊なパラメータをピックアップして説明する．

- self.epoch_from ：   
   再学習する際の開始エポック数．
   学習済みのモデル「discriminator_param_xxxx.h5」と「generator_param_xxxx.h5」を読み込み，学習を再開する．
   それらの"xxxx"に対応する値を設定しなければならない．再学習させない場合は"0"に設定する．

- self.model_save_cycle ：  
   ネットワークモデルを保存するエポック数の周期．"1"なら毎エポックでモデルを保存する．
   
   
##  実行

   1.   "segan.py"内のメイン関数部(一番下)から，「train(args)」(学習)と「test(args)」(推論)のどちらかをコメントアウトする．
   
```
# Training
train(args)

# Test
#test(args)
```

   2.  "segan.py"を実行する．
   
### 学習時の動作
   
train(args)関数を実行すると，"pkl"フォルダに学習用データセット（xxxx.pkl）が生成される．  
また self.model_save_cycle で設定されたエポックごとに"params"フォルダにネットワークモデル（xxxx.h5）が保存される．
   
### 推論時の動作

train(args)関数を実行すると，"pkl"フォルダに推論用データセット（xxxx.pkl）が生成される．    
また，処理結果として以下のwavファイルが生成される．
   
   - clean.wav ：原音声 wavファイル
   - noisy.wav ：雑音重畳 wavファイル
   - output.wav ：処理結果 wavファイル
   
