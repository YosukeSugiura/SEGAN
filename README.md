# SEGAN (NNabla)

ソースコードの使い方

## ダウンロードと設定

   1.   "segan.py", "settings.py", "data.py" をダウンロードし，同じディレクトリに保存する．
   
   2.   ディレクトリ内に"data", "pkl", "params" フォルダを作成する．
   
   3.   下記のサイトから４つのデータセット(zip)をダウンロードし，解凍する．  
   
          - [clean_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_trainset_wav.zip)
          - [noisy_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_trainset_wav.zip)
          - [clean_trainset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_testset_wav.zip)
          - [noisy_testset_wav.zip](http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_testset_wav.zip)
          
    4. "data"フォルダ内に解凍したフォルダを保存する．
         
    5. すべてのwavファイルのサンプリング周波数を16kHzに変換する．  
       例えば 以下のサイトなどで．
       http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_testset_wav.zip
              
    6. "segan.py"を実行する．
    
##


