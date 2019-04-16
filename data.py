#
#	Data Creator for SEGAN
#
#

from __future__ import absolute_import

import math
import wave
import array
import joblib
import glob

from numba import jit
import numpy as np
import numpy.random as rd
from scipy.signal import lfilter

from parameters import parameters

import os

# Low Pass Filter for de-emphasis
@jit
def de_emph(y, preemph=0.95):
    if preemph <= 0:
        return y
    return lfilter(1,[1, -preemph], y)

def data_loader(preemph=0.95):
    """
    Read wav files or Load pkl files including wav information
	"""

	# Parameters 読み取り
    args = parameters()

	##  Pklファイルなし → wav読み込み + Pklファイル作成
    ## -------------------------------------------------
    if not os.access(args.clean_pkl_path + '/clean.pkl', os.F_OK):

        ##  Wav ファイルの読み込み
	    # wavファイルパスの獲得
        cname = glob.glob(args.clean_wav_path + '/*.wav')
        nname = glob.glob(args.noisy_wav_path + '/*.wav')
        l = len(cname)  # ファイル数

        # Clean wav の読み込み
        i = 1
        cdata = []
        for cname_ in cname:
            cfile = wave.open(cname_, 'rb')
            cdata.append(np.frombuffer(cfile.readframes(-1), dtype='int16'))
            cfile.close()

            print(' Load Clean wav... #%d / %d' % (i, l))
            i+=1

        cdata = np.concatenate(cdata, axis=0)               # データのシリアライズ
        cdata = cdata - preemph * np.roll(cdata, 1)         # プリエンファシス
        cdata = cdata.astype(np.float32)                    # データ量圧縮(メモリに余裕があるなら消す)
        L = args.len // 2                                   # フレーム長の半分(8192サンプル)
        D = len(cdata) // L                                 # 0.5s(8192サンプル)毎に分割
        cdata = cdata[:D * L].reshape(D, L)                 # (1,:) --> (D, 8192)

        print(' Clean wav is Loaded !!')

        # Noisy wav の読み込み
        i = 1
        ndata = []
        for nname_ in nname:
            nfile = wave.open(nname_, 'rb')
            ndata.append(np.frombuffer(nfile.readframes(-1), dtype='int16'))
            nfile.close()

            print(' Load Noisy wav... #%d / %d' % (i, l))
            i += 1

        ndata = np.concatenate(ndata, axis=0)               # データのシリアライズ化
        ndata = ndata - preemph * np.roll(ndata, 1)         # プリエンファシス
        ndata = ndata.astype(np.float32)                    # データ量圧縮(メモリに余裕があるなら消す)
        L = args.len // 2                                   # フレーム長の半分(8192サンプル)
        D = len(ndata) // L                                 # 0.5s(8192サンプル)毎に分割
        ndata = ndata[:D * L].reshape(D, L)                 # (1,:) --> (D, 8192)

        print(' Now Creating Pkl file...')

        ##  Pklファイルの作成
		# クリーンpklの作成
        with open(args.clean_pkl_path + '/clean.pkl', 'wb') as f:
            joblib.dump(cdata, f, protocol=-1,compress=3)

        # ノイズpklの作成
        with open(args.noisy_pkl_path + '/noisy.pkl', 'wb') as f:
            joblib.dump(ndata, f, protocol=-1,compress=3)

        print(' Pkl file is Created !!')

	##  Pklファイルあり → ロード
    ## -------------------------------------------------
    else:
        # クリーン音声Pklのロード
        print(' Loading Clean wav...')
        with open(args.clean_pkl_path + '/clean.pkl', 'rb') as f:
            cdata = joblib.load(f)

        # ノイジィ音声Pklのロード
        print(' Loading Noisy wav...')
        with open(args.noisy_pkl_path + '/noisy.pkl', 'rb') as f:
            ndata = joblib.load(f)

    return cdata, ndata


class create_batch:
    """
    Creating Batch Data
    """

    ## 	パラメータおよびデータの保持
    def __init__(self, clean_data, noisy_data, batches):
        print(' Create batch object from waveform...')

        # 正規化
        def normalize(data):
            return (1. / 32767.) * data  # [-32768 ~ 32768] -> [-1 ~ 1]

        # データの整形
        self.clean = np.expand_dims(normalize(clean_data),axis=1)     # (D,8192,1) -> (D,1,8192)
        self.noisy = np.expand_dims(normalize(noisy_data),axis=1)     # (D,8192,1) -> (D,1,8192)

        # ランダムインデックス生成 (データ取り出し用)
        ind = np.array(range(len(clean_data)-1))
        rd.shuffle(ind)

        # パラメータの読み込み
        self.batch = batches
        self.batch_num = math.ceil(len(clean_data)/batches)         # 1エポックあたりの学習数
        self.rnd = np.r_[ind,ind[:self.batch_num*batches-len(clean_data)+1]] # 足りない分は巻き戻して利用
        self.len = len(clean_data)                                  # データ長
        self.index = 0                                              # 読み込み位置

    ## 	データの取り出し
    def next(self, i):

        # データのインデックス指定
        # 各バッチではじめの8192サンプル分のインデックス
        index = self.rnd[ i * self.batch : (i + 1) * self.batch ]

        # データ取り出し
        return np.concatenate((self.clean[index],self.clean[index+1]),axis=2), \
               np.concatenate((self.noisy[index],self.noisy[index+1]),axis=2)


class create_batch_test:
    """
    Creating Batch Data
    """

    ## 	パラメータおよびデータの保持
    def __init__(self, clean_data, noisy_data, start_time = 0, stop_time=20):
        print(' Create batch object from waveform...')

        def normalize(data):
            return (1. / 32767.) * data  # [-32768 ~ 32768] -> [-1 ~ 1]

        # データを設定
        f_len = clean_data.shape[1] * 2     # 1データ長: 8192*2 = 16384
        self.clean = np.expand_dims(normalize(clean_data[start_time:stop_time]).reshape(-1, f_len), axis=1)
        self.noisy = np.expand_dims(normalize(noisy_data[start_time:stop_time]).reshape(-1, f_len), axis=1)
        self.len = len(clean_data)


def wav_write(filename, x, fs=16000):

    # x = de_emph(x)      # De-emphasis using LPF

    x = x * 32767       # denormalized
    x = x.astype('int16')               # cast to int
    w = wave.Wave_write(filename)
    w.setparams((1,     # channel
                 2,     # byte width
                 fs,    # sampling rate
                 len(x),  # number of frames
                 'NONE',
                 'not compressed' # no compression
    ))
    w.writeframes(array.array('h', x).tobytes())
    w.close()

    return 0