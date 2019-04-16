# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from six.moves import range

import os
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.initializer as I
from nnabla.ext_utils import get_extension_context

#   Figure 関連
import matplotlib.pyplot as plt

from parameters import parameters
import data as dt



# -------------------------------------------
#   Generator ( Encoder + Decoder )
#   - output estimated clean wav
# -------------------------------------------
def Generator(Noisy, z):
    """
    Building generator network
        [Arguments]
        Noisy : Noisy speech waveform (Batch, 1, 16384)
        
        Output : (Batch, 1, 16384)
    """

    ##  Sub-functions
    ## ---------------------------------
    # Convolution
    def conv(x, output_ch, karnel=(32,), pad=(15,), stride=(2,), name=None):
        return PF.convolution(x, output_ch, karnel, pad=pad, stride=stride, name=name)

    # deconvolution
    def deconv(x, output_ch, karnel=(32,), pad=(15,), stride=(2,), name=None):
        return PF.deconvolution(x, output_ch, karnel, pad=pad, stride=stride, name=name)

    # Activation Function
    def af(x):
        return PF.prelu(x)

    # Concantate input and skip-input
    def concat(x, h, axis=1):
        return F.concatenate(x, h, axis=axis)

    ##  Main Processing
    ## ---------------------------------
    with nn.parameter_scope("gen"):
        # Genc : Encoder in Generator
        enc1    = af(conv(Noisy, 16, name="enc1"))   # Input:(16384, 1) --> (16, 8192) *convolutionの結果は自動的に(フィルタ数, 出力サイズ)にreshapeされる
        enc2    = af(conv(enc1, 32, name="enc2"))    # (16, 8192) --> (32, 4096)
        enc3    = af(conv(enc2, 32, name="enc3"))    # (32, 4096) --> (32, 2048)
        enc4    = af(conv(enc3, 64, name="enc4"))    # (32, 2048) --> (64, 1024)
        enc5    = af(conv(enc4, 64, name="enc5"))    # (64, 1024) --> (64, 512)
        enc6    = af(conv(enc5, 128, name="enc6"))   # (64, 512) --> (128, 256)
        enc7    = af(conv(enc6, 128, name="enc7"))   # (128, 256) --> (128, 128)
        enc8    = af(conv(enc7, 256, name="enc8"))   # (128, 128) --> (256, 64)
        enc9    = af(conv(enc8, 256, name="enc9"))   # (256, 64) --> (256, 32)
        enc10   = af(conv(enc9, 512, name="enc10"))  # (256, 32) --> (512, 16)
        enc11   = af(conv(enc10, 1024, name="enc11"))# (512, 16) --> (1024, 8)

        # Latent Variable (concat random sequence)
        with nn.parameter_scope("latent"):
            C = F.concatenate(enc11, z, axis=1) # (1024, 8) --> (2048, 8)

		# Gdec : Decoder in Generator
        # Concatenate skip input for each layer
        dec1    = concat(af(deconv(C, 512, name="dec1")), enc10)     # (2048, 8) --> (512, 16) >> [concat](1024, 16)
        dec2    = concat(af(deconv(dec1, 256, name="dec2")), enc9)   # (1024, 16) --> (256, 32)
        dec3    = concat(af(deconv(dec2, 256, name="dec3")), enc8)   # (512, 32) --> (256, 64)
        dec4    = concat(af(deconv(dec3, 128, name="dec4")), enc7)   # (512, 128) --> (128, 256)
        dec5    = concat(af(deconv(dec4, 128, name="dec5")), enc6)   # (512, 128) --> (128, 256)
        dec6    = concat(af(deconv(dec5, 64, name="dec6")), enc5)    # (512, 256) --> (64, 512)
        dec7    = concat(af(deconv(dec6, 64, name="dec7")), enc4)    # (128, 512) --> (64, 1024)
        dec8    = concat(af(deconv(dec7, 32, name="dec8")), enc3)    # (128, 1024) --> (32, 2048)
        dec9    = concat(af(deconv(dec8, 32, name="dec9")), enc2)    # (64, 2048) --> (32, 4096)
        dec10   = concat(af(deconv(dec9, 16, name="dec10")), enc1)   # (32, 4096) --> (16, 8192)
        dec11   = deconv(dec10, 1, name="dec11")                     # (32, 8192) --> (1, 16384)

    return F.tanh(dec11)


# -------------------------------------------
#   Discriminator
# -------------------------------------------
def Discriminator(Noisy, Clean, test=False, output_hidden=False):
    """
    Building discriminator network
        Noisy : (Batch, 1, 16384)
        Clean : (Batch, 1, 16384)
        Output : (Batch, 1, 16384)
    """

    ##  Sub-functions
    ## ---------------------------------
    # Convolution + Batch Normalization
    def n_conv(x, output_ch, karnel=(31,), pad=(15,), stride=(2,), name=None):
        return PF.batch_normalization(
            PF.convolution(x, output_ch, karnel, pad=pad, stride=stride, name=name),
            batch_stat=not test,
            name=name)

    # Activation Function
    def af(x):
        return F.leaky_relu(x)

    ##  Main Processing
    ## ---------------------------------
    Input = F.concatenate(Noisy,Clean, axis=1)
    # Dis : Discriminator
    with nn.parameter_scope("dis"):
        dis1    = af(n_conv(Input, 32, name="dis1"))     # Input:(2, 16384) --> (16, 16384)
        dis2    = af(n_conv(dis1, 64, name="dis2"))      # (16, 16384) --> (32, 8192)
        dis3    = af(n_conv(dis2, 64, name="dis3"))      # (32, 8192) --> (32, 4096)
        dis4    = af(n_conv(dis3, 128, name="dis4"))     # (32, 4096) --> (64, 2048)
        dis5    = af(n_conv(dis4, 128, name="dis5"))     # (64, 2048) --> (64, 1024)
        dis6    = af(n_conv(dis5, 256, name="dis6"))     # (64, 1024) --> (128, 512)
        dis7    = af(n_conv(dis6, 256, name="dis7"))     # (128, 512) --> (128, 256)
        dis8    = af(n_conv(dis7, 512, name="dis8"))     # (128, 256) --> (256, 128)
        dis9    = af(n_conv(dis8, 512, name="dis9"))     # (256, 128) --> (256, 64)
        dis10   = af(n_conv(dis9, 1024, name="dis10"))   # (256, 64) --> (512, 32)
        dis11   = n_conv(dis10, 2048, name="dis11")      # (512, 32) --> (1024, 16)
        f       = F.sigmoid(PF.affine(dis11, 1))        # (1024, 16) --> (1,)

    return f


# -------------------------------------------
#   Loss funcion (sub functions)
# -------------------------------------------
def SquaredError_Scalor(x, val=1):
    return F.squared_error(x, F.constant(val, x.shape))

def AbsoluteError_Scalor(x, val=1):
    return F.absolute_error(x, F.constant(val, x.shape))

# -------------------------------------------
#   Loss funcion
# -------------------------------------------
def Loss_dis(dval_real, dval_fake):
    E_real = F.mean( SquaredError_Scalor(dval_real, val=1) )    # real:Disから1を出力するように
    E_fake = F.mean( SquaredError_Scalor(dval_fake, val=0) )    # fake:Disから0を出力するように
    return E_real + E_fake

def Loss_gen(wave_fake, wave_true, dval_fake, lmd=100):
    E_fake = F.mean( SquaredError_Scalor(dval_fake, val=1) )	# fake:Disから1を出力するように
    E_wave = F.mean( F.absolute_error(wave_fake, wave_true) )  	# 再構成性能の向上
    return E_fake / 2 + lmd * E_wave

# -------------------------------------------
#   Train processing
# -------------------------------------------
def train(args):

    ##  Create network
    # Variables
    noisy 		= nn.Variable([args.batch_size, 1, 16384])  # Input
    clean 		= nn.Variable([args.batch_size, 1, 16384])  # Desire
    z           = nn.Variable([args.batch_size, 1024, 8])   # Random Latent Variable
    # Generator
    genout 	    = Generator(noisy, z)                       # Predicted Clean
    genout.persistent = True                                # Not to clear at backward
    loss_gen 	= Loss_gen(genout, clean, Discriminator(noisy, genout))
    loss_ae     = F.mean(F.absolute_error(genout, clean))
    # Discriminator
    fake_dis 	= genout.get_unlinked_variable(need_grad=True)
    loss_dis    = Loss_dis(
        Discriminator(noisy, clean),        # real
        Discriminator(noisy, fake_dis))     # fake

    ##  Solver
    # RMSprop.
    solver_gen = S.RMSprop(args.learning_rate)
    solver_dis = S.RMSprop(args.learning_rate)
    # Adam
    #solver_gen = S.Adam(args.learning_rate)
    #solver_dis = S.Adam(args.learning_rate)
    # set parameter
    with nn.parameter_scope("gen"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope("dis"):
        solver_dis.set_parameters(nn.get_parameters())

    ##  Create monitor
    import nnabla.monitor as M
    monitor = M.Monitor(args.monitor_path)
    monitor_loss_gen = M.MonitorSeries("Generator loss", monitor, interval=1)
    monitor_loss_dis = M.MonitorSeries("Discriminator loss", monitor, interval=1)
    monitor_time = M.MonitorTimeElapsed("Time", monitor, interval=100)

    ##  Load data & Create batch
    clean_data, noisy_data = dt.data_loader()           # データの読み込み
    baches = dt.create_batch(clean_data, noisy_data, args.batch_size)# バッチ生成関数の初期化
    del clean_data, noisy_data

    # for plot
    ax = np.linspace(0, 1, 16384)

    ##  Pre-train
    ##----------------------------------------------------
    print('== Start Pre-Training ==')

    # Process pre-train
    if args.pretrain:

        fig = plt.figure()          # open fig object

        ##  Retrain : load from past parameter
        if args.retrain:
            # Load generator parameter
            with nn.parameter_scope("gen"):
                nn.load_parameters(os.path.join(args.model_save_path, "generator_pre_param_%06d.h5" % args.pre_epoch))

        ##  Solver
        # Adam
        solver_ae = S.Adam(args.pre_earning_rate, beta1=0.5)
        # set parameter
        with nn.parameter_scope("gen"):
            solver_ae.set_parameters(nn.get_parameters())

        #  Epoch iteration
        for i in range(args.pre_epoch):

            print('--------------------------------')
            print(' Epoch :: %d/%d' % (i + 1, args.epoch))
            print('--------------------------------')

            #  Batch iteration
            for j in range(baches.batch_num):
                print('  Pre-Train (Epoch.%d) - %d/%d' % (i + 1, j + 1, baches.batch_num))

                # Set Batch
                clean.d, noisy.d = baches.next(j)
                z.d = np.random.randn(*z.shape)

                # Update Generator by Auto-Encoder
                solver_ae.zero_grad()
                loss_ae.forward(clear_no_need_grad=True)
                loss_ae.backward(clear_buffer=True)
                solver_ae.weight_decay(args.weight_decay)
                solver_ae.update()

                # Display
                if (j + 1) % 10 == 0:
                    # Display loss value
                    genout.forward(clear_buffer=True)
                    print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    print('  Epoch #%d, %d/%d  Loss ::' % (i + 1, j + 1, baches.batch_num))
                    print('     Reconstruction Error = %.2f' % loss_ae.d)
                    print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                    # Plot wave
                    plt.cla()
                    plt.plot(ax, genout.d[0, 0, :])
                    plt.plot(ax, clean.d[0, 0, :], color='crimson')
                    plt.show(block=False)
                    plt.pause(0.001)

        ##  Save generator parameter
        with nn.parameter_scope("gen"):
            nn.save_parameters(os.path.join(args.model_save_path, "generator_pre_param_%06d.h5" % args.pre_epoch))

        ##  Save waveform to png
        plt.cla()
        plt.plot(ax, genout.d[0, 0, :])
        plt.plot(ax, clean.d[0, 0, :], color='crimson')
        plt.show(block=False)
        plt.savefig(os.path.join(args.model_save_path, "figs/plt_pre_%06d.png" % args.pre_epoch))


    ##  Train
    ##----------------------------------------------------
    print('== Start Training ==')

    # retrain = True : Load past-trained parameter
    if args.retrain:

        print(' Retrain parameter from past-trained network')

        # load generator parameter
        with nn.parameter_scope("gen"):
            nn.load_parameters(os.path.join(args.model_save_path, "generator_pre_param_%06d.h5" % args.pre_epoch)) # Pre-Train
            # nn.load_parameters(os.path.join(args.model_save_path, "generator_param_%06d.h5" % args.epoch)) # Not Pre-Train
        # load discriminator parameter
        with nn.parameter_scope("dis"):
            nn.load_parameters(os.path.join(args.model_save_path, "discriminator_param_%06d.h5" % args.epoch))

    fig = plt.figure() # open fig object

    #   Epoch iteration
    for i in range(args.epoch):

        print('--------------------------------')
        print(' Epoch :: %d/%d' % (i + 1, args.epoch))
        print('--------------------------------')

        #  Batch iteration
        for j in range(baches.batch_num):
            print('  Train (Epoch.%d) - %d/%d' % (i+1, j+1, baches.batch_num))

            # Batch setting
            clean.d, noisy.d = baches.next(j)
            z.d = np.random.randn(*z.shape)
            #z.d = np.zeros(z.shape)

            # update Generator
            solver_gen.zero_grad()
            loss_gen.forward(clear_no_need_grad=True)
            loss_gen.backward(clear_buffer=True)
            solver_gen.weight_decay(args.weight_decay)
            solver_gen.update()

            # update Discriminator
            solver_dis.zero_grad()
            loss_dis.forward(clear_no_need_grad=True)
            loss_dis.backward(clear_buffer=True)
            solver_dis.weight_decay(args.weight_decay)
            solver_dis.update()

            # Monitoring
            monitor_loss_gen.add(i + 1, loss_gen.d.copy())  # generator
            monitor_loss_dis.add(i+1, loss_dis.d.copy())    # discriminator
            monitor_time.add(j+1)                           # elapsed time

            # Display
            if (j+1) % 10 == 0:
                # Display
                loss_ae.forward(clear_buffer =True)
                genout.forward(clear_buffer =True)
                print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('  Epoch #%d, %d/%d  Loss ::' % (i + 1, j + 1, baches.batch_num))
                print('     Gen = %.2f' % loss_gen.d)
                print('     Dis = %.4f' % loss_dis.d)
                print('     Reconstruction Error = %.4f' % loss_ae.d)
                print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                # Plot
                plt.cla()                               # clear fig object
                plt.plot(ax, genout.d[0, 0, :])         # output waveform
                plt.plot(ax, clean.d[0, 0, :], color='crimson') # clean waveform
                plt.show(block=False)                   # update fig
                plt.pause(0.01)                         # pause for drawing

    ## Save parameters
    # save generator parameter
    with nn.parameter_scope("gen"):
        nn.save_parameters(os.path.join(args.model_save_path, "generator_param_%06d.h5" % args.epoch))
    # save discriminator parameter
    with nn.parameter_scope("dis"):
        nn.save_parameters(os.path.join(args.model_save_path, "discriminator_param_%06d.h5" % args.epoch))


def test(args):

    ##  Load data & Create batch
    clean_data, noisy_data = dt.data_loader()
    baches_test = dt.create_batch_test(clean_data, noisy_data, start_time=1000, stop_time=1020)
    del clean_data, noisy_data

    ##  Create network
    # Variables
    noisy_t     = nn.Variable(baches_test.noisy.shape)          # Input
    z           = nn.Variable([baches_test.noisy.shape[0], 1024, 8])  # Random Latent Variable
    # Network (Only Generator)
    output_t    = Generator(noisy_t, z)

    ##  Load parameter
    # load generator
    with nn.parameter_scope("gen"):
        nn.load_parameters(os.path.join(args.model_save_path, "generator_param_%06d.h5" % args.epoch))

    ##  Validation
    noisy_t.d = baches_test.noisy
    #z.d = np.random.randn(*z.shape)
    z.d = np.zeros(z.shape)

    output_t.forward()
    # C.forward(clear_buffer =True)

    ##  Create wav files
    clean = baches_test.clean.flatten()
    output = output_t.d.flatten()
    dt.wav_write('clean.wav', baches_test.clean.flatten(), fs=16000)
    dt.wav_write('input.wav', baches_test.noisy.flatten(), fs=16000)
    dt.wav_write('output.wav', output_t.d.flatten(), fs=16000)

    ##  Plot
    fig = plt.figure()                      # create fig object
    plt.clf()                               # clear fig object
    ax = np.linspace(0, 1, len(output))     # axis
    plt.plot(ax, output)                    # output waveform
    plt.plot(ax, clean, color='crimson')    # clean waveform
    plt.savefig(os.path.join(args.model_save_path, "figs/test_%06d.png" % args.epoch))# save fig to png
    plt.show()      # Stop

    # import csv
    # for i, bat in enumerate(C.d):
    #     with open(os.path.join(args.model_save_path, 'latent%d.csv' % i), 'w') as f:
    #         writer = csv.writer(f, lineterminator='\n')
    #         for row_ in bat:
    #             writer.writerow(row_)



if __name__ == '__main__':

    # GPU connection
    ctx = get_extension_context('cudnn', device_id=0)
    nn.set_default_context(ctx)

    # Load parameters
    args = parameters()

    # Training
    #   1. Pre-train for only generator
    #       -- if "pretrain"
    #           - if "retrain"     -> load trained-generator & restart pre-train
    #           - else             -> initialize generator & start pre-train
    #       -- else                -> nothing
    #   2. Train
    #       -- if "retrain"        -> load trianed-generator and trained-discriminator & restart train
    #       -- else                -> start train (* normal case)
    # train(args)

    # Test
    test(args)

