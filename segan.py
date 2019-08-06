#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from six.moves import range

import os
import numpy as np

#   NNabla
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.initializer as I
from nnabla.ext_utils import get_extension_context  # GPU

#   PyQT Graph
import pyqtgraph as pg

#   Others
from settings import settings
import data as dt
from display import display, figout, pesq_core

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
    def conv(x, output_ch, karnel=(32,), pad=(15,), stride=(2,), name=None, w_init=None, b_init=None):
        return PF.convolution(x, output_ch, karnel, pad=pad, stride=stride, name=name,
                              w_init=w_init, b_init=b_init)

    # deconvolution
    def deconv(x, output_ch, karnel=(32,), pad=(15,), stride=(2,), name=None):
        return PF.deconvolution(x, output_ch, karnel, pad=pad, stride=stride, name=name)

    # Activation Function
    def af(x, name=None):
        return PF.prelu(x, name=name)

    def af2(x, name=None):
        return F.tanh(x)

    # Concantate input and skip-input
    def concat(x, h, axis=1):
        return F.concatenate(x, h, axis=axis)

    ##  Main Processing
    ## ---------------------------------
    with nn.parameter_scope("gen"):
        # Genc : Encoder in Generator
        enc1    = af(conv(Noisy, 16, name="enc1"))   # Input:(16384, 1) --> (16, 8192) *convolution reshapes output to (No. of Filter, Output Size) automatically
        enc2    = af(conv(enc1, 32, name="enc2"))    # (16, 8192) --> (32, 4096)
        enc3    = af(conv(enc2, 32, name="enc3"))    # (32, 4096) --> (32, 2048)
        enc4    = af(conv(enc3, 64, name="enc4"))    # (32, 2048) --> (64, 1024)
        enc5    = af(conv(enc4, 64, name="enc5"))    # (64, 1024) --> (64, 512)
        enc6    = af(conv(enc5, 128, name="enc6"))   # (64, 512) --> (128, 256)
        enc7    = af(conv(enc6, 128, name="enc7"))   # (128, 256) --> (128, 128)
        enc8    = af(conv(enc7, 256, name="enc8"))   # (128, 128) --> (256, 64)
        enc9    = af(conv(enc8, 256, name="enc9"))   # (256, 64) --> (256, 32)
        enc10   = af(conv(enc9, 512, name="enc10"))  # (256, 32) --> (512, 16)
        enc11   = af2(conv(enc10, 1024, name="enc11",
                    w_init=I.ConstantInitializer(), b_init=I.ConstantInitializer()))# (512, 16) --> (1024, 8)

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
        dec11   = F.tanh(deconv(dec10, 1, name="dec11"))             # (32, 8192) --> (1, 16384)

    return dec11


# -------------------------------------------
#   Discriminator
# -------------------------------------------
def Discriminator(Noisy, Clean, test=False, output_hidden=False, name="dis"):
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
    with nn.parameter_scope(name):
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
        f       = PF.affine(dis11, 1)                    # (1024, 16) --> (1,)

    return f

# -------------------------------------------
#   Loss funcion
# -------------------------------------------
def Loss_dis(dval_real, dval_fake):

    def SquaredError_Scalor(x, val=1):
        return F.squared_error(x, F.constant(val, x.shape))

    E_real = F.mean( SquaredError_Scalor(dval_real, val=1) )    # real
    E_fake = F.mean( SquaredError_Scalor(dval_fake, val=0) )    # fake
    return E_real + E_fake

def Loss_gen(wave_fake, wave_true, dval_fake, lmd=100):

    def SquaredError_Scalor(x, val=1):
        return F.squared_error(x, F.constant(val, x.shape))

    E_fake = F.mean( SquaredError_Scalor(dval_fake, val=1) )	# fake
    E_wave = F.mean( F.absolute_error(wave_fake, wave_true) )  	# Reconstruction Performance
    return E_fake / 2 + lmd * E_wave

# -------------------------------------------
#   Train processing
# -------------------------------------------
def train(args):

    ##  Sub-functions
    ## ---------------------------------
    ## Save Models
    def save_models(epoch_num, cle_disout, fake_disout, losses_gen, losses_dis, losses_ae):

        # save generator parameter
        with nn.parameter_scope("gen"):
            nn.save_parameters(os.path.join(args.model_save_path, 'generator_param_{:04}.h5'.format(epoch_num + 1)))

        # save discriminator parameter
        with nn.parameter_scope("dis"):
            nn.save_parameters(os.path.join(args.model_save_path, 'discriminator_param_{:04}.h5'.format(epoch_num + 1)))

        # save results
        np.save(os.path.join(args.model_save_path, 'disout_his_{:04}.npy'.format(epoch_num + 1)), np.array([cle_disout, fake_disout]))
        np.save(os.path.join(args.model_save_path, 'losses_gen_{:04}.npy'.format(epoch_num + 1)), np.array(losses_gen))
        np.save(os.path.join(args.model_save_path, 'losses_dis_{:04}.npy'.format(epoch_num + 1)), np.array(losses_dis))
        np.save(os.path.join(args.model_save_path, 'losses_ae_{:04}.npy'.format(epoch_num + 1)), np.array(losses_ae))

    ## Load Models
    def load_models(epoch_num, gen=True, dis=True):

        # load generator parameter
        with nn.parameter_scope("gen"):
            nn.load_parameters(os.path.join(args.model_save_path, 'generator_param_{:04}.h5'.format(args.epoch_from)))

        # load discriminator parameter
        with nn.parameter_scope("dis"):
            nn.load_parameters(os.path.join(args.model_save_path, 'discriminator_param_{:04}.h5'.format(args.epoch_from)))

    ## Update parameters
    class updating:

        def __init__(self):
            self.scale = 8 if args.halfprec else 1

        def __call__(self, solver, loss):
            solver.zero_grad()                                  # initialize
            loss.forward(clear_no_need_grad=True)               # calculate forward
            loss.backward(self.scale, clear_buffer=True)      # calculate backward
            solver.scale_grad(1. / self.scale)                # scaling
            solver.weight_decay(args.weight_decay * self.scale) # decay
            solver.update()                                     # update


    ##  Inital Settings
    ## ---------------------------------

    ##  Create network
    #   Clear
    nn.clear_parameters()
    #   Variables
    noisy 		= nn.Variable([args.batch_size, 1, 16384], need_grad=False)  # Input
    clean 		= nn.Variable([args.batch_size, 1, 16384], need_grad=False)  # Desire
    z           = nn.Variable([args.batch_size, 1024, 8], need_grad=False)   # Random Latent Variable
    #   Generator
    genout = Generator(noisy, z)                       # Predicted Clean
    genout.persistent = True                # Not to clear at backward
    loss_gen 	= Loss_gen(genout, clean, Discriminator(noisy, genout))
    loss_ae     = F.mean(F.absolute_error(genout, clean))
    #   Discriminator
    fake_dis 	= genout.get_unlinked_variable(need_grad=True)
    cle_disout  = Discriminator(noisy, clean)
    fake_disout  = Discriminator(noisy, fake_dis)
    loss_dis    = Loss_dis(Discriminator(noisy, clean),Discriminator(noisy, fake_dis))

    ##  Solver
    # RMSprop.
    # solver_gen = S.RMSprop(args.learning_rate_gen)
    # solver_dis = S.RMSprop(args.learning_rate_dis)
    # Adam
    solver_gen = S.Adam(args.learning_rate_gen)
    solver_dis = S.Adam(args.learning_rate_dis)
    # set parameter
    with nn.parameter_scope("gen"):
        solver_gen.set_parameters(nn.get_parameters())
    with nn.parameter_scope("dis"):
        solver_dis.set_parameters(nn.get_parameters())

    ##  Load data & Create batch
    clean_data, noisy_data = dt.data_loader()
    batches     = dt.create_batch(clean_data, noisy_data, args.batch_size)
    del clean_data, noisy_data

    ##  Initial settings for sub-functions
    fig     = figout()
    disp    = display(args.epoch_from, args.epoch, batches.batch_num)
    upd     = updating()

    ##  Train
    ##----------------------------------------------------

    print('== Start Training ==')

    ##  Load "Pre-trained" parameters
    if args.epoch_from > 0:
        print(' Retrain parameter from pre-trained network')
        load_models(args.epoch_from, dis=False)
        losses_gen  = np.load(os.path.join(args.model_save_path, 'losses_gen_{:04}.npy'.format(args.epoch_from)))
        losses_dis  = np.load(os.path.join(args.model_save_path, 'losses_dis_{:04}.npy'.format(args.epoch_from)))
        losses_ae   = np.load(os.path.join(args.model_save_path, 'losses_ae_{:04}.npy'.format(args.epoch_from)))
    else:
        losses_gen  = []
        losses_ae   = []
        losses_dis  = []

    ## Create loss loggers
    point       = len(losses_gen)
    loss_len    = (args.epoch - args.epoch_from) * ((batches.batch_num+1)//10)
    losses_gen  = np.append(losses_gen, np.zeros(loss_len))
    losses_ae   = np.append(losses_ae, np.zeros(loss_len))
    losses_dis  = np.append(losses_dis, np.zeros(loss_len))

    ##  Training
    for i in range(args.epoch_from, args.epoch):

        print('')
        print(' =========================================================')
        print('  Epoch :: {0}/{1}'.format(i + 1, args.epoch))
        print(' =========================================================')
        print('')

        #  Batch iteration
        for j in range(batches.batch_num):
            print('  Train (Epoch. {0}) - {1}/{2}'.format(i+1, j+1, batches.batch_num))

            ##  Batch setting
            clean.d, noisy.d = batches.next(j)
            #z.d = np.random.randn(*z.shape)
            z.d = np.zeros(z.shape)

            ##  Updating
            upd(solver_gen, loss_gen)       # update Generator
            upd(solver_dis, loss_dis)       # update Discriminator

            ##  Display
            if (j+1) % 10 == 0:
                # Get result for Display
                cle_disout.forward()
                fake_disout.forward()
                loss_ae.forward(clear_no_need_grad=True)

                # Display text
                disp(i, j, loss_gen.d, loss_dis.d, loss_ae.d)

                # Data logger
                losses_gen[point] = loss_gen.d
                losses_ae[point]  = loss_ae.d
                losses_dis[point] = loss_dis.d
                point = point + 1

                # Plot
                fig.waveform(noisy.d[0,0,:], genout.d[0,0,:], clean.d[0,0,:])
                fig.loss(losses_gen[0:point-1], losses_ae[0:point-1], losses_dis[0:point-1])
                fig.histogram(cle_disout.d, fake_disout.d)
                pg.QtGui.QApplication.processEvents()


        ## Save parameters
        if ((i+1) % args.model_save_cycle) == 0:
            save_models(i, cle_disout.d, fake_disout.d, losses_gen[0:point-1], losses_dis[0:point-1], losses_ae[0:point-1])  # save model
            exporter = pg.exporters.ImageExporter(fig.win.scene())  # Call pg.QtGui.QApplication.processEvents() before exporters!!
            exporter.export(os.path.join(args.model_save_path, 'plot_{:04}.png'.format(i + 1))) # save fig

    ## Save parameters (Last)
    save_models(args.epoch-1, cle_disout.d, fake_disout.d, losses_gen, losses_dis, losses_ae)


def test(args):

    ##  Load data & Create batch
    clean_data, noisy_data = dt.data_loader(test=True, need_length=True)
    # Batch
    #  - Proccessing speech interval can be adjusted by "start_frame" and "start_frame".
    #  - "None" -> All speech in test dataset.
    baches_test = dt.create_batch_test(clean_data, noisy_data, start_frame=None, stop_frame=None)
    del clean_data, noisy_data

    ##  Create network
    # Variables
    noisy_t     = nn.Variable(baches_test.noisy.shape)          # Input
    z           = nn.Variable([baches_test.noisy.shape[0], 1024, 8])  # Random Latent Variable
    # Network (Only Generator)
    output_t = Generator(noisy_t, z)

    ##  Load parameter
    # load generator
    with nn.parameter_scope("gen"):
        print(args.epoch)
        nn.load_parameters(os.path.join(args.model_save_path, "generator_param_{:04}.h5".format(args.epoch)))

    ##  Validation
    noisy_t.d = baches_test.noisy
    #z.d = np.random.randn(*z.shape)
    z.d = np.zeros(z.shape)             # zero latent valiables

    output_t.forward()

    ##  Create wav files
    dt.wav_write('clean.wav', baches_test.clean.flatten(), fs=16000)
    dt.wav_write('input_segan.wav', baches_test.noisy.flatten(), fs=16000)
    dt.wav_write('output_segan.wav', output_t.d.flatten(), fs=16000)
    print('finish!')
    

if __name__ == '__main__':

    ## Load settings
    args = settings()

    ## GPU connection
    if args.halfprec:
        # - Float 16-bit precision mode : When GPU memory often gets stack, please use it.
        ctx = get_extension_context('cudnn', device_id=args.device_id, type_config='half')
    else:
        # - Float 32-bit precision mode :
        ctx = get_extension_context('cudnn', device_id=args.device_id)

    ## Training or Prediction
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
        # PESQ score = 2.8472938394546508  : (2019.7.18)
