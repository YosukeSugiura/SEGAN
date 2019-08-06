#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np

from scipy.io import wavfile        # Sound
from pypesq import pypesq           # PESQ : install from https://github.com/ludlows/python-pesq

#   PyQTGraph
import pyqtgraph as pg
import pyqtgraph.exporters as pgex

# -------------------------------------------
#   PESQ
# -------------------------------------------
## Display PESQ
def pesq_score(clean_wav, reconst_wav, split_num=100, band='nb'):

    rate, ref = wavfile.read(clean_wav)
    rate, deg = wavfile.read(reconst_wav)

    ref_s     = np.array_split(ref, split_num)
    deg_s     = np.array_split(deg, split_num)

    scores    = []

    print('PESQ Calculation...')
    for i in range(split_num):
        if i%10 == 0:
            print('  No. {0}/{1}...'.format(i, split_num))
        scores.append(pypesq(rate, ref_s[i], deg_s[i], band))


    score = np.average(np.array(scores))


    print('  ---------------------------------------------------')
    print('  PESQ score = {0}'.format(score))
    print('  ---------------------------------------------------')

    return 0


# -------------------------------------------
#   Sub Classes for Display
# -------------------------------------------
## Display progress in console
class display:

    # Remaining Time Estimation
    class time_estimation:

        def __init__(self, epoch_from, epoch, batch_num):
            self.start = time.time()
            self.epoch = epoch
            self.epoch_from = epoch_from
            self.batch = batch_num
            self.all = batch_num * (epoch - epoch_from)

        def __call__(self, epoch_num, batch_num):
            elapse = time.time() - self.start
            amount = (batch_num + 1) + (epoch_num - self.epoch_from) * self.batch
            remain = elapse / amount * (self.all - amount)

            hours, mins = divmod(int(elapse), 3600)
            mins, sec = divmod(mins, 60)
            hours_e, mins_e = divmod(int(remain), 3600)
            mins_e, sec_e = divmod(mins_e, 60)

            elapse_time = [int(hours), int(mins), int(sec)]
            remain_time = [int(hours_e), int(mins_e), int(sec_e)]

            return elapse_time, remain_time

    def __init__(self, epoch_from, epoch, batch_num):

        self.tm = self.time_estimation(epoch_from, epoch, batch_num)
        self.batch = batch_num

    def __call__(self, epoch, trial, loss_gen, loss_dis, loss_ae):

        elapse_time, remain_time = self.tm(epoch, trial)
        print('  ---------------------------------------------------')
        print('  [ Epoch  # {0},    Trials  # {1}/{2} ]'.format(epoch + 1, trial + 1, self.batch))
        print('    +  Generator Loss        = {:.4f}'.format(loss_gen))
        print('    +  Discriminator Loss    = {:.4f}'.format(loss_dis))
        print('    +  Reconstruction Error  = {:.4f}'.format(loss_ae))
        print('    -------------------------')
        print('    +  Elapsed Time            : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(elapse_time))
        print('    +  Expected Remaining Time : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(remain_time))
        print('  ---------------------------------------------------')

## Create figure object and plot
class figout:
    def __init__(self):

        ## Create Graphic Window
        self.win = pg.GraphicsWindow(title="Test")
        self.win.resize(800, 600)
        self.win.setWindowTitle('pyqtgraph example: Plotting')
        self.win.setBackground("#FFFFFFFF")
        pg.setConfigOptions(antialias=True)     # Anti-Aliasing for clear plotting

        ## Graph Layout
        #   1st Col: Waveform
        self.p1 = self.win.addPlot(colspan=2, title="Waveform")
        self.p1.addLegend()
        self.c11 = self.p1.plot(pen=(255, 0, 0), name="Input")
        self.c12 = self.p1.plot(pen=(0, 255, 0), name="Reconstructed")
        self.c13 = self.p1.plot(pen=(0, 0, 255), name="Clean")
        self.win.nextRow()
        #   2nd Col-1: Loss
        self.p2 = self.win.addPlot(title="Loss Curve")
        self.p2.addLegend()
        self.p2.setLogMode(False, True)      # Log-scale display
        self.c21 = self.p2.plot(pen=(255, 0, 0), name="Generator")
        self.c22 = self.p2.plot(pen=(0, 255, 0), name="Reconstruction")
        self.c23 = self.p2.plot(pen=(0, 0, 255), name="Discriminator")
        #   2nd Col-2: Histogram
        self.p3 = self.win.addPlot(title="Histogram of Discriminator output")
        self.p3.addLegend()
        self.c31 = self.p3.plot(pen=(128, 0, 0), stepMode=True, fillLevel=0, brush=(255,0,0,150), name="True Input")
        self.c32 = self.p3.plot(pen=(0, 128, 0), stepMode=True, fillLevel=0, brush=(0,255,0,150), name="False Input")
        self.win.nextRow()

    def waveform(self, noisy, genout, clean, stride=10):
        self.c11.setData(noisy[0:-1:stride])
        self.c12.setData(genout[0:-1:stride])
        self.c13.setData(clean[0:-1:stride])

    def loss(self, losses_gen, losses_ae, losses_dis, stride=5):
        if len(losses_gen) < 100:
            stride = 1
        self.c21.setData(losses_gen[0:-1:stride])
        self.c22.setData(losses_ae[0:-1:stride])
        self.c23.setData(losses_dis[0:-1:stride])

    def histogram(self, true_disout, fake_disout):
        y_t, x_t = np.histogram(true_disout, bins=np.linspace(0, 1, 60))
        y_f, x_f = np.histogram(fake_disout, bins=np.linspace(0, 1, 60))
        self.c31.setData(x_t,y_t)
        self.c32.setData(x_f,y_f)


    def save(self, path):
        pg.QtGui.QApplication.processEvents()
        exporter = pgex.ImageExporter(self.win.scene())
        exporter.export(path)  # save fig


