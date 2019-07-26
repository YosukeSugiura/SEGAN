#
# Settings for SEGAN
#

class settings:

    def __init__(self):

        # General Settings
        self.len                = 2 ** 14           # Input Size (len = 16384)
        self.device_id          = 0                 # GPU ID (init:0)
        self.random_seed        = 0
        self.halfprec           = True              # 16Bit or not

        # Parameters
        self.batch_size 	    = 150               # Batch size
        self.epoch              = 100               # Epoch
        self.learning_rate_pre_dae = 0.001         # Learning Rate (Denoising Auto Encoder)
        self.learning_rate_gen  = 0.0001            # Learning Rate (Generator)
        self.learning_rate_dis  = 0.000001          # Learning Rate (Discriminator)
        self.weight_decay_rate 	= 0.05              # Regularization rate (weight decay = learning_rate * weight_decay_rate)
        self.alpha              = 0.0001
        self.lp                 = 1.0

        # Retrain
        self.epoch_from         = 0                 # Epoch No. from that Retraining starts (init:0)

        # Save path
        self.model_save_path    = 'params_prop'          # Network model path
        self.model_save_cycle   = 2                      # Epoch cycle for saving model (init:1)
        self.result_save_path   = 'monitor'              # Network model path

        # Save wav path
        self.wav_save_path      = 'pred'

        # Wave files
        self.clean_train_path   = './data/clean_trainset_wav'     # Folder containing clean wav (train)
        self.noisy_train_path   = './data/noisy_trainset_wav'     # Folder containing noisy wav (train)
        self.clean_test_path    = './data/clean_testset_wav'      # Folder containing clean wav (test)
        self.noisy_test_path    = './data/noisy_testset_wav'      # Folder containing noisy wav (test)

        # # Pkl files for train
        self.train_pkl_path     = 'pkl'             # Folder of pkl files for train
        self.train_pkl_clean    = 'train_clean.pkl' # File name of "Clean" pkl for train
        self.train_pkl_noisy    = 'train_noisy.pkl' # File name of "Noisy" pkl for train

        # # Pkl files for test
        self.test_pkl_path      = 'pkl'             # Folder of pkl files for test
        self.test_pkl_clean     = 'test_clean.pkl'  # File name of "Clean" pkl for test
        self.test_pkl_noisy     = 'test_noisy.pkl'  # File name of "Noisy" pkl for test


        # additional
        self.learning_rate_msk = 0.00005  # Learning Rate