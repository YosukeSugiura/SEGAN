#
# Parameterset
#

# パラメータを詰め込む用変数
class parameters:

    def __init__(self):

        # General Settings
        self.len            = 2 ** 14           # 入力のサンプル数 (len = 16384)

        # Parameters
        self.batch_size 	= 200               # バッチサイズ
        self.pre_epoch      = 1                 # 事前学習のエポック
        self.epoch          = 60                # エポック (1エポック=1回全データ学習)
        self.pre_earning_rate = 0.0001          # 事前学習の学習率
        self.learning_rate_gen = 0.0001         # 学習率 (Generator)
        self.learning_rate_dis = 0.00001        # 学習率 (Discriminator)
        self.weight_decay 	= 0.0001            # 正則化パラメータ

        # Train Condition
        self.pretrain       = False             # 事前学習を行う or not
        self.retrain        = True              # 過去に学習したパラメータからの再開 or not

        # Retrain
        self.epoch_from     = 37                # 再学習する際の開始エポック数

        # test
        self.epoch_test     = 37                # 再学習する際の開始エポック数

        # Save path & Load path
        self.monitor_path = 'tmp.monitor'       # モニタ関連保存場所
        self.model_save_path= self.monitor_path # パラメータ保存先
        self.clean_wav_path = '../01 Data/SEGAN/clean_trainset_wav_16k'  # 学習用クリーンwavのパス
        self.noisy_wav_path = '../01 Data/SEGAN/noisy_trainset_wav_16k'  # 学習用ノイジィwavのパス
        self.clean_pkl_path = 'pkl'             # 学習用クリーンpklのパス
        self.noisy_pkl_path = 'pkl'             # 学習用ノイジィpklのパス
