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
        self.epoch          = 60                # エポック (1エポック=1回全データ学習)
        self.learning_rate_gen = 0.0001         # 学習率 (Generator)
        self.learning_rate_dis = 0.00001        # 学習率 (Discriminator)
        self.weight_decay 	= 0.0001            # 正則化パラメータ

        # Train Condition
        self.retrain        = True              # 過去に学習したパラメータからの再開 or not

        # Retrain
        self.epoch_from     = 37                # 再学習する際の開始エポック数

        # Save path & Load path
        self.model_save_path= 'params' # パラメータ保存先
        self.clean_wav_path = 'wav/clean_trainset_wav_16k'  # 学習用クリーンwavのパス
        self.noisy_wav_path = '.wav/noisy_trainset_wav_16k'  # 学習用ノイジィwavのパス
        self.clean_pkl_path = 'pkl'             # 学習用クリーンpklのパス
        self.noisy_pkl_path = 'pkl'             # 学習用ノイジィpklのパス
