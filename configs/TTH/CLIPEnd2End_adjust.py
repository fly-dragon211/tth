# CLIPEnd2End_adjust
from .. import base_config as BaseConfig
import numpy as np


class config(BaseConfig.config):
    model_name = 'End2EndClip'  # chosen model

    # txt encoder and transform
    text_encoding = {
        'bow_encoding': {'name': 'nobow_nsw'},  # [nobow_nsw, bow_nsw]
        'w2v_encoding': {'name': 'now2v_nsw'},  # [now2v_nsw, w2v_nsw]
        'rnn_encoding': {'name': 'nogru_mean'},  # [gru_mean, bigru_mean, nogru_mean]
        'bert_encoding': {'name': 'noBert',  # [noBert, bert-base-uncased, \bert_name, ...]
                          'dir_name': 'bert-base-uncased'
                          },
        'CLIP_encoding': {'name': 'ViT-B/32',  # [noCLIP, ViT-B/32, \CLIP_name, ...]
                          'dir_name': 'CLIP_ViT-B32'
                          },
        'NetVLAD_encoding': {'name': 'noNetVLAD'},  # [noNetVLAD, NetVLAD]
    }
    vid_feats = []
#

    float16 = True
    # end2end 学习，输入 frame/video 原始文件
    frame_loader = True
    # if text_encoding includes CLIP
    clip_opt = {
        'size': 512, 'transform_batch_norm': True, 'transform_dropout': 0.0,
        'transform_activation': 'tanh', 'frozen': False,
    }
    # ********************************萌萌哒分界线******************
    # Attack
    attack_scales = [1024]  #  "[1024], [300,400,500,600,700,800,900,1024], [300,350,400,450,500,550,600,650,700,750,800,850,900,950,1024]
    attack_iters = 300
    attack_lr = 0.01
    attack_lam = 1
    attack_sigma_blur = 0.0  # no blur if 0.0
    attack_mode = 'global'  # "global | tensor | hist"
    attack_variant = ""
    patch_ratio = 0.35
    only_keyword = False

    # For Attention params
    def adjust_parm(self, value):
        a = []
        for i, each in enumerate(value.split('_')):
            a.append(eval(each))
        sample_types = ['uniform', 'random']
        self.frame_sample_type_train = sample_types[a[0]]
        self.sample_frame = a[1] if a[1] > 0 else 1
        self.clip_opt['frozen'] = True if a[2] == 1 else False
        pass