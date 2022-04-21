# coding=utf-8

import torch
import sys

sys.path.append('../')
import model.clip as clip
import numpy as np
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import BertTokenizer, BertModel

import util
from loss import *
from bigfile import BigFile
from generic_utils import Progbar


def get_we(vocab, w2v_dir):
    """
    得到 word2vec 模型，n x 500
    :param vocab:
    :param w2v_dir:
    :return: word2vec 参数，[11286, 500]
    """
    w2v = BigFile(w2v_dir)
    ndims = w2v.ndims
    nr_words = len(vocab)
    words = [vocab[i] for i in range(nr_words)]
    we = np.random.uniform(low=-1.0, high=1.0, size=(nr_words, ndims))

    renamed, vecs = w2v.read(words)
    for i, word in enumerate(renamed):
        idx = vocab.find(word)
        we[idx] = vecs[i]

    return torch.Tensor(we)


def _initialize_weights(m):
    """Initialize module weights
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm1d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def to_device_and_float16(x: torch.Tensor):
    x = x.to(device)
    # if float16:
    #     x = x.half()
    return x


class IdentityNet(nn.Module):
    """
    直接返回源数据
    """

    def __init__(self, opt):
        super(IdentityNet, self).__init__()

    def forward(self, input_x):
        """Extract image feature vectors."""
        return input_x


class TransformNet(nn.Module):
    """
    fc_layers: (dim_in, dim_out)
    加入 BatchNorm, activation, dropout
    """

    def __init__(self, fc_layers, opt=None, dropout=None, batch_norm=None, activation=None, fc=True):
        super(TransformNet, self).__init__()

        if opt is not None:
            if batch_norm is None:
                batch_norm = opt.batch_norm
            if activation is None:
                activation = opt.activation
            if dropout is None:
                dropout = opt.dropout
        if fc:
            self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        else:
            self.fc1 = None
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(fc_layers[1])
        else:
            self.bn1 = None

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

        if dropout is not None and dropout > 1e-3:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        self.apply(_initialize_weights)

    def forward(self, input_x):
        """
        一般来说顺序：-> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        不过有了 bn 一般不用 dropout
        https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        """
        features = input_x.to(device)
        if self.fc1 is not None:
            features = self.fc1(features)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        if self.bn1 is not None:
            features = self.bn1(features)

        return features


class VisTransformNet(TransformNet):
    """
    把拼接的 video_emb 映射到公共空间
    """

    def __init__(self, opt):
        super(VisTransformNet, self).__init__((np.sum(list(opt.vis_fc_layers[0].values())), opt.vis_fc_layers[1]), opt)

    def forward(self, vis_input: dict, txt_emb=None, vis_frame_feat_dict_input=None):
        """
        一般来说顺序：-> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        不过有了 bn 一般不用 dropout
        https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        """
        if type(vis_input) == dict:
            vis_feature = to_device_and_float16(torch.cat(list(vis_input.values()), dim=1))
        else:
            vis_feature = to_device_and_float16(vis_input)
        features = self.fc1(vis_feature)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        if self.bn1 is not None:
            features = self.bn1(features)

        return features


class TxtEncoder(nn.Module):
    def __init__(self, opt):
        super(TxtEncoder, self).__init__()

    def forward(self, caption_feat_dict, task3=False):
        output = {}
        output['text_features'] = caption_feat_dict['caption']

        return output


class GruTxtEncoder(TxtEncoder):
    def _init_rnn(self, opt):
        if opt.rnn_size == 0:
            return
        self.rnn = nn.GRU(int(opt.we_dim), int(opt.rnn_size), int(opt.rnn_layer), batch_first=True, bidirectional=False)

    def __init__(self, opt):
        super().__init__(opt)
        self.bigru = False
        self.pooling = opt.pooling
        self.rnn_size = opt.rnn_size
        self.t2v_idx = opt.t2v_idx
        self.we = nn.Embedding(len(self.t2v_idx.vocab), opt.we_dim)
        if opt.we_dim == 500:
            self.we.weight = nn.Parameter(opt.we)  # initialize with a pre-trained 500-dim w2v

        self._init_rnn(opt)

    def forward(self, caption_feat_dict, task3=False):
        txt_input = caption_feat_dict['caption']
        batch_size = len(txt_input)

        # caption encoding
        idx_vecs = [self.t2v_idx.encoding(caption) for caption in txt_input]
        lengths = [len(vec) for vec in idx_vecs]

        x = to_device_and_float16(torch.zeros(batch_size, max(lengths)).long())
        for i, vec in enumerate(idx_vecs):
            end = lengths[i]
            x[i, :end] = torch.Tensor(vec)

        # caption embedding
        x = self.we(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)

        if self.pooling == 'mean':
            # out = torch.zeros(batch_size, padded[0].shape[-1]).to(device)
            out = x.new_zeros((batch_size, padded[0].shape[-1])).to(device)
            for i, ln in enumerate(lengths):
                out[i] = torch.mean(padded[0][i][:ln], dim=0)
        elif self.pooling == 'last':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            out = torch.gather(padded[0], 1, I).squeeze(1)
        elif self.pooling == 'mean_last':
            # out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            for i, ln in enumerate(lengths):
                out1[i] = torch.mean(padded[0][i][:ln], dim=0)

            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            out2 = torch.gather(padded[0], 1, I).squeeze(1)
            out = torch.cat((out1, out2), dim=1)

        output = {}
        output['text_features'] = out

        return output


class BiGruTxtEncoder(GruTxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim, opt.rnn_size, opt.rnn_layer, batch_first=True, bidirectional=True)

    def __init__(self, opt):
        super().__init__(opt)
        self.bigru = True


class BoWTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super(BoWTxtEncoder, self).__init__(opt)
        self.t2v_bow = opt.t2v_bow

    def forward(self, caption_feat_dict,task3=False):
        txt_input = caption_feat_dict['caption']
        t = np.empty((len(txt_input), self.t2v_bow.ndims), )
        for i, caption in enumerate(txt_input):
            t[i] = self.t2v_bow.encoding(caption)

        # bow_out = torch.Tensor([self.t2v_bow.encoding(caption) for caption in txt_input]).to(device)
        bow_out = to_device_and_float16(torch.Tensor(t))
        # print(bow_out.shape)
        output = {}
        output['text_features'] = bow_out

        return output


class W2VTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super(W2VTxtEncoder, self).__init__(opt)
        self.t2v_w2v = opt.t2v_w2v

    def forward(self, caption_feat_dict, task3=False):
        txt_input = caption_feat_dict['caption']
        t = np.empty((len(txt_input), self.t2v_w2v.ndims), )
        for i, caption in enumerate(txt_input):
            t[i] = self.t2v_w2v.encoding(caption)

        w2v_out = to_device_and_float16(torch.Tensor(t))
        output = {}
        output['text_features'] = w2v_out

        return output


class BertTxtEncoder(nn.Module):
    """
    Bert encoder
    """

    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.bert_name = opt.text_encoding['bert_encoding']['name']  # 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name, do_lower_case=opt.bert_do_lower_case)
        self.BertModel = BertModel.from_pretrained(self.bert_name)

    def forward(self, caption_feat_dict, task3=False):
        if 'bert_encoding' in caption_feat_dict and self.opt.bert_frozen:
            features = caption_feat_dict['bert_encoding']
        else:
            txt_input = caption_feat_dict['caption']
            encoded_input = self.tokenizer(txt_input, return_tensors='pt', padding=True, truncation=True)
            for each in encoded_input:
                encoded_input[each] = to_device_and_float16(encoded_input[each])
            if self.opt.bert_frozen:
                with torch.no_grad():
                    bert_output = self.BertModel(**encoded_input)
            else:
                bert_output = self.BertModel(**encoded_input)
            features = bert_output['pooler_output']
        output = {}
        output['text_features'] = features

        return output


class CLIPEncoder(nn.Module):
    """
    CLIP encoder.
    transform text and image into features.
    """

    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.Clip_name = opt.text_encoding['CLIP_encoding']['name']
        self.frozen = opt.clip_opt['frozen']
        self.dim = opt.clip_opt['size']
        self.tokenizer = clip.tokenize
        self.simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.ClipModel, self.preprocess = clip.load(self.Clip_name, device=device, jit=False)

    def forward(self, caption_feat_dict, vis_origin_frame_tuple=None, task3=False,
                frame_agg_method='mean'):
        """

        :param caption_feat_dict:
        :param vis_origin_frame_tuple: ([sample_frame, 3, 224, 224], ...)
        :param task3:
        :return: (batch_size, dim)
        """
        output = {}
        # For text encoding
        if caption_feat_dict is not None:
            if 'CLIP_encoding' in caption_feat_dict and self.frozen:
                text_features = caption_feat_dict['CLIP_encoding']
            else:
                txt_input = caption_feat_dict['caption']
                text = to_device_and_float16(self.tokenizer(txt_input))
                if self.frozen and (not task3):
                    with torch.no_grad():
                        text_features = self.ClipModel.encode_text(text)
                else:
                    text_features = self.ClipModel.encode_text(text)
            output['text_features'] = text_features

        # For visual encoding
        if vis_origin_frame_tuple is not None:
            batch_size = len(vis_origin_frame_tuple)
            origin_frames = to_device_and_float16(torch.cat(vis_origin_frame_tuple, dim=0))

            if self.frozen:
                with torch.no_grad():
                    frame_features = self.ClipModel.encode_image(origin_frames)
            else:
                frame_features = self.ClipModel.encode_image(origin_frames)
            frame_features = frame_features.reshape((batch_size, -1, self.dim))
            if frame_agg_method == 'mean':
                visual_features = torch.mean(frame_features, dim=1)
            else:
                raise Exception("frame_agg_method is not applied.")

            output['visual_features'] = visual_features

        return output

    def get_visual_features(self, origin_frames):
        frame_features = self.ClipModel.encode_image(to_device_and_float16(origin_frames))
        return frame_features


class MultiScaleTxtEncoder(TxtEncoder):
    """
    多个 txt net concatenate 叠加输出

    """

    def init_txt_encoder(self, opt):
        self.bow_encoding, self.w2v_encoding, self.rnn_encoding, \
        self.bert_encoding, self.CLIP_encoding, self.NetVLAD_encoding = (
            opt.text_encoding['bow_encoding']['name'],
            opt.text_encoding['w2v_encoding']['name'],
            opt.text_encoding['rnn_encoding']['name'],
            opt.text_encoding['bert_encoding']['name'],
            opt.text_encoding['CLIP_encoding']['name'],
            opt.text_encoding['NetVLAD_encoding']['name'],
        )
        self.space_dict = {}  # encoder: space_dimension
        self.txt_encoder_num = 0
        self.encoder = nn.Module()

        # gru
        self.rnn_encoding, opt.pooling = self.rnn_encoding.split('_', 1)
        if self.rnn_encoding == 'gru':
            self.space_dict['rnn_encoder'] = opt.rnn_size
            self.txt_encoder_num += 1
            self.encoder.add_module('rnn_encoder', GruTxtEncoder(opt))
        elif self.rnn_encoding == 'bigru':
            self.space_dict['rnn_encoder'] = opt.rnn_size * 2
            self.txt_encoder_num += 1
            self.encoder.add_module('rnn_encoder', BiGruTxtEncoder(opt))
        elif self.rnn_encoding == 'nogru':
            pass

        # bert
        if self.bert_encoding == 'noBert':
            pass
        else:
            self.space_dict['bert_encoder'] = opt.bert_size
            self.txt_encoder_num += 1
            self.encoder.add_module('bert_encoder', BertTxtEncoder(opt))

        # w2v, bow
        if 'no' not in self.bow_encoding:
            self.space_dict['bow_encoder'] = opt.t2v_bow.ndims
            self.txt_encoder_num += 1
            self.encoder.add_module('bow_encoder', BoWTxtEncoder(opt))
        if 'no' not in self.w2v_encoding:
            self.space_dict['w2v_encoder'] = opt.t2v_w2v.ndims
            self.txt_encoder_num += 1
            self.encoder.add_module('w2v_encoder', W2VTxtEncoder(opt))

        # CLIP
        if 'no' not in self.CLIP_encoding:
            self.space_dict['CLIP_encoder'] = opt.clip_opt['size']
            self.txt_encoder_num += 1
            self.encoder.add_module('CLIP_encoder', CLIPEncoder(opt))

        # encoder_names
        self.encoder_name_list = []
        for name, parm in self.encoder.named_modules():
            if '.' in name or name == '':  # if it is children name, continue
                continue
            self.encoder_name_list.append(name)

    def init_transform(self, opt, suffix=''):
        common_space_dim = opt.txt_fc_layers[1]
        if 'transform_layer' not in dict(self.named_modules()):
            self.transform_layer = nn.Module()

        dropout = opt.dropout
        batch_norm = opt.batch_norm
        activation = opt.activation

        if 'no' not in self.rnn_encoding:
            if 'bigru' in self.rnn_encoding:
                rnn_transform = TransformNet((self.space_dict['rnn_encoder'], common_space_dim), None, dropout,
                                             batch_norm, activation)
                self.transform_layer.add_module('rnn_encoder' + '_transform' + suffix, rnn_transform)
            else:
                rnn_transform = TransformNet((self.space_dict['rnn_encoder'], common_space_dim), None, dropout,
                                             batch_norm, activation)
                self.transform_layer.add_module('rnn_encoder' + '_transform' + suffix, rnn_transform)

        if self.bert_encoding == 'noBert':
            pass
        else:
            bert_transform = TransformNet((opt.bert_size, common_space_dim), None, opt.bert_transform_dropout,
                                          opt.bert_transform_batch_norm, opt.bert_transform_activation)
            self.transform_layer.add_module('bert_encoder' + '_transform' + suffix, bert_transform)

        if 'no' not in self.w2v_encoding:
            w2v_transform = TransformNet((opt.t2v_w2v.ndims, common_space_dim), None, dropout,
                                         batch_norm, activation)
            self.transform_layer.add_module('w2v_encoder' + '_transform' + suffix, w2v_transform)

        if 'no' not in self.bow_encoding:
            bow_transform = TransformNet((opt.t2v_bow.ndims, common_space_dim), None, dropout,
                                         batch_norm, activation)
            self.transform_layer.add_module('bow_encoder' + '_transform' + suffix, bow_transform)

        if 'no' not in self.CLIP_encoding:
            if "CLIP_encoder" in self.opt.txt_no_transform:
                CLIP_transform = TransformNet((
                    opt.clip_opt['size'], common_space_dim), None,
                    opt.clip_opt['transform_dropout'], opt.clip_opt['transform_batch_norm'],
                    False, False)
            else:

                CLIP_transform = TransformNet((
                    opt.clip_opt['size'], common_space_dim), None,
                    opt.clip_opt['transform_dropout'], opt.clip_opt['transform_batch_norm'],
                    opt.clip_opt['transform_activation'])
            self.transform_layer.add_module('CLIP_encoder' + '_transform' + suffix, CLIP_transform)

        if 'no' not in self.NetVLAD_encoding:
            NetVLAD_transform = TransformNet((opt.t2v_w2v.ndims * opt.NetVLAD_opt['num_clusters'], common_space_dim),
                                             None, dropout,
                                             batch_norm, activation)
            self.transform_layer.add_module('NetVLAD_encoder' + '_transform' + suffix, NetVLAD_transform)


    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.init_txt_encoder(opt)

    def forward(self, caption_feat_dict, task3=False):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        out_feature = []
        encoder_module_dict = dict(self.encoder.named_modules())
        for name in self.encoder_name_list:
            txt_features = encoder_module_dict[name](caption_feat_dict, task3=task3)['text_features']
            txt_features = to_device_and_float16(txt_features)
            out_feature.append(txt_features)

        out = torch.cat(out_feature, dim=1)
        return out


class MultiScaleTxtNet(nn.Module):
    def _init_encoder(self, opt):
        self.encoder = MultiScaleTxtEncoder(opt)

    def _init_transformer(self, opt):
        self.transformer = TransformNet(
            opt.txt_fc_layers, opt, opt.dropout, opt.batch_norm,
            opt.activation)

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self._init_encoder(self.opt)

        self.opt.txt_fc_layers[0] = 0
        for name in self.encoder.space_dict:
            self.opt.txt_fc_layers[0] += self.encoder.space_dict[name]

        self._init_transformer(self.opt)

    def forward(self, caption_feat_dict, task3=False):
        features = self.encoder(caption_feat_dict, task3)
        features = self.transformer(features)
        return features


# ****************************萌萌哒分界线****************************************
class W2VVPP(nn.Module):
    """
    w2v++ 加入预测 Bert 作为 txt encoder

        w2v++ 最主要的net, self.vis_net为视频特征转换网络，可以使用 ircsn finetune 作为 vis 输入
        self.txt_net为多网络拼接的 文本查询转换网络
        """

    def _init_vis_net(self, opt):
        self.vis_net = VisTransformNet(opt)

    def _init_txt_net(self, opt):
        self.txt_net = MultiScaleTxtNet(opt)
        if opt.txt_fc_same_with_vis_fc:
            if self.txt_net.transformer.fc1.weight.shape[1] == self.vis_net.fc1.weight.shape[1]:
                self.txt_net.transformer = self.vis_net
            else:
                raise Exception("txt_fc is not matching vis_fc ")

    def __init__(self, opt):
        super().__init__()
        self.scaler = GradScaler()
        if opt is None:
            return
        self._init_vis_net(opt)
        self._init_txt_net(opt)

        self.opt = opt
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = True

        self.criterion = MarginRankingLoss(margin=opt.margin,
                                           measure=opt.measure,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction,
                                           device=device)

        self.params = list(self.parameters())  # 所有 params

        # 设置学习率
        params_special = []
        params_usual = []
        for name, parm in list(self.named_parameters()):
            if ('BertModel' in name) or ('csn_model' in name) or ('ClipModel' in name):
                params_special.append(parm)
            else:
                params_usual.append(parm)
        params = [{'params': params_usual},
                  {'params': params_special, 'lr': opt.lr / 20}]

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                              torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                         patience=2)]

        self.iters = 0

    def compute_loss(self, vis_embs, txt_embs, vis_embs_multi_labels, txt_embs_multi_labels, labels_embs):
        """Compute the loss given pairs of image and caption embeddings
        """
        if len(vis_embs.shape) == len(txt_embs.shape) == 2:
            triplet_loss = self.criterion(txt_embs, vis_embs)
            multi_label_loss_vis = 0
            multi_label_loss_txt = 0
            multi_label_triplet_loss = 0
            loss = triplet_loss + multi_label_loss_vis + multi_label_loss_txt + multi_label_triplet_loss
            loss_items = {
                'triplet_loss': triplet_loss
            }
        elif len(vis_embs.shape) == len(txt_embs.shape) == 3:
            triplet_loss_multi_head = 0
            for each in range(vis_embs.size(1)):
                triplet_loss_multi_head += self.criterion(txt_embs[:, each, :], vis_embs[:, each, :])
            loss = triplet_loss_multi_head
            loss_items = {
                'triplet_loss': triplet_loss_multi_head
            }
        else:
            raise Exception("vis_embs dims are not equal to txt_embs dims")
        return loss, loss_items

    def cal_foward(self, train_data):
        (vis_input, caption_feat_dict, labels_input,
         vis_frame_feat_dict_input,
         vis_origin_frame_tuple) = (
            train_data['vis_feats'], train_data['captions'],
            train_data['captions_task2'], train_data['vis_frame_feat_dict'],
            train_data['vis_origin_frame_tuple']
        )
        if vis_frame_feat_dict_input == {}:
            vis_frame_feat_dict_input = None
        # compute the embeddings
        txt_embs = self.txt_net(caption_feat_dict)
        vis_embs = self.vis_net(vis_input, txt_emb=txt_embs,
                                vis_frame_feat_dict_input=vis_frame_feat_dict_input)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, loss_items = self.compute_loss(vis_embs, txt_embs, 0, 0, 0)
        # print("triplet_loss and multi_label_loss_vis", loss_items, end='\r')

        return loss, loss_items

    def forward(self, train_data, epoch=None):
        """One training step given vis_feats and captions.
        """

        self.iters += 1

        if float16:
            # 前向过程(model + loss)开启 autocast
            with autocast():
                loss, loss_items = self.cal_foward(train_data)

            # Scales loss，这是因为半精度的数值范围有限，因此需要用它放大
            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                clip_grad_norm_(self.params, self.grad_clip)

            # scaler.step() unscale之前放大后的梯度，但是scale太多可能出现inf或NaN
            # 故其会判断是否出现了inf/NaN
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 如果检测到出现了inf或者NaN，就跳过这次梯度更新，同时动态调整scaler的大小
            self.scaler.step(self.optimizer)
            # 查看是否要更新scaler,这个要注意不能丢
            self.scaler.update()
        else:
            loss, loss_items = self.cal_foward(train_data)
            # compute gradient and do SGD step
            loss.backward()
            if self.grad_clip > 0:
                clip_grad_norm_(self.params, self.grad_clip)
            self.optimizer.step()

        return loss_items

    def get_txt2vis_matrix(self, txt_embs, vis_embs, measure='cosine'):
        if len(vis_embs.shape) == len(txt_embs.shape) == 2:
            txt2vis_sim = self.compute_sim(txt_embs, vis_embs, measure, device)

        elif len(vis_embs.shape) == len(txt_embs.shape) == 3:
            for j, each in enumerate(range(vis_embs.size(1))):
                txt2vis_sim_temp = self.compute_sim(txt_embs[:, each, :], vis_embs[:, each, :], measure,
                                                    device).unsqueeze(0)
                txt2vis_sims = txt2vis_sim_temp if j == 0 else torch.cat(
                    (txt2vis_sims, txt2vis_sim_temp), dim=0)

            txt2vis_sim = torch.mean(txt2vis_sims, dim=0)

        return txt2vis_sim

    @util.timer
    def predict(self, txt_loader, vis_loader, measure, record_emb=False):
        if vis_loader.dataset.length > 5e4:
            return self.predict_batch(txt_loader, vis_loader, measure, record_emb)
        self.eval()

        txt_ids = []
        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.vis_net(vis_input, vis_frame_feat_dict_input=vis_frame_feat_dict).cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.txt_net(caption_feat_dict)

                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break
                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, self.vis_ids

    @util.timer
    def predict_batch(self, txt_loader, vis_loader, measure, record_emb=False):
        """
        predict similarity each batch.
        :param txt_loader:
        :param vis_loader:
        :param measure:
        :param record_emb:
        :return:
        """
        print("predict_batch !")
        self.eval()

        txt_ids = []
        vis_ids = []
        pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))

        with torch.no_grad():
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
                # if i > 1:
                #     txt_ids.extend(batch_txt_ids)
                #     continue

                txt_embs = self.txt_net(caption_feat_dict)  # a dict

                for j, output_dict in enumerate(vis_loader):

                    (vis_input, idxs, batch_vis_ids, vis_frame_feat_dict) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict']
                    )
                    vis_embs = self.vis_net(vis_input, vis_frame_feat_dict_input=vis_frame_feat_dict)

                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    if i == 0:
                        vis_ids.extend(batch_vis_ids)

                    pbar.add(len(batch_vis_ids) * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, vis_ids


    @staticmethod
    def compute_sim(query_embs, retro_embs, measure='cosine', device=torch.device('cuda')):
        query_embs = query_embs.to(device)
        retro_embs = retro_embs.to(device)
        if measure == 'cosine':
            return cosine_sim(query_embs, retro_embs)
        elif measure == 'hist':
            return hist_sim(query_embs, retro_embs)
        elif measure == 'euclidean':
            raise Exception('Not implemented')
        else:
            raise Exception('%s is invalid' % measure)

    @property
    def learning_rate(self):
        """Return learning rate"""
        lr_list = []
        for param_group in self.optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list

    def lr_step(self, val_value):
        """
        降低学习率
        :param val_value:
        :return:
        """
        self.lr_schedulers[0].step()
        self.lr_schedulers[1].step(val_value)

    def change_raw_global_emb_weight(self):
        # 更改 raw_global_emb_weight 比例
        try:
            if hasattr(self.txt_net, 'attention_layer'):
                if hasattr(self.txt_net.attention_layer, 'get_raw_global_emb_weight'):
                    # 指数级别衰减
                    # new_global_emb_weight = self.opt.txt_attention_global_decay_rate * \
                    #                         self.txt_net.attention_layer.get_raw_global_emb_weight()
                    # self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)

                    # 线性衰减
                    new_global_emb_weight = self.opt.txt_attention_global_decay_rate - \
                                            1 + self.txt_net.attention_layer.get_raw_global_emb_weight()
                    if new_global_emb_weight < 0:
                        new_global_emb_weight = 0
                    self.txt_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                else:
                    print("txt_net.attention_layer doesn't have get_raw_global_emb_weight meathod")
                print("new_global_emb_weight: ", new_global_emb_weight)
            else:
                print("txt_net doesn't have attention_layer")
        except Exception as e:
            print(e)

        try:
            if hasattr(self.vis_net, 'attention_layer'):
                if hasattr(self.vis_net.attention_layer, 'get_raw_global_emb_weight'):
                    # 指数级别衰减
                    # new_global_emb_weight = self.opt.vis_attention_global_decay_rate * \
                    #                         self.vis_net.attention_layer.get_raw_global_emb_weight()
                    # self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                    # 线性衰减
                    new_global_emb_weight = self.opt.vis_attention_global_decay_rate - \
                                            1 + self.vis_net.attention_layer.get_raw_global_emb_weight()
                    if new_global_emb_weight < 0:
                        new_global_emb_weight = 0
                    self.vis_net.attention_layer.change_raw_global_emb_weight(new_global_emb_weight)
                print("new_global_emb_weight: ", new_global_emb_weight)
            else:
                print("vis_net doesn't have attention_layer")
        except Exception as e:
            print(e)




class CLIP端到端系列():
    pass


class W2VVPP_FrozenClip(W2VVPP):
    """
    w2v++ clip 直接算

        w2v++ 最主要的net, self.vis_net为视频特征转换网络，可以使用 ircsn finetune 作为 vis 输入
        self.txt_net为多网络拼接的 文本查询转换网络
        """

    class SimpleVisMudule(nn.Module):
        def __init__(self):
            """
            简单返回
            """
            super().__init__()

        def forward(self, vis_input: dict, txt_emb=None, vis_frame_feat_dict_input=None):
            vis_feature = torch.cat(list(vis_input.values()), dim=1)  # batch_size, vis_feature_concat

            return vis_feature

    class SimpleTxtMudule(nn.Module):

        def __init__(self, opt):
            super().__init__()
            self.encoder = CLIPEncoder(opt)  # return a dict

        def forward(self, caption_feat_dict):
            features = self.encoder(caption_feat_dict)['text_features']
            return features

    def _init_txt_net(self, opt):
        self.txt_net = W2VVPP_FrozenClip.SimpleTxtMudule(opt)

    def _init_vis_net(self, opt):
        self.vis_net = W2VVPP_FrozenClip.SimpleVisMudule()

    def __init__(self, opt):
        super().__init__(opt)

    def forward(self, train_data, epoch):
        """One training step given vis_feats and captions.
        """
        self.iters += 1

        loss_items = {'triplet_loss': torch.Tensor(1)}

        return loss_items


class End2EndClip(W2VVPP):
    """
    w2v端到端 clip model

        输入视频帧信息和原始文本，使用 clip 模型计算匹配。
        """

    def __init__(self, opt):
        super().__init__(None)
        self.clip_model = CLIPEncoder(opt)
        self.clip_frozen = opt.clip_opt['frozen']

        self.opt = opt
        self.grad_clip = opt.grad_clip
        if torch.cuda.is_available():
            cudnn.benchmark = True

        self.criterion = MarginRankingLoss(margin=opt.margin,
                                           measure=opt.measure,
                                           max_violation=opt.max_violation,
                                           cost_style=opt.cost_style,
                                           direction=opt.direction,
                                           device=device)

        self.params = list(self.parameters())  # 所有 params

        # 设置学习率
        params_special = []
        params_usual = []
        for name, parm in list(self.named_parameters()):
            if ('BertModel' in name) or ('csn_model' in name) or ('ClipModel' in name):
                params_special.append(parm)
            else:
                params_usual.append(parm)
        params = [{'params': params_usual},
                  {'params': params_special, 'lr': opt.lr / 100}]

        if opt.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=opt.lr)
        elif opt.optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(params, lr=opt.lr)

        self.lr_schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=opt.lr_decay_rate),
                              torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5,
                                                                         patience=2)]

        self.iters = 0

    @util.timer
    def predict(self, txt_loader, vis_loader, measure, record_emb=False):
        """
        :param txt_loader:
        :param vis_loader:
        :param measure:
        :param record_emb: record the video_all_embs and accelerate the prediction.
        :return:
        """
        self.eval()

        txt_ids = []

        pbar_video = Progbar(len(vis_loader.dataset))
        if not hasattr(self, "video_all_embs"):
            self.video_all_embs = None
            self.video_idxs_list = []

        with torch.no_grad():
            # First, get the embeddings of all videos
            if not record_emb or self.video_all_embs == None:
                self.video_all_embs = None
                self.video_idxs_list = []
                self.vis_ids = []

                for j, output_dict in enumerate(vis_loader):
                    (vis_input, idxs, batch_vis_ids,
                     vis_frame_feat_dict, vis_origin_frame_tuple
                     ) = (
                        output_dict['vis_feat_dict'], output_dict['idxs'],
                        output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                        output_dict['vis_origin_frame_tuple']
                    )
                    pbar_video.add(len(idxs))
                    self.video_idxs_list.append(idxs)
                    vis_embs = self.clip_model(
                        caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
                    )['visual_features'].cpu()

                    self.video_all_embs = vis_embs if self.video_all_embs is None else \
                        torch.cat((self.video_all_embs, vis_embs), dim=0)

                    self.vis_ids.extend(batch_vis_ids)

            pbar = Progbar(len(txt_loader.dataset) * len(vis_loader.dataset))
            # Get the similarity matrix
            scores = torch.zeros((len(txt_loader.dataset), len(vis_loader.dataset)))
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):

                txt_embs = self.clip_model(caption_feat_dict)['text_features']

                for idxs in self.video_idxs_list:
                    bs = vis_loader.batch_size
                    vis_embs = to_device_and_float16(self.video_all_embs[idxs])
                    if vis_embs.shape[0] == 0:
                        break

                    score = self.get_txt2vis_matrix(txt_embs, vis_embs, measure=measure).float()
                    if i != len(txt_loader)-1:
                        scores[(i * len(txt_idxs)):((i + 1) * len(txt_idxs)), idxs] = score.cpu()
                    else:
                        scores[-len(txt_idxs):, idxs] = score.cpu()

                    pbar.add(bs * len(batch_txt_ids))

                txt_ids.extend(batch_txt_ids)

        return scores.detach().numpy(), txt_ids, self.vis_ids

    def cal_foward(self, train_data):
        (vis_input, caption_feat_dict, labels_input,
         vis_frame_feat_dict_input,
         vis_origin_frame_tuple) = (
            train_data['vis_feats'], train_data['captions'],
            train_data['captions_task2'], train_data['vis_frame_feat_dict'],
            train_data['vis_origin_frame_tuple']
        )
        if vis_frame_feat_dict_input == {}:
            vis_frame_feat_dict_input = None
        # compute the embeddings
        output = self.clip_model(caption_feat_dict,
                                   vis_origin_frame_tuple=vis_origin_frame_tuple,
                                )
        vis_embs, txt_embs = output['visual_features'], output['text_features']
        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss, loss_items = self.compute_loss(vis_embs, txt_embs, 0, 0, 0)
        # print("triplet_loss and multi_label_loss_vis", loss_items, end='\r')

        return loss, loss_items

    def visual_features(self, vis_origin_frame_tensor):
        """
        For attack. Get the visual features.
        :param vis_origin_frame_tensor: [batch, 3, 224, 224]
        :return: [batch, 512]
        """
        if vis_origin_frame_tensor.shape[-1] != 224:
            from torchvision.transforms import Resize
            vis_origin_frame_tensor = Resize([224, 224])(vis_origin_frame_tensor)

        visual_features = self.clip_model.get_visual_features(vis_origin_frame_tensor)
        return visual_features

    def text_features(self, caption_list):
        """
        For attack. Get the text features.
        :param caption_list:
        :return:
        """
        caption_feat_dict = {'caption': caption_list}
        text_features = self.clip_model.forward(caption_feat_dict=caption_feat_dict)['text_features']
        return text_features

    def forward(self, train_data, epoch=None):
        """One training step given vis_feats and captions.
        """
        if self.clip_frozen:
            self.iters += 1
            loss_items = {'triplet_loss': torch.Tensor(1)}
            self.optimizer.zero_grad()
            return loss_items
        else:
            super().forward(train_data, epoch=None)


def get_model(name, device_, config):
    global device
    global float16
    device = device_
    float16 = config.float16

    NAME_TO_MODELS = {
        'W2VVPP': W2VVPP,
        'End2EndClip': End2EndClip,

    }
    assert name in NAME_TO_MODELS, '%s not supported.' % name

    model_ = NAME_TO_MODELS[name](config)
    model_ = model_.float().to(device_)
    return model_


if __name__ == '__main__':
    global device
