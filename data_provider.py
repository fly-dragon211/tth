# coding=utf-8
import torch
import torch.utils.data as data
from torchvision.datasets import Kinetics400
from prefetch_generator import BackgroundGenerator
import numpy as np
import pickle
import os
from bigfile import BigFile
from textlib import TextTool, Vocabulary, negation_augumentation
from torchvision.transforms import Compose, Resize, CenterCrop, TenCrop, Lambda, ToTensor, Normalize, RandomResizedCrop
import PIL
import model.clip as clip
import random
import util
import re


class DataLoaderX(torch.utils.data.DataLoader):
    pass
    # def __iter__(self):
    #     return BackgroundGenerator(super().__iter__())


def generate_sent_masks(source_lengths):
    """ Generate sentence masks for encoder hidden states.
        returns enc_masks (Tensor): Tensor of sentence masks of shape (b, max_seq_length),where max_seq_length = max source length """
    max_seq_length = max(source_lengths)
    batch_size = len(source_lengths)
    enc_masks = torch.zeros(batch_size, max_seq_length, dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks


# 这些是得到 dataloader 列表的后处理
def collate_vision(data):
    vis_feat_tuple, idxs, vis_ids, vis_frame_feat_tuple, vis_origin_frame_tuple = list(zip(*data))
    # 得到多视频特征字典
    vis_feat_dict = {}
    if vis_feat_tuple[0] != {}:
        for name in vis_feat_tuple[0].keys():
            vis_feat_dict[name] = torch.stack([each[name] for each in vis_feat_tuple], 0)

    # 视频帧特征字典，由于帧数不统一，使用 0 填充，并且输出 mask_tensor 矩阵
    vis_frame_feat_dict = {}  # (batch_size, max_length, embedding_size)
    if vis_frame_feat_tuple[0] != {}:
        # 得到 source_lengths 列表
        name = list(vis_frame_feat_tuple[0].keys())[0]
        source_lengths = [each[name].shape[0] for each in vis_frame_feat_tuple]
        mask_tensor = generate_sent_masks(source_lengths)
        vis_frame_feat_dict['mask_tensor'] = mask_tensor
        batch_size, max_length = mask_tensor.shape

        for name in vis_frame_feat_tuple[0].keys():
            vis_frame_feat_dict[name] = torch.zeros(
                batch_size, max_length, vis_frame_feat_tuple[0][name].shape[-1]
            )
            for index, each in enumerate(vis_frame_feat_tuple):
                vis_frame_feat_dict[name][index][0:source_lengths[index]] = each[name]

    # 视频帧原始数据
    if vis_origin_frame_tuple[0] != None:
        pass

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行

    output_dict = {
        'vis_feat_dict': vis_feat_dict, 'idxs': idxs, 'vis_ids': vis_ids,
        'vis_frame_feat_dict': vis_frame_feat_dict, 'vis_origin_frame_tuple': vis_origin_frame_tuple
    }
    return output_dict


def collate_text(data):
    data.sort(key=lambda x: len(TextTool.tokenize(x[0]['caption'])), reverse=True)
    caption_dict_tuples, idxs, cap_ids = list(zip(*data))

    # 得到多特征 caption 字典
    caption_feat_dict = {}
    for name in caption_dict_tuples[0].keys():
        if name == 'caption':
            caption_feat_dict[name] = [each[name] for each in caption_dict_tuples]
        else:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    return caption_feat_dict, idxs, cap_ids


def collate_pair(data):
    data.sort(key=lambda x: len(TextTool.tokenize(x[1]['caption'])), reverse=True)
    vis_feat_tuple, caption_dict_tuples, vis_muti_feat, caption_labels_task2, \
    idxs, vis_ids, cap_ids, vis_frame_feat_tuple, caption_labels_task3, mask_task3, \
    vis_origin_frame_tuple = list(zip(*data))

    if mask_task3[0] is not None:
        mask_task3 = np.array((mask_task3))

        index_task3 = np.where(mask_task3 > -1)[0]
        caption_labels_task3 = list(caption_labels_task3)
        caption_labels_task3 = [caption_labels_task3[i] for i in index_task3]
    # 视频特征字典
    vis_feat_dict = {}
    if vis_feat_tuple[0] != {}:
        for name in vis_feat_tuple[0].keys():
            vis_feat_dict[name] = torch.stack([each[name] for each in vis_feat_tuple], 0)

    # 视频帧特征字典，由于帧数不统一，使用 0 填充，并且输出 mask_tensor 矩阵
    vis_frame_feat_dict = {}
    if vis_frame_feat_tuple[0] != {}:
        # 得到 source_lengths 列表
        name = list(vis_frame_feat_tuple[0].keys())[0]
        source_lengths = [each[name].shape[0] for each in vis_frame_feat_tuple]
        mask_tensor = generate_sent_masks(source_lengths)
        vis_frame_feat_dict['mask_tensor'] = mask_tensor
        batch_size, max_length = mask_tensor.shape

        for name in vis_frame_feat_tuple[0].keys():
            vis_frame_feat_dict[name] = torch.zeros(
                batch_size, max_length, vis_frame_feat_tuple[0][name].shape[-1]
            )
            for index, each in enumerate(vis_frame_feat_tuple):
                vis_frame_feat_dict[name][index][0:source_lengths[index]] = each[name]

    if vis_muti_feat[0] is not None:
        vis_muti_feat = torch.stack(vis_muti_feat, 0)

    # 文本特征字典
    caption_feat_dict = {}
    for name in caption_dict_tuples[0].keys():
        if name == 'caption':
            caption_feat_dict[name] = [each[name] for each in caption_dict_tuples]
        else:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)
    caption_task3_feat_dict = {}
    if mask_task3[0] is not None:
        for name in caption_labels_task3[0].keys():
            if name == 'caption':
                caption_task3_feat_dict[name] = [each[name] for each in caption_labels_task3]
            else:
                caption_task3_feat_dict[name] = torch.stack([each[name] for each in caption_labels_task3], 0)

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    output = {'vis_feats': vis_feat_dict, 'vis_muti_feat': vis_muti_feat,
              'vis_frame_feat_dict': vis_frame_feat_dict,
              'vis_origin_frame_tuple': vis_origin_frame_tuple,
              'captions': caption_feat_dict, 'captions_task2': caption_labels_task2,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids,
              'captions_task3': caption_task3_feat_dict, "captions_task3_mask": mask_task3}
    return output


def collate_pair_frame_list(data):
    """
    输出的 视频帧特征 是一个 list，效率不高，已弃用
    :param data:
    :return:
    """
    data.sort(key=lambda x: len(TextTool.tokenize(x[1]['caption'])), reverse=True)
    vis_feat_tuple, caption_dict_tuples, vis_muti_feat, caption_labels_task2, \
    idxs, vis_ids, cap_ids, vis_frame_feat_tuple = list(zip(*data))
    # 视频特征字典
    vis_feat_dict = {}
    for name in vis_feat_tuple[0].keys():
        vis_feat_dict[name] = torch.stack([each[name] for each in vis_feat_tuple], 0)

    # 视频帧特征字典，由于帧数不统一，里面是列表
    vis_frame_feat_dict = {}
    if vis_frame_feat_tuple[0] != {}:
        for name in vis_frame_feat_tuple[0].keys():
            vis_frame_feat_dict[name] = [each[name] for each in vis_frame_feat_tuple]

    if vis_muti_feat[0] is not None:
        vis_muti_feat = torch.stack(vis_muti_feat, 0)

    # 文本特征字典
    caption_feat_dict = {}
    for name in caption_dict_tuples[0].keys():
        if name == 'caption':
            caption_feat_dict[name] = [each[name] for each in caption_dict_tuples]
        else:
            caption_feat_dict[name] = torch.stack([each[name] for each in caption_dict_tuples], 0)

    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    output = {'vis_feats': vis_feat_dict, 'vis_muti_feat': vis_muti_feat,
              'vis_frame_feat_dict': vis_frame_feat_dict,
              'captions': caption_feat_dict, 'captions_task2': caption_labels_task2,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids}
    return output


def collate_pair_subset(data):
    data.sort(key=lambda x: len(TextTool.tokenize(x[1])), reverse=True)
    vis_feats, captions, captions_task2, idxs, vis_ids, cap_ids = list(zip(*data))
    vis_feats = torch.stack(vis_feats, 0)
    idxs = list(idxs)  # 如果是 pin_memory = False 必须要这样,否则evaluation.py 无法执行
    idxs = np.array(idxs) - np.array(idxs).min()
    output = {'vis_feats': vis_feats, 'captions': captions, 'captions_task2': captions_task2,
              'idxs': idxs, 'vis_ids': vis_ids, 'cap_ids': cap_ids}
    return output


class ImageDataset(data.Dataset):
    def __init__(self, id_path_file, oversample=False, sample_frame=8, sample_type='uniform'):
        """
        :param id_path_file: similar to "video5027_200  ImageData/video5027/video5027_200.jpg \n ..."
        :param oversample:
        :param sample_type: ['uniform', 'random', ...]
        # 均匀取 sample_frame 帧，随机选 sample_frame 帧.
        """

        self.sample_frame = sample_frame
        self.sample_type = sample_type
        collection_path = os.path.dirname(id_path_file)
        data = list(map(str.strip, open(id_path_file).readlines()))
        self.image_ids = [x.split()[0] for x in data]
        self.file_names = [os.path.join(collection_path, x.split()[1]) for x in data]

        # Get the mapping of video_id to image path
        self.video2Image_path = {}
        for each in data:
            image_id, image_path = each.split()[0], os.path.join(collection_path, each.split()[1])
            video_id = "_".join(image_id.split('_')[:-1])
            if video_id == '':
                video_id = image_id  # this is image dataset
            if video_id not in self.video2Image_path:
                self.video2Image_path[video_id] = []
            self.video2Image_path[video_id].append(image_path)
        # rank the image_paths
        for video_id in self.video2Image_path:
            try:
                self.video2Image_path[video_id].sort(key=lambda x: int(os.path.basename(x).split('.')[0].split("_")[-1]))
            except ValueError:
                self.video2Image_path[video_id].sort(
                    key=lambda x: os.path.basename(x).split('.')[0].split("_")[-1])


        # _, self.preprocess_clip = clip.load("ViT-B/32", device="cpu")
        self.preprocess_clip_toTensor = Compose([
            # Resize(256),
            # CenterCrop(224),
            Resize(512),
            CenterCrop(512),
            lambda image: image.convert("RGB"),
            ToTensor(),
        ])
        self.meta = {'mean': [0.48145466, 0.4578275, 0.40821073], 'std': [0.26862954, 0.26130258, 0.27577711]}
        self.preprocess_clip_fromTensor = Compose([
            Resize(224),
            Normalize(self.meta['mean'], self.meta['std']),
    ])

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        file_name = self.file_names[index]
        image = PIL.Image.open(file_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.preprocess_clip_toTensor(image)
        image = self.preprocess_clip_fromTensor(image)
        return image_id, image

    def __len__(self):
        return len(self.image_ids)

    def get_image_from_videoid_with_clip(self, video_id):
        images = None  # (image_num, 3, 224, 224)
        image_ids = []
        frame_indexs = []  # The index of chosen frames
        # video_id missing
        if video_id not in self.video2Image_path:
            print(video_id, "is missing in id.imagepath.txt file")
            image_ids = ["%s_%d" % (video_id, 0) for each in range(0, self.sample_frame)]
            images = torch.ones((self.sample_frame, 3, 224, 224))
            return image_ids, images

        if self.sample_type == 'uniform' or len(self.video2Image_path[video_id]) <= self.sample_frame:
            frame_indexs = np.linspace(0, len(self.video2Image_path[video_id]) - 1,
                                       self.sample_frame, dtype=int)
        elif self.sample_type == 'random':
            frame_indexs = random.sample(list(np.arange(0, len(self.video2Image_path[video_id]))), self.sample_frame)
            frame_indexs.sort()
        else:
            raise Exception("Sample_type is not implemented!")


        for index in frame_indexs:
            each = self.video2Image_path[video_id][index]
            try:
                image = self.preprocess_clip_toTensor(PIL.Image.open(each))
                image = self.preprocess_clip_fromTensor(image).unsqueeze(0)  # (1, 3, 224, 224)

            except Exception as e:
                print(e)
                image = torch.ones((1, 3, 224, 224))

            if images is None:
                images = image
            else:
                images = torch.cat((images, image), dim=0)
            image_ids.append(os.path.basename(each).split('.')[0])

        return frame_indexs, images

    def get_image_from_image_id(self, image_names):
        """

        :param image_names: [video8883_325.jpg / video8883_325, ...]
        :return:
        """
        image_paths = []
        images = None
        for image_name in image_names:
            video_id = "_".join(image_name.split('_')[:-1])
            for each in self.video2Image_path[video_id]:
                if each[-4:] not in image_name:
                    image_name = image_name + each[-4:]
                if image_name in each:
                    image_paths.append(each)
        try:
            assert len(image_paths) == len(image_names)
        except Exception as e:
            print(e)
            exit(1)

        for image_path in image_paths:
            image = self.preprocess_clip_toTensor(PIL.Image.open(each))
            image = self.preprocess_clip_fromTensor(image).unsqueeze(0)  # (1, 3, 224, 224)
            if images is None:
                images = image
            else:
                images = torch.cat((images, image), dim=0)

        return images, image_paths

    def _get_imagePreTensor_from_videoid_with_clip(self, video_id):
        images_preprocess = None  # (image_num, 3, 224, 224)

        if self.sample_type == 'uniform' or len(self.video2Image_path[video_id]) <= self.sample_frame:
            frame_indexs = np.linspace(0, len(self.video2Image_path[video_id]) - 1,
                                       self.sample_frame, dtype=int)
        elif self.sample_type == 'random':
            frame_indexs = random.sample(list(np.arange(0, len(self.video2Image_path[video_id]))), self.sample_frame)
            frame_indexs.sort()
        else:
            raise Exception("Sample_type is not implemented!")


        for index in frame_indexs:
            each = self.video2Image_path[video_id][index]
            try:
                image_preprocess = self.preprocess_clip_toTensor(PIL.Image.open(each)).unsqueeze(0)  # (1, 3, 224, 224)
            except Exception as e:
                print(e)
                image_preprocess = torch.ones((1, 3, 224, 224))

            if images_preprocess is None:
                images_preprocess = image_preprocess
            else:
                images_preprocess = torch.cat((images_preprocess, image_preprocess), dim=0)

        return frame_indexs, images_preprocess

    def _get_imagePostTensor_from_imagePreTensor_with_clip(self, images_preprocess):
        # for each in range(images_preprocess.shape[0]):
        #     images_preprocess[each] = self.preprocess_clip_fromTensor(images_preprocess[each])
        images_preprocess = self.preprocess_clip_fromTensor(images_preprocess)
        return images_preprocess


class VisionDataset(data.Dataset):
    """
    得到视频的 Dataset
    """

    def __init__(self, params):
        # 视频特征字典
        self.vis_feat_file = None  # 默认无 视频特征
        if params['vis_feat_files'] is not None:
            if len(params['vis_feat_files']) > 0:
                self.vis_feat_file_dict = params['vis_feat_files']
                self.vis_feat_file = self.vis_feat_file_dict[list(self.vis_feat_file_dict.keys())[0]]

        # 视频按帧特征字典
        self.multi_frame_feat = False  # 默认无 视频按帧特征
        # 帧级别格式: frame_name tensors ...
        if 'vis_frame_feat_dicts' in params:
            if params['vis_frame_feat_dicts'] is not None:
                self.max_frame = params['max_frame']  # 最大出现帧数
                self.multi_frame_feat = True
                self.vis_frame_feat_dict = params['vis_frame_feat_dicts']
                self.visual_id2frame_id_dict = self.__get_visual_id2frame_id_dict__(self.vis_frame_feat_dict)

        # self.vis_ids = self.vis_feat_file.names if params.get('vis_ids', None) is None else params['vis_ids']
        self.vis_ids = params.get('vis_ids', None)

        # Faster-rcnn 特征
        self.muti_feat = False  # 默认无 视频Faster-rcnn特征
        if 'vis_muti_feat_dicts' in params:
            if params['vis_muti_feat_dicts'] is not None:
                self.vis_muti_feat_dicts = params['vis_muti_feat_dicts']
                self.muti_feat = True
        self.length = len(self.vis_ids)

        # 原始帧数据
        self.frame_loader = False
        if 'config' in params:
            if params['config'].frame_loader:
                self.frame_loader = True
                if 'sample_type' in params:
                    sample_type = params['sample_type']
                else:
                    sample_type = 'uniform'
                self.ImageDataset = ImageDataset(
                    params['frame_id_path_file'],
                    sample_frame=params['config'].sample_frame,
                    sample_type=sample_type,
                )

    def __get_visual_id2frame_id_dict__(self, vis_frame_feat_dict):
        visual_id2frame_id_dict = {}
        for each in vis_frame_feat_dict:
            frameid_list = vis_frame_feat_dict[each].names
            visual_id2frame_id_dict[each] = {}
            # 得到 videoid 对应的 frame id
            for frame_id in frameid_list:
                video_id = "_".join(frame_id.split('_')[0:-1])
                if video_id not in visual_id2frame_id_dict[each]:
                    visual_id2frame_id_dict[each][video_id] = []
                visual_id2frame_id_dict[each][video_id].append(frame_id)

        # rank the frame_id
        for each_name in visual_id2frame_id_dict:
            for each_video_id in visual_id2frame_id_dict[each_name]:
                visual_id2frame_id_dict[each_name][each_video_id].sort(key=lambda x: int(x.split("_")[-1]))
        return visual_id2frame_id_dict

    def __getitem__(self, index):
        vis_id = self.vis_ids[index]
        visual_output = self.get_feat_by_id(vis_id)
        vis_tensor_dict = visual_output['vis_tensor_dict']
        vis_frame_tensor_dict = visual_output['vis_frame_tensor_dict']
        vis_origin_frame_tensor = visual_output['vis_origin_frame_tensor']

        return vis_tensor_dict, index, vis_id, vis_frame_tensor_dict, vis_origin_frame_tensor

    def get_feat_by_id(self, vis_id):
        # 视频特征字典
        vis_tensor_dict = {}
        if self.vis_feat_file is not None:
            for each in self.vis_feat_file_dict.keys():
                # try:
                #     vis_tensor_dict[each] = torch.Tensor(self.vis_feat_file_dict[each].read_one(vis_id))
                # except Exception as e:
                #     print(e)
                #     print('Vis id: ', vis_id)
                #     print('feature name: ', each)
                vis_tensor_dict[each] = torch.Tensor(self.vis_feat_file_dict[each].read_one(vis_id))


        # 视频按帧特征字典
        vis_frame_tensor_dict = {}
        if self.multi_frame_feat:
            for featname in self.visual_id2frame_id_dict:
                video_frame_list = self.visual_id2frame_id_dict[featname][vis_id]
                if len(video_frame_list) > self.max_frame:
                    video_frame_list = video_frame_list[0:self.max_frame]
                vis_frame_tensor_dict[featname] = torch.Tensor(
                    self.vis_frame_feat_dict[featname].read(video_frame_list)[1])
        # 视频原始帧信息
        vis_origin_frame_tensor = None
        if self.frame_loader:
            frame_ids, vis_origin_frame_tensor = self.ImageDataset.get_image_from_videoid_with_clip(vis_id)

        vis_output_dict = {'vis_tensor_dict': vis_tensor_dict,
                           'vis_frame_tensor_dict': vis_frame_tensor_dict,
                           'vis_muti_feat_tensor': None,
                           'vis_origin_frame_tensor': vis_origin_frame_tensor,
                           }
        if self.muti_feat:
            vis_muti_feat_list = self.vis_muti_feat_dicts[vis_id]
            vis_muti_feat_tensor = torch.Tensor(vis_muti_feat_list)
            vis_output_dict['vis_muti_feat_tensor'] = vis_muti_feat_tensor

        return vis_output_dict

    def __len__(self):
        return self.length


class TextDataset(data.Dataset):
    """
    得到 文字的 Dataset, self.get_caption_by_id(cap_id)可以得到第几个 caption.
    """

    def __init__(self, params, task3=False, capfile_task2=False, capfile_task3=False):
        capfile = params['capfile']

        # 读取预先计算特征
        try:
            self.pre_calculate_feat_files = self.get_precalculate_file(params['config'],
                                                                       os.path.dirname(params['capfile']))
        except Exception as e:
            print("Read the pre-calculated feature error!", e)
            self.pre_calculate_feat_files = {}

        if task3 and 'CLIP_encoding' in self.pre_calculate_feat_files:
            self.pre_calculate_feat_files.pop('CLIP_encoding')
        if capfile_task2:
            capfile = params['capfile_task2']
        elif capfile_task3:
            capfile = params['capfile_task3']

        self.capfile_task3 = capfile_task3
        self.capfile_task2 = capfile_task2

        self.captions = {}
        self.cap_ids = []
        if capfile_task3:
            # mask 0：negtive 1：positive
            self.mask_task3 = {}
            with open(capfile, 'r') as reader:
                lines = reader.readlines()
                random.shuffle(lines)
                for line in lines:
                    cap_idfull, caption = line.strip().split(None, 1)
                    cap_id, cap_id2 = cap_idfull.split('#')
                    cap_id = cap_id + '#' + cap_id2.split("F")[0]
                    if "p" in cap_idfull:
                        self.mask_task3[cap_id] = 1

                    else:
                        self.mask_task3[cap_id] = 0

                    if cap_id not in self.captions:
                        self.captions[cap_id] = [caption]
                        self.cap_ids.append(cap_id)
                    else:
                        self.captions[cap_id].append(caption)
        else:
            with open(capfile, 'r') as reader:
                for line in reader.readlines():
                    if line.strip() == "":
                        continue
                    if len(line.strip().split(None, 1)) < 2:
                        cap_id = line.strip().split(None, 1)[0]
                        caption = ''
                    else:
                        cap_id, caption = line.strip().split(None, 1)
                    self.captions[cap_id] = caption
                    self.cap_ids.append(cap_id)

        # visId2captionId = {}
        self.visId2captionId = {}
        for each in self.cap_ids:
            if each.split('#')[0] not in self.visId2captionId:
                self.visId2captionId[each.split('#')[0]] = []
            self.visId2captionId[each.split('#')[0]].append(each)

        self.length = len(self.cap_ids)

    def get_precalculate_file(self, config, TextPath):
        precalculate_feat_files = {}
        for each_encoding_name in config.text_encoding:
            if 'no' in config.text_encoding[each_encoding_name]['name']:
                continue
            each_encoding_dict = config.text_encoding[each_encoding_name]
            if 'dir_name' in each_encoding_dict:
                precalculate_feat_files[each_encoding_name] = BigFile(
                    os.path.join(TextPath, each_encoding_dict['dir_name']))
        return precalculate_feat_files

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        caption_dict = self.get_caption_dict_by_id(cap_id)
        return caption_dict, index, cap_id

    def get_caption_dict_by_id(self, cap_id):
        caption_dict = {}

        pop_list = []
        for each in self.pre_calculate_feat_files:

            try:
                caption_dict[each] = torch.Tensor(self.pre_calculate_feat_files[each].read_one(cap_id))
            except Exception as e:
                print("{}, Read the pre-calculated feature error!".format(each), e)
                pop_list.append(each)
        for each in pop_list:
            self.pre_calculate_feat_files.pop(each)

        caption_dict["caption"] = self.captions[cap_id]
        return caption_dict

    def get_falsecaption_by_id(self, cap_id):
        caption_dict = {}
        if cap_id in self.captions:
            caption = self.captions[cap_id]
            caption = random.choice(caption)
            mask = self.mask_task3[cap_id]
            pop_list = []
            for each in self.pre_calculate_feat_files:
                pop_list.append(each)
            for each in pop_list:
                self.pre_calculate_feat_files.pop(each)

            caption_dict["caption"] = caption
        else:
            mask = -1
            caption_dict["caption"] = None

        return caption_dict, mask

    def __len__(self):
        return self.length


class PairDataset(data.Dataset):
    """
    得到 vis_feat, caption, capfile_task2, index, vis_id, cap_id
    """

    def __init__(self, params):
        """

        :param params: params['vis_muti_feat_dicts']: Faster-rcnn 特征
        """
        self.params = params
        self.visData = VisionDataset(params)

        if params['capfile_task2'] is None:
            self.txtData_task2 = None
        else:
            self.txtData_task2 = TextDataset(params, capfile_task2=True)
        if params['capfile_task3'] is None:
            self.txtData_task3 = None
            self.txtData = TextDataset(params)
        else:
            self.txtData = TextDataset(params, task3=True)
            self.txtData_task3 = TextDataset(params, task3=True, capfile_task3=True)
            self.txtData_augmentation = self.get_negation_augumentation(self.txtData.captions,
                                                                        self.txtData_task3.mask_task3)
        self.cap_ids = self.txtData.cap_ids
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        vis_id = self.get_visId_by_capId(cap_id)

        caption_dict = self.txtData.get_caption_dict_by_id(cap_id)  # cap_id: 'video7768#14'
        # 多视频特征
        vis_output_dict = self.visData.get_feat_by_id(vis_id)
        vis_feat_dict = vis_output_dict['vis_tensor_dict']
        vis_frame_feat_dict = vis_output_dict['vis_frame_tensor_dict']  # 多视频frame特征
        # 原始视频帧
        vis_origin_frame_tensor = vis_output_dict['vis_origin_frame_tensor']

        vis_muti_feat = None
        if vis_output_dict['vis_muti_feat_tensor'] is not None:
            vis_muti_feat = vis_output_dict['vis_muti_feat_tensor']

        # task2
        if self.txtData_task2 is None:
            caption_labels_task2 = None
        else:
            caption_labels_task2 = self.txtData_task2.get_caption_dict_by_id(
                vis_id)  # 由于 task2 名词去掉了‘#’，可以使用video_id 来查找
        if self.txtData_task3 is None:
            caption_labels_task3 = None
            mask_task3 = None
        else:
            caption_labels_task3, mask_task3 = self.txtData_task3.get_falsecaption_by_id(cap_id)

            if mask_task3 == 1:
                caption = random.choice(self.txtData_augmentation[cap_id])

        output_dict = {

        }
        return vis_feat_dict, caption_dict, vis_muti_feat, caption_labels_task2, index, vis_id, cap_id, \
               vis_frame_feat_dict, caption_labels_task3, mask_task3, vis_origin_frame_tensor

    def get_visId_by_capId(self, cap_id):
        vis_id = cap_id.split('#', 1)[0]
        return vis_id

    def get_negation_augumentation(self, captions, mask):
        dataset = {}
        for capid, neg_mask in mask.items():
            if neg_mask:
                dataset[capid] = negation_augumentation(captions[capid])
        return dataset

    def __len__(self):
        return self.length


def vis_provider(params):
    data_loader = DataLoaderX(dataset=VisionDataset(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              sampler=params.get('sampler', None),
                                              collate_fn=collate_vision)
    return data_loader


def txt_provider(params):
    data_loader = DataLoaderX(dataset=TextDataset(params, task3=params.get('task3')),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              sampler=params.get('sampler', None),
                                              collate_fn=collate_text)
    return data_loader


def pair_provider(params):
    data_loader = DataLoaderX(dataset=PairDataset(params),
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              sampler=params.get('sampler', None),
                                              collate_fn=collate_pair,
                                              )
    return data_loader


def pair_provider_subset(params, induce):
    subset = torch.utils.data.dataset.Subset(PairDataset(params), induce)
    if params['sampler'] is not None:
        params['sampler'] = torch.utils.data.distributed.DistributedSampler(subset, shuffle=True)
        print(params)
    data_loader = DataLoaderX(subset,
                                              batch_size=params.get('batch_size', 1),
                                              shuffle=params.get('shuffle', False),
                                              pin_memory=params.get('pin_memory', False),
                                              num_workers=params.get('num_workers', 0),
                                              sampler=params.get('sampler', None),
                                              collate_fn=collate_pair_subset)
    return data_loader


    return data_loader

if __name__ == '__main__':
    import os

    data_path = '/data2/hf/VisualSearch'
    collection = 'tgif-msrvtt10k'
    vid_feat = 'mean_resnext101_resnet152'
    vid_feat_dir = os.path.join(data_path, collection, 'FeatureData', vid_feat)

    vis_loader = vis_provider({'vis_feat_files': vid_feat_dir, 'batch_size': 100, 'num_workers': 2})

    for i, (feat_vecs, idxs, vis_ids) in enumerate(vis_loader):
        print(i, feat_vecs.shape, len(idxs))
        break

    capfile = os.path.join(data_path, collection, 'TextData', '%s.caption.txt' % collection)

    txt_loader = txt_provider({'capfile': capfile, 'batch_size': 100, 'num_workers': 2})

    for i, (captions, idxs, cap_ids) in enumerate(txt_loader):
        print(i, captions, len(cap_ids))
        print([len(cap) for cap in captions])
        break

    capfile_task2 = os.path.join(data_path, collection, 'TextData', '%s.caption.nouns.txt' % collection)
    pair_loader = pair_provider({'vis_feat_files': vid_feat_dir, 'capfile': capfile,
                                 'capfile_task2': capfile_task2, 'batch_size': 100, 'num_workers': 2, 'shuffle': True})
    for i, (vis_feats, captions, captions_task2, idxs, vis_ids, cap_ids) in enumerate(pair_loader):
        print(i, vis_feats.shape, captions[:10], len(cap_ids))
        print("next")
        print(idxs)
        print(vis_ids)
        print(cap_ids)
        print(captions_task2)
        # print [len(cap) for cap in captions]
        break
