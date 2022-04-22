# coding=utf-8
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys
import time
import json
import argparse
import random
import re

import numpy as np

import util
import evaluation
import data_provider as data
import model.TTH as tth
from common import *
from loss import l2norm
from model.model import get_model
from bigfile import BigFile
from generic_utils import Progbar
import matplotlib.pyplot as plt
from torchvision import transforms
import importlib


def parse_args():
    parser = argparse.ArgumentParser('TTH attack')
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,
                        help='path to datasets. (default: %s)'%ROOT_PATH)
    parser.add_argument('testCollection', type=str, default='flickr30ktest_add_ad',
                        help='test collection')
    parser.add_argument('--attack_trainData', type=str, default='flickr30ktrain',
                        help='train collection')
    parser.add_argument('model_path', type=str, default='None',
                        help='Path to load the model.')
    parser.add_argument('sim_name', type=str,
                        help='sub-folder where computed similarities are saved')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1],
                        help='overwrite existed vocabulary file. (default: 0)')
    parser.add_argument('--query_sets', type=str, default='flickr30ktest_add_ad.caption.txt',
                        help='test query sets,  flickr30ktest_add_ad.caption.txt.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of a predicting mini-batch')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of data loader workers.')
    parser.add_argument("--device", default=0, type=str, help="cuda:n or cpu (default: 0)")
    parser.add_argument('--config_name', type=str, default='TTH.CLIPEnd2End_adjust',
                        help='model configuration file. (default: TTH.CLIPEnd2End_adjust')
    parser.add_argument('--parm_adjust_config', type=str, default='None',
                        help='the config parm you need to set. (default: None')

    parser.add_argument('--task3_caption', type=str, default='no_task3_caption',
                        help='the suffix of task3 caption.(It looks like "caption.false ") Default is false.')

    args = parser.parse_args()
    return args


def get_eval_from_matrix(t2i_matrix: np.ndarray, vis_ids, txt_ids, i2t_matrix=None, t2multi_v=False):
    """

    :param t2i_matrix:
    :param vis_ids:
    :param txt_ids:
    :param i2t_matrix:
    :param t2multi_v: 一个文本对应多个图片
    :return:
    """
    evaluation_result = {}
    if t2multi_v:
        vis_ids = [vis_id.split('#')[0] for vis_id in vis_ids]

    if t2i_matrix is not None:
        inds = np.argsort(t2i_matrix, axis=1)

        label_matrix = np.zeros(inds.shape)

        for index in range(inds.shape[0]):
            ind = inds[index][::-1]
            gt_index = np.where(np.array(vis_ids)[ind] == txt_ids[index].split('#')[0])[0]
            label_matrix[index][gt_index] = 1

        (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
        evaluation_result['eval_tuple'] = (r1, r5, r10, medr, meanr, mir, mAP)
    if i2t_matrix is not None:
        inds = np.argsort(i2t_matrix, axis=1)
        label_matrix = np.zeros(inds.shape)
        txt_ids = [txt_id.split('#')[0] for txt_id in txt_ids]
        for index in range(inds.shape[0]):
            ind = inds[index][::-1]
            label_matrix[index][np.where(
                np.array(txt_ids)[ind] == vis_ids[index])[0]] = 1
        (r1, r5, r10, medr, meanr, mir, mAP) = evaluation.eval(label_matrix)
        evaluation_result['eval_tuple_i2t'] = (r1, r5, r10, medr, meanr, mir, mAP)

    return evaluation_result


def concept_in_caption(target_concept, caption):
    if (" %s " % target_concept in caption) or \
            (" %s." % target_concept in caption) or \
            ("%s " % target_concept in caption):
        return True
    elif (" %ss " % target_concept in caption) or \
            (" %ss." % target_concept in caption) or \
            ("%ss " % target_concept in caption):
        return True
    else:
        return False



def get_images2concept_attack(opt, config):
    def image2concept_attack_eval_recall(
            model, benign_vis_indexs, target_txt_ids, adversarial_tensors,
            vis_loader, txt_loader, benign_vis_tensors):
        """
        Trojan-horse attack，get test metrics。
        :param model:
        :param benign_vis_indexs:
        :param target_txt_ids: eg.['2682194299.jpg#0','2682194299.jpg#1']
        :param adversarial_tensors: [n, 3, 224, 224]. The visual tensor that not be normalized.
        :param vis_loader:
        :param txt_loader:
        :return:
        """
        # ----------------
        # vis_id_all = vis_loader.dataset.vis_ids
        # adversarial = vis_loader.dataset.ImageDataset.get_image_from_videoid_with_clip(vis_id_all[5])[1]  # 1, 3, 224, 224
        # -----------------

        # post precess for adversarial_tensor
        adversarial_tensors = vis_loader.dataset.ImageDataset._get_imagePostTensor_from_imagePreTensor_with_clip(
            adversarial_tensors)
        benign_vis_tensors = vis_loader.dataset.ImageDataset._get_imagePostTensor_from_imagePreTensor_with_clip(
            benign_vis_tensors)
        benign_vis_ids = []
        for each in benign_vis_indexs:
            benign_vis_ids.append(vis_loader.dataset.vis_ids[each])
        # Get targeted visual index
        target_vis_index = []
        target_vis_ids = set([each.split("#")[0] for each in target_txt_ids])
        for i, each in enumerate(vis_loader.dataset.vis_ids):
            if each in target_vis_ids:
                target_vis_index.append(i)
        target_vis_ids = list(np.array(vis_loader.dataset.vis_ids)[target_vis_index])
        # Get targeted text index
        target_txt_index = []
        for i, each in enumerate(txt_loader.dataset.cap_ids):
            if each in target_txt_ids:
                target_txt_index.append(i)

        # Get txt_ids and caption_feat_dict
        benign_txt_ids = []
        for each in benign_vis_ids:
            benign_txt_ids.extend(txt_loader.dataset.visId2captionId[each])
        benign_caption_feat_dict = {
            'caption': [txt_loader.dataset.get_caption_dict_by_id(each)["caption"] for each in benign_txt_ids]}
        target_caption_feat_dict = {
            'caption': [txt_loader.dataset.get_caption_dict_by_id(each)["caption"] for each in target_txt_ids]}
        print()

        with torch.no_grad():
            # Get the embedding of texts
            benign_txt_embs = model.text_features(benign_caption_feat_dict['caption'])
            target_txt_embs = model.text_features(target_caption_feat_dict['caption'])
            text_all_embs = None
            text_all_ids = []
            pbar_text = Progbar(len(txt_loader.dataset))
            for i, (caption_feat_dict, txt_idxs, batch_txt_ids) in enumerate(txt_loader):
                pbar_text.add(len(batch_txt_ids))
                text_all_ids.extend(batch_txt_ids)
                txt_emb = model.clip_model(caption_feat_dict)['text_features'].cpu()
                text_all_embs = txt_emb if text_all_embs is None else \
                    torch.cat((text_all_embs, txt_emb), dim=0)

            # Get the embeddings of adversarial videos/images
            adversarial_embs = model.visual_features(adversarial_tensors)
            benign_vis_embs = model.visual_features(benign_vis_tensors)

            # Get the embeddings of all videos/images
            pbar_video = Progbar(len(vis_loader.dataset))
            video_all_embs = None
            video_idxs_list = []
            vis_all_ids = []
            for j, output_dict in enumerate(vis_loader):
                (vis_input, idxs, batch_vis_ids,
                 vis_frame_feat_dict, vis_origin_frame_tuple
                 ) = (
                    output_dict['vis_feat_dict'], output_dict['idxs'],
                    output_dict['vis_ids'], output_dict['vis_frame_feat_dict'],
                    output_dict['vis_origin_frame_tuple']
                )
                pbar_video.add(len(idxs))
                video_idxs_list.append(idxs)
                __vis_embs__ = model.clip_model(
                    caption_feat_dict=None, vis_origin_frame_tuple=vis_origin_frame_tuple,
                )['visual_features'].cpu()

                video_all_embs = __vis_embs__ if video_all_embs is None else \
                    torch.cat((video_all_embs, __vis_embs__), dim=0)

                vis_all_ids.extend(batch_vis_ids)

            str_return = ""
            result_list = []
            # ************************************************************************
            target_txt_emb_mean = target_txt_embs.mean(dim=0, keepdim=True)
            target_mean_cos = model.get_txt2vis_matrix(target_txt_emb_mean, target_txt_embs).mean().detach().data
            result_str = 'mean cos sim: %.3f ' % target_mean_cos
            print(result_str)
            str_return += '\n' + result_str
            result_list.append(float(target_mean_cos))

            # *************************** T to I ******************************************
            # R10 of truly relevant images w/o TTH
            target_mAP_sim_matrix = model.get_txt2vis_matrix(target_txt_embs, video_all_embs).cpu().data.numpy()
            (r1, r5, r10, medr, meanr, mir, mAP) = get_eval_from_matrix(target_mAP_sim_matrix, vis_all_ids, target_txt_ids)['eval_tuple']
            print('R10 of truly relevant images w/o TTH: %.3f ' % r10)
            str_return += '\n' + 'R10 of truly relevant images w/o TTH: %.3f ' % r10
            result_list.extend([mAP, r1, r5, r10])

            # R10 of truly relevant images w/ TTH
            adv_video_all_embs = torch.cat((video_all_embs.cpu(), adversarial_embs.cpu()), dim=0)
            target_mAP_sim_matrix = model.get_txt2vis_matrix(target_txt_embs, adv_video_all_embs).cpu().data.numpy()
            adv_vis_all_ids = vis_all_ids + ["adversarialVideo#%d" % each for each in range(len(adversarial_embs))]
            (r1, r5, r10, medr, meanr, mir, mAP) = get_eval_from_matrix(target_mAP_sim_matrix, adv_vis_all_ids, target_txt_ids)['eval_tuple']
            print('R10 of truly relevant images w/ TTH: %.3f ' % r10)
            str_return += '\n' + 'targeted query get target result mAP: %.3f ' % mAP
            result_list.extend([mAP, r1, r5, r10])

            # R10 of Benign images w/o TTH
            adv_video_all_embs = torch.cat((video_all_embs.cpu(), benign_vis_embs.cpu()), dim=0)
            target_mAP_sim_matrix = model.get_txt2vis_matrix(target_txt_embs, adv_video_all_embs).cpu().data.numpy()
            adv_vis_all_ids = vis_all_ids + ["adversarialVideo#%d" % each for each in range(len(benign_vis_embs))]
            adv_target_txt_ids = ["adversarialVideo#%d" % each for each in range(len(target_txt_ids))]
            (r1, r5, r10, medr, meanr, mir, mAP) = get_eval_from_matrix(target_mAP_sim_matrix, adv_vis_all_ids,
                                       adv_target_txt_ids, t2multi_v=True)['eval_tuple']
            print('R10 of Benign images w/o TTH: %.3f ' % r10)
            str_return += '\n' + 'targeted query get target result mAP: %.3f ' % mAP
            result_list.extend([mAP, r1, r5, r10])

            # R10 of Novel images w/ TTH
            adv_video_all_embs = torch.cat((video_all_embs.cpu(), adversarial_embs.cpu()), dim=0)
            adv_vis_all_ids = vis_all_ids + ["adversarialVideo#%d" % each for each in range(len(adversarial_embs))]
            adv_target_txt_ids = ["adversarialVideo#%d" % each for each in range(len(target_txt_ids))]
            adv_sim_matrix = model.get_txt2vis_matrix(target_txt_embs, adv_video_all_embs).cpu().data.numpy()
            (r1, r5, r10, medr, meanr, mir, mAP) = get_eval_from_matrix(adv_sim_matrix, adv_vis_all_ids,
                                       adv_target_txt_ids, t2multi_v=True)['eval_tuple']

            print('R10 of Novel images w/ TTH: %.3f ' % r10)
            str_return += '\n' + 'adv-t mAP: %.3f ' % mAP
            result_list.extend([mAP, r1, r5, r10])

            eval_output_dict = {
                'video_all_embs': video_all_embs.cpu(),
                'adversarial_embs': adversarial_embs.cpu(),
                'vis_all_ids': vis_all_ids,
                'target_txt_ids': target_txt_ids,
                'target_txt_embs': target_txt_embs.cpu()
            }

            return str_return, result_list, eval_output_dict

    rootpath = opt.rootpath
    testCollection = opt.testCollection
    # cuda number
    device = torch.device("cuda:{}".format(opt.device)
                          if (torch.cuda.is_available() and opt.device != "cpu") else "cpu")

    model_name = config.model_name

    if opt.task3_caption == "no_task3_caption":
        task3 = False
    else:
        task3 = True
    if hasattr(config, 't2v_w2v') and hasattr(config.t2v_w2v, 'w2v'):
        w2v_feature_file = os.path.join(rootpath, 'word2vec', 'flickr', 'vec500flickr30m', 'feature.bin')
        config.t2v_w2v.w2v.binary_file = w2v_feature_file

    # Construct the model
    model = get_model(model_name, device, config)


    if config.clip_opt['frozen'] == False:
        # load the pretrained model if clip is not frozen
        # Load checkpoint
        logger.info('loading model...')
        resume_file = os.path.join(opt.model_path)
        if '~' in resume_file:
            resume_file = resume_file.replace('~', os.path.expanduser('~'))
            opt.model_path = resume_file
        if not os.path.exists(resume_file):
            logging.info(resume_file + '\n not exists.')
            sys.exit(0)
        checkpoint = torch.load(resume_file, map_location='cpu')
        # checkpoint.pop('config'); checkpoint.pop('opt')
        # torch.save( checkpoint,
        #     resume_file.replace("model_best.pth", "model_best_small.pth"))
        best_perf = checkpoint['best_perf']
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {}, best_perf {})"
              .format(resume_file, epoch, best_perf))

    model = model.to(device)
    vis_feat_files = {y: BigFile(os.path.join(rootpath, testCollection, 'FeatureData', y))
                                   for y in config.vid_feats}
    # 视频帧特征文件
    vis_frame_feat_dicts = None

    vis_ids = list(map(str.strip, open(os.path.join(rootpath, testCollection, 'VideoSets', testCollection + '.txt'))))
    # 视频帧文件
    if config.frame_loader:
        frame_id_path_file = os.path.join(rootpath, testCollection, 'id.imagepath.txt')
    else:
        frame_id_path_file = None

    vis_loader = data.vis_provider({'vis_feat_files': vis_feat_files, 'vis_ids': vis_ids, 'pin_memory': False,
                                    'vis_frame_feat_dicts': vis_frame_feat_dicts,
                                    'max_frame': config.max_frame,
                                    'sample_type': config.frame_sample_type_test,
                                    'config': config,
                                    'frame_id_path_file': frame_id_path_file,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers})

    query_set = opt.query_sets.split(',')[0]

    output_dir = os.path.join(rootpath, testCollection, 'Attack', query_set, opt.sim_name)

    util.makedirs(output_dir)  # 创建文件夹


    capfile = os.path.join(rootpath, testCollection, 'TextData', query_set)
    # load text data
    txt_loader = data.txt_provider({'capfile': capfile, 'pin_memory': False, 'config': config,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers, 'task3': task3})
    capfile_attack_train = os.path.join(rootpath, opt.attack_trainData, 'TextData', "%s.caption.txt" % opt.attack_trainData)
    txt_loader_attack = data.txt_provider({'capfile': capfile_attack_train, 'pin_memory': False, 'config': config,
                                    'batch_size': opt.batch_size, 'num_workers': opt.num_workers, 'task3': task3})

    # ******************************萌萌哒分界线****************************************
    # Attack
    whether_plt = False
    # copy arguments
    train_scales, iters, lr, \
    lam, sigma_blur, mode, variant, patch_ratio, only_keyword= \
        config.attack_scales, config.attack_iters, config.attack_lr, \
        config.attack_lam, config.attack_sigma_blur, config.attack_mode, \
        config.attack_variant, config.patch_ratio, config.only_keyword
    im_size = {'flick30k': 1024}
    scale_factors = [x / im_size['flick30k'] for x in train_scales]
    # meta data
    model.meta = vis_loader.dataset.ImageDataset.meta
    model.meta['architecture'] = 'CLIP'
    model.norm = l2norm

    # log file
    log = open(output_dir + "/log_" + 'attack' + ".txt", 'a')
    log_result = open(output_dir + "/log_" + 'attack_result' + ".txt", 'a+')
    log_result.write(str('\n' + time.asctime(time.localtime(time.time()))) + '\n')
    log_result.close()
    # params for rerun if no converge
    max_trials, multiply_rate_iters, divide_rate_lr = 10, 2, 5

    # load target (query) image
    benign_vis_index = []
    for i, each in enumerate(vis_loader.dataset.vis_ids):
        if re.sub(r'[0-9]+', '', each) == "ad.jpg":
            benign_vis_index.append((i, each))
    benign_vis_index.sort(key=lambda x: int(x[1].split('.')[0][2:]))
    benign_vis_index = [each[0] for each in benign_vis_index]
    benign_vis_index = np.array(benign_vis_index[0:20])  # ad index
    target_concepts = "jacket dress floor female motorcycle policeman cow waiter swimming reading run dancing floating smiling climbing feeding blue front little green yellow pink navy maroon"
    target_concepts = [each for each in target_concepts.strip().split()]

    for target_concept in target_concepts:
        try:
            log_result = open(output_dir + "/log_" + 'attack_result' + ".txt", 'a+')
            print("target_concept: ", target_concept)

            if only_keyword:
                print("Only input one keyword as target_caption_feature. ")
                target_caption_features = model.text_features([target_concept, ]).cpu()
                target_caption_feature = target_caption_features.mean(dim=0, keepdim=True)
            else:
                target_caption_dict_train = {}  # 训练集中包含 concept 的 caption
                with torch.no_grad():
                    for i, txt_id in enumerate(txt_loader_attack.dataset.captions):
                        caption = txt_loader_attack.dataset.captions[txt_id]
                        caption = caption.lower()
                        if concept_in_caption(target_concept, caption):
                            target_caption_dict_train[txt_id] = txt_loader_attack.dataset.captions[txt_id]
                    if len(target_caption_dict_train) >= 500:
                        target_caption_features = model.text_features(random.sample(list(target_caption_dict_train.values()), 500)).cpu()
                    else:
                        target_caption_features = model.text_features(list(target_caption_dict_train.values())).cpu()
                    target_caption_feature = target_caption_features.mean(dim=0, keepdim=True)

            target_caption_dict = {}  # 测试集中包含 concept 的 caption
            with torch.no_grad():
                for i, txt_id in enumerate(txt_loader.dataset.captions):
                    caption = txt_loader.dataset.captions[txt_id]
                    caption = caption.lower()
                    if concept_in_caption(target_concept, caption):
                        target_caption_dict[txt_id] = txt_loader.dataset.captions[txt_id]

            benign_vis_tensors = None
            benign_vis_ids = []
            for each in benign_vis_index:
                benign_vis_id = vis_loader.dataset.vis_ids[each]
                benign_vis_ids.append(benign_vis_id)
                temp = vis_loader.dataset.ImageDataset._get_imagePreTensor_from_videoid_with_clip(benign_vis_id)[1]  # 未归一化的原始图片
                benign_vis_tensors = temp if benign_vis_tensors is None else \
                    torch.cat((benign_vis_tensors, temp), dim=0)


            # attack
            t = time.time()
            trials = 0
            converged = False
            while not converged and trials < max_trials:
                alr = lr / divide_rate_lr**trials  # reducing lr after every failure
                aiters = int(iters * multiply_rate_iters**trials)  # increase iterations after every failure

                adversarial_tensors, loss_perf, loss_distort, converged = \
                    tth.tth_patch_multi_carrier([model, ], scale_factors, target_caption_feature, benign_vis_tensors,
                                                mode=mode, num_steps=aiters, lr=alr, lam=lam, sigma_blur=sigma_blur, verbose=True,
                                                device=device, target_type='embedding', patch_ratio=patch_ratio)

                trials += 1
            adversarial_tensors = adversarial_tensors.cpu()
            # time and log
            log.write("performance loss  {:6f} distortion loss {:6f} total loss {:6f}\n".format(
                loss_perf.item(), (loss_distort).item(), (loss_distort+loss_perf).item())); log.flush()
            if trials == max_trials:
                print("Failed...")
                log.write("Failed...")

            result_str, mAP_list, eval_output_dict = image2concept_attack_eval_recall(
                model, benign_vis_index, list(target_caption_dict.keys()),
                adversarial_tensors, vis_loader, txt_loader, benign_vis_tensors)

            for each in [target_concept] + [round(each, 3) for each in mAP_list]:
                log_result.write(str(each) + '\t')
            log_result.write('\n')
            log_result.close()
            print("Save to ", output_dir)


            if whether_plt:
                video_all_embs, adversarial_embs, vis_all_ids, target_txt_ids, target_txt_embs = \
                    (eval_output_dict['video_all_embs'], eval_output_dict['adversarial_embs'], eval_output_dict['vis_all_ids'],
                     eval_output_dict['target_txt_ids'], eval_output_dict['target_txt_embs'],
                     )
                # draw_tsne_text_and_vis(txt_loader, adversarial_embs, target_caption_features, target_concept, model, output_dir)
                # continue
                # ***************************************************************************
                # 画出攻击后图片的样子
                # benign_image = transforms.ToPILImage()(benign_vis_tensors[0]).convert('RGB')
                # one_target_image = list(target_caption_dict.keys())[0].split('#')[0]
                # one_target_image = vis_loader.dataset.ImageDataset._get_imagePreTensor_from_videoid_with_clip(one_target_image)[1]
                # one_target_image = transforms.ToPILImage()(one_target_image[0]).convert('RGB')
                #
                # fig, ax = plt.subplots(3, 2, dpi=300, figsize=(14, 7))
                # for i, each_image in enumerate([benign_image, one_target_image]):
                #     ax[0, i].imshow(each_image)
                # for i, each_image in enumerate(adversarial_tensors[0:4]):
                #     i = i+2
                #     adversarial_image = transforms.ToPILImage()(each_image).convert('RGB')
                #     print((i//2, i%2))
                #     ax[i//2, i%2].imshow(adversarial_image)
                #
                # ax[0,0].set_xlabel("benign_image")
                # ax[0,1].set_xlabel("one target_image \n target_concept: %s" % target_concept)
                # ax[1,0].set_xlabel("adversarial_image")
                #
                # fig.suptitle(result_str+'\n target_concept: %s' % target_concept)
                # plt.show()
                # adversarial_image.save('/data/hf/VisualSearch/flickr30k/test_%s.jpg' % target_concept)
                # # plt.savefig('/data/hf/VisualSearch/msrvtt10k/Frame_weights_plot/%s_%d_cam.jpg' % (video_id, caption_index))
                # plt.close()

                # ***************************************************************************

                # 画出攻击前后，adv-t-mAP 最大的两组 query 前10张检索结果
                # 首先找到攻击后 adv-t-mAP 最大的两个结果
                adv_mAP_list = []
                adv_video_all_embs = torch.cat((video_all_embs.cpu(), adversarial_embs.cpu()), dim=0)
                adv_vis_all_ids = vis_all_ids + ["adversarialVideo#%d" % each for each in range(len(adversarial_embs))]
                adv_target_txt_ids = ["adversarialVideo#%d" % each for each in range(len(target_txt_ids))]
                adv_sim_matrix = model.get_txt2vis_matrix(target_txt_embs,
                                                          adv_video_all_embs).cpu().data.numpy()
                for txt_index in range(len(adv_target_txt_ids)):
                    adv_mAP_list.append(get_eval_from_matrix(adv_sim_matrix[[txt_index]], adv_vis_all_ids,
                                         [adv_target_txt_ids[txt_index]], t2multi_v=True)['eval_tuple'][1])
                mAP_inds = np.argsort(adv_mAP_list)[::-1]  # 前两个即为最大 mAP 的 txt_index
                inds = np.argsort(adv_sim_matrix, axis=1)[:, ::-1]


                # 画出攻击前后 adv-t-mAP 最大的两组 query 前10张检索结果
                top = 10
                fig, ax = plt.subplots(2*top, 10, dpi=300)
                vis_tensors = [[] for each in range(top*2)]
                for i_, each in enumerate(inds[mAP_inds[0:top]]):  # visual 排序
                    for j, each_index in enumerate(each):
                        vis_id_ = adv_vis_all_ids[each_index]
                        if 'adversarialVideo' in vis_id_:
                            # print(j, each_index)
                            continue
                        temp = \
                        vis_loader.dataset.ImageDataset._get_imagePreTensor_from_videoid_with_clip(
                            vis_id_)[1][0].numpy()  # 未归一化的原始图片
                        vis_tensors[i_].append(temp)
                        if len(vis_tensors[i_]) == 10:
                            break
                for i_, each in enumerate(inds[mAP_inds[0:top]]):  # visual 排序
                    i_ += top
                    # print("second", i_)
                    for each_index in each:
                        vis_id_ = adv_vis_all_ids[each_index]
                        if 'adversarialVideo' in vis_id_:
                            vis_index = int(vis_id_.split("#")[-1])
                            # vis_id_ = "ad%s.jpg" % vis_index
                            temp = adversarial_tensors[vis_index].numpy()  # 未归一化的原始图片
                        else:
                            temp = \
                                vis_loader.dataset.ImageDataset._get_imagePreTensor_from_videoid_with_clip(
                                    vis_id_)[1][0].numpy()  # 未归一化的原始图片
                        vis_tensors[i_].append(temp)
                        if len(vis_tensors[i_]) == 10:
                            break

                vis_tensors = torch.Tensor(np.array(vis_tensors))

                for i in range(2*top):
                    for j in range(10):
                        image = transforms.ToPILImage()(vis_tensors[i, j]).convert('RGB')
                        print((i, j))
                        ax[i, j].imshow(image)
                        ax[i, j].set_xticks([])
                        ax[i, j].set_yticks([])
                fig.suptitle(txt_loader.dataset.get_caption_dict_by_id(target_txt_ids[mAP_inds[0]])['caption']
                             +'\n'+ txt_loader.dataset.get_caption_dict_by_id(target_txt_ids[mAP_inds[1]])['caption'])

                util.makedirs(os.path.join(output_dir, target_concept))
                fig.show()
                fig.savefig(os.path.join(output_dir, target_concept, '%s_result.jpg' % target_concept))

                with open(os.path.join(output_dir, target_concept, 'caption.txt'), 'w') as f:
                    for i in range(top):
                        f.write("caption %d: " % i)
                        f.write(txt_loader.dataset.get_caption_dict_by_id(target_txt_ids[mAP_inds[i]])['caption'])
                        f.write('\n')
                for i in range(top):
                    for j in range(10):
                        image = transforms.ToPILImage()(vis_tensors[i, j]).convert('RGB')
                        util.makedirs(os.path.join(output_dir, target_concept, 'caption_%d_org'% i))
                        image.save(os.path.join(output_dir, target_concept, 'caption_%d_org'% i, 'rank_%d.jpg'% j))
                for i in range(top, top+top):
                    for j in range(10):
                        image = transforms.ToPILImage()(vis_tensors[i, j]).convert('RGB')
                        util.makedirs(os.path.join(output_dir, target_concept, 'caption_%d_adv'% (i-top)))
                        image.save(os.path.join(output_dir, target_concept, 'caption_%d_adv'% (i-top), 'rank_%d.jpg'% j))


        except Exception as e:
            print(e)


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent=2))

    np.random.seed(2)
    torch.manual_seed(2)
    random.seed(2)

    if '~' in opt.rootpath:
        opt.rootpath = opt.rootpath.replace('~', os.path.expanduser('~'))
    # set the config parm you adjust
    config = importlib.import_module('configs.%s' % opt.config_name).config()
    if opt.parm_adjust_config != 'None':
        config.adjust_parm(opt.parm_adjust_config)

    get_images2concept_attack(opt, config)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print()
        # CLIP
        sys.argv = "TTH_attack.py --device 1 flickr30ktest_add_ad " \
                   "/home/hf/hf_code/VisualSearch/flickr30ktrain/w2vvpp_train/flickr30kval/ICDE.CLIPEnd2End_adjust/runs_1_0_1_0_0_seed_2/model_best.pth.tar " \
                   "msrvtt10ktrain/msrvtt10kval/test " \
                   "--attack_trainData flickr30ktrain " \
                   "--config_name TTH.CLIPEnd2End_adjust " \
                   "--parm_adjust_config 0_1_1 " \
                   "--rootpath ~/VisualSearch --batch_size 256 " \
                   "--query_sets flickr30ktest_add_ad.caption.txt " \
                   "--overwrite 0".split(' ')

    main()
