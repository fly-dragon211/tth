# Targeted Trojan-Horse Attacks on Language-based Image Retrieval

Source code of our TTH paper:  [Targeted Trojan-Horse Attacks on Language-based Image Retrieval](https://arxiv.org/abs/2202.03861). This project implements TTH for [CLIP](https://github.com/openai/CLIP)  and CLIP-flickr on Flickr30k.

![image-20220422124016610](image/image-20220422124016610.png)

## Environment

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

```sh
conda create -n tth python==3.8
conda activate tth
git clone https://github.com/fly-dragon211/tth.git
cd tth
pip install -r requirements.txt
```



## Data prepare

### Dataset

We put the dataset files on `~/VisualSearch`.

```sh
mkdir ~/VisualSearch
unzip -q "TTH_VisualSearch.zip" -d "~/VisualSearch/"
```

Readers need to download Flickr30k dataset and move the image files to `~/VisualSearch/flickr30k/flickr30k-images/`. The Flickr30k is available on [official website](http://shannon.cs.illinois.edu/DenotationGraph/) or Baidu Yun (https://pan.baidu.com/s/1r0RVUwctJsI0iNuVXHQ6kA  提取码：hrf3).



### CLIP-flickr and CLIP-coco models

We provide the CLIP model which finetuned on Flickr30k and MSCOCO:

Baidu Yun: https://pan.baidu.com/s/1n8Sa7Fr9-G9KbZ3-FxS1_g?pwd=sbsv 提取码: sbsv 

Readers can move the model files to `~/VisualSearch/flickr30k`

## TTH attack

**CLIP** 

```sh
 python TTH_attack.py \
 --device 0 flickr30ktest_add_ad None flickr30ktrain/flickr30kval/test \
 --attack_trainData flickr30ktrain --config_name TTH.CLIPEnd2End_adjust \
 --parm_adjust_config 0_1_1 --rootpath ~/VisualSearch \
 --batch_size 256 --query_sets flickr30ktest_add_ad.caption.txt
```



 R10 of truly relevant images and novel images w.r.t. specific queries. LBIR setup: CLIP + Flickr30ktest. Adversarial patches are learned with Flickr30ktrain as training data. The clear drop of R10 for truley relevant images and the clear increase of R10 for novel images show the success of the proposed method for making TTH attacks

![image-20220422125243619](image/image-20220422125243619.png)

**CLIP-*flickr***

```shell
 CLIP_flickr="~/VisualSearch/flickr30k/CLIP-flickr.tar"
 
 python TTH_attack.py \
 --device 0 flickr30ktest_add_ad ${CLIP_flickr} flickr30ktrain/flickr30kval/test \
 --attack_trainData flickr30ktrain --config_name TTH.CLIPEnd2End_adjust \
 --parm_adjust_config 0_1_0 --rootpath ~/VisualSearch \
 --batch_size 256 --query_sets flickr30ktest_add_ad.caption.txt
```

 R10 of truly relevant images and novel images w.r.t. specific queries. LBIR setup: CLIP-flickr + Flickr30ktest. 

![image-20220422125609322](image/image-20220422125609322.png)



## References

```
@article{hu2022targeted,
  title={Targeted Trojan-Horse Attacks on Language-based Image Retrieval},
  author={Hu, Fan and Chen, Aozhu and Li, Xirong},
  journal={arXiv},
  year={2022}
}
```

### Contact

If you enounter any issue when running the code, please feel free to reach us either by creating a new issue in the github or by emailing

- Fan Hu ([hufan_hf@ruc.edu.cn](mailto:hufan_hf@ruc.edu.cn))
- Aozhu Chen ([caz@ruc.edu.cn](mailto:caz@ruc.edu.cn))
