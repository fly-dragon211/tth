# Targeted Trojan-Horse Attacks on Language-based Image Retrieval



Source code of our TTH paper:  [Targeted Trojan-Horse Attacks on Language-based Image Retrieval](https://arxiv.org/abs/2202.03861). This project uses [CLIP](https://github.com/openai/CLIP) to as Attack model.

## Environment

We used Anaconda to setup a deep learning workspace that supports PyTorch. Run the following script to install all the required packages.

```sh
conda create -n tth python==3.8
conda activate tth
git clone https://github.com/li-xirong/sea.git
cd tth
pip install -r requirements.txt
```



## Data prepare

We put the dataset files on `~/VisualSearch`.

```sh
mkdir ~/VisualSearch
unzip -q "TTH_VisualSearch.zip" -d "~/VisualSearch/"
```

You need to download Flickr30k dataset and move the image files to `~/VisualSearch/flickr30k/flickr30k-images/`. The Flickr30k is available on [official website](http://shannon.cs.illinois.edu/DenotationGraph/) or Baidu Yun (https://pan.baidu.com/s/1r0RVUwctJsI0iNuVXHQ6kA  hrf3).



## TTH attack



```sh
 python TTH_attack.py \
 --device 0 flickr30ktest_add_ad None flickr30ktrain/flickr30kval/test \
 --attack_trainData flickr30ktrain --config_name TTH.CLIPEnd2End_adjust \
 --parm_adjust_config 0_1_1 --rootpath ~/VisualSearch \
 --batch_size 256 --query_sets flickr30ktest_add_ad.caption.txt
```



## A demo

Coming soon.



## References

```
@article{hu2022targeted,
  title={Targeted Trojan-Horse Attacks on Language-based Image Retrieval},
  author={Hu, Fan and Chen, Aozhu and Li, Xirong},
  journal={arXiv},
  year={2022}
}
```

