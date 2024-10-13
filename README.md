# THISNet: Tooth Instance Segmentation on 3D Dental Models via Highlighting Tooth Regions
Official PyTorch implementation of the paper entitled 'THISNet: Tooth Instance Segmentation on 3D Dental Models via Highlighting Tooth Regions'.

# Installation
1. install requirements
- Python >=3.6
- PyTorch = 1.8.0
- pymeshlab = 2022.2
- plyfile = 0.7.4

2. conda env installation
```
conda env create -f environment.yml -n thisnet
```
# Usage
You can train and test the model by running the following commands:
```
# Training steps:
python train.py
# Testing steps:
python test.py
```
# Contact
If you have any technical questions, please contact:
- Wechat：lIpen9chen9
- E-mail: lipengchengme@163.com

# Citation
If you find THISNet is useful in your research or applications, please consider giving us a star &#127775; and citing THISNet by the following BibTeX entry.
```
    @article{li2023thisnet,
      title={THISNet: Tooth Instance Segmentation on 3D Dental Models via Highlighting Tooth Regions},
      author={Li, Pengcheng and Gao, Chenqiang and Liu, Fangcen and Meng, Deyu and Yan, Yan},
      journal={IEEE Transactions on Circuits and Systems for Video Technology},
      page={5229-5241},
      year={2023},
      publisher={IEEE}
      }
```

# Acknowledgment 
Our project is partially based on [TSGCNet](https://github.com/ZhangLingMing1/TSGCNet) and [SparseInst](https://github.com/hustvl/SparseInst),
and we sincerely thanks for their code and contribution to the community!

# License
This code is only freely available for non-commercial research use. If you have other purpose, please contact:
- Chenqiang Gao
- E-mail: gaochq6@mail.sysu.edu.cn
- Copyright: Chongqing University of Posts and Telecommunications
