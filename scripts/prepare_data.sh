#! /bin/bash

### Download data for both activitynet and charades dataset
wget http://cvlab.postech.ac.kr/research/LGI/LGI_data.tar.gz
tar zxvf LGI_data.tar.gz
mv LGI data
rm LGI_data.tar.gz

### Download data for activitynet dataset
#wget http://cvlab.postech.ac.kr/~jonghwan/data/LGI/anet_ann_data.tar.gz # annotation + (preprocessed ones)
#mv LGI data
#rm anet_ann_data.tar.gz
# please download the C3D features from http://activity-net.org/challenges/2016/download.html
# move the C3D feature file into data/LGI/feats/

### Download data for charades dataset
#wget http://cvlab.postech.ac.kr/research/LGI/charades_data.tar.gz
#tar zxvf charades_data.tar.gz
#mv charades data
#rm charades_data.tar.gz
