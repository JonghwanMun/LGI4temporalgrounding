#! /bin/bash

wget http://cvlab.postech.ac.kr/research/LGI/LGI_data.tar.gz
tar zxvf LGI_data.tar.gz
mv LGI data
rm LGI_data.tar.gz

wget http://cvlab.postech.ac.kr/research/LGI/charades_data.tar.gz
tar zxvf charades_data.tar.gz
mv charades data
rm charades_data.tar.gz
