#! /bin/zsh
python3 ../run.py --model TVAE --dataset cifar100_imb --dirname TVAE_default_imb10 --epoch 50 --lr 1e-4 --datapath ../ --batch_size 128 --imb 10
python3 ../run.py --model TVAE --dataset cifar100_imb --dirname TVAE_default_imb50 --epoch 50 --lr 1e-4 --datapath ../ --batch_size 128 --imb 50
python3 ../run.py --model TVAE --dataset cifar100_imb --dirname TVAE_default_imb100 --epoch 50 --lr 1e-4 --datapath ../ --batch_size 128 --imb 100