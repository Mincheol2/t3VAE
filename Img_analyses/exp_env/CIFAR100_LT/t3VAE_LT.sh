#! /bin/zsh
python3 ../run.py --model t3VAE --dataset cifar100_imb --dirname t3VAE_nu10_imb_10 --nu 10 --epoch 50 --lr 1e-4 --datapath ../ --imb 10
python3 ../run.py --model t3VAE --dataset cifar100_imb --dirname t3VAE_nu10_imb_50 --nu 10 --epoch 50 --lr 1e-4 --datapath ../ --imb 50
python3 ../run.py --model t3VAE --dataset cifar100_imb --dirname t3VAE_nu10_imb_100 --nu 10 --epoch 50 --lr 1e-4 --datapath ../ --imb 100
