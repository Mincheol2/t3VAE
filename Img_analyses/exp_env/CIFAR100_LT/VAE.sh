#! /bin/zsh
python3 ../run.py --model VAE --dataset cifar100_imb --dirname VAE_default_imb100 --batch_size 128 --epoch 50 --lr 1e-4 --datapath ../ --imb 100
python3 ../run.py --model VAE --dataset cifar100_imb --dirname VAE_sig1.5_imb100 --batch_size 128 --epoch 50 --lr 1e-4 --prior_sigma 1.5 --datapath ../ --imb 100
python3 ../run.py --model VAE --dataset cifar100_imb --dirname betaVAE_beta0.05_imb100 --batch_size 128 --epoch 50 --lr 1e-4 --beta_weight 0.05 --datapath ../ --imb 100
