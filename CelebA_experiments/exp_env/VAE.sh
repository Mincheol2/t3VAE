#! /bin/zsh
python3 ../run.py --model VAE --dirname VAE_default --batch_size 128 --epoch 50 --lr 1e-4 --datapath ../
python3 ../run.py --model VAE --dirname VAE_sig1.5 --batch_size 128 --epoch 50 --lr 1e-4 --prior_sigma 1.5 --datapath ../
python3 ../run.py --model VAE --dirname betaVAE_beta0.05 --batch_size 128 --epoch 50 --lr 1e-4 --beta_weight 0.05 --datapath ../
