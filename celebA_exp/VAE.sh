#! /bin/zsh
python3 run.py --model VAE --dirname VAE_default2 --batch_size 128 --epoch 50 --lr 1e-4 
python3 run.py --model VAE --dirname VAE_sig1.5 --batch_size 128 --epoch 50 --lr 1e-4 --recon_sigma 1.5 

python3 run.py --model VAE --dirname VAE_sig1.5 --batch_size 128 --epoch 50 --lr 1e-4 --beta 0.1
