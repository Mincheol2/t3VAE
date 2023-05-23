#! /bin/zsh
python3 ../run.py --model TtVAE --dirname t3VAE_nu2.5 --nu 2.5 --epoch 50 --lr 3e-4 --datapath ../
python3 ../run.py --model TtVAE --dirname t3VAE_nu5 --nu 5 --epoch 50 --lr 3e-4 --datapath ../
python3 ../run.py --model TtVAE --dirname t3VAE_nu10 --nu 10 --epoch 50 --lr 3e-4 --datapath ../