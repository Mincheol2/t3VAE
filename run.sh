#! /bin/zsh
python3 run.py --model TtAE --nu 3 --dataset celebA --qdim 64 --dirname celebA_TtAE --epoch 20 --lr 1e-3  --reg_weight 0.00025




# History
# python3 run.py --model VAE --dataset celebA --qdim 64 --dirname celebA_TtAE --epoch 20 --lr 1e-3  --reg_weight 0.00025
# VAE gen이 약간 blurry한 느낌? lr 높여보자.