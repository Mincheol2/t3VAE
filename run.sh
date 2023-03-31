#! /bin/zsh
python3 run.py --model VampPrior --dataset celebA --qdim 64 --dirname celebA_Vamp --epoch 20 --lr 1e-3  --reg_weight 0.00025




# History
# --qdim 64 --dirname celebA_TtAE --epoch 20 --lr 1e-3 --reg_weight 0.00025
# Recon은 확실히 VAE << TtAE (nu=3)
# VAE gen은 약간 blurry한 느낌? lr 높여보자.
# TtAE gen은 여전히 잘 안된다.