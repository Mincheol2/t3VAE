#! /bin/zsh
python3 ../run.py --model FactorVAE --dataset cifar100_imb --dirname FactorVAE_default_imb10 --batch_size 128 --epoch 50 --lr 1e-4 --TC_gamma 20 --imb 10
python3 ../run.py --model FactorVAE --dataset cifar100_imb --dirname FactorVAE_default_imb50 --batch_size 128 --epoch 50 --lr 1e-4 --TC_gamma 20 --imb 50
python3 ../run.py --model FactorVAE --dataset cifar100_imb --dirname FactorVAE_default_imb100 --batch_size 128 --epoch 50 --lr 1e-4 --TC_gamma 20 --imb 100
