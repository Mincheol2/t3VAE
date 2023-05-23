#! /bin/zsh
python3 run.py --model TtVAE --dataset mnist --dirname MNIST_TEST --nu 2.5 --epoch 10 --lr 1e-4

# python3 run.py --model TtVAE --dirname FINAL_Tt_nu10_2 --nu 10 --epoch 50 --lr 3e-4 --reg_weight 0.001
# python3 run.py --model FactorVAE --dirname FINAL_FACTOR_2 --batch_size 128 --epoch 50 --lr 1e-4 --TC_gamma 6.4

# python3 run.py --model TtVAE --dirname new_Tt_l1norm0.1 --batch_size 128 --nu 5 --epoch 10 --lr 4e-4 --reg_weight 0.02



# python3 run.py --model TtVAE --dirname new_Tt_nu25 --nu 2.5 --epoch 50 --lr 3e-4 --reg_weight 0.003

# python3 run.py --model TtVAE --dirname new_Tt_nu5_5e4 --batch_size 128 --nu 5 --epoch 50 --lr 3e-4 --reg_weight 0.005 --scheduler_gamma 0.95




## Final ##
# python3 run.py --model VAE --dirname new_VAE_FID2 --batch_size 128 --epoch 60 --lr 1e-4 
# python3 run.py --model VAE --dirname new_VAE_sig1.5 --batch_size 128 --epoch 30 --lr 1e-4 --recon_sigma 1.5
# python3 run.py --model FactorVAE --dirname FactorVAE_Try1 --batch_size 128 --epoch 30 --lr 1e-4 --TC_gamma 6.4
