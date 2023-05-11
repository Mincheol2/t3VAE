# python3 run_1D.py --dirname "A" --sample_nu_list 1.0 1.0 --model_nu_list 2.0001 2.01 3.0 4.0 5.0 --batch_size 128 --lr 5e-3 
# python3 run_1D.py --dirname "A" --sample_nu_list 1.0 2.0 --model_nu_list 2.0001 2.01 3.0 4.0 5.0 --batch_size 128 --lr 5e-3 
# python3 run_1D.py --dirname "A" --sample_nu_list 2.0 2.0 --model_nu_list 2.0001 2.01 3.0 4.0 5.0 --batch_size 128 --lr 5e-3 
# python3 run_1D.py --dirname "A_1" --sample_nu_list 1.0 1.0 --recon_sigma 1.0 --model_nu_list 2.0001 2.001 2.01 2.1 3.0 --num_layers 32 --batch_size 128 --boot_iter 999 --model_init_seed 10
# python3 run_1D.py --dirname "A_2" --sample_nu_list 1.0 1.0 --recon_sigma 1.0 --model_nu_list 2.0001 2.001 2.01 2.1 3.0 --num_layers 64 --batch_size 128 --boot_iter 999 --model_init_seed 10
# python3 run_1D.py --dirname "A_3" --sample_nu_list 1.0 1.0 --recon_sigma 1.0 --model_nu_list 2.0001 2.001 2.01 2.1 3.0 --num_layers 128 --batch_size 128 --boot_iter 999 --model_init_seed 10
# python3 run_1D_rev.py --dirname "D_1" --sample_nu_list 10.0 10.0 --recon_sigma 1.0 --model_nu_list 5.0  --boot_iter 999 --xlim 30
# python3 run_1D_rev.py --dirname "D_2" --sample_nu_list 10.0 10.0 --recon_sigma 1.0 --model_nu_list 5.0  --boot_iter 999 --xlim 30
# python3 run_1D_rev.py --dirname "D_2" --sample_nu_list 10.0 10.0 --recon_sigma 1.0 --model_nu_list 5.0  --boot_iter 999 --xlim 30
# python3 run_1D_rev.py --dirname "Cauchy_1" --recon_sigma 0.01 --sample_nu_list 1.0 1.0 --num_layers 64 --epochs 100 --model_nu_list 2.0001 2.001 2.01 2.1 3.0  --boot_iter 999 --xlim 200
# python3 run_1D_rev.py --dirname "Cauchy_2" --recon_sigma 0.01 --sample_nu_list 2.0 2.0 --num_layers 64 --epochs 100 --model_nu_list 2.0001 2.001 2.01 2.1 3.0  --boot_iter 999 --xlim 200

# python3 run_1D_rev.py --dirname "B_1" --recon_sigma 0.0001 --K 1 --sample_nu_list 2.0 --mu_list 3.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 2.1 2.2 --boot_iter 999 --xlim 100 --model_init_seed 10000
# python3 run_1D_rev.py --dirname "B_2" --recon_sigma 1 --K 1 --sample_nu_list 2.0 --mu_list 3.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 2.1 2.2 --boot_iter 999 --xlim 100 --model_init_seed 10000
# python3 run_1D_rev.py --dirname "B_3" --recon_sigma 1 --sample_nu_list 5.0 5.0 --var_list 4.0 1.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers  64 --epochs 100 --model_nu_list 3.0 5.0 7.0 10.0 --boot_iter 999 --xlim 30
python3 run_1D_rev.py --dirname "B_4" --recon_sigma 1 --sample_nu_list 5.0 5.0 --var_list 4.0 1.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers  64 --epochs 100 --model_nu_list 12.0 14.0 16.0 18.0 20.0 --boot_iter 999 --xlim 30
# python3 run_1D_rev.py --dirname "B_4" --recon_sigma 0.1 --sample_nu_list 5.0 5.0 --num_layers 64 --epochs 100 --model_nu_list 12.0 14.0 16.0 18.0 20.0  --boot_iter 999 --xlim 30
