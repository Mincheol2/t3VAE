# python3 run_1D.py --dirname "B" --sample_nu_list 5.0 5.0   --model_nu_list 3.0 5.0 6.0 10.0 11.0
# python3 run_1D.py --dirname "B" --sample_nu_list 5.0 10.0  --model_nu_list 3.0 5.0 6.0 10.0 11.0
# python3 run_1D.py --dirname "B" --sample_nu_list 10.0 10.0 --model_nu_list 3.0 5.0 6.0 10.0 11.0

# python3 run_1D.py --dirname "B_1" --recon_sigma 1.0 --sample_nu_list 10.0 10.0 --model_nu_list 2.01 3.0 9.0 10.0 11.0 20.0 --num_layers 32 --boot_iter 999 --model_init_seed 10
# python3 run_1D.py --dirname "B_2" --recon_sigma 1.0 --sample_nu_list 10.0 10.0 --model_nu_list 2.01 3.0 9.0 10.0 11.0 20.0 --num_layers 64 --boot_iter 999 --model_init_seed 10
# python3 run_1D.py --dirname "B_3" --recon_sigma 1.0 --sample_nu_list 10.0 10.0 --model_nu_list 2.01 3.0 9.0 10.0 11.0 20.0 --num_layers 128 --boot_iter 999 --model_init_seed 10

# python3 run_1D_rev.py --dirname "E_1" --sample_nu_list 1.0 1.0 --var_list 4.0 4.0 --recon_sigma 1.0 --model_nu_list 2.0001 2.001 2.01 2.1 3.0 4.0  --boot_iter 9999 
# python3 run_1D_rev.py --dirname "E_2" --sample_nu_list 2.0 2.0 --var_list 1.0 4.0 --recon_sigma 1.0 --model_nu_list 2.0001 2.001 2.01 2.1 3.0 4.0  --boot_iter 9999 
# python3 run_1D_rev.py --dirname "E_3" --sample_nu_list 2.0 2.0 --num_layers 512 --epochs 100 --var_list 1.0 4.0 --recon_sigma 1.0 --model_nu_list 2.0001 2.001 2.01 2.1 3.0 4.0  --boot_iter 999 
# python3 run_1D_rev.py --dirname "E_4" --sample_nu_list 2.0 2.0 --num_layers 512 --epochs 100 --var_list 1.0 4.0 --recon_sigma 1.0 --model_nu_list 5.0 6.0 7.0 8.0 9.0 10.0  --boot_iter 999 
# python3 run_1D_rev.py --dirname "F_1" --sample_nu_list 5.0 5.0 --num_layers 512 --epochs 100 --var_list 1.0 4.0 --recon_sigma 1.0 --model_nu_list 2.01 2.1 3.0 4.0 5.0  --boot_iter 999 
# python3 run_1D_rev.py --dirname "F_2" --sample_nu_list 5.0 5.0 --num_layers 512 --epochs 100 --var_list 1.0 4.0 --recon_sigma 1.0 --model_nu_list 6.0 7.0 8.0 9.0 10.0  --boot_iter 999 
# python3 run_1D_rev.py --dirname "F_3" --sample_nu_list 5.0 5.0 --num_layers 32 --epochs 100 --var_list 1.0 1.0 --recon_sigma 1.0 --model_nu_list 6.0 7.0 8.0 9.0 10.0  --boot_iter 999 --xlim 30

# python3 run_1D_rev.py --dirname "G_1" --recon_sigma 0.1 --sample_nu_list 5.0 5.0 --num_layers 64 --epochs 100 --model_nu_list 2.01 2.1 3.0 4.0 5.0  --boot_iter 999 --xlim 30
# python3 run_1D_rev.py --dirname "G_2" --recon_sigma 0.1 --sample_nu_list 5.0 5.0 --num_layers 64 --epochs 100 --model_nu_list 6.0 7.0 8.0 9.0 10.0  --boot_iter 999 --xlim 30
# python3 run_1D_rev.py --dirname "G_3" --recon_sigma 0.1 --sample_nu_list 5.0 5.0 --num_layers 64 --epochs 100 --model_nu_list 12.0 14.0 16.0 18.0 20.0  --boot_iter 999 --xlim 30

# python3 run_1D_rev.py --dirname "H_1" --recon_sigma 0.04 --sample_nu_list 5.0 5.0 --num_layers 64 --epochs 50 --model_nu_list 3.0 5.0 7.0 9.0 11.0  --boot_iter 999 --xlim 25
# python3 run_1D_rev.py --dirname "H_2" --recon_sigma 0.04 --sample_nu_list 3.0 3.0 --num_layers 64 --epochs 50 --model_nu_list 3.0 5.0 7.0 9.0 11.0  --boot_iter 999 --xlim 25

# python3 run_1D_rev.py --dirname "H_3" --recon_sigma 0.04 --sample_nu_list 2.0 2.0 --num_layers 64 --epochs 50 --model_nu_list 2.0001 2.001 2.01 2.1 3.0  --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "H_4" --recon_sigma 0.04 --sample_nu_list 2.0 5.0 --mu_list 0.0 20.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 64 --epochs 50 --model_nu_list 3.0 5.0 7.0 9.0 11.0  --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "H_5" --recon_sigma 0.09 --sample_nu_list 2.0 5.0 --mu_list 0.0 15.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 128 --epochs 100 --model_nu_list 2.01 2.1 2.3 2.5 2.8  --boot_iter 1999 --xlim 100
# python3 run_1D_rev.py --dirname "H_6" --recon_sigma 0.09 --sample_nu_list 2.0 5.0 --mu_list 0.0 15.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 128 --epochs 100 --model_nu_list 3.0 5.0 7.0 9.0 11.0  --boot_iter 1999 --xlim 100
# python3 run_1D_rev.py --dirname "H_6" --recon_sigma 0.09 --sample_nu_list 2.0 0.0 --mu_list 0.0 10.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 128 --epochs 100 --model_nu_list 2.1 3.0 5.0 7.0 10.0  --boot_iter 999 --xlim 100

# python3 run_1D_rev.py --dirname "H_7" --recon_sigma 0.09 --sample_nu_list 2.0 0.0 --mu_list 0.0 7.5 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 128 --epochs 100 --model_nu_list 2.1 3.0 5.0 7.0 10.0  --boot_iter 999 --xlim 100

# python3 run_1D_rev.py --dirname "I_1" --recon_sigma 0.04 --sample_nu_list 2.0 0.0 --mu_list 0.0 7.5 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 32 --epochs 100 --model_nu_list 2.1 3.0 4.0 5.0 7.0   --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "I_2" --recon_sigma 0.04 --sample_nu_list 2.0 0.0 --mu_list 0.0 5.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 32 --epochs 100 --model_nu_list 2.1 3.0 4.0 5.0 7.0   --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "I_3" --recon_sigma 0.04 --sample_nu_list 2.0 0.0 --mu_list 0.0 7.5 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 64 --epochs 100 --model_nu_list 2.1 3.0 4.0 5.0 7.0   --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "I_4" --recon_sigma 0.04 --sample_nu_list 2.0 0.0 --mu_list 0.0 5.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 64 --epochs 100 --model_nu_list 2.1 3.0 4.0 5.0 7.0   --boot_iter 999 --xlim 100

# python3 run_1D_rev.py --dirname "I_5" --recon_sigma 0.01 --sample_nu_list 2.0 0.0 --mu_list 0.0 5.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers  64 --epochs 100 --model_nu_list 2.1 2.5 3.0 --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "I_6" --recon_sigma 0.01 --sample_nu_list 2.0 0.0 --mu_list 0.0 5.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 128 --epochs 100 --model_nu_list 2.1 2.5 3.0 --boot_iter 999 --xlim 100

# python3 run_1D_rev.py --dirname "I_07" --recon_sigma 0.025 --sample_nu_list 2.0 0.0 --mu_list 0.0 5.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers  64 --epochs 100 --model_nu_list 2.01 2.1 2.2 2.5 3.0 --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "I_08" --recon_sigma 0.025 --sample_nu_list 2.0 0.0 --mu_list 0.0 5.0 --train_N_list 80000 20000 --test_N_list 80000 20000 --num_layers 128 --epochs 100 --model_nu_list 2.01 2.1 2.2 2.5 3.0 --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "I_09" --recon_sigma 0.025 --K 1 --sample_nu_list 2.0 --mu_list 0.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 2.01 2.1 2.2 2.5 3.0 --boot_iter 999 --xlim 100
# python3 run_1D_rev.py --dirname "I_10" --recon_sigma 0.025 --K 1 --sample_nu_list 2.0 --mu_list 0.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers 128 --epochs 100 --model_nu_list 2.01 2.1 2.2 2.5 3.0 --boot_iter 999 --xlim 100

# python3 run_1D_rev.py --dirname "lognormal_01" --recon_sigma 0.025 --K 1 --sample_nu_list 0 --mu_list 0.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 2.01 3.0 5.0 10.0 --boot_iter 999 --xlim 100 --sample_type "lognormal"
# python3 run_1D_rev.py --dirname "lognormal_02" --recon_sigma 0.04 --K 1 --sample_nu_list 0 --mu_list 1.0 --var_list 2.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 3.0 5.0 7.0 --boot_iter 999 --xlim 100 --sample_type "lognormal" --model_init_seed 8000
# python3 run_1D_rev.py --dirname "lognormal_03" --recon_sigma 0.04 --K 1 --sample_nu_list 0 --mu_list 2.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 2.1 3.0 5.0 7.0 --boot_iter 999 --xlim 100 --sample_type "lognormal" --model_init_seed 8000

# python3 run_1D_rev.py --dirname "I_12" --recon_sigma 0.025 --K 1 --sample_nu_list 2.0 --mu_list 0.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 2.1 2.2 --boot_iter 999 --xlim 100 --model_init_seed 10000
python3 run_1D_rev.py --dirname "I_13" --recon_sigma 0.025 --K 1 --sample_nu_list 2.0 --mu_list 0.0 --var_list 1.0 --train_N_list 100000 --test_N_list 100000 --num_layers  64 --epochs 100 --model_nu_list 2.1 2.2 --boot_iter 999 --xlim 100 --model_init_seed 10000
