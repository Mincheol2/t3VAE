#! /bin/zsh

python3 main.py --dataset celeb_crop64 --zdim 64 --dirname celeb64_crop_VAE --epoch 20
python3 main.py --dataset celeb_crop64 --nu 2.5 --zdim 64 --dirname "celeb64_crop_gae2.5" --epoch 20
python3 main.py --dataset celeb_crop64 --nu 2.5 --zdim 64 --dirname "celeb64_crop_noflat_gae2.5" --epoch 20 --flat n