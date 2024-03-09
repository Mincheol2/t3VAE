#! /bin/zsh
python3 ../run.py --model TiltedVAE --dirname tilt_default_imb10 --batch_size 128 --tilt 50 --epoch 50 --lr 1e-4 --datapath ../ --imb 10
python3 ../run.py --model TiltedVAE --dirname tilt_default_imb50 --batch_size 128 --tilt 50 --epoch 50 --lr 1e-4 --datapath ../ --imb 50
python3 ../run.py --model TiltedVAE --dirname tilt_default_imb100 --batch_size 128 --tilt 50 --epoch 50 --lr 1e-4 --datapath ../ --imb 100
