# python3 run.py --test_N 5000000  --gen_N 5000000
# python3 hypothesis_test.py 

python3 run.py --dirname "test" --epochs 6
python3 hypothesis_test.py --dirname "test"

