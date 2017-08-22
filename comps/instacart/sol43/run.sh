protoc -I=comps/instacart/sol43 --python_out=comps/instacart/sol43 comps/instacart/sol43/insta.proto
~/anaconda3/bin/python main.py --comp instacart --sol 43 --input_path ~/ml/instacart/input --data_path comps/instacart/data
