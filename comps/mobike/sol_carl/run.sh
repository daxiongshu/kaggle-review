# please run this script from the root folder: ./kaggle-review
# sh comps/mobike/sol_carl/run.sh


# preprocess
python comps/mobike/sol_carl/split.py
python comps/mobike/sol_carl/sort_bytime.py

# cv for coordinate data
pypy main.py --comp mobike --sol carl --task prepare_cv_coord --input_path ../input
~/anaconda3/bin/python main.py --comp mobike --net xgb --sol carl --task cv_coord
pypy comps/mobike/sol_carl/post.py comps/mobike/sol_carl/data/cv_coord_18-19.csv
pypy comps/mobike/sol_carl/post.py comps/mobike/sol_carl/data/cv_coord_20-24.csv
pypy main.py --comp mobike --sol carl --task eval_coord 

# sub for coordinate data
pypy main.py --comp mobike --sol carl --task prepare_sub_coord --input_path ../input
~/anaconda3/bin/python main.py --comp mobike --net xgb --sol carl --task sub_coord --input_path ../input

# cv for hash data
pypy main.py --comp mobike --sol carl --task prepare_cv_hash --input_path ../input
pypy comps/mobike/sol_carl/distance.py 
pypy comps/mobike/sol_carl/sample.py
~/anaconda3/bin/python main.py --comp mobike --net xgb --sol carl --task cv_hash
pypy comps/mobike/sol_carl/post.py comps/mobike/sol_carl/cv.csv
pypy main.py --comp mobike --sol carl --task eval_hash --pred_path cv_sub.csv

# sub for hash data
pypy main.py --comp mobike --sol carl --task prepare_sub_hash --input_path ../input
pypy comps/mobike/sol_carl/distance.py
~/anaconda3/bin/python main.py --comp mobike --net xgb --sol carl --task sub_hash
pypy comps/mobike/sol_carl/post.py comps/mobike/sol_carl/sub.csv
pypy comps/mobike/sol_carl/merge_sub.py
