rem python main.py --comp personal --sol baobao --task xgb_sub_stage1_bag --pred_path comps/personal/baobao/sub_stage1_bag.csv --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold -1

rem python main.py --comp personal --sol baobao --task eval --pred_path comps/personal/baobao/sub_stage1_bag.csv --data_path comps/personal/baobao/data

python main.py --comp personal --sol baobao --task xgb_sub_stage2_bag --pred_path comps/personal/baobao/sub_stage2_bag.csv --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold -1
