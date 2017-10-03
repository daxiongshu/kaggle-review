python main.py --comp personal --sol baobao --task xgb_cv_meta --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold 0 --load_path comps/personal/baobao/weights/train_d2v___7.npy

python main.py --comp personal --sol baobao --task xgb_cv_meta --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold 1 --load_path comps/personal/baobao/weights/train_d2v___7.npy

python main.py --comp personal --sol baobao --task xgb_cv_meta --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold 2 --load_path comps/personal/baobao/weights/train_d2v___7.npy

python main.py --comp personal --sol baobao --task xgb_cv_meta --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold 3 --load_path comps/personal/baobao/weights/train_d2v___7.npy

python main.py --comp personal --sol baobao --task xgb_sub_stage1_meta --pred_path comps/personal/baobao/sub_stage1.csv --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold -1

python main.py --comp personal --sol baobao --task eval --pred_path comps/personal/baobao/sub_stage1.csv --data_path comps/personal/baobao/data
