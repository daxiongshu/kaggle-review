rem python main.py --comp carvana --metric dice_coef --batch_size 32 --classes 1 --opt adam --learning_rate 0.0001 --save_path comps/carvana/btb1/weights --net car_zf_unet --task train_random --sol btb1 --width 224 --height 224 --run_name random 

python main.py --comp carvana --metric dice_coef --batch_size 32 --classes 1 --opt adam --learning_rate 0.0001 --save_path comps/carvana/btb1/weights --net car_zf_unet --task train_cv --sol btb1 --width 224 --height 128 --load_path comps/carvana/btb1/weights/carvana_random_car_zf_unet_700.npy --split_path comps/carvana/btb1/data/split.npy --folds 4 --input_path ../input/train 


rem tfrecords cost too much disk space for 1920x1080
rem python main.py --comp carvana --sol btb1 --split_path comps/carvana/btb1/data/split.npy --record_path comps/carvana/btb1/data/train.tfrecords --width 1918 --height 1280 --color 3 --classes -1 --task cv_train --folds 4 --fold 0 --input_path ../input/train  
