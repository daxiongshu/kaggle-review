rem train with 3 folds and validate on 1 fold
python main.py --comp carvana --metric dice_coef --batch_size 32 --classes 1 --opt adam --learning_rate 0.001 --save_path comps/carvana/btb1/weights --net zf_unet --task train_cv --sol btb1 --width 224 --height 128 --split_path comps/carvana/btb1/data/split.npy --folds 4 --input_path ../input/train --epochs 10 --fold 0 --visualize image,mask,grad --log_path comps/carvana/btb1/logs/run1

rem predict 1 fold
python main.py --comp carvana --metric dice_coef --batch_size 32 --classes 1 --net zf_unet --task predict_cv --sol btb1 --width 224 --height 128 --split_path comps/carvana/btb1/data/split.npy --input_path ../input/train --pred_path comps/carvana/btb1/pred_cv --load_path comps/carvana/btb1/backup/va_0.987/cv_weights/carvana_run_zf_unet_20.npy --fold 0 --folds 4

rem write rle in parallel for cv
python main.py --comp carvana --sol btb1 --pred_path comps/carvana/btb1/pred_cv --threshold 0.9 --task post_sub 

rem predict test data
python main.py --comp carvana --metric dice_coef --batch_size 32 --classes 1 --net zf_unet --task test --sol btb1 --width 224 --height 128 --input_path ../input/test --pred_path comps/carvana/btb1/pred --load_path comps/carvana/btb1/backup/va_0.987/cv_weights/carvana_run_zf_unet_20.npy

rem write rle in parallel for test
python main.py --comp carvana --sol btb1 --pred_path comps/carvana/btb1/pred --threshold 0.9 --task post_sub

rem get the final sub file
C:\Users\Jiwei\Downloads\installer\pypy2-v5.8.0-win32\pypy.exe comps/carvana/btb1/write.py comps/carvana/btb1/pred img,rle_mask sub.csv 
