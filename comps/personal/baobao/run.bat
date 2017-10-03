rem python main.py --comp personal --sol baobao --task clean --input_path ../input
C:\Users\Jiwei\Downloads\installer\pypy2-v5.8.0-win32\pypy.exe main.py --comp personal --sol baobao --task write_gene_text --input_path ../input --data_path comps/personal/baobao/data
rem python main.py --comp personal --sol baobao --task preprocess --input_path ../input --data_path comps/personal/baobao/data --folds 4 

rem python main.py --comp personal --sol baobao --task train_embedding --input_path ../input --data_path comps/personal/baobao/data --window_size 5 --opt adam --learning_rate 0.001 --lambdax 0.0001 --batch_size 4096 --embedding_size 64 --epochs 100 --save_path comps/personal/baobao/weights

rem python main.py --comp personal --sol baobao --task train_d2v --input_path ../input --data_path comps/personal/baobao/data --d2v_size 128 --opt adam --learning_rate 0.001 --lambdax 0.0001 --batch_size 4096 --embedding_size 64 --epochs 100 --save_path comps/personal/baobao/weights --load_path comps/personal/baobao/weights/train_embedding___17.npy 

rem python main.py --comp personal --sol baobao --task show_d2v --log_path comps/personal/baobao/logs/d2v --input_path ../input --data_path comps/personal/baobao/data  --load_path comps/personal/baobao/weights/train_d2v___7.npy --embedding_size 64 --d2v_size 128

rem python main.py --comp personal --sol baobao --task train_nn --input_path ../input --data_path comps/personal/baobao/data  --load_path comps/personal/baobao/weights/train_d2v___8.npy --d2v_size 128 --opt adam --learning_rate 0.001 --lambdax 0.0001 --batch_size 4096 --save_path comps/personal/baobao/weights --fold 0 --epochs 10000 --classes 9 --folds 4 --pred_path comps/personal/baobao/nn_pred.csv --save_epochs 1000 --keep_prob 0.5

rem python main.py --comp personal --sol baobao --task predict_nn --input_path ../input --data_path comps/personal/baobao/data  --load_path comps/personal/baobao/weights/train_nn___600.npy --d2v_size 128 --batch_size 4096 --fold 0 --classes 9 --folds 4 --pred_path comps/personal/baobao/nn_pred.csv

rem python main.py --comp personal --sol baobao --task eval --input_path ../input --data_path comps/personal/baobao/data --fold 0 --folds 4 --pred_path comps/personal/baobao/nn_pred.csv

rem python main.py --comp personal --sol baobao --task show_embedding --log_path comps/personal/baobao/logs --input_path ../input --data_path comps/personal/baobao/data --window_size 5 --load_path comps/personal/baobao/weights/train_embedding___300.npy --embedding_size 64

rem python main.py --comp personal --sol baobao --task train_rnn --fold 0 --log_path comps/personal/baobao/logs --input_path ../input --data_path comps/personal/baobao/data --seq_len 50 --load_path comps/personal/baobao/weights/train_embedding___14.npy --embedding_size 64 --opt adam --learning_rate 0.001 --lambdax 0.0001 --batch_size 4096 --save_path comps/personal/baobao/weights --run_name fold0 --cell LSTM --epochs 1000 --classes 9 --folds 4 --num_unit 64 


rem python main.py --comp personal --sol baobao --task train_cnn --fold -1 --log_path comps/personal/baobao/logs --input_path ../input --data_path comps/personal/baobao/data --seq_len 2048 --load_path comps/personal/baobao/weights/train_embedding___14.npy --embedding_size 64 --opt adam --learning_rate 0.01 --lambdax 0.0001 --batch_size 1024 --save_path comps/personal/baobao/weights --run_name all  --epochs 10 --classes 9 --folds 4 --window_size 32 --verbosity 10

rem python main.py --comp personal --sol baobao --task test_cnn_stage1 --input_path ../input --data_path comps/personal/baobao/data --seq_len 2048 --load_path comps/personal/baobao/weights/train_cnn_all__2.npy --embedding_size 64 --batch_size 1024  --epochs 10 --classes 9 --window_size 32 --verbosity 10 --pred_path comps/personal/baobao/cnn_pred_stage1.csv --verbosity 10

rem python main.py --comp personal --sol baobao --task eval --pred_path comps/personal/baobao/cnn_pred_stage1_sub.csv --data_path comps/personal/baobao/data

rem python main.py --comp personal --sol baobao --task test_cnn_stage2 --input_path ../input --data_path comps/personal/baobao/data --seq_len 2048 --load_path comps/personal/baobao/weights/train_cnn_all__2.npy --embedding_size 64 --batch_size 1024  --epochs 10 --classes 9 --window_size 32 --verbosity 10 --pred_path comps/personal/baobao/cnn_pred_stage2.csv --verbosity 10

rem python main.py --comp personal --sol baobao --task xgb_cv --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold 3

rem python main.py --comp personal --sol baobao --task xgb_sub_stage2 --pred_path comps/personal/baobao/sub_stage2.csv --input_path ../input --data_path comps/personal/baobao/data --folds 4 --fold -1
