#~/anaconda3/bin/python main.py --comp income --sol learn_tf --task poke --input_path ~/ml/learn-tf/lattice/input/ --data_path comps/income/learn_tf/data
~/anaconda3/bin/python main.py --comp income --sol learn_tf --task cv --input_path ~/ml/learn-tf/lattice/input/ --data_path comps/income/learn_tf/data --net wide --save_path comps/income/learn_tf/weights --batch_size 128 --learning_rate 0.01

