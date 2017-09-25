import argparse

def get_parser():
    ################################################################
    #common flags
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt',help="optimizor")
    parser.add_argument("--seed",help="random seed")
    parser.add_argument('--run_name', help='name of the experiment')
    parser.add_argument("--classes",help="number of classes")
    parser.add_argument("--fold",help="index of fold")
    parser.add_argument("--folds",help="number of folds")
    parser.add_argument('--log_path',help='Log path')
    parser.add_argument('--batch_size',help="batch size")
    parser.add_argument('--threads',help="number of CPU threads")
    parser.add_argument("--augmentation",help="data augmentation")
    parser.add_argument("--learning_rate",help="Learning rate")
    parser.add_argument("--threshold",help="threshold")
    parser.add_argument("--keep_prob",help="keep prob")
    parser.add_argument("--metric",help="evaluating metric")
    parser.add_argument("--lambdax",help="lambda for L2 regularization")
    parser.add_argument("--epsilon",help="epsilon in RL")
    parser.add_argument("--color",help="Color channels")
    parser.add_argument("--epochs",help="number of epochs")
    parser.add_argument("--pre_epochs",help="pretrained number of epochs")
    parser.add_argument('--comp', help='name of the competition')
    parser.add_argument('--sol', help='name of the solution')
    parser.add_argument('--task', help='name of the task')
    parser.add_argument('--visualize',help='visualize options')
    parser.add_argument('--save_path',help='path to save weights')
    parser.add_argument('--save_epochs',help='for how many epochs are weights saved')
    parser.add_argument('--load_path', help='path to load weights')
    parser.add_argument('--net',help='net name')
    parser.add_argument('--pred_path',help='path of prediction files')
    parser.add_argument('--record_path',help='path of tf record')
    parser.add_argument('--input_path',help='input path')
    parser.add_argument('--data_path',help='input path')
    parser.add_argument("--width",help="width of image to resize to")
    parser.add_argument("--height",help="height of image to resize to")
    parser.add_argument('--add_paths',help='additional input paths')
    parser.add_argument('--add_record_paths',help='additional records')
    parser.add_argument("--momentum",help="momentum")
    #####################################################################

    #####################################################################
    #paper specific flags

    #####################################################################
    return parser

