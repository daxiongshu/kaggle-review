import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


flags = tf.app.flags
flags.DEFINE_integer("start",None,"start point")
flags.DEFINE_integer("sample_size",0,"sample size")
flags.DEFINE_integer("num_units",0,"number of units")
flags.DEFINE_integer("seq_len",10,"length of the sequence")
flags.DEFINE_integer("embedding_size",0,"embedding size")
flags.DEFINE_string('opt', 'sgd', 'optimizer')
flags.DEFINE_string('sol', None, 'id of the solution')
flags.DEFINE_string('cell', 'BASIC_LSTM', 'name of the rnn cells')
flags.DEFINE_integer("seed",0,"seed")
flags.DEFINE_integer("seeds",0,"seeds")
flags.DEFINE_string('run_name', 'run', 'name of the experiment')
flags.DEFINE_string('split_path', None, 'path of split file')
flags.DEFINE_string('embedding_path', None, 'Embedding path')
flags.DEFINE_integer("classes",2,"number of classes")
flags.DEFINE_integer("fold",None,"index of fold")
flags.DEFINE_integer("num_folds",None,"number of folds")
flags.DEFINE_string('log_path', None, 'Log path')
flags.DEFINE_integer('batch_size',64,"batch size")
flags.DEFINE_integer('threads',4,"number of threads")
flags.DEFINE_integer("augmentation",0,"data augmentation")
flags.DEFINE_float("learning_rate",0.1,"Learning rate")
flags.DEFINE_float("threshold",0.2,"threshold")
flags.DEFINE_float("keep_prob",1.0,"keep prob")
flags.DEFINE_string("metric","Accuracy","metric")
flags.DEFINE_float("lambdax",0.0001,"lambda for L2 regularization")
flags.DEFINE_float("epsilon",0.01,"epsilon in RL")
flags.DEFINE_integer("color",3,"Color channels")
flags.DEFINE_integer("epochs",10,"number of epochs")
flags.DEFINE_integer("pre_epochs",0,"pretrained number of epochs")
flags.DEFINE_string('comp', None, 'name of the competition')
flags.DEFINE_string('task', None, 'train or test')
flags.DEFINE_string('visualize',None,'visualize verbosity')
flags.DEFINE_string('save_path', 'weights', 'path to save weights')
flags.DEFINE_integer('save_epochs',1,'for how many epochs are weights saved')
flags.DEFINE_string('load_path', None , 'path to load weights')
flags.DEFINE_string('net', 'lenet', 'net name')
flags.DEFINE_string('pred_path', 'result.csv', 'name of prediction files')
flags.DEFINE_string('record_path', None, 'path of tf record')
flags.DEFINE_string('data_path', None, 'path of other data')
flags.DEFINE_string('input_path', None, 'input path')
flags.DEFINE_integer("width",256,"width of image to resize to")
flags.DEFINE_integer("height",256,"height of image to resize to")
flags.DEFINE_string('add_paths', None, 'additional input paths')
flags.DEFINE_string('add_record_paths', None, 'additional records')
flags.DEFINE_float("momentum",0.0,"momentum")
FLAGS = flags.FLAGS

def print_args():
    dic = FLAGS.__flags
    print()
    keys = sorted(dic.keys())
    for i in keys:
        #print(i,dic[i])
        print('{:>23} {:>23}'.format(i, str(dic[i])))
    print()


def main(_):
    print_args()
    if FLAGS.comp == "instacart":
        from comps.instacart.instacart import run_instacart
        run_instacart(FLAGS)
    else:
        print("Unknown competion %s"%FLAGS.comp)
        assert False
if __name__ == "__main__":
    tf.app.run()
