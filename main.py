import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flags import FLAGS

def print_args():
    dic = FLAGS.__flags
    print()
    keys = sorted(dic.keys())
    print("common flags:")
    for i in keys:
        if dic[i]:
            print('{:>23} {:>23}'.format(i, str(dic[i])))
    print()


def main(_):
    print_args()
    if FLAGS.comp == "instacart":
        from comps.instacart.run import run_sol
    elif FLAGS.comp == "carvana":
        from comps.carvana.run import run_sol
    elif FLAGS.comp == "mobike":
        from comps.mobike.run import run_sol
    else:
        print("Unknown competion %s"%FLAGS.comp)
        assert False
    print("run competition %s solution %s"%(FLAGS.comp,FLAGS.sol))
    run_sol(FLAGS)

if __name__ == "__main__":
    tf.app.run()
