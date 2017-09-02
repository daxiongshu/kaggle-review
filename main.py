import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_args(args = None):
    try: 
        dic = FLAGS.__flags
    except:
        dic = vars(args)
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
    try:
        import tensorflow as tf
        from tf_flags import FLAGS
        tf.app.run()
    except:
        from flags import get_parser
        parser = get_parser()
        args = parser.parse_args() 
        FLAGS = vars(args)
        main(None)
