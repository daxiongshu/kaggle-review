import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
FLAGS = None

def print_args():
    try: 
        dic = FLAGS.__flags
    except:
        dic = vars(FLAGS)#vars(args)
    keys = sorted(dic.keys())
    print("\ncommon flags:")
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
    elif FLAGS.comp == "porto":
        from comps.porto.run import run_sol
    elif FLAGS.comp == "mobike":
        from comps.mobike.run import run_sol
    elif FLAGS.comp == "zillow":
        from comps.zillow.run import run_sol
    elif FLAGS.comp == "wiki":
        from comps.wiki.run import run_sol
    elif FLAGS.comp == "eng_norm":
        from comps.eng_norm.run import run_sol
    elif FLAGS.comp == "russian_norm":
        from comps.russian_norm.run import run_sol
    elif FLAGS.comp == "caesars":
        from comps.caesars.run import run_sol
    elif FLAGS.comp == "personal":
        from comps.personal.run import run_sol
    elif FLAGS.comp == "income":
        from comps.income.run import run_sol
    else:
        print("Unknown competion %s"%FLAGS.comp)
        assert False
    print("run competition %s solution %s"%(FLAGS.comp,FLAGS.sol))
    run_sol(FLAGS)

if __name__ == "__main__":
    try:
        import tensorflow as tf
        use_flags = "tf"
    except:
        use_flags = "arg"

    if use_flags == "tf": 
        from tf_flags import FLAGS
        tf.app.run()
    else:
        from flags import get_parser
        parser = get_parser()
        args = parser.parse_args() 
        FLAGS = args
        main(None)
