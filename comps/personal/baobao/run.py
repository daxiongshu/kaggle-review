def run(flags):
    task = flags.task
    if task == "preprocess":
    	from comps.personal.baobao.preprocess import preprocess
    	preprocess(flags)
    elif task == "clean":
        from comps.personal.baobao.clean import clean
        clean(flags)
    elif task == 'write_gene_text':
        from comps.personal.baobao.write_gene import write_gene_text_pypy
        write_gene_text_pypy(flags,window=10,bar=10)
    elif task == 'write_variation_text':
        from comps.personal.baobao.write_gene import write_gene_text_pypy
        write_gene_text_pypy(flags,tag="Variation",window=5,bar=0)
    elif task == "train_embedding":
    	from comps.personal.baobao.embedding import train_embedding
    	train_embedding(flags)
    elif task == "train_d2v":
        from comps.personal.baobao.d2v import train_d2v
        train_d2v(flags)
    elif task == "show_embedding":
        from comps.personal.baobao.embedding import show_embedding
        show_embedding(flags)  
    elif task == "show_d2v":
        from comps.personal.baobao.d2v import show_d2v
        show_d2v(flags)
    elif task == "train_rnn":
        from comps.personal.baobao.rnn import train_rnn
        train_rnn(flags)
    elif task == "train_nn":
        from comps.personal.baobao.nn import train_nn
        train_nn(flags)
    elif task == "eval":
        from comps.personal.baobao.eval import eval
        eval(flags)
        from comps.personal.baobao.xgb import post_cv
        post_cv(flags)
    elif task == "predict_nn":
        from comps.personal.baobao.nn import predict_nn
        predict_nn(flags)
    elif task == "train_cnn":
        from comps.personal.baobao.cnn import train_cnn
        train_cnn(flags)
    elif "test_cnn" in task:
        from comps.personal.baobao.cnn import test_cnn
        test_cnn(flags)
    elif "xgb_cv" in task:
        from comps.personal.baobao.xgb import cv
        cv(flags)
    elif "xgb_sub" in task:
        from comps.personal.baobao.xgb import sub
        sub(flags)


