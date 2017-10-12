import tensorflow as tf
from comps.income.income_db import incomeDB
from utils.tf_utils.utils import tf_input_fn,tf_columns 


def build_estimator(model_dir, model_type,
    base_columns=[],crossed_columns=[],deep_columns=[]):
    """Build an estimator."""
    if model_type == "wide":
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=base_columns + crossed_columns)
    elif model_type == "deep":
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir,
            feature_columns=deep_columns,
            hidden_units=[100, 50])
    else:
        m = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=crossed_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100, 50])
    return m

def cv(flags, train_steps):
    """Train and evaluate the model."""
    myDB = incomeDB(flags)
    model_dir = flags.save_path
    model_type = flags.net
    path = flags.input_path
    epochs = flags.epochs
    train,test = myDB.data['train'],myDB.data['test']
    B = flags.batch_size

    cols = tf_columns(train.drop('target',axis=1)) 
    m = build_estimator(model_dir, model_type, base_columns=cols)
    # set num_epochs to None to get infinite stream of data.
    m.train(
        input_fn=tf_input_fn(train, ycol='target', batch_size=B, 
            epochs=epochs, shuffle=True, threads=8),
        steps=train_steps)
    # set steps to None to run evaluation until all data consumed.
    results = m.evaluate(
        input_fn=tf_input_fn(test, ycol='target', epochs=1,
            batch_size=B,shuffle=False),
        steps=None)
    print("model directory = %s" % model_dir)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


