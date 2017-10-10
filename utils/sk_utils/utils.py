from sklearn.datasets import load_svmlight_file
#mem = Memory("./mycache")

#@mem.cache
def load_svm(name):
    data = load_svmlight_file(name)
    return data[0], data[1]

