from comps.carvana.btb1.data import tfCarData

def run(flags):
    data = tfCarData(flags)
    data.write_tfrecords()
