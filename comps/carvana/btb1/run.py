from comps.carvana.btb1.data import tfCarData
from comps.carvana.btb1.car_zf_unet import carZF_UNET 

def run(flags):

    data = None
    if flags.record_path is not None:
        data = tfCarData(flags)
        data.write_tfrecords()

    if flags.net == "car_zf_unet":
        model = carZF_UNET(flags)

    if flags.task == "train_random":
        model.train(mode="random")

    elif flags.task == "train_cv":
        model.train(mode="cv")
