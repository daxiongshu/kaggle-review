from comps.carvana.btb1.data import tfCarData
from models.tf_models.unet.zf_unet import ZF_UNET
from comps.carvana.btb1.post_sub import post_sub_all

def run(flags):

    if flags.task == "post_sub":
        post_sub_all(flags.pred_path,flags.threshold)
        return

    data = tfCarData(flags)
    if flags.record_path is not None:
        data.write_tfrecords()

    if flags.net == "zf_unet":
        model = ZF_UNET(flags,data)

    if flags.task == "train_random":
        model.trainPL()

    elif flags.task == "train_cv":
        model.trainPL(mode="cv",do_va=True)

    else:
        model.predictPL()
