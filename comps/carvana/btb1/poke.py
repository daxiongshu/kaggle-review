from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.misc import imresize

def show_one_img_mask(data):
    w,h = 1918,1280
    a = randint(0,31)
    path = "../input/test"
    data = np.load(data).item()
    name,masks = data['name'][a],data['pred']
    img = Image.open("%s/%s"%(path,name))
    #img.show()
    plt.imshow(img)
    plt.show()
    mask = np.squeeze(masks[a])
    mask = imresize(mask,[h,w]).astype(np.float32)
    print(mask.shape,mask[0])
    img = Image.fromarray(mask*256)#.resize([w,h])
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    show_one_img_mask("comps/carvana/btb1/pred/1120.npy")
