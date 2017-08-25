#based on https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model
import cv2
import numpy as np
import random

def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gen_random_image(w,h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0+1, 255)
    light_color1 = random.randint(dark_color1+1, 255)
    light_color2 = random.randint(dark_color2+1, 255)
    center_0 = random.randint(0, h)
    center_1 = random.randint(0, w)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(h):
        for j in range(w):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, mask

def random_batch_generator(batch_size,w,h):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = gen_random_image(w=w,h=h)
            image_list.append(img)
            mask_list.append(mask)

        image_list = np.array(image_list, dtype=np.float32)
        #image_list = image_list.transpose((0, 3, 1, 2))
        #image_list = preprocess_batch(image_list)
        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0
        yield image_list, mask_list, 0

#def preprocess_batch(batch):
#    batch /= 256
#    batch -= 0.5
#    return batch

if __name__ == "__main__":
    #print(cv2.IMREAD_COLOR,cv2.IMREAD_GRAYSCALE,cv2.IMREAD_UNCHANGED)
    # 1 0 -1

    #im = cv2.imread('C:/Users/Jiwei/Pictures/me2.png',1)
    img,mask = gen_random_image(224,224) 
    show_image(img)
