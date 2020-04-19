import os
import glob
import argparse
import numpy as np
from scipy import misc
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import random

class generator_inference():
    def __init__(self,args):
        # Input shape
        self.generator =  load_model(args.generator)
        self.lr_height = self.generator.layers[0].input_shape[1]
        self.lr_width = self.generator.layers[0].input_shape[2]
        self.output_dir = args.outputdir

    def dataloader(self,img_path_list):
        imgs_lr = []
        for img_path in img_path_list:
            img = misc.imread(img_path, mode = 'RGB').astype(np.float)

            img_res = (self.lr_height, self.lr_width)
            low_h, low_w = img_res
            img_lr = misc.imresize(img, (low_h, low_w), interp = 'bicubic')
            imgs_lr.append(img_lr)

            fig = plt.figure()
            plt.imshow(img_lr)
            plt.axis('off')
            fig.savefig(os.path.join(self.output_dir,os.path.basename(img_path)),bbox_inches='tight',pad_inches=0)
            plt.close()

        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        return imgs_lr



    def inference(self,img_path_list,batch_size):
        name_recorder = []
        fake_hrs = None
        while len(img_path_list)>0:
            if len(img_path_list)>batch_size: #inference with batch sample
                sub_img_ind = random.sample(range(len(img_path_list)),batch_size)
                sub_img_list = [img_path_list[i] for i in sub_img_ind]
                for i in sub_img_list:img_path_list.remove(i)
            else:
                sub_img_list = img_path_list
                img_path_list = []

            imgs_lr = self.dataloader(sub_img_list)
            print('batch generating...')
            fake_hr = self.generator.predict(imgs_lr)

            if fake_hrs is None:
                fake_hrs = fake_hr
            else:
                fake_hrs = np.concatenate((fake_hrs,fake_hr),axis=0)

            for n in sub_img_list:name_recorder.append(os.path.basename(n))

        # Rescale images 0 - 255
        fake_rgbs = None
        for i in range(fake_hrs.shape[0]):
            fake_hr_i = np.round((fake_hrs[i] + 1) * 255 / 2)
            fake_rgb = cv2.cvtColor(fake_hr_i, cv2.COLOR_BGR2RGB)
            if fake_rgbs is None:
                fake_rgbs = np.expand_dims(fake_rgb, axis=0)
            else:
                fake_rgbs = np.concatenate((fake_rgbs,np.expand_dims(fake_rgb, axis=0)),axis=0)

        return fake_rgbs,name_recorder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str,default = './testset', help = 'image path')
    parser.add_argument('--outputdir', type = str, default = './outputs', help = 'output directory')
    parser.add_argument('--generator', type = str, help = 'model path')
    args = parser.parse_args()

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)

    g = generator_inference(args)
    image_path = args.image_path
    img_path_list  = glob.glob(os.path.join(image_path,'*.jpg'))

    fake_rgbs,name_recorder = g.inference(img_path_list,batch_size=3)
    for i in range(fake_rgbs.shape[0]):
        img_name = 'gen_'+name_recorder[i]
        cv2.imwrite(os.path.join(args.outputdir,img_name), fake_rgbs[i])


