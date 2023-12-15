import os.path
import torchvision


def save_img(img, im_dir, im_name):
    folder_dir = os.path.join(im_dir, os.path.split(im_name)[0])
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    save_path = os.path.join(im_dir, im_name)
    torchvision.utils.save_image(img, save_path)
