import shutil

from encode_image import main
import os
import argparse
import random

train_path = '/home/josh/bd4auth/StegaStamp/imagenette/train'
test_path = '/home/josh/bd4auth/StegaStamp/imagenette/test'
encoded_train_path = '/home/josh/bd4auth/StegaStamp/imagenette/encoded_train'
# B_encoded_train_path = '/home/josh/bd4auth/StegaStamp/imagenette/B_encoded_train'
encoded_test_path = '/home/josh/bd4auth/StegaStamp/imagenette/encoded_test'
# B_encoded_test_path = '/home/josh/bd4auth/StegaStamp/imagenette/B_encoded_test'


def encoded_data(source_path, target_path):
    for cls in os.listdir(source_path):
        absolute_cls_path = os.path.join(source_path, cls)
        args = argparse.Namespace(
            model='/home/josh/bd4auth/StegaStamp/saved_models/mytrain_new/saved_model.pb',  # 要去encode里面改
            image=None,
            images_dir=absolute_cls_path,
            save_dir=os.path.join(target_path, cls),
            secret='targetA'
        )
        main(args)


def remove_residual(path):
    for cls in os.listdir(path):
        for image in os.listdir(os.path.join(path, cls)):
            if 'residual' in image:
                # print(os.path.join(os.path.join(path, cls), image))
                os.remove(os.path.join(os.path.join(path, cls), image))


def B_encoded_data(source_path, target_path):
    for cls in os.listdir(source_path):
        # if cls in random.sample(os.listdir(source_path), 5):
        # if cls in ['n01440764', 'n02102040', '']:
        absolute_cls_path = os.path.join(source_path, cls)
        args = argparse.Namespace(
            model=None,  # 要去encode里面改
            image=None,
            images_dir=absolute_cls_path,
            save_dir=os.path.join(target_path, cls),
            secret='B'
        )
        main(args)


def random_select_num_image(source_path, target_path, n_image):
    for cls in os.listdir(source_path):
        cls_path = os.path.join(source_path, cls)
        print(cls_path)
        images = os.listdir(cls_path)
        sample = random.sample(images, n_image)
        for image in sample:
            if not os.path.exists(os.path.join(target_path, cls)):
                os.makedirs(os.path.join(target_path, cls))
            shutil.copy(os.path.join(cls_path, image), os.path.join(target_path, cls))


def add_10_trian_image():
    pass


if __name__ == '__main__':
    # random_select_num_image(train_path, '/home/josh/bd4auth/StegaStamp/imagenette/train_600', n_image=600)
    # encoded_data('/home/josh/bd4auth/StegaStamp/imagenette/train_600', '/home/josh/bd4auth/StegaStamp/imagenette/bd_train_600')
    # remove_residual(encoded_train_path)
    # B_encoded_data(test_path, B_encoded_test_path)

    # yt_train_dir = '/home/josh/bd4auth/backdoor/youtube/Valid_data'
    num = 60
    random_select_num_image('/home/josh/bd4auth/backdoor/youtube/Train_data',
                            f'/home/josh/bd4auth/backdoor/youtube/Train_data_{num}', num)
    encoded_data(f'/home/josh/bd4auth/backdoor/youtube/Train_data_{num}',
                 f'/home/josh/bd4auth/backdoor/youtube/bd_Train_data_{num}')
