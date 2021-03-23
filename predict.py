from utils.helpers import check_related_path, get_colored_info, color_encode
from utils.utils import load_image, decode_one_hot
from keras_applications import imagenet_utils
from builders import builder
from PIL import Image
import numpy as np
import argparse
import sys
import cv2
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


model = 'FCN-8s'
base_model = 'ResNet50'
csv_file = 'CamVid/class_dict.csv'
num_classes = 32
crop_height = 256
crop_width = 256
weights = 'weights/{Weight-name}.h5'
image_path = 'predictions/{Image-name}.png'
color_encode1 = True

paths = check_related_path(os.getcwd())

if not os.path.exists(image_path):
    raise ValueError('The path \'{image_path}\' does not exist the image file.'.format(image_path=image_path))

net, base_model = builder(num_classes, (crop_height, crop_width), model, base_model)

print('Loading the weights...')
if weights is None:
    net.load_weights(filepath=os.path.join(
        paths['weigths_path'], '{model}_based_on_{base_model}.h5'.format(model=model, base_model=base_model)))
else:
    if not os.path.exists(weights):
        raise ValueError('The weights file does not exist in \'{path}\''.format(path=weights))
    net.load_weights(weights)

print("\n***** Begin testing *****")
print("Model -->", model)
print("Base Model -->", base_model)
print("Num Classes -->", num_classes)

print("")

# load_images
image_names = list()
if os.path.isfile(image_path):
    image_names.append(image_path)
else:
    for f in os.listdir(image_path):
        image_names.append(os.path.join(image_path, f))
    image_names.sort()

# get color info
if csv_file is None:
    csv_file = os.path.join('CamVid', 'class_dict.csv')
else:
    csv_file = csv_file

_, color_values = get_colored_info(csv_file)

for i, name in enumerate(image_names):
    sys.stdout.write('\rRunning test image %d / %d' % (i + 1, len(image_names)))
    sys.stdout.flush()

    image = cv2.resize(load_image(name), dsize=(crop_width, crop_height))
    image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last', mode='torch')

    # image processing
    if np.ndim(image) == 3:
        image = np.expand_dims(image, axis=0)
    assert np.ndim(image) == 4

    # get the prediction
    prediction = net.predict(image)

    if np.ndim(prediction) == 4:
        prediction = np.squeeze(prediction, axis=0)

    # decode one-hot
    prediction = decode_one_hot(prediction)

    # color encode
    if color_encode1:
        prediction = color_encode(prediction, color_values)

    # get PIL file
    prediction = Image.fromarray(np.uint8(prediction))

    # save the prediction
    _, file_name = os.path.split(name)
    prediction.save(os.path.join(paths['prediction_path'], file_name))
