from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.callbacks import LearningRateScheduler
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils import utils
from builders import builder
import tensorflow as tf
import argparse
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


backend = tf.keras.backend


def categorical_crossentropy_with_logits(y_true, y_pred):
    # compute cross entropy
    cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

    # compute loss
    loss = backend.mean(backend.sum(cross_entropy, axis=[1, 2]))
    return loss


model = 'FCN-8s'
base_model = 'ResNet50'
dataset = 'CamVid'
loss = categorical_crossentropy_with_logits
num_classes = 32
random_crop = False
crop_height = 256
crop_width = 256
batch_size = 5
valid_batch_size = 1
num_epochs = 100
initial_epoch = 0
h_flip = False
v_flip = False
brightness = None
rotation = 0.
zoom_range = 0.
channel_shift = 0.
data_aug_rate = 0.
checkpoint_freq = 1
validation_freq = 1
num_valid_images = 20
data_shuffle = True
random_seed = None
weights = None
steps_per_epoch = None
lr_scheduler = 'step_decay'
lr_warmup = False
learning_rate = 3e-4
# adagrad optimizer =>
optimizer = tf.keras.optimizers.Adagrad

paths = check_related_path(os.getcwd())

train_image_names, train_label_names, valid_image_names, valid_label_names, _, _ = get_dataset_info(dataset)

net, base_model = builder(num_classes, (crop_height, crop_width), model, base_model)

net.summary()

if weights is not None:
    print('Loading the weights...')
    net.load_weights(weights)

total_iterations = len(train_image_names) * num_epochs // batch_size
wd_dict = utils.get_weight_decays(net)
ordered_values = []
weight_decays = utils.fill_dict_in_order(wd_dict, ordered_values)

optimizers = {'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
              'nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate),
              'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99)}

# loss = {'ce': categorical_crossentropy_with_logits}

if lr_warmup and num_epochs - 5 <= 0:
    raise ValueError('num_epochs must be larger than 5 if lr warm up is used.')

lr_decays = {'step_decay': step_decay(learning_rate, num_epochs - 5 if lr_warmup else num_epochs,
                                      warmup=lr_warmup),
             }
lr_decay = lr_decays[lr_scheduler]

steps_per_epoch = len(train_image_names) // batch_size if not steps_per_epoch else steps_per_epoch

validation_steps = num_valid_images // valid_batch_size

net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=[MeanIoU(num_classes)])

train_gen = ImageDataGenerator(random_crop=random_crop,
                               rotation_range=rotation,
                               brightness_range=brightness,
                               zoom_range=zoom_range,
                               channel_shift_range=channel_shift,
                               horizontal_flip=v_flip,
                               vertical_flip=v_flip)

valid_gen = ImageDataGenerator()

train_generator = train_gen.flow(images_list=train_image_names,
                                 labels_list=train_label_names,
                                 num_classes=num_classes,
                                 batch_size=batch_size,
                                 target_size=(crop_height, crop_width),
                                 shuffle=data_shuffle,
                                 seed=random_seed,
                                 data_aug_rate=data_aug_rate)

valid_generator = valid_gen.flow(images_list=valid_image_names,
                                 labels_list=valid_label_names,
                                 num_classes=num_classes,
                                 batch_size=valid_batch_size,
                                 target_size=(crop_height, crop_width))

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(paths['checkpoint_path'],
                          '{model}_basedon_{base}_'.format(model=model, base=base_model) +
                          'miou_{value_mean_io_u:04f}_' + 'ep_{epochs:02d}.h5'),
    save_best_only=True, period=checkpoint_freq, monitor='val_mean_io_u', mode='max')

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=paths['logs_path'])

learning_rate_scheduler = LearningRateScheduler(lr_decay, learning_rate, lr_warmup, steps_per_epoch,
                                                verbose=1)

callbacks = [model_checkpoint, tensorboard, learning_rate_scheduler]

print("\n***** Begin training *****")
print("Dataset -->", dataset)
print("Num Images -->", len(train_image_names))
print("Model -->", model)
print("Base Model -->", base_model)
print("Num Epochs -->", num_epochs)
print("Initial Epoch -->", initial_epoch)
print("Batch Size -->", batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tData Augmentation Rate -->", data_aug_rate)
print("\tChannel Shift -->", channel_shift)

print("")

net.fit_generator(train_generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=num_epochs,
                  callbacks=callbacks,
                  validation_data=valid_generator,
                  validation_steps=validation_steps,
                  validation_freq=validation_freq,
                  max_queue_size=10,
                  workers=os.cpu_count(),
                  use_multiprocessing=False,
                  initial_epoch=initial_epoch)

net.save(filepath=os.path.join(
    paths['weights_path'], '{model}_basedon_{base_model}.h5'.format(model=model, base_model=base_model)))
