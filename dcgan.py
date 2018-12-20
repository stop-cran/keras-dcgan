from keras import backend as K
from keras.datasets import mnist
from keras.engine import Layer, InputSpec
from keras.layers import Dense, Dropout, Reshape, Input, Lambda, concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.utils.vis_utils import model_to_dot

from IPython.display import SVG
from PIL import Image



%matplotlib inline
import argparse
import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import scipy
import tensorflow as tf


class UniformNoiseSource1D(Layer):
    def __init__(self, size=2, minval=0.0, maxval=1.0, seed=None, **kwargs):
        super(UniformNoiseSource1D, self).__init__(**kwargs)
        self.size = int(size)
        self.minval = minval
        self.maxval = maxval
        self.seed = seed
        self.input_spec = InputSpec(ndim=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.size)

    def call(self, inputs):
        return K.random_uniform(self.compute_output_shape(tf.shape(inputs)),
                                minval = self.minval,
                                maxval = self.maxval,
                                seed   = self.seed)

    def get_config(self):
        config = {'size': self.size, 'minval': self.minval, 'maxval': self.maxval, 'seed': self.seed}
        base_config = super(UniformNoiseSource1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def generator_model():
    input_layer  = Input(shape=(4, ))
    noise_layer  = UniformNoiseSource1D(size = 96, minval = -1.0, maxval = 1.0)(input_layer)
    concat_input = concatenate([input_layer, noise_layer])
    
    layer = Dense(1024)(concat_input)
    layer = Activation('tanh')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(128*7*7)(layer)
    layer = BatchNormalization()(layer)
    layer = Activation('tanh')(layer)
    layer = Reshape((7, 7, 128), input_shape=(128*7*7,))(layer)
    layer = Dropout(0.1)(layer)
    layer = UpSampling2D(size=(2, 2))(layer)
    layer = Conv2D(64, (5, 5), padding='same')(layer)
    layer = Activation('tanh')(layer)
    layer = Dropout(0.05)(layer)
    layer = UpSampling2D(size=(2, 2))(layer)
    layer = Conv2D(1, (5, 5), padding='same')(layer)
    layer = Activation('tanh')(layer)
    
    return Model(inputs=[input_layer], outputs=[layer])

def discriminator_model():
    input_layer = Input(shape=(28, 28, 1))
    layer = Conv2D(128, (5, 5), padding='same')(input_layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(256, (5, 5))(layer)
    layer = Activation('relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(0.15)(layer)
    layer = Flatten()(layer)
    layer = Dense(2048)(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(4)(layer)
    layer = Activation('sigmoid')(layer)
    return Model(inputs=[input_layer], outputs=[layer])


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    model.add(d)
    return model


def load_models():
    d = discriminator_model()
    g = generator_model()

    if (os.path.isfile('weights/generator')):
        g.load_weights('weights/generator')
        print('loaded generator weights')

    if (os.path.isfile('weights/discriminator')):
        d.load_weights('weights/discriminator')
        print('loaded discriminator weights')

    d.trainable = False
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    return (d, g, d_on_g)


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def save_png(image_array, name):
    file_name = 'images/' + name + '.png'
    print('Saving ', file_name, '...')
    Image.fromarray(
        (
            (combine_images(image_array) + 1)*0.5*255
        ).astype(np.uint8)
    ).save(file_name)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32)*2/255 - 1
    return (
        x_train[:, :, :, None],
        np.unpackbits(
            (np.expand_dims(y_train, axis=1) + 1).astype(np.uint8),
            axis=1)[:,4:])


def digits_batch(BATCH_SIZE):
    return np.unpackbits(
        np.random.uniform(1, 11, size=(BATCH_SIZE, 1)).astype(np.uint8),
        axis=1)[:,4:]

def digits_batch_range(BATCH_SIZE):
    return np.unpackbits(
        np.expand_dims(
            np.repeat(np.arange(1, 11, dtype=np.uint8), BATCH_SIZE),
            axis=1),
        axis=1)[:,4:]

def load_epoch_batch_number(epoch_batch_filename):
    if (os.path.isfile(epoch_batch_filename)):
        with open(epoch_batch_filename) as f:
            content = f.readlines()
            if (len(content) == 2):
                return (int(content[0]), int(content[1]))
    return (0, 0)

def save_epoch_batch_number(epoch_batch_filename, epoch, batch):
    with open(epoch_batch_filename, 'w+') as f:
        f.write("%d\n%d" % (epoch, batch))

def train(EPOCH_COUNT, BATCH_SIZE):
    print ('Beginning train %d epoch, batch size is %d ...' % (EPOCH_COUNT, BATCH_SIZE))
    (x_train, y_train) = load_data()
    (d, g, d_on_g) = load_models()
    batch_count = int(x_train.shape[0]/BATCH_SIZE) + 1

    if not os.path.exists('images'):
        os.makedirs('images')
    if not os.path.exists('weights'):
        os.makedirs('weights')
        
    (begin_epoch, begin_batch) = load_epoch_batch_number('weights/epoch_batch.txt')
    print("Starting from epoch %d, batch %d" % (begin_epoch, begin_batch))
    
    if begin_epoch == 0 and begin_batch == 0:
        with open("weights/losses.txt", "w") as losses:
            losses.write("epoch\tbatch\td_loss\tg_loss\n")

    
    for epoch in range(begin_epoch, EPOCH_COUNT):
        print("Epoch is", epoch)
        print("Number of batches", batch_count)
        began = False
        for index in range(begin_batch, batch_count):
            if not began or g_loss < 1.2 * d_loss:
                x_train_batch = x_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
                y_train_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

                # train discriminator to recognize real digits and give zero score to generated digits
                x = np.concatenate((x_train_batch, g.predict(y_train_batch, verbose=0)))
                y = np.concatenate((y_train_batch, np.zeros(y_train_batch.shape)))
                d_loss = d.train_on_batch(x, y)

            if not began or d_loss < 1.2 * g_loss:
                d.trainable = False

                x = digits_batch(BATCH_SIZE)
                np.random.shuffle(x)
                g_loss = d_on_g.train_on_batch(x, x)

                d.trainable = True
                
            began = True

            if index % 50 == 0:
                print("batch %d d_loss : %f" % (index, d_loss))
                print("batch %d g_loss : %f" % (index, g_loss))
                save_png(g.predict(digits_batch_range(10), verbose=0),
                         "%d_%d" % (epoch, index))
                g.save_weights('weights/generator', True)
                d.save_weights('weights/discriminator', True)
                save_epoch_batch_number('weights/epoch_batch.txt', epoch, index)
                with open("weights/losses.txt", "a") as losses:
                    losses.write("%d\t%d\t%f\t%f\n" % (epoch, index, d_loss, g_loss))

        begin_batch = 0


def generate(BATCH_SIZE, numbers, nice=False):
    print ('Generating images...')
    g = generator_model()
    g.load_weights('weights/generator')
    print('loaded generator weights')
    g.compile(loss='binary_crossentropy', optimizer="SGD")

    if not os.path.exists('images/generated'):
        os.makedirs('images/generated')
    save_png(g.predict(np.tile(numbers, (BATCH_SIZE,1)), verbose=1), 'generated/generated_image')


def generate_nice(BATCH_SIZE, numbers):
    print ('Generating nice images...')
    (d, g, d_on_g) = load_models()

    generated_images = g.predict(np.tile(numbers, (BATCH_SIZE*20,1)), verbose=1)
    scores = d.predict(generated_images, verbose=1)
    nice_images = next(zip(*sorted(zip(generated_images, scores),
                            key=lambda x: np.linalg.norm(np.array(x[1]) - numbers))))[0:BATCH_SIZE]
    if not os.path.exists('images/generated'):
        os.makedirs('images/generated')
    save_png(np.array(nice_images), 'generated/generated_image')


def generate_transform(digits, BATCH_SIZE, SELECT_SIZE):
    def digitToArray(digit):
        return np.unpackbits(np.array([[digit + 1]], dtype=np.uint8), axis=1)[0][4:].astype(np.float)

    print ('Generating transform...')
    (d, g, d_on_g) = load_models()

    digits = list(digits)
    digits_count = len(digits) - 1
    all_numbers = np.zeros((BATCH_SIZE*SELECT_SIZE*digits_count, 4))

    for index in range(digits_count):
        for n in range(BATCH_SIZE*SELECT_SIZE):
            x = float(n)/(BATCH_SIZE*SELECT_SIZE)
            all_numbers[BATCH_SIZE*SELECT_SIZE*index+n] = digitToArray(digits[index])*(1-x) + digitToArray(digits[index+1])*x
    if not os.path.exists('images/generated'):
        os.makedirs('images/generated')

    generated_images = g.predict(all_numbers, verbose=1)
    scores = d.predict(generated_images, verbose=1)

    images = []
    nice_image = np.array([])
    for n in range(BATCH_SIZE*digits_count):
        generated_images_b = generated_images[n*SELECT_SIZE:(n+1)*SELECT_SIZE]
        scores_b = scores[n*SELECT_SIZE:(n+1)*SELECT_SIZE]
        numbers = all_numbers[n*SELECT_SIZE][0:10]
        if nice_image.shape[0]!=0:
            n_size=int(SELECT_SIZE/10)
            generated_images_b = next(zip(*sorted(zip(generated_images_b, scores_b),
                                          key=lambda x: np.linalg.norm(np.array(x[1]) - numbers))))[0:n_size]
            nice_image = list(sorted(iter(list(generated_images_b)),
                                     key=lambda x: np.linalg.norm((x.reshape(28,28) - nice_image.reshape(28,28)).reshape(28*28))
                                    )
                             )[0]
        else:
            nice_image = next(zip(*sorted(zip(generated_images_b, scores_b),
                                          key=lambda x: np.linalg.norm(np.array(x[1]) - numbers))))[0]
        Image.fromarray(
            (
                (nice_image + 1)*0.5*255
            ).astype(np.uint8).reshape((28,28))

        ).save('images/generated/temp.png')
        images.append(imageio.imread('images/generated/temp.png'))
    imageio.mimsave('images/generated/'+str(digits[0])+'-'+str(digits[digits_count])+'.gif', images, fps=BATCH_SIZE/2)
    os.remove('images/generated/temp.png')

def save_train_gif():
    def filekey(file):
        index = file.find('_')
        try:
            return int(file[0:index])*1000 + int(file[index+1:file.find('.')])
        except Exception:
            return -1

    images = []
    for file in sorted(os.listdir('images'), key=filekey):
        if os.path.isfile('images/' + file) and file.endswith('.png'):
            images.append(imageio.imread('images/' + file))
    imageio.mimsave('images/generated/train.gif', np.array(images), fps=10)

def show_model(model):
    return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

def plot_losses():
    xy = np.loadtxt('weights/losses.txt', skiprows=1)
    #xy = np.log(xy)

    t = xy[:,0] + xy[:,1]*0.001
    plt.plot(t, xy[:,2], '-', label='generator loss')
    plt.plot(t, xy[:,3], '-', label='discriminator loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Price event')
    plt.rcParams["figure.figsize"] = [12,8]
    plt.grid(True)
    plt.legend()
    plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--transform_length", type=int, default=32)
    parser.add_argument("--epoch_count", type=int, default=100)
    parser.add_argument("--digit", type=int)
    parser.add_argument("--digit_from", type=int)
    parser.add_argument("--digit_to", type=int)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(args.epoch_count, args.batch_size)
    elif args.mode == "generate" and args.nice == True:
        generate_nice(args.batch_size, digitToArray(args.digit))
    elif args.mode == "generate" and args.nice == False:
        generate(args.batch_size, digitToArray(args.digit))
    elif args.mode == "transform":
        generate_transform([int(args.digit_from), int(args.digit_to)], args.batch_size, args.transform_length)