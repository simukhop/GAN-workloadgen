import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#matplotlib inline
import keras
from keras.layers import Dense, Dropout, Input, Reshape
from keras.models import Model,Sequential
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv("data/smaller-cern-100001.csv")
    data = data[['wb']]
    data.loc[data.wb > 0, 'class'] = '1'
    data.loc[data.wb == 0, 'class'] = '0'
    #print(data.head())

#Split train and test
    y=data['class']
    x=data['wb']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
    print(x_train.head())
    return (x_train, y_train, x_test, y_test)

def adam_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def create_generator():
    model=Sequential()
    model.add(Dense(256,input_shape=(1,)))
    model.add(LeakyReLU(0.2))

    #model.add(Dense(units=256))
    #model.add(LeakyReLU(0.2))

    model.add(Dense(512))
    model.add(LeakyReLU(0.2))

    #model.add(Dense(units=1024))
    #model.add(LeakyReLU(0.2))

    model.add(Dense(784, activation='relu'))
    model.add(Reshape(1))
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return model
g=create_generator()
g.summary()

def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=784,input_dim=1))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    #discriminator_input = Input (shape=(1,))

    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return discriminator
d =create_discriminator()
d.summary()


#Create the GAN using the previously created MLP discriminator and generator
def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(1,))
    x = generator(gan_input)
    #x = x.reshape((1,x.shape[1]))
    gan_output= discriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan
gan = create_gan(d,g)
gan.summary()

