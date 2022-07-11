import numpy as np
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import matplotlib.pyplot as plt
import time
from keras.utils import multi_gpu_model
from keras import optimizers
from keras.models import load_model

HEIGHT = 31
WIDTH = 71
FRAMES = 12

SEQUENCE = np.load('sequence_array.npz')['sequence_array']  # load array
print(SEQUENCE[0])
print('Data loaded.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

NUMBER = len(SEQUENCE) #115*12
s=SEQUENCE
s=s.reshape(NUMBER,HEIGHT,WIDTH)

SEQUENCE = SEQUENCE.reshape(NUMBER, HEIGHT, WIDTH, 1)
BASIC_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, HEIGHT, WIDTH, 1))
NEXT_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, HEIGHT, WIDTH, 1))


for i in range(FRAMES):
    print(i)
    BASIC_SEQUENCE[:, i, :, :, :] = SEQUENCE[i:i+NUMBER-FRAMES]
    NEXT_SEQUENCE[:, i, :, :, :] = SEQUENCE[i+1:i+NUMBER-FRAMES+1]

def w_mse(y_true, y_pred):
    less1=K.less(y_true, 1/6)
    less2=K.less(y_true, 0.25)
    less3=K.less(y_true, 1/3)
    less4=K.less(y_true, 2/3)
    less5=K.less(y_true, 0.75)
    less6=K.less(y_true, 5/6)
    greater1=K.greater_equal(y_true, 1/6)
    greater2=K.greater_equal(y_true, 0.25)
    greater3=K.greater_equal(y_true, 1/3)
    greater4=K.greater_equal(y_true, 2/3)
    greater5=K.greater_equal(y_true, 0.75)
    greater6=K.greater_equal(y_true, 5/6)
    cond1=less1
    cond2=K.equal(greater1,less2)
    cond3=K.equal(greater2,less3)
    cond4=K.equal(greater3,less4)
    cond5=K.equal(greater4,less5)
    cond6=K.equal(greater5,less6)
    cond7=greater6
    cond1=K.cast(cond1,K.floatx())
    cond2=K.cast(cond2,K.floatx())
    cond3=K.cast(cond3,K.floatx())
    cond4=K.cast(cond4,K.floatx())
    cond5=K.cast(cond5,K.floatx())
    cond6=K.cast(cond6,K.floatx())
    cond7=K.cast(cond7,K.floatx())
    cond=50*cond1+20*cond2+10*cond3+1*cond4+10*cond5+20*cond6+30*cond7
    return K.mean(cond*K.square(y_true-y_pred))

'''
def w_mse(y_true, y_pred):
    less1=K.less(y_true, 0.4)
    less2=K.less(y_true, 0.7)
    greater1=K.greater_equal(y_true, 0.4)
    greater2=K.greater_equal(y_true, 0.7)
    cond1=less1
    cond2=K.equal(greater1,less2)
    cond3=greater2
    cond1=K.cast(cond1,K.floatx())
    cond2=K.cast(cond2,K.floatx())
    cond3=K.cast(cond3,K.floatx())
    cond=10*cond1+cond2+10*cond3
    return K.mean(cond*K.square(y_true-y_pred))
'''
loss = lambda y_true, y_pred: w_mse(y_true, y_pred)

seq = Sequential()

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(None, HEIGHT, WIDTH, 1), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))

# using multiple GPUs, batch_size:10->3
parallel_model = multi_gpu_model(seq, gpus=2)
sgd=optimizers.SGD(lr=0.01,clipnorm=1)
#parallel_model.compile(loss='mean_squared_error', optimizer='adadelta')
parallel_model.compile(loss=loss, optimizer='adadelta')
parallel_model.fit(BASIC_SEQUENCE[:1200], NEXT_SEQUENCE[:1200], batch_size=10, epochs=10, validation_split=0.05)

##using single GPU
#seq.compile(loss='mean_squared_error', optimizer='adadelta')
#seq.fit(BASIC_SEQUENCE[:250], NEXT_SEQUENCE[:250], batch_size=3, epochs=10, validation_split=0.05)

seq.save('outputs/nice_model.h5')
#seq=load_model('nice_model.h5')

'''
# predict drivern by 15 years, validation
track_series=parallel_model.predict(BASIC_SEQUENCE[1200:])
#track_series=seq.predict(BASIC_SEQUENCE[1200:]) #after load model
np.save('outputs/track_series.npy',track_series)
'''

#predict driven by month
track_month=np.zeros((15*12, HEIGHT, WIDTH,1))
for which in range((1200-FRAMES),(15*12+1200-FRAMES)):
    track_base = BASIC_SEQUENCE[which]
    # predict by month
    new_pos = parallel_model.predict(track_base[np.newaxis, ::, ::, ::, ::])
    print(new_pos.shape)
    new = new_pos[::, -1, ::, ::, ::]
    track_month[which-(1200-FRAMES)] = new

print(track_month.shape)
track_month=np.squeeze(track_month,axis=None)
track_month=track_month.reshape(-1,WIDTH)
#np.save('outputs/predict_mn.npy',track_month)
np.savetxt('outputs/predict_mn.txt',track_month)

'''
#predict driven by 12 month
track_yr=np.zeros((15*12, HEIGHT, WIDTH,1))
for which in range((1200-FRAMES),(15*12+1200-FRAMES)):
    track_base = BASIC_SEQUENCE[which]
    for t in range(1,4):
         track_base=np.concatenate((track_base,BASIC_SEQUENCE[which+3*t]),axis=0)
    # predict by month
    new_pos = parallel_model.predict(track_base[np.newaxis, ::, ::, ::, ::])
    print(new_pos.shape)
    new = new_pos[::, -1, ::, ::, ::]
    track_yr[which-(1200-FRAMES)] =  new

print(track_yr.shape)
#np.save('outputs/predict_yr.npy',track_yr)
track_yr=np.squeeze(track_yr,axis=None)
track_yr=track_yr.reshape(-1,WIDTH)
np.savetxt('outputs/predict_yr.txt',track_yr)


#predict driven by 3 month, example
t=BASIC_SEQUENCE[43]
np.squeeze(t)
new_pos = seq.predict(t[np.newaxis, ::, ::, ::, ::])
np.squeeze(new_pos)
new= new_pos[::, -1, ::, ::, ::]
np.squeeze(new)

which = 1296
pos=np.zeros([])
track = BASIC_SEQUENCE[which][:3, ::, ::, ::]
for j in range(12):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    print(new_pos.shape)
    pos=np.append(pos, new_pos)
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

np.save('track.npy',track)

# And then compare the predictions
# to the ground truth
for i in range(FRAMES):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    if i >= 3:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)
    toplot = track[i, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)
    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = NEXT_SEQUENCE[which][i - 1, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    plt.savefig('%i_animate.png' % (i + 1))
'''

