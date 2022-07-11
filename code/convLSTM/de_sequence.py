import numpy as np

HEIGHT = 31
WIDTH = 71
FRAMES = 3

SEQUENCE = np.load('sequence_array.npz')['sequence_array']  # load array
print(SEQUENCE[0])
print('Data loaded.')
NUMBER = len(SEQUENCE) #115*12
SEQUENCE = SEQUENCE.reshape(NUMBER, HEIGHT, WIDTH, 1)
BASIC_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, HEIGHT, WIDTH, 1))
NEXT_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, HEIGHT, WIDTH, 1))
for i in range(FRAMES):
    print(i)
    BASIC_SEQUENCE[:, i, :, :, :] = SEQUENCE[i:i+NUMBER-FRAMES]
    NEXT_SEQUENCE[:, i, :, :, :] = SEQUENCE[i+1:i+NUMBER-FRAMES+1]

track_month=np.zeros((15*12,HEIGHT,WIDTH))
for which in range(1200,(15*12+1200-3)):
    t1 = BASIC_SEQUENCE[which][0, ::, ::, ::]
    t1=np.squeeze(t1,axis=None)
    track_month[which-1200] = t1


t1 = NEXT_SEQUENCE[(15*12+1200-4)][::, ::, ::, ::]
t1=np.squeeze(t1,axis=None)
track_month[(15*12-3):(15*12)] = t1

track_month=track_month.reshape(-1,71)
np.savetxt('results/spei.txt',track_month)



