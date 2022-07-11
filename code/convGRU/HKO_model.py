import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import sys
import cv2
import os
import time
from forecaster import Forecaster
from encoder import Encoder
from BMSELoss import BMSELoss
import pickle
from scipy import misc

input_num_seqs = 10
output_num_seqs = 10
hidden_size = 3
input_channels_img = 1
output_channels_img = 1
size_image = 240
max_epoch = 10
cuda_flag = False
kernel_size = 3
batch_size = 4
lr= 0.0001
momentum = 0.5
stride=5
IMAGE_BASEPATH = '/Users/xinhuang/Documents/Lab/WORK/drought/convGRU/code/testData/' 
IMAGE_PATH_train = os.path.join(IMAGE_BASEPATH,'train/')
IMAGE_PATH_vali = os.path.join(IMAGE_BASEPATH,'vali/')
IMAGE_PATH_test = os.path.join(IMAGE_BASEPATH,'test/')
IMAGE_SAVEPATH = os.path.join(IMAGE_BASEPATH,'pred/')

class HKOModel(nn.Module):
    def __init__(self, inplanes, input_num_seqs, output_num_seqs):
        super(HKOModel, self).__init__()
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        self.encoder = Encoder(inplanes=inplanes, num_seqs=input_num_seqs)
        self.forecaster = Forecaster(num_seqs=output_num_seqs)

    def forward(self, data):
        self.encoder.init_h0()
        for time in xrange(self.input_num_seqs):
            self.encoder(data[time])
        all_pre_data = []
        self.forecaster.set_h0(self.encoder)
        for time in xrange(self.output_num_seqs):
            pre_data = self.forecaster(None)
            # print h_next.size()
            all_pre_data.append(pre_data)

        return all_pre_data


def train_by_stype(model, loss, optimizer, x_val, y_val):
    # model.encoder.init_h0()
    # for time in xrange(model.input_num_seqs):
    #     h_next = model.encoder(x_val[time])
    #
    # all_pre_data = []
    #
    # model.forecaster.set_h0(model.encoder)
    #
    # for time in xrange(model.output_num_seqs):
    #     pre_data, h_next = model.forecaster(None)
    #     # print h_next.size()
    #     all_pre_data.append(pre_data)

    fx = model.forward(x_val)
    all_loss = 0

    for pre_id in range(len(fx)):
        output = loss.forward(fx[pre_id], y_val[pre_id])
        all_loss += output
        optimizer.zero_grad()
        output.backward(retain_graph=True)
        optimizer.step()
        # if pre_id == 1:
        #     print 'loss 1:',output
    return all_loss.cuda().data[0], fx


def train(model, loss, optimizer, x_val, y_val):
    # x = Variable(x_val.cuda(), requires_grad=False)
    # y = Variable(y_val.cuda(), requires_grad=False)
    optimizer.zero_grad()
    fx = model.forward(x_val)
    output = 0
    # t_y = fx.cpu().data.numpy().argmax(axis=1)
    # acc = 1. * np.mean(t_y == y_val.numpy())
    for pre_id in range(len(fx)):
        output += loss.forward(fx[pre_id], y_val[pre_id])
        # if pre_id == 1:
        #     print 'loss 1:',output
    output/=10.
    output.backward()
    optimizer.step()

    return output.cuda().data[0], fx


def verify(model, loss, x_val, y_val):
    fx = model.forward(x_val)
    output = 0
    for pre_id in range(len(fx)):
        output += loss.forward(fx[pre_id], y_val[pre_id])
    return output.cuda().data[0]


def sample(imgPath,mode):
    fList=os.listdir(imgPath)
    fList.sort()
    if mode=='random':
        batch_num=len(fList)//batch_size
        imgList=fList[:batch_num*batch_size]
        imgList=np.reshape(imgList,(batch_num,batch_size))
        l=np.arange(batch_num)
        np.random.shuffle(l)
        imgList=imgList[l]
    else:
        batch_num=len(fList)-batch_size+1
        f_in=[np.arange(batch_size)+x for x in range(batch_num)]
        imgList=[fList[f_in[i][j]] for i in range(batch_num) for j in range(batch_size)]
        imgList=np.reshape(imgList,(batch_num,batch_size))
    imgList=imgList.T
    return imgList
	

def load_data(imgList,imgPath,nSeq_in):                                            
    SEQUENCE=np.zeros((batch_size,input_num_seqs,size_image*2,size_image))          
    for i in range(batch_size):
    	for j in range(input_num_seqs):                                          
            image_array = misc.imread(os.path.join(imgPath, imgList[i,j+nSeq_in]))
            SEQUENCE[i,j,...] = image_array           
            print(imgList[i+nSeq_in,j], time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    SEQUENCE=SEQUENCE/255.
    return SEQUENCE


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_t = lr
    lr_t = lr_t * (0.3 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_t


def touch_dir(path):
    result = False
    try:
        path = path.strip().rstrip("\\")
        if not os.path.exists(path):
            os.makedirs(path)
            result = True
        else:
            result = True
    except:
        result = False
    return result


def test(input_channels_img, output_channels_img, size_image, max_epoch, model, cuda_test):
    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    optimizer = optim.Adam(model.parameters(), lr=(lr), weight_decay=0.005)
    print(model)
    print(optimizer)
    print(criterion)
    valiList=sample(IMAGE_PATH_vali,'sequence')
    testList=sample(IMAGE_PATH_test,'sequence')
    batch_num_vali=valiList.shape[0]//input_num_seqs
    batch_num_test=testList.shape[0]//input_num_seqs
    for i in range(max_epoch):
        trainList=sample(IMAGE_PATH_train,'random')
        batch_num_train=trainList.shape[0]//input_num_seqs
        print('epoch :%s', i)
        model.train()
        all_error = 0.
        for j in range(batch_num_train):
            batch_img=load_data(trainList,IMAGE_PATH_train,j*input_num_seqs)
            input_image = batch_img[:,:size_image, ...] 
            target_image = batch_img[:,size_image:,...] 
            input_image = torch.from_numpy(input_image).float()
            input_gru = Variable(input_image.cuda())
            target_image = torch.from_numpy(target_image).float()
            target_gru = Variable(target_image.cuda())

            error, pre_list = train(model, criterion, optimizer, input_gru, target_gru)
            all_error += error
            print(j, ' : ', error)
        print('epoch train %d %f' % (i, all_error / batch_num))
        # print model.encoder.conv1_act
        # params = model.state_dict()
        # print params.keys()
        # print params['encoder.conv1_act.0.weight']
        model.eval()
        all_error = 0.
        for j in range(batch_num_vali):
            batch_img=load_data(valiList,IMAGE_PATH_vali,j*input_num_seqs)
            input_image = batch_img[:,:size_image, ...] 
            target_image = batch_img[:,size_image:,...] 
            input_image = torch.from_numpy(input_image).float()
            input_gru = Variable(input_image.cuda())
            target_image = torch.from_numpy(target_image).float()
            target_gru = Variable(target_image.cuda())
        
            error = verify(model, criterion, input_gru, target_gru)
            all_error += error
            print(j, ' : ', error)
        print('epoch test %d %f' % (i, all_error / batch_num))
    model.eval()
    touch_dir(IMAGE_SAVEPATH)
    all_error = 0.
    for j in range(batch_num_test):
        batch_img=load_data(testList,IMAGE_PATH_test,j*input_num_seqs)
        input_image = batch_img[:,:size_image, ...] 
        target_image = batch_img[:,size_image:,...] 
        input_image_t = torch.from_numpy(input_image).float()
        input_gru = Variable(input_image_t.cuda())
        target_image = torch.from_numpy(target_image).float()
        target_gru = Variable(target_image.cuda())
        fx = model.forward(input_gru)
        output=0
        for pre_id in range(len(fx)):
            output += criterion.forward(fx[pre_id],target_gru[pre_id])
            temp_xx = fx[pre_id].cpu().data.numpy()
            for b_id in range(len(batch_size)):
                tmp_img = temp_xx[0, b_id, ...]
                tmp_img = tmp_img * 255.
                dy=input_num_seqs*batch_size*pre_id+b_id+1
                cv2.imwrite(os.path.join(IMAGE_SAVEPATH, '2012_%s.png' % dy), tmp_img)
        

if __name__ == '__main__':
    m = HKOModel(inplanes=batch_size, input_num_seqs=input_num_seqs, output_num_seqs=output_num_seqs)
    #m = m.cuda()
    #imgList=sample(IMAGE_PATH_train,'sequence')

   # trainList=sample(IMAGE_PATH_test,'sequence')
   # batch_num_train=trainList.shape[0]//input_num_seqs
   # j=0
   # batch_img=load_data(trainList,IMAGE_PATH_test,j*input_num_seqs)
   # input_image = batch_img[:,:,:size_image, ...]
   # target_image = batch_img[:,:,size_image:,...] 
   # print(input_image[0,0,0,0])
   # input_image = torch.from_numpy(input_image).float()
   # #input_gru = Variable(input_image.cuda())
   # target_image = torch.from_numpy(target_image).float()
   # #target_gru = Variable(target_image.cuda())
    #test(input_channels_img, output_channels_img, size_image, max_epoch, model=m, cuda_test=cuda_flag)
