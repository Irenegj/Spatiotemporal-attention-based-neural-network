import sys
from Framework import STA_STIN
import datetime
import argparse
import os
import shutil
import time
import math
import Engine as Engine
import numpy as np
import torch
import matplotlib.pyplot as plt
import random

torch.backends.cudnn.benchmark = True
if_cuda = False
parser = argparse.ArgumentParser(description='Spatial-temporal attention based route risk assessment model')
parser.add_argument('--model', type=str, default='STA_STIN',
                    help='type of recurrent net')
parser.add_argument('--database',type=str,default='',
                    help='type of recurrent net')
parser.add_argument('--N_SAM', type=int, default=5,
                    help='The number of ground-based SAMs')
parser.add_argument('--N_RADAR', type=int, default=3,
                    help='The number of ground-based radars')
parser.add_argument('--input_size', type=int, default=6,
                    help='Length of input data = length of UAV attribute + length of threat attribute')
parser.add_argument('--effect_size', type=int, default=20,
                    help='Influence vector length')
parser.add_argument('--output_size', type=int, default=1,
                    help='Output vector length')
parser.add_argument('--nlayers_1', type=int, default=1,
                    help='Number of layers of the GRU layer in part of Relation')
parser.add_argument('--nlayers_2', type=int, default=1,
                    help='Number of layers ofthe GRU layer in part of Evaluation')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.01,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--snum', type=int, default=2,
                    help='Proportion of rout points')
parser.add_argument('--train_long', type=int, default=422400,
                    help='Training data quantity = number of route * length of route')
parser.add_argument('--valid_long', type=int, default=80000,
                    help='Valid data quantity')
parser.add_argument('--test_long', type=int, default=2000,
                    help='Test data quantity')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=4000, metavar='N',
                    help='test batch size')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                    help='eval batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--cudnn', action='store_true',
                    help='use cudnn optimized version.')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--name', type=str, default=None,
                    help='name for this experiment. generates folder with the name if specified.')
args = parser.parse_args()
device = torch.device("cuda" if args.cuda else "cpu")

def batchify(data, bsz,n1):
    nbatch = np.size(data,0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1,n1,args.input_size).transpose(0,2).contiguous()
    return data.to(device)
def batchify_lable(data, bsz):
    nbatch = np.size(data,0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1,1).transpose(0,1).contiguous()
    return data.to(device)
def get_data():
    engine = Engine.engine(args.N_SAM,args.N_RADAR,args.database,args.snum)
    engine.connect()
    X11,X22 = engine.get_data()
    label_ideal = torch.Tensor(engine.get_label_real())*10
    return X11,X22,label_ideal
def data_processing(data,n1):
    train_data = batchify(data[0:args.train_long], args.batch_size,n1)
    val_data = batchify(data[args.train_long:args.train_long+args.valid_long], args.eval_batch_size,n1)
    test_data = batchify(data[args.train_long+args.valid_long:args.train_long+args.valid_long+args.test_long], args.test_batch_size,n1)
    return train_data,val_data,test_data
def data_processing_lable(data):
    train_data = batchify_lable(data[0:args.train_long], args.batch_size)
    val_data = batchify_lable(data[args.train_long:args.train_long+args.valid_long], args.eval_batch_size)
    test_data = batchify_lable(data[args.train_long+args.valid_long:args.train_long+args.valid_long+args.test_long], args.test_batch_size)
    return train_data,val_data,test_data
'''Get data from database'''
data_X1,data_X2,label = get_data()
'''Data processing'''
data_X1 = torch.Tensor(data_X1)
data_X2 = torch.Tensor(data_X2)
train_data_X1,val_data_X1 ,test_data_X1 = data_processing(data_X1,args.N_RADAR)
train_data_X2,val_data_X2 ,test_data_X2 = data_processing(data_X2,args.N_SAM)
train_label,val_data_label ,test_data_label = data_processing_lable(label)
'''Save args to logger'''
folder_name = str(datetime.datetime.now())[:-7]
if args.name is not None:
    folder_name = str(args.name) + '!' + folder_name
folder_name = folder_name.replace(':','')
os.mkdir(folder_name)
for file in os.listdir(os.getcwd()):
    if file.endswith(".py"):
        shutil.copy2(file, os.path.join(os.getcwd(), folder_name))
logger_train = open(os.path.join(os.getcwd(), folder_name, 'train_log.txt'), 'w+')
logger_test = open(os.path.join(os.getcwd(), folder_name, 'test_log.txt'), 'w+')
logger_train.write(str(args) + '\n')
'''Build the model'''
if args.model == "STA_STIN":
    model = STA_STIN.Model_layer(args.input_size, args.effect_size, args.output_size,
                            args.nlayers_1, args.nlayers_2, args.bptt).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
def get_batch(data_source_X1,data_source_X2,label,i):
    seq_len = min(args.bptt, len(label)-i)
    X1 = (data_source_X1.transpose(0,1))[i:i + seq_len].transpose(0,1)
    X2 = (data_source_X2.transpose(0,1))[i:i + seq_len].transpose(0,1)
    target = label[i:i + seq_len]
    return X1,X2,target
def evaluate(data_source_X1,data_source_X2,label_source):
    '''Turn on evaluation mode'''
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, label_source.size(0), args.bptt):
            data_X1, data_X2, data_label = get_batch(data_source_X1, data_source_X2, label_source, i)
            output,_1,_2 = model(data_X1, data_X2)
            loss_fn = torch.nn.MSELoss(reduce=False, size_average=False).to(device)
            loss = loss_fn(output.transpose(0, 1).view(args.eval_batch_size, 1),
                            data_label[args.bptt-1].transpose(0, 1).view(args.eval_batch_size, 1)).to(device)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss)
            total_loss += loss.item()
    return total_loss, output, data_label
def testit(data_source_X1,data_source_X2,label_source):
    '''Turn on evaluation mode'''
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, label_source.size(0), args.bptt):
            data_X1, data_X2, data_label = get_batch(data_source_X1, data_source_X2, label_source, i)
            output,_1,_2_ = model(data_X1, data_X2)
            loss_fn = torch.nn.MSELoss(reduce=False, size_average=False).to(device)
            loss = loss_fn(output.transpose(0, 1).view(args.test_batch_size, 1),
                            data_label[args.bptt-1].transpose(0, 1).view(args.test_batch_size, 1)).to(device)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss)
            total_loss += loss.item()
    return total_loss, output, data_label
def train():
    '''Turn on train mode'''
    model.train()
    total_loss = 0.
    forward_elapsed_time = 0.
    start_time = time.time()
    judge = 0
    for batch, i in enumerate(range(0,train_label.size(0), args.bptt)):
        judge+=1
        data_X1, data_X2, data_label = get_batch(train_data_X1, train_data_X2, train_label, i)
        if if_cuda:
            torch.cuda.synchronize()
        forward_start_time = time.time()
        '''Zero the gradients before running the backward pass'''
        model.zero_grad()
        output,_1,_2= model(data_X1, data_X2)
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=True).to(device)
        loss = loss_fn(output.transpose(0, 1).view(args.batch_size,1), data_label[args.bptt-1].transpose(0, 1).view(args.batch_size,1)).to(device)
        total_loss += loss.item()
        if if_cuda:
            torch.cuda.synchronize()
        forward_elapsed = time.time() - forward_start_time
        forward_elapsed_time += forward_elapsed
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            printlog = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | forward ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data_label) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, forward_elapsed_time * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss))
            print(printlog)
            logger_train.write(printlog + '\n')
            logger_train.flush()
            start_time = time.time()
            forward_elapsed_time = 0.
    return total_loss/judge

def render_loss(list1,list2):
    '''Draw the line chart of the trends of training and verification errors'''
    x = []
    y1 = []
    y2 = []
    p = max(max(list1),max(list2))
    pp = min(min(list1),min(list2))
    for i in range(0, args.epochs):
        x.append(i)
        y1.append(list1[i])
        y2.append(list2[i])
    plt.xlim((-1, args.epochs))
    plt.ylim((pp-0.01, p+0.01))
    plt.xlabel('number')
    plt.ylabel('p')
    plt.plot(x, y1, 'r', label='train')
    plt.plot(x, y2, 'b', label='val')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), folder_name,'img_train'), format='png')
    plt.show()

'''Loop over epochs'''
lr = args.lr
best_val_loss = None
train_loss_list = []
val_loss_list = []
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        total_loss = train()
        val_loss,val_output,val_true = evaluate(val_data_X1,val_data_X2,val_data_label)
        train_loss_list.append(total_loss)
        val_loss_list.append(val_loss)
        otime = time.time() - epoch_start_time
        print('-' * 89)
        testlog = '| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.6f}| valid loss {:5.6f} | valid ppl {:8.2f}'.format(
            epoch, otime, total_loss, val_loss, math.exp(val_loss))
        print(testlog)
        logger_test.write(testlog + '\n')
        logger_test.flush()
        print('-' * 89)
        scheduler.step(val_loss)
        '''Save the model if the validation loss is the best we've seen so far.'''
        if not best_val_loss or val_loss < best_val_loss:
            with open(os.path.join(os.getcwd(), folder_name, args.save), 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
with open(os.path.join(os.getcwd(), folder_name, args.save), 'rb') as f:
    model = torch.load(f)
    if args.cudnn:
        model.rnn.flatten_parameters()
'''Run on test data'''
test_loss,output,true_output = testit(test_data_X1,test_data_X2,test_data_label)
print('=' * 89)
testlog = '| End of training | test loss {:5.4f} | test ppl {:8.2f}| output {:s}| true_output {:s}'.format(
    test_loss,math.exp(test_loss),str(output),str(true_output[args.bptt-1]))
print(testlog)
logger_test.write(testlog + '\n')
logger_test.flush()
print('=' * 89)
render_loss(train_loss_list,val_loss_list)













