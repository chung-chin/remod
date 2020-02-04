from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

from scipy import signal
import scipy
import math
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.gridspec import GridSpec

import time
import os

import random
from data_utils_tf import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#SNR_list = {0.4, 0.375, 0.35}
SNR_list = {0.35}
SNR_int = 100

random.seed(1)

sys.path.append('..')

batch_size = 256
initial_data = []

print(SNR_list)

check = True     # check all variables

def extrapolation(x1, y1, x2, y2, x3):
    sp = np.shape(y1)
    ##print(sp)
    y1 = np.reshape(y1,[-1])
    y2 = np.reshape(y2,[-1])
    
    y3 = ((y2-y1)*(x3-x2)/(x2-x1)) + y2 
    y3 = np.reshape(y3,sp)
    
    return y3



tf.reset_default_graph()

with tf.Session() as sess:

    
    saver = tf.train.import_meta_graph('../SNR40/model.ckpt.meta')
    
    load_path = saver.restore(sess, '../SNR40/model.ckpt')
    print("Model restored from ../SNR40/model.ckpt")
    
    #tf.summary.FileWriter('./events/', graph=tf.get_default_graph())
    
    graph = tf.get_default_graph()
    
    k1_1 = sess.run(graph.get_tensor_by_name('conv1d/kernel/read:0'))
    b1_1 = sess.run(graph.get_tensor_by_name('conv1d/bias/read:0'))
    
    k2_1 = sess.run(graph.get_tensor_by_name('conv1d_1/kernel/read:0'))
    b2_1 = sess.run(graph.get_tensor_by_name('conv1d_1/bias/read:0'))
    
    k3_1 = sess.run(graph.get_tensor_by_name('conv1d_2/kernel/read:0'))
    b3_1 = sess.run(graph.get_tensor_by_name('conv1d_2/bias/read:0'))
    
    k4_1 = sess.run(graph.get_tensor_by_name('conv1d_3/kernel/read:0'))
    b4_1 = sess.run(graph.get_tensor_by_name('conv1d_3/bias/read:0'))

    
tf.reset_default_graph()
    
with tf.Session() as sess:

    
    saver = tf.train.import_meta_graph('../SNR37/model.ckpt.meta')
    
    load_path = saver.restore(sess, '../SNR37/model.ckpt')
    print("Model restored from ../SNR37/model.ckpt")
    
    graph = tf.get_default_graph()
    
    k1_2 = sess.run(graph.get_tensor_by_name('conv1d/kernel/read:0'))
    b1_2 = sess.run(graph.get_tensor_by_name('conv1d/bias/read:0'))
    
    k2_2 = sess.run(graph.get_tensor_by_name('conv1d_1/kernel/read:0'))
    b2_2 = sess.run(graph.get_tensor_by_name('conv1d_1/bias/read:0'))
    
    k3_2 = sess.run(graph.get_tensor_by_name('conv1d_2/kernel/read:0'))
    b3_2 = sess.run(graph.get_tensor_by_name('conv1d_2/bias/read:0'))
    
    k4_2 = sess.run(graph.get_tensor_by_name('conv1d_3/kernel/read:0'))
    b4_2 = sess.run(graph.get_tensor_by_name('conv1d_3/bias/read:0'))
    
    
    
tf.reset_default_graph()
    

#print(k1_1)
#print(k1_2)

x1 = 0.4
x2 = 0.375
x3 = 0.35

k1_ = extrapolation(x1, k1_1, x2, k1_2, x3)
#print(k1_.shape)
k2_ = extrapolation(x1, k2_1, x2, k2_2, x3)

k3_ = extrapolation(x1, k3_1, x2, k3_2, x3)

k4_ = extrapolation(x1, k4_1, x2, k4_2, x3)
############
b1_ = extrapolation(x1, b1_1, x2, b1_2, x3)
#print(b1_.shape)
b2_ = extrapolation(x1, b2_1, x2, b2_2, x3)

b3_ = extrapolation(x1, b3_1, x2, b3_2, x3)

b4_ = extrapolation(x1, b4_1, x2, b4_2, x3)


######################
### Construct TF graph
######################
tf.reset_default_graph()

DIM = 8192
drop_prob = 0
b_size = 256   # batch size
lr = 0.0003    # learning rate
epochs = 2000     # max epoch
cutoff = 1.e-7
patience_ = 5

X = tf.placeholder(tf.float64, shape=(None, DIM))
Y = tf.placeholder(tf.float64, shape=(None, 1))
drop_prob_ = tf.placeholder(tf.float64)
bs         = tf.placeholder(tf.int64)


train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=10000).batch(bs).repeat()
test_ds  = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(buffer_size=10000).batch(bs)


## Use one common iterator
iter = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
features, labels = iter.get_next()

# create the initialisation operations
train_init_op = iter.make_initializer(train_ds)
test_init_op = iter.make_initializer(test_ds)








feature = tf.reshape(features, [-1, DIM,1])
##logits = net(features, drop_prob, DIM)
   
def nnconv(x, f, k, d, s, p, ps, ip, n, act):   #f=output channels, k=kernel size, ip=input channels
    #kernel = tf.random_normal([1,k, ip, f], stddev=.01, dtype=tf.float64)
    #out = tf.nn.conv2d(x, kernel, strides=[1,1,s,1], padding='VALID', name=n, dilations=[1,1,1,1])
    #out = tf.nn.avg_pool(out, ksize=[1,1,p,1], strides=[1,1,ps,1], padding='VALID')
    
    kernel_ = tf.Variable(tf.random_normal([k, ip, f], stddev=.01, dtype=tf.float64))
    bias_ = tf.Variable(tf.random_normal([f], stddev=.01, dtype=tf.float64))
    out_ = tf.nn.conv1d(x, kernel_, stride=s, dilations=d, padding='VALID', name=n)
    out_ = tf.nn.bias_add(out_, bias_)
    out_ = tf.layers.average_pooling1d(out_, pool_size= p, strides= ps, padding='valid')

    return act(out_)


kernel1 = tf.Variable(tf.random_normal([16, 1, 64], stddev=.01, dtype=tf.float64))
kernel1 = tf.assign(kernel1, k1_)
bias1 = tf.Variable(tf.random_normal([64], stddev=.01, dtype=tf.float64))
bias1 = tf.assign(bias1, b1_)
h1 = tf.nn.conv1d(feature, kernel1, stride=1, dilations=1, padding='VALID', name='h1')
h1 = tf.nn.bias_add(h1, bias1)
h1 = tf.layers.average_pooling1d(h1, pool_size= 4, strides= 4, padding='valid')
h1 = tf.nn.relu(h1)

kernel2 = tf.Variable(tf.random_normal([16, 64, 128], stddev=.01, dtype=tf.float64))
kernel2 = tf.assign(kernel2, k2_)
bias2 = tf.Variable(tf.random_normal([128], stddev=.01, dtype=tf.float64))
bias2 = tf.assign(bias2, b2_)
h2 = tf.nn.conv1d(h1, kernel2, stride=1, dilations=2, padding='VALID', name='h2')
h2 = tf.nn.bias_add(h2, bias2)
h2 = tf.layers.average_pooling1d(h2, pool_size= 4, strides= 4, padding='valid')
h2 = tf.nn.relu(h2)

kernel3 = tf.Variable(tf.random_normal([16, 128, 256], stddev=.01, dtype=tf.float64))
kernel3 = tf.assign(kernel3, k3_)
bias3 = tf.Variable(tf.random_normal([256], stddev=.01, dtype=tf.float64))
bias3 = tf.assign(bias3, b3_)
h3 = tf.nn.conv1d(h2, kernel3, stride=1, dilations=2, padding='VALID', name='h3')
h3 = tf.nn.bias_add(h3, bias3)
h3 = tf.layers.average_pooling1d(h3, pool_size= 4, strides= 4, padding='valid')
h3 = tf.nn.relu(h3)

kernel4 = tf.Variable(tf.random_normal([32, 256, 512], stddev=.01, dtype=tf.float64))
kernel4 = tf.assign(kernel4, k4_)
bias4 = tf.Variable(tf.random_normal([512], stddev=.01, dtype=tf.float64))
bias4 = tf.assign(bias4, b4_)
h4 = tf.nn.conv1d(h3, kernel4, stride=1, dilations=2, padding='VALID', name='h4')
h4 = tf.nn.bias_add(h4, bias4)
h4 = tf.layers.average_pooling1d(h4, pool_size= 4, strides= 4, padding='valid')
h4 = tf.nn.relu(h4)


dim = h4.get_shape().as_list()
fcnn = dim[1]*dim[2]
h4 = tf.reshape(h4, [-1, fcnn])
    
h5          = tf.layers.dense(h4, 128, activation=tf.nn.relu)
h6          = tf.layers.dense(h5, 64,  activation=tf.nn.relu)
##h6          = tf.nn.dropout(h6, rate= drop_prob)
yhat_linear = tf.layers.dense(h6,  1, activation=None)

logits = yhat_linear





# Compute predictions
with tf.name_scope('eval'):
    predict_prob = tf.sigmoid(logits, name="sigmoid_tensor")
    predict_op   = tf.cast( tf.round(predict_prob), tf.int32 )

with tf.name_scope('loss'):
    ## with reduction compared to tf.nn.softmax_cross_entropy_with_logits_v2 
    loss_op = tf.losses.sigmoid_cross_entropy(logits=logits, multi_class_labels=labels)

with tf.name_scope('adam_optimizer'):
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op)
    ##optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss_op)

##correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
##accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

label = tf.reshape(labels,[-1])

_, accuracy    = tf.metrics.accuracy(labels=labels, predictions=predict_op  )
_, sensitivity = tf.metrics.recall(labels=labels, predictions=predict_op  )

_, fp = tf.metrics.false_positives(labels=labels, predictions=predict_op  )
_, fn = tf.metrics.false_negatives(labels=labels, predictions=predict_op  )
_, tp = tf.metrics.true_positives(labels=labels, predictions=predict_op  )
_, tn = tf.metrics.true_negatives(labels=labels, predictions=predict_op  )

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=100)

#################
### check variable
#################
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope("summaries_%s"% var.name.replace("/", "_").replace(":", "_")):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

### check all variables
if check:
    vars = 0
    for v in tf.global_variables():
        print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Whole size: %.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.global_variables()) ) )

    vars = 0
    for v in tf.trainable_variables():
        print (v)
        #variable_summaries(v)
        vars += np.prod(v.get_shape().as_list())
    print("Model size: %.3f MB | Var # : %d" % (8*vars/(1024**2), len(tf.trainable_variables()) ) )

    vars = 0
    for v in tf.local_variables():
        print (v)
        vars += np.prod(v.get_shape().as_list())
    print("Local var size: %.3f Bytes | Var # : %d" % (8*vars, len(tf.local_variables()) ) )



################# input initial data #####################
##if 0:
t1 = time.time()

with tf.Session() as sess:
    sess.run(init)

    ##for snr in [10]:
    for snr in SNR_list:
        sess.run(tf.local_variables_initializer())
        ##sess.run(init)
        
        address = 'SNR%s' %(int(snr*SNR_int))
        mkdir_checkdir(path = "./%s" %address)
        
        train_acc = []
        train_sen = []
        epoch = 1      # ini epoch
    
        print('Start Training at SNR= %s !' %(snr), '\n')
    
        f = open('../pycbc_inidata/initial_data_%s.pkl' %int(snr*SNR_int), 'rb')
        train_dict_, test_dict_ = pickle.load(f)
        f.close()
    
        train = train_dict_['%s' %int(snr*SNR_int)]
    
        num_examples= train.shape[0]
        print('num_examples: ', num_examples, end='\n\n')
    
        y_train = np.array(~train.sigma.isnull() +0)
        y_train = np.reshape(y_train, [-1,1])
        X_train = np.array(Normolise(train.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
        print('Label for training:', y_train.shape)
        print('Dataset for training:', X_train.shape, end='\n\n')
    
        if 0:
            for i in range(10):
                feature, label = iter.get_next()
                print (feature.shape, label.shape)


        ckpt_path = './SNR%s/model.ckpt' %(int(snr*SNR_int))

        sess.run(train_init_op, feed_dict={X:X_train, Y:y_train, bs:b_size})
        
        steps = int(num_examples/b_size)
        patience = 0

        ##while epoch < epochs:
        while patience < patience_:
            print('epoch= ', epoch)
            
            ##if epoch == 5:
            ##    lr = 0.0003

            if epoch > 2:
                lr /= (1+0.01*epoch)
                
            for i in range(steps):
                _, loss, acc, sen = sess.run( [optimizer, loss_op, accuracy, sensitivity] )
                print('loss for SNR=%s : %s' %(str(snr),loss))
                
                if loss < cutoff:
                    if patience > patience_:
                        break
                    patience += 1
                else:
                    patience = 0
            epoch += 1

            print('accuracy: %s,  sensitivity: %s' %(acc,sen))
            train_acc.append(acc)
            train_sen.append(sen)
            
        save_path = saver.save(sess, ckpt_path)
        print("Model saved in path: %s" % save_path)
        
        np.save('./SNR%s/train_accuracy.npy' %(int(snr*SNR_int)), train_acc)
        np.save('./SNR%s/train_sensitivity.npy' %(int(snr*SNR_int)), train_sen)


t2 = time.time()
t3 = t2 - t1
###############
### Testing ###
###############
SNR_list_ = np.linspace(0, 0.4, num=9*2-1).tolist()[-10-1::-1]
print('SNR_list_: ', SNR_list_)
with tf.Session() as sess:
    sess.run(init)
    
    auc_frame = []
    fpr_frame = []
    tpr_frame = []
    acc_frame = []
    sen_frame = []

    ##for snr in [10]:
    for snr in SNR_list:
        address = 'SNR%s' %(int(snr*SNR_int))
    
        load_path = saver.restore(sess, './SNR%s/model.ckpt' %(int(snr*SNR_int)))
        print("Model restored from /SNR%s/model.ckpt" %(int(snr*SNR_int))) 

        test_acc = []
        test_sen = []
        tpr_list = []
        fpr_list = []
        auc_list = []

        for snr_ in SNR_list:
            sess.run(tf.local_variables_initializer())
            
            test_acc_ = []
            test_sen_ = []
            test_pre_ = []
            test_lab = []
            
            f_ = open('../pycbc_inidata/initial_data_%s.pkl' %int(snr_*SNR_int), 'rb')
            train_dict__, test_dict__ = pickle.load(f_)
            f_.close()
            
            test  = test_dict__['%s' %int(snr_*SNR_int)]
            
            num_examples= test.shape[0]
            ##print('num_examples: ', num_examples, end='\n\n')
            
            y_test = np.array(~test.sigma.isnull() +0)
            y_test = np.reshape(y_test, [-1,1])
            X_test = np.array(Normolise(test.drop(['mass','positions','gaps','max_peak','sigma','SNR_mf','SNR_mf0'],axis=1)))
            print('Label for testing:', y_test.shape)
            print('Dataset for testing:', X_test.shape, end='\n\n')
            
            sess.run(test_init_op, feed_dict={X:X_test, Y:y_test, bs:b_size})
            
            steps = int(num_examples/b_size)
            
            for i in range(steps):
                pre, loss, acc, sen, ttp, ttn, tfp, tfn, tlb = sess.run([predict_prob, loss_op, accuracy, sensitivity, tp, tn, fp, fn, label ])
                
                test_pre_.extend(pre)
                test_acc_.append(acc)
                test_sen_.append(sen)
                test_lab.extend(tlb)
            
            test_pre_ = np.array(test_pre_)
            test_pre_ = np.reshape(test_pre_, [-1])
            print('pre_shape: ', test_pre_.shape)
            #h = len(test_pre_)
            test_lab_ = np.array(test_lab)
            print('label_shape: ', test_lab_.shape)
            fpr, tpr, _ = roc_curve(test_lab_,test_pre_)
            print('tpr: ', tpr.shape)
            auc_ = metrics.auc(fpr, tpr)
            
            auc_list.append(auc_)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

            acc_ = np.mean(test_acc_)
            sen_ = np.mean(test_sen_)
            test_acc.append(acc_)
            test_sen.append(sen_)
            
            print('Test for SNR=%s, loss: %s, acc: %s, sen: %s' %(str(snr_),loss,acc_,sen_),end='\n\n')
            
        acc_frame.append(test_acc)
        sen_frame.append(test_sen)
        tpr_frame.append(tpr_list)
        fpr_frame.append(fpr_list)
        auc_frame.append(auc_list)
        
    mkdir_checkdir(path = "./output")
    np.save('./output/test_accuracy',acc_frame)
    np.save('./output/test_sensitivity',sen_frame)
    np.save('./output/tpr',tpr_frame)
    np.save('./output/fpr',fpr_frame)
    np.save('./output/AUC',auc_frame)
    
    print('finished !')
    print('time: ',t3)










