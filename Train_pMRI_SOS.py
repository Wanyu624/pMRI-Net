# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:52:45 2019

@author: 啊咧啊咧
email : wanyu.bian@ufl.edu

"""


import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import Utils
import glob
from time import time
from PIL import Image
import math
import tensorflow.contrib.slim as slim
#from skimage.metrics import peak_signal_noise_ratio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # # 0--2， 1--0， 2--1
ckpt_model_number = 1000

CS_ratio = 31.6
PhaseNumber = 3
ntrain = 15
ntest = 15
global_step = tf.Variable(tf.constant(0))   
EpochNum = ckpt_model_number
batch_size = 1
size = 320

def psnr(imag1, imag2):

    mse = np.mean( ( abs(imag1) - abs(imag2) ) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = abs(imag1).max()
    relative_error = np.linalg.norm( abs(imag1) - abs(imag2), 'fro' )/np.linalg.norm( abs(imag1), 'fro')
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), relative_error  


#def psnr(imag1, imag2): #same
#    temp = (abs(imag1) - abs(imag2)).flatten()
#    mse = np.mean( ( temp.conj().T*temp ) )
#    if mse == 0:
#        return 100
#    PIXEL_MAX = abs(imag1).max()
#    relative_error = np.linalg.norm( abs(imag1) - abs(imag2), 'fro' )/np.linalg.norm( abs(imag1), 'fro')
#    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse)), relative_error  

# Define a placeholder for input values
Mask = sio.loadmat('masks_pd/uniform_31_56.mat')
mask = Utils.removePEOversampling( Utils.removeFEOversampling( Mask['mask']))
m = np.tile( np.expand_dims(np.expand_dims( mask.astype(np.float32), axis=0), axis=1), (1, 15, 1, 1))

target = tf.placeholder(shape=[None, size, size], dtype=tf.complex64)#u
coil_imgs = tf.placeholder(shape=[None, 15, size, size], dtype=tf.complex64)#si
k_space = tf.placeholder(shape=[None, 15, size, size], dtype=tf.complex64)#f

def mriForwardOp(img, sampling_mask):
    with tf.variable_scope('mriForwardOp'):
        # centered Fourier transform
        Fu = Utils.fftc2d(img)
        # apply sampling mask
        kspace = tf.complex(tf.real(Fu) * sampling_mask, tf.imag(Fu) * sampling_mask)
        return kspace

def mriAdjointOp(f, sampling_mask):
    with tf.variable_scope('mriAdjointOp'):
        # apply mask and perform inverse centered Fourier transform
        Finv = Utils.ifftc2d(tf.complex(tf.real(f) * sampling_mask, tf.imag(f) * sampling_mask))

        return Finv

def add_con2d_weight_k(w_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_k_%d' % order_no)
    return Weights

ui_0 = Utils.ifftc2d(k_space)
x_input = tf.tile( tf.expand_dims(target, axis=1), multiples = [1, 15, 1, 1])

ATf = mriAdjointOp( k_space, m)#(?, 15, 320, 320)
ATf_real = tf.real(ATf)
ATf_imag = tf.imag(ATf)

def add_con2d_weight_G(w_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_G_%d' % order_no)
    return Weights

def add_con2d_weight(w_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_J_%d' % order_no)
    return Weights

def ista_block(input_layer, layer_no):
    step_real = tf.Variable(0.01, dtype=tf.float32)
    step_imag = tf.Variable(0.01, dtype=tf.float32)    
    soft_thr_real = tf.Variable(0.000, dtype=tf.float32)#0.0001 0.00005 
    soft_thr_imag = tf.Variable(0.000, dtype=tf.float32)
    conv_size = 32
    filter_size = 9

    Weights555 = add_con2d_weight([3, 3, 15, 64], 1555)
    Weights55 = add_con2d_weight([3, 3, 64, 64], 155)
    Weights666 = add_con2d_weight([3, 3, 64, 64], 1666)
    Weights66 = add_con2d_weight([3, 3, 64, 1], 166)
    
    Weights0 = add_con2d_weight_G([filter_size, filter_size, 1, conv_size], 0)

    Weights1 = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 1)
    Weights11 = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 11)
    
    Weights2 = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 2)
    Weights22 = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 22)
    
    Weights3 = add_con2d_weight_G([filter_size, filter_size, conv_size, 1], 3)

    Weights888 = add_con2d_weight([3, 3, 1, 64], 1888)
    Weights88 = add_con2d_weight([3, 3, 64, 64], 188)    
    Weights999 = add_con2d_weight([3, 3, 64, 64], 1999)
    Weights99 = add_con2d_weight([3, 3, 64, 15], 199)
#______________________________________________________
    
    Weights555_ = add_con2d_weight([3, 3, 15, 64], 9155)
    Weights55_ = add_con2d_weight([3, 3, 64, 64], 915)
    Weights666_ = add_con2d_weight([3, 3, 64, 64], 9166)
    Weights66_ = add_con2d_weight([3, 3, 64, 1], 916)
    
    Weights0_ = add_con2d_weight_G([filter_size, filter_size, 1, conv_size], 90)

    Weights1_ = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 91)
    Weights11_ = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 911)

    Weights2_ = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 92)
    Weights22_ = add_con2d_weight_G([filter_size, filter_size, conv_size, conv_size], 922)
    
    Weights3_ = add_con2d_weight_G([filter_size, filter_size, conv_size, 1], 93)

    Weights888_ = add_con2d_weight([3, 3, 1, 64], 9188)
    Weights88_ = add_con2d_weight([3, 3, 64, 64], 918)    
    Weights999_ = add_con2d_weight([3, 3, 64, 64], 9199)
    Weights99_ = add_con2d_weight([3, 3, 64, 15], 919)

#_______________________________________________________________________________________________________________________________________    
    
    Au =  mriForwardOp(input_layer[-1], m)
    ATAu = mriAdjointOp( Au, m)#(?, 15, 320, 320)
    ATAu_real = tf.real(ATAu)
    ATAu_imag = tf.imag(ATAu)
    
    x1_real = tf.add( tf.real(input_layer[-1]) - tf.scalar_mul(step_real, ATAu_real), tf.scalar_mul(step_real, ATf_real)) # X_k - lambda*A^T(AX -fi)
    x1_imag = tf.add( tf.imag(input_layer[-1]) - tf.scalar_mul(step_imag, ATAu_imag), tf.scalar_mul(step_imag, ATf_imag))#(?, 15, 320, 320)
     
#SOS
    
    Ju_abs = tf.sqrt(tf.square(x1_real) + tf.square(x1_imag) )
    Ju_0 = tf.sqrt(tf.reduce_sum(tf.square(Ju_abs), 1))
    Ju_0 = tf.expand_dims(Ju_0, axis= 1)
     
    x2 = tf.transpose(Ju_0, perm=[0, 2, 3, 1])
    
#g    
    x3_real = tf.nn.conv2d(x2, Weights0, strides=[1, 1, 1, 1], padding='SAME') 
    x3_imag = tf.nn.conv2d(x2, Weights0_, strides=[1, 1, 1, 1], padding='SAME')

    x4_real = tf.nn.relu(tf.nn.conv2d(x3_real, Weights1, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x3_imag, Weights1_, strides=[1, 1, 1, 1], padding='SAME'))
    x4_imag = tf.nn.relu(tf.nn.conv2d(x3_real, Weights1_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x3_imag, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    
    x44_real = tf.nn.conv2d(x4_real, Weights11, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x4_imag, Weights11_, strides=[1, 1, 1, 1], padding='SAME')
    x44_imag = tf.nn.conv2d(x4_real, Weights11_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x4_imag, Weights11, strides=[1, 1, 1, 1], padding='SAME') #(?, 320, 320, 32)
#S
    x5_real = tf.multiply(tf.sign(x44_real), tf.nn.relu(tf.abs(x44_real) - soft_thr_real))
    x5_imag = tf.multiply(tf.sign(x44_imag), tf.nn.relu(tf.abs(x44_imag) - soft_thr_imag))
#g~    
    x6_real = tf.nn.relu(tf.nn.conv2d(x5_real, Weights2, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x5_imag, Weights2_, strides=[1, 1, 1, 1], padding='SAME') )
    x6_imag = tf.nn.relu(tf.nn.conv2d(x5_real, Weights2_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x5_imag, Weights2, strides=[1, 1, 1, 1], padding='SAME') )
    
    x66_real = tf.nn.conv2d(x6_real, Weights22, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x6_imag, Weights22_, strides=[1, 1, 1, 1], padding='SAME')
    x66_imag = tf.nn.conv2d(x6_real, Weights22_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x6_imag, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x7_real = tf.nn.conv2d(x66_real, Weights3, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x66_imag, Weights3_, strides=[1, 1, 1, 1], padding='SAME')
    x7_imag = tf.nn.conv2d(x66_real, Weights3_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x66_imag, Weights3, strides=[1, 1, 1, 1], padding='SAME')#(?, 320, 320, 1)

#J~
    x88_real =  tf.nn.relu(tf.nn.conv2d(x7_real, Weights888, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x7_imag, Weights888_, strides=[1, 1, 1, 1], padding='SAME'))
    x88_imag =  tf.nn.relu(tf.nn.conv2d(x7_real, Weights888_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x7_imag, Weights888, strides=[1, 1, 1, 1], padding='SAME'))
    
    x8_real =  tf.nn.relu(tf.nn.conv2d(x88_real, Weights88, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x88_imag, Weights88_, strides=[1, 1, 1, 1], padding='SAME'))#(?, 320, 320, 15)
    x8_imag =  tf.nn.relu(tf.nn.conv2d(x88_real, Weights88_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x88_imag, Weights88, strides=[1, 1, 1, 1], padding='SAME'))
    
    x99_real =  tf.nn.relu(tf.nn.conv2d(x8_real, Weights999, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x8_imag, Weights999_, strides=[1, 1, 1, 1], padding='SAME'))
    x99_imag =  tf.nn.relu(tf.nn.conv2d(x8_real, Weights999_, strides=[1, 1, 1, 1], padding='SAME') + tf.nn.conv2d(x8_imag, Weights999, strides=[1, 1, 1, 1], padding='SAME'))
    
    x9_real =  tf.nn.conv2d(x99_real, Weights99, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x99_imag, Weights99_, strides=[1, 1, 1, 1], padding='SAME')#(?, 320, 320, 15)
    x9_imag =  tf.nn.conv2d(x99_real, Weights99_, strides=[1, 1, 1, 1], padding='SAME') - tf.nn.conv2d(x99_imag, Weights99, strides=[1, 1, 1, 1], padding='SAME')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    x_real =  tf.transpose( x9_real, perm=[0, 3, 1, 2]) + x1_real#(?, 15, 320, 320) b_k+ r_k(u_k) 
    x_imag =  tf.transpose( x9_imag, perm=[0, 3, 1, 2]) + x1_imag#(?, 15, 320, 320)  
    
    new = tf.complex(x_real, x_imag)#(?, 15, 320, 320)         

    Ju_abs = tf.sqrt(tf.square(x_real) + tf.square(x_imag) )
    Ju = tf.sqrt(tf.reduce_sum(tf.square(Ju_abs), 1))
    print(Ju.shape)             
    return [new, Ju_0, Ju, step_real, step_imag, soft_thr_real, soft_thr_imag]

def inference_(input_u, n, reuse):
    layers = []
    layers.append(input_u)
    for i in range(n):
        with tf.variable_scope('conv_%d' % i, reuse=reuse):
            [ phase_i, Ju_0, Ju, step_real, step_imag, soft_thr_real, soft_thr_imag] = ista_block(layers, i)
            layers.append(phase_i)

    return [layers, Ju, step_real, step_imag, soft_thr_real, soft_thr_imag]

def compute_cost(Prediction, Ju, PhaseNumber):

    true_abs = tf.sqrt(tf.square(tf.real(coil_imgs)) + tf.square(tf.imag(coil_imgs)) + 1e-12)
    true_sum = tf.sqrt(tf.reduce_sum(tf.square(true_abs), 1))
#    ui_0_abs = tf.sqrt(tf.square(tf.real(ui_0)) + tf.square(tf.imag(ui_0)) + 1e-12)
#    ui_0_sum = tf.sqrt(tf.reduce_sum(tf.square(ui_0_abs), 1))
#    cost_0 = tf.reduce_mean(tf.abs(ui_0_sum - true_sum))
    
    cost = tf.reduce_mean(tf.abs( Ju - true_sum))

    pred_abs = tf.sqrt(tf.square(tf.real(Prediction[-1])) + tf.square(tf.imag(Prediction[-1])))
    pred_sum = tf.sqrt(tf.reduce_sum(tf.square(pred_abs), 1))
    cost_ui = tf.reduce_mean(tf.abs( pred_sum - true_sum))

#    ui_real =  tf.abs((tf.real(Prediction[-1]) - tf.real(coil_imgs)))
#    ui_imag =  tf.abs((tf.imag(Prediction[-1]) - tf.imag(coil_imgs)))
#    cost_ui = tf.reduce_mean( ui_real + ui_imag )
    
    # ssim
    output_abs = tf.expand_dims(tf.abs(Ju), -1)
    target_abs = tf.expand_dims(tf.abs(target), -1)
    L = tf.reduce_max(target_abs, axis=(1, 2, 3), keepdims=True) - tf.reduce_min(target_abs, axis=(1, 2, 3),
                                                                                 keepdims=True)
    ssim = Utils.ssim(output_abs, target_abs, L=L)   

    # MSE_VN  prediction vs. target 8.0   
    target_abs = tf.sqrt(tf.real((target) * tf.conj(target)) + 1e-12)
    output_abs = tf.sqrt(tf.real((Ju) * tf.conj(Ju)) + 1e-12)
    energy = tf.reduce_mean(tf.reduce_sum(((output_abs - target_abs) ** 2))) / batch_size 
           
    return [cost, ssim, cost_ui, energy]


learning_rate = tf.train.exponential_decay(learning_rate= 0.0003,
                                       global_step=global_step,
                                       decay_steps= 100,
                                       decay_rate=0.95, staircase=False)    

[Prediction, Ju, step_real, step_imag, soft_thr_real, soft_thr_imag] = inference_(ui_0, PhaseNumber, reuse=False)

#cost0 = tf.reduce_mean(tf.square(X0 - X_output))

[cost, ssim, cost_ui, energy] = compute_cost(Prediction, Ju, PhaseNumber)

t=1
cost_all =  cost + t*cost_ui

optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam').minimize(cost_all)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)

sess = tf.Session(config=config)
sess.run(init)

#ISTA_logits = slim.get_variables_to_restore()
#init_Weights = 'Phase3_ratio_0_31' + '/CS_Saved_Model_580.ckpt' 
#init_, init_feeddic = slim.assign_from_checkpoint(init_Weights, ISTA_logits, ignore_missing_vars = True)
#sess.run(init_, init_feeddic)

model_dir = 'Phase%d_ratio_0_%d' % (PhaseNumber, CS_ratio)
log_file_name = "Log_output_%s.txt" % (model_dir)

print("...................................")
print("Phase Number is %d, CS ratio is %.2f" % (PhaseNumber, CS_ratio))
print("...................................\n")
print('Load Data...')


data = sio.loadmat('Train_Data_pd/data.mat' )

U = data['U']

F = data['Y']

Ui = data['Ui']


for epoch_i in range(0, EpochNum+1):
    randidx_all = np.random.permutation(ntrain)
    for batch_i in range(ntrain // batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
    
        u = U[randidx, :, :]
        ui = Ui[randidx, :, :, :]
        f = F[randidx, :, :, :]
        
        feed_dict = {coil_imgs: ui, target: u, k_space: f} #(batch_size, 15,320, 320) 
        sess.run(optm_all, feed_dict=feed_dict)
        
    output_data = "[%02d/%02d] cost_all: %.7f, cost_true: %.7f,energy: %.5f, ssim: %.5f, lr: %.6f, step_r:%.5f, step_i:%.5f, thr_r:%.5f, thr_i:%.5f  \n" % (epoch_i, EpochNum,#, theta: %.4f
                   sess.run(cost_all, feed_dict=feed_dict), sess.run(cost, feed_dict=feed_dict), sess.run(energy, feed_dict=feed_dict), sess.run(ssim, feed_dict=feed_dict) ,
                   sess.run(learning_rate, feed_dict={global_step:epoch_i}) , 
                   sess.run(step_real, feed_dict=feed_dict), sess.run(step_imag, feed_dict=feed_dict),
                   sess.run(soft_thr_real, feed_dict=feed_dict), sess.run(soft_thr_imag, feed_dict=feed_dict))#, theta 
    print(output_data)
    
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if epoch_i % 10 == 0:
        saver.save(sess, './%s/CS_Saved_Model_%d.ckpt' % (model_dir, epoch_i), write_meta_graph=False)
        
print("Training Phase%d Finished" % ( PhaseNumber))

