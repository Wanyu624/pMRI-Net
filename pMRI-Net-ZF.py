# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:52:45 2019

@author: 啊咧啊咧
email : wanyu.bian@ufl.edu

Implementation of Initial ZF, real-valued Conv & ReLU
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # # 0--2， 1--0， 2--1
ckpt_model_number = 3000

CS_ratio = 31.6
PhaseNumber = 5
ntrain = 526
global_step = tf.Variable(tf.constant(0))   
EpochNum = ckpt_model_number
batch_size = 2
size = 320

def psnr(imag1, imag2):
    mse = np.mean( ( abs(imag1) - abs(imag2) ) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = abs(imag1).max()
    relative_error = np.linalg.norm( abs(imag1) - abs(imag2), 'fro' )/np.linalg.norm( abs(imag1), 'fro')
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), relative_error

# Define a placeholder for input values
Mask = sio.loadmat('masks_pd/COR_PD_iPat4_masks.mat')
mask = Utils.removePEOversampling( Utils.removeFEOversampling( Mask['mask']))
m = np.tile( np.expand_dims(np.expand_dims( mask.astype(np.float32), axis=0), axis=1), (1, 15, 1, 1))

target = tf.placeholder(shape=[None, size, size], dtype=tf.complex64)#u
coil_imgs = tf.placeholder(shape=[None, 15, size, size], dtype=tf.complex64)#ui
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
    
ui_0 =  Utils.ifftc2d( k_space)
x_input = tf.tile( tf.expand_dims(target, axis= 1), multiples = [1, 15, 1, 1])
ui_true = coil_imgs

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
    step_real = tf.Variable(0.1, dtype=tf.float32)
    step_imag = tf.Variable(0.1, dtype=tf.float32)    
    soft_thr_real = tf.Variable(0.0001, dtype=tf.float32)#0.0001 0.00005 
    soft_thr_imag = tf.Variable(0.0001, dtype=tf.float32)
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
     
    x2_real = tf.transpose(x1_real, perm=[0, 2, 3, 1])
    x2_imag = tf.transpose(x1_imag, perm=[0, 2, 3, 1])#(?, 320, 320, 15)
        
#J    
    x0_real =  tf.nn.relu(tf.nn.conv2d(x2_real, Weights555, strides=[1, 1, 1, 1], padding='SAME'))
    x0_imag =  tf.nn.relu(tf.nn.conv2d(x2_imag, Weights555_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x0_real =  tf.nn.relu(tf.nn.conv2d(x0_real, Weights55, strides=[1, 1, 1, 1], padding='SAME'))#(?, 320, 320, 1)
    x0_imag =  tf.nn.relu(tf.nn.conv2d(x0_imag, Weights55_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x0_real =  tf.nn.relu(tf.nn.conv2d(x0_real, Weights666, strides=[1, 1, 1, 1], padding='SAME'))
    x0_imag =  tf.nn.relu(tf.nn.conv2d(x0_imag, Weights666_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x0_real =  tf.nn.conv2d(x0_real, Weights66, strides=[1, 1, 1, 1], padding='SAME')#(?, 320, 320, 1)
    x0_imag =  tf.nn.conv2d(x0_imag, Weights66_, strides=[1, 1, 1, 1], padding='SAME')

    J_real = tf.reshape(x0_real, shape = [-1, 320, 320 ])
    J_imag = tf.reshape(x0_imag, shape = [-1, 320, 320 ])
    
    Ju_0 = tf.abs(tf.complex(J_real, J_imag)) #(?, 320, 320)
    print(Ju_0.shape)
#g    
    x3_real = tf.nn.conv2d( x0_real, Weights0, strides=[1, 1, 1, 1], padding='SAME')
    x3_imag = tf.nn.conv2d( x0_imag, Weights0_, strides=[1, 1, 1, 1], padding='SAME')

    x4_real = tf.nn.relu(tf.nn.conv2d(x3_real, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x4_imag = tf.nn.relu(tf.nn.conv2d(x3_imag, Weights1_, strides=[1, 1, 1, 1], padding='SAME'))
    x44_real = tf.nn.conv2d(x4_real, Weights11, strides=[1, 1, 1, 1], padding='SAME')
    x44_imag = tf.nn.conv2d(x4_imag, Weights11_, strides=[1, 1, 1, 1], padding='SAME')#(?, 320, 320, 32)
#S
    x5_real = tf.multiply(tf.sign(x44_real), tf.nn.relu(tf.abs(x44_real) - soft_thr_real))
    x5_imag = tf.multiply(tf.sign(x44_imag), tf.nn.relu(tf.abs(x44_imag) - soft_thr_imag))
#g~    
    x6_real = tf.nn.relu(tf.nn.conv2d(x5_real, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x6_imag = tf.nn.relu(tf.nn.conv2d(x5_imag, Weights2_, strides=[1, 1, 1, 1], padding='SAME'))
    x66_real = tf.nn.conv2d(x6_real, Weights22, strides=[1, 1, 1, 1], padding='SAME')
    x66_imag = tf.nn.conv2d(x6_imag, Weights22_, strides=[1, 1, 1, 1], padding='SAME')

    x7_real = tf.nn.conv2d(x66_real, Weights3, strides=[1, 1, 1, 1], padding='SAME')
    x7_imag = tf.nn.conv2d(x66_imag, Weights3_, strides=[1, 1, 1, 1], padding='SAME')#(?, 320, 320, 1)

#J~
    x88_real =  tf.nn.relu(tf.nn.conv2d(x7_real, Weights888, strides=[1, 1, 1, 1], padding='SAME'))
    x88_imag =  tf.nn.relu(tf.nn.conv2d(x7_imag, Weights888_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x8_real =  tf.nn.relu(tf.nn.conv2d(x88_real, Weights88, strides=[1, 1, 1, 1], padding='SAME'))#(?, 320, 320, 15)
    x8_imag =  tf.nn.relu(tf.nn.conv2d(x88_imag, Weights88_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x99_real =  tf.nn.relu(tf.nn.conv2d(x8_real, Weights999, strides=[1, 1, 1, 1], padding='SAME'))
    x99_imag =  tf.nn.relu(tf.nn.conv2d(x8_imag, Weights999_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x9_real =  tf.nn.conv2d(x99_real, Weights99, strides=[1, 1, 1, 1], padding='SAME')#(?, 320, 320, 15)
    x9_imag =  tf.nn.conv2d(x99_imag, Weights99_, strides=[1, 1, 1, 1], padding='SAME')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    x_real =  tf.transpose( x9_real, perm=[0, 3, 1, 2]) + x1_real#(?, 15, 320, 320) b_k+ r_k(u_k) 
    x_imag =  tf.transpose( x9_imag, perm=[0, 3, 1, 2]) + x1_imag#(?, 15, 320, 320)  
    
    new = tf.complex(x_real, x_imag)#(?, 15, 320, 320)       

    #Ju    
    x00_real =  tf.nn.relu(tf.nn.conv2d(x9_real + x2_real, Weights555, strides=[1, 1, 1, 1], padding='SAME'))
    x00_imag =  tf.nn.relu(tf.nn.conv2d(x9_imag + x2_imag, Weights555_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x00_real =  tf.nn.relu(tf.nn.conv2d(x00_real, Weights55, strides=[1, 1, 1, 1], padding='SAME'))#(?, 320, 320, 1)
    x00_imag =  tf.nn.relu(tf.nn.conv2d(x00_imag, Weights55_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x00_real =  tf.nn.relu(tf.nn.conv2d(x00_real, Weights666, strides=[1, 1, 1, 1], padding='SAME'))
    x00_imag =  tf.nn.relu(tf.nn.conv2d(x00_imag, Weights666_, strides=[1, 1, 1, 1], padding='SAME'))
    
    x00_real =  tf.nn.conv2d(x00_real, Weights66, strides=[1, 1, 1, 1], padding='SAME')#(?, 320, 320, 1)
    x00_imag =  tf.nn.conv2d(x00_imag, Weights66_, strides=[1, 1, 1, 1], padding='SAME')

    J_real = tf.reshape(x00_real, shape = [-1, 320, 320 ])
    J_imag = tf.reshape(x00_imag, shape = [-1, 320, 320 ])
    
    Ju = tf.complex(J_real, J_imag) #(?, 320, 320)
    print(Ju.shape)   
      
    return [new, Ju_0, Ju, step_real, step_imag, soft_thr_real, soft_thr_imag]

def inference_(input_u, n, reuse):
    layers = []
    layers_J = []
    layers.append(input_u)
    for i in range(n):
        with tf.variable_scope('conv_%d' % i, reuse=reuse):
            [ phase_i, Ju_0, Ju, step_real, step_imag, soft_thr_real, soft_thr_imag] = ista_block(layers, i)
            if i == 0:
                layers_J.append(Ju_0)
            else:
                layers.append(phase_i)
                layers_J.append(Ju)

    return [layers, layers_J, step_real, step_imag, soft_thr_real, soft_thr_imag]

def compute_cost(Prediction, Ju, PhaseNumber):
    
    true_abs = tf.sqrt(tf.square(tf.real(ui_true)) + tf.square(tf.imag(ui_true)))
    true_sum = tf.sqrt(tf.reduce_sum(tf.square(true_abs), 1))
    cost = tf.reduce_mean(tf.abs( tf.abs(Ju[-1]) - true_sum))

    pred_abs = tf.sqrt(tf.square(tf.real(Prediction[-1])) + tf.square(tf.imag(Prediction[-1])))
    pred_sum = tf.sqrt(tf.reduce_sum(tf.square(pred_abs), 1))
    cost_ui = tf.reduce_mean(tf.abs( pred_sum - true_sum))

    cost_ui_r = tf.reduce_mean(tf.square( tf.real(Prediction[-1] - ui_true)))
    cost_ui_i = tf.reduce_mean(tf.square( tf.imag(Prediction[-1] - ui_true)))
    cost_ui = cost_ui_r + cost_ui_i
    ui_r = tf.reduce_mean(tf.square( tf.real( ui_true)))
    ui_i = tf.reduce_mean(tf.square( tf.imag( ui_true)))
    ui = ui_r + ui_i
    
    ui_rmse =  cost_ui / ui
    
    # ssim
    output_abs = tf.expand_dims(tf.abs(Ju[-1]), -1)
    target_abs = tf.expand_dims(tf.abs(target), -1)
    L = tf.reduce_max(target_abs, axis=(1, 2, 3), keepdims=True) - tf.reduce_min(target_abs, axis=(1, 2, 3),
                                                                                 keepdims=True)
    ssim = Utils.ssim(output_abs, target_abs, L=L)   

    # MSE_VN  prediction vs. target 8.0   
    target_abs = tf.sqrt(tf.real((target) * tf.conj(target)) + 1e-12)
    output_abs = tf.sqrt(tf.real((Ju[-1]) * tf.conj(Ju[-1])) + 1e-12)
    energy = tf.reduce_mean(tf.reduce_sum(((output_abs - target_abs) ** 2))) / batch_size 
           
    return [cost, ssim, cost_ui, ui_rmse]

learning_rate = tf.train.exponential_decay(learning_rate= 0.0001,
                                       global_step=global_step,
                                       decay_steps= 100,
                                       decay_rate=0.992, staircase=False)    

[Prediction, Ju,  step_real, step_imag, soft_thr_real, soft_thr_imag] = inference_(ui_0, PhaseNumber, reuse=False)

#cost0 = tf.reduce_mean(tf.square(X0 - X_output))

[cost, ssim, cost_ui, ui_rmse] = compute_cost(Prediction, Ju, PhaseNumber)

t=0.00001
cost_all = cost + t*cost_ui

optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam').minimize(cost_all)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)

sess = tf.Session(config=config)
sess.run(init)

#ISTA_logits = slim.get_variables_to_restore()
#init_Weights = 'pd_Phase5_loss' + '/CS_Saved_Model_580.ckpt' 
#init_, init_feeddic = slim.assign_from_checkpoint(init_Weights, ISTA_logits, ignore_missing_vars = True)
#sess.run(init_, init_feeddic)

model_dir = 'pd_Phase%d_ZF' % (PhaseNumber)
log_file_name = "Log_output_%s.txt" % (model_dir)
#___________________________________________________________________________________Train

print("...................................")
print("Phase Number is %d" % (PhaseNumber))
print("...................................\n")
print('Load Data...')

data = sio.loadmat('Train_Data_pd/data.mat' )

U = data['U']

F = data['Y']

Ui = data['Ui']

for epoch_i in range(EpochNum+1):
    randidx_all = np.random.permutation(ntrain)
    for batch_i in range(ntrain // batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
    
        u = U[randidx, :, :]
        ui = Ui[randidx, :, :, :]
        f = F[randidx, :, :, :]
        
        feed_dict = {coil_imgs: ui, target: u, k_space: f} #(batch_size, 15,320, 320) 
        sess.run(optm_all, feed_dict=feed_dict)
        
    output_data = "[%02d/%02d]  cost_true: %.7f, lr: %.10f, step_r:%.5f \n" % (epoch_i, EpochNum,#, theta: %.4f
                   sess.run(cost, feed_dict=feed_dict),
                   sess.run(learning_rate, feed_dict={global_step:epoch_i}) , 
                   sess.run(step_real, feed_dict=feed_dict))#, theta 
    print(output_data)
    
    output_file = open(log_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if epoch_i >= 200:
        saver.save(sess, './%s/CS_Saved_Model_%d.ckpt' % (model_dir, epoch_i), write_meta_graph=False)
    else:
        if epoch_i % 50 == 0:
            saver.save(sess, './%s/CS_Saved_Model_%d.ckpt' % (model_dir, epoch_i), write_meta_graph=False)
        
print("Training Phase%d Finished" % ( PhaseNumber))

#___________________________________________________________________________________Test
print('Load Test Data...')
    
data = sio.loadmat('Train_Data_pd/data.mat' )

U = data['U']

F = data['Y']

Ui = data['Ui']


ntest = U.shape[0]
print(ntest)
PSNR_All = np.zeros([1, ntest], dtype=np.float32)
ERROR_All = np.zeros([1, ntest], dtype=np.float32)
#SSIM_All = np.zeros([1, ntest], dtype=np.float32)
#MSE_All = np.zeros([1, ntest], dtype=np.float32)
TIME_All = np.zeros([1, ntest], dtype=np.float32)
saver.restore(sess, './%s/CS_Saved_Model_%d.ckpt' % (model_dir, ckpt_model_number))

result_file_name = "PSNR_Results_pd.txt"

idx_all = np.arange(ntest)
for imag_no in range(ntest):
    randidx = idx_all[imag_no:imag_no+1]
    u = U[randidx, :, :]
    ui = Ui[randidx, :, :, :]
    f = F[randidx, :, :, :]
    
    feed_dict = { coil_imgs: ui, target: u, k_space: f} #(batch_size, 15,320, 320) 
    
    start = time()
    Prediction_value = sess.run(Ju[-1], feed_dict=feed_dict)
    end = time()
    
    rec = np.reshape(Prediction_value, (size,size))
    
    reference = np.reshape( u, ( size,size))

    rec_PSNR, relative_error =  psnr( reference , rec ) 
    #ssim_value = sess.run(ssim, feed_dict=feed_dict)
    #energy_value = sess.run(energy, feed_dict=feed_dict)
    cost_value = sess.run(cost, feed_dict=feed_dict)

    result = "Run time for %s:%.4f, PSNR:%.4f, relative_error:%.6f, loss:%.6f \n" % (imag_no+1, (end - start), rec_PSNR, relative_error, cost_value)
    print(result)
    
    #output_file.write(result)
    
    im_rec_name = "%s_rec_zf_%d.mat" % (imag_no+1, ckpt_model_number)  
    
    # save mat file
    Utils.saveAsMat(rec, im_rec_name, 'result',  mat_dict=None)
    
    # enhance image and save as png
    v_min, v_max = Utils.getContrastStretchingLimits(np.abs(rec),
                                                        saturated_pixel=0.002)
    volume_enhanced = Utils.normalize(np.abs(rec), v_min=v_min, v_max=v_max)
    
    Utils.imsave(volume_enhanced, im_rec_name)
    
    PSNR_All[0, imag_no] = rec_PSNR 
    ERROR_All[0, imag_no] = relative_error
#    SSIM_All[0, imag_no] = ssim_value
#    MSE_All[0, imag_no] = energy_value

output_data = "Phase%d, Avg REC PSNR is %.4f dB, Avg relative error is %.6f dB, ckpt NO. is %d %.6f \n" % (PhaseNumber, np.mean( PSNR_All), np.mean(ERROR_All), 
                                                                                                                     ckpt_model_number, t)
print(output_data)
output_file = open(result_file_name, 'a')
output_file.write(output_data)
output_file.close()


sess.close()

print("Reconstruction READY")