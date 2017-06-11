"""
The data required for this code can be download from https://drive.google.com/open?id=0B5eGOMdyHn2mWDYtQzlQeGNKa2s 
$tar -xvzf training_data.tgz

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import json
import pickle
import argparse

import numpy as np
import tensorflow as tf

from utils import TextLoader
from utils import get_vocab_embedding
from model import BasicLSTM

options = tf.GPUOptions()
options.per_process_gpu_memory_fraction = 0.3
tf.logging.set_verbosity(tf.logging.ERROR)
max_model_keep = 250

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/Holmes_Training_Data/',
                        help='data directory containing input data')
    parser.add_argument('--log_dir', type=str, default='./logs2/',
                       help='directory containing tensorboard logs')    
    parser.add_argument('--save_dir', type=str, default='./save2/',
                        help='directory to store checkpointed models')
    parser.add_argument('--embedding_file', type=str, 
                        default='./data/GoogleNews-vectors-negative300.bin', 
                        help='pretrained word embeddings')
    parser.add_argument('--earlyStop_log_filename', type=str, default='earlyStop_log.txt',
                        help='filename of the early stop log')
    parser.add_argument('--rnn_size', type=int, default=400,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers in the RNN')
    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help=' 1 - dropout rate')
    parser.add_argument('--num_sampled', type=int, default=5000,
                        help='number of negative examples to sample')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=40,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=8,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=2000,
                       help='save frequency' )
    parser.add_argument('--check_strip_length', type=int, default=5,
                       help='check eraly stop criteria strip length' )
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    parser.add_argument('--GL_threshold0', type=float, default=3.0,
                        help=' earlt stop GL threshold')
    parser.add_argument('--PG_threshold0', type=float, default=0.3,
                        help=' earlt stop PG threshold')
    parser.add_argument('--UP_threshold0', type=int, default=2,
                        help=' earlt stop UP threshold')

    parser.add_argument('--GL_threshold1', type=float, default=4.0,
                        help=' earlt stop GL threshold')
    parser.add_argument('--PG_threshold1', type=float, default=0.4,
                        help=' earlt stop PG threshold')
    parser.add_argument('--UP_threshold1', type=int, default=3,
                        help=' earlt stop UP threshold')

    parser.add_argument('--GL_threshold2', type=float, default=5.0,
                        help=' earlt stop GL threshold')
    parser.add_argument('--PG_threshold2', type=float, default=0.5,
                        help=' earlt stop PG threshold')
    parser.add_argument('--UP_threshold2', type=int, default=4,
                        help=' earlt stop UP threshold')

    parser.add_argument('--GL_threshold3', type=float, default=6.0,
                        help=' earlt stop GL threshold')
    parser.add_argument('--PG_threshold3', type=float, default=0.6,
                        help=' earlt stop PG threshold')
    parser.add_argument('--UP_threshold3', type=int, default=5,
                        help=' earlt stop UP threshold')
    args = parser.parse_args()
    
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 
    
    train(args)

#early stop criteria
def checkEarlyStopGL(valLoss, valLoss_opt):
    _GL = (valLoss/valLoss_opt-1)*100
    return _GL
def checkEarlyStopPQ(valLoss, valLoss_opt, trainLossList):
    _GL = (valLoss/valLoss_opt-1)*100
    minLoss = min(trainLossList)
    _Pk = (sum(trainLossList)/(len(trainLossList)*minLoss) -1)*1000
    _PQ = _GL/_Pk
    return _PQ
def checkEarlyStopUP(valLoss, lastStripValLoss, successiveHit):
    _successiveHit = successiveHit
    if valLoss > lastStripValLoss:
        _successiveHit += 1
    else:
        _successiveHit = 0
    return _successiveHit

def printEarlyStopLog2File(args, largestGL, largestPG, largestUP, _step, _epoch, _start):
    _file = open(args.log_dir+args.earlyStop_log_filename, 'a')
    _file.write("=================== Epoch : "+str(_epoch)+"======================\n")
    _file.write("===================== Step : "+str(_step)+"========================\n")
    _file.write("GL => step:"+str(largestGL[1])+"\tvalue:"+str(largestGL[0])+"\n")
    _file.write("PG => step:"+str(largestPG[1])+"\tvalue:"+str(largestPG[0])+"\n")
    _file.write("UP => step:"+str(largestUP[1])+"\tvalue:"+str(largestUP[0])+"\n")
    _file.write("cost time : "+str(time.time()-_start)+" secs\n")
    _file.close()

def print2LogFile(args, criteria, _step, _epoch, _start):
    _file = open(args.log_dir+args.earlyStop_log_filename, 'a')
    _file.write("=================== Epoch : "+str(_epoch)+"======================\n")
    _file.write("===================== Step : "+str(_step)+"========================\n")
    _file.write("Save "+criteria+" to model")
    _file.write("cost time : "+str(time.time()-_start)+" secs\n")
    _file.close()

def train(args):
    # Data Preparation
    # ====================================

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size
    print("Number of sentences: {}" .format(data_loader.num_data))
    print("Vocabulary size: {}" .format(args.vocab_size))

    
    # Check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = pickle.load(f)
        need_be_same=["rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = pickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"
    
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.words, data_loader.vocab), f)
    
    """
    embedding_matrix = get_vocab_embedding(args.save_dir, data_loader.words, args.embedding_file)
    print("Embedding matrix shape:",embedding_matrix.shape)
    """
    
    
    # Training
    # ====================================
    with tf.Graph().as_default():
        with tf.Session(config = tf.ConfigProto(gpu_options = options)) as sess:
            model = BasicLSTM(args)
          
            # Define training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.cost, tvars), args.grad_clip)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            
            # Keep track of gradient values and sparsity
            grad_summaries = []
            for g, v in zip(grads, tvars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            
            # Summary for loss
            loss_summary = tf.summary.scalar("loss", model.cost)

            # Train summaries
            merged = tf.summary.merge_all()
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)
            train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

            # saver = tf.train.Saver(tf.global_variables())
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_model_keep)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Restore model
            if args.init_from is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)

            # Start training
            print("Start training")
            valLoss_opt = 100000000.0
            lastStripValLoss = 100000000.0
            successiveHit = 0
            trainLossList = list()
            largestGL = [-1000,0]
            largestPG = [-1000,0]
            largestUP = [0,0]
            total_start_time = time.time()
            for epoch in range(args.num_epochs):
                data_loader.reset_batch_pointer()
                state = sess.run(model.initial_state)
                for i in range(data_loader.num_batches):
                    start = time.time()
                    #training
                    x_batch, y_batch = data_loader.next_batch()
                    feed_dict = {model.x: x_batch, model.y: y_batch, model.keep_prob: args.keep_prob }
                    _, step, summary, loss, equal = sess.run([train_op, global_step, merged, model.cost, model.equal], feed_dict)
                    print("training step {}, epoch {}, batch {}/{}, loss: {:.4f}, accuracy: {:.4f}, time/batch: {:.3f}"
                        .format(step, epoch, i, data_loader.num_batches, loss, np.mean(equal), time.time()-start))
                    train_writer.add_summary(summary, step)
                    trainLossList.append(loss)


                    current_step = tf.train.global_step(sess, global_step)
                    #validing
                    if current_step % args.check_strip_length == 0 and current_step > 0 and epoch>0:
                        start = time.time()
                        x_batch_valid, y_batch_valid = data_loader.get_first_batch_as_valid()
                        total_valid_loss = 0.0
                        total_valid_equal = 0.0
                        for index in range(len(x_batch_valid)):
                            feed_dict_valid = {model.x: x_batch_valid[index], model.y: y_batch_valid[index], model.keep_prob: args.keep_prob }
                            valid_loss, valid_equal = sess.run([model.cost, model.equal], feed_dict_valid)
                            total_valid_loss += valid_loss
                            total_valid_equal += valid_equal
                        total_valid_loss /= len(x_batch_valid)
                        total_valid_equal /= len(x_batch_valid)

                        print("================================= step {} ===================================".format(step))
                        print("validing step {}, epoch {}, loss: {:.4f}, accuracy: {:.4f}, time/batch: {:.3f}"
                            .format(step, epoch, total_valid_loss, np.mean(total_valid_equal), time.time()-start))
                        
                        _GL = checkEarlyStopGL(total_valid_loss, valLoss_opt)
                        _PG = checkEarlyStopPQ(total_valid_loss, valLoss_opt, trainLossList)
                        _UP = checkEarlyStopUP(total_valid_loss, lastStripValLoss, successiveHit)
                        if _GL > largestGL[0]:
                            largestGL[0] = _GL
                            largestGL[1] = current_step
                        if _PG > largestPG[0]:
                            largestPG[0] = _PG
                            largestPG[1] = current_step
                        if _UP > largestUP[0]:
                            largestUP[0] = _UP
                            largestUP[1] = current_step
                        print("Criteria GL : "+str(_GL))
                        print("Criteria PG : "+str(_PG))
                        print("Criteria UP : "+str(_UP))
                        print("==============================================================================")
                        #save model
                        #check GL criteria
                        if _GL > args.GL_threshold0:
                            args.GL_threshold0 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_GL0.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved GL0 model checkpoint to {}".format(path))
                            print2LogFile(args, "GL0", current_step, epoch, total_start_time)
                        if _GL > args.GL_threshold1:
                            args.GL_threshold1 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_GL1.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved GL1 model checkpoint to {}".format(path))
                            print2LogFile(args, "GL1", current_step, epoch, total_start_time)
                        if _GL > args.GL_threshold2:
                            args.GL_threshold2 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_GL2.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved GL2 model checkpoint to {}".format(path))
                            print2LogFile(args, "GL2", current_step, epoch, total_start_time)
                        if _GL > args.GL_threshold3:
                            args.GL_threshold3 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_GL3.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved GL3 model checkpoint to {}".format(path))
                            print2LogFile(args, "GL3", current_step, epoch, total_start_time)
                        #check PG criteria
                        if _PG > args.PG_threshold0:
                            args.PG_threshold0 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_PG0.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved PG0 model checkpoint to {}".format(path))
                            print2LogFile(args, "PG0", current_step, epoch, total_start_time)
                        if _PG > args.PG_threshold1:
                            args.PG_threshold1 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_PG1.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved PG1 model checkpoint to {}".format(path))
                            print2LogFile(args, "PG1", current_step, epoch, total_start_time)
                        if _PG > args.PG_threshold2:
                            args.PG_threshold2 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_PG2.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved PG2 model checkpoint to {}".format(path))
                            print2LogFile(args, "PG2", current_step, epoch, total_start_time)
                        if _PG > args.PG_threshold3:
                            args.PG_threshold3 = 10000000.0
                            checkpoint_path = os.path.join(args.save_dir, 'model_PG3.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved PG3 model checkpoint to {}".format(path))
                            print2LogFile(args, "PG3", current_step, epoch, total_start_time)
                        #check UP criteria
                        if _UP > args.UP_threshold0:
                            args.UP_threshold0 = 1000
                            checkpoint_path = os.path.join(args.save_dir, 'model_UP0.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved UP0 model checkpoint to {}".format(path))
                            print2LogFile(args, "UP0", current_step, epoch, total_start_time)
                        if _UP > args.UP_threshold1:
                            args.UP_threshold1 = 1000
                            checkpoint_path = os.path.join(args.save_dir, 'model_UP1.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved UP1 model checkpoint to {}".format(path))
                            print2LogFile(args, "UP1", current_step, epoch, total_start_time)
                        if _UP > args.UP_threshold2:
                            args.UP_threshold2 = 1000
                            checkpoint_path = os.path.join(args.save_dir, 'model_UP2.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved UP2 model checkpoint to {}".format(path))
                            print2LogFile(args, "UP2", current_step, epoch, total_start_time)
                        if _UP > args.UP_threshold3:
                            args.UP_threshold3 = 1000
                            checkpoint_path = os.path.join(args.save_dir, 'model_UP3.ckpt')
                            path = saver.save(sess, checkpoint_path, global_step = current_step)
                            print("Saved UP3 model checkpoint to {}".format(path))
                            print2LogFile(args, "UP3", current_step, epoch, total_start_time)


                        #setting variables
                        if total_valid_loss < valLoss_opt:
                            valLoss_opt = total_valid_loss
                        lastStripValLoss = total_valid_loss
                        successiveHit = _UP
                        trainLossList = list()

                    # current_step = tf.train.global_step(sess, global_step)
                    if current_step % args.save_every == 0 or (epoch == args.num_epochs-1 
                        and i == data_loader.num_batches-1): #save for the last result
                        checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                        path = saver.save(sess, checkpoint_path, global_step = current_step)
                        print("Saved model checkpoint to {}".format(path))

                        printEarlyStopLog2File(args, largestGL, largestPG, largestUP, current_step, epoch, total_start_time)
                        print("print early stop log to file : "+args.earlyStop_log_filename)
                        print("cost time : "+str(time.time()-total_start_time)+" secs")
                        #reset 
                        largestGL = [-1000,0]
                        largestPG = [-1000,0]
                        largestUP = [0,0]

            train_writer.close()

if __name__ == '__main__':
    main()

