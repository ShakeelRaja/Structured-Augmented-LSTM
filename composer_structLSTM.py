"""
Shakeel ur Rehman Raja
Final Project for MSc Data Science (2016/17). City, University of london  

StructLSTM - (Structure Augmented LSTM model)

The code is inspired by the dual softmax model developed by Yoav Zimmerman, 
at http://yoavz.com/music_rnn/. The baseline model has been developed by applying 
a few midifcations to the original code available at 
https://github.com/yoavz/music_rnn

This script integrates a complete feature generator including a duration counter 
metrical features for the dataset. 

This code performs following function on datasets created in preprocess.py
    Loads a trained model based on given configuration file
    conditions the trained model with random sequences augmented with structural features
    samples from the output of the model and continues feature augmentation
    The final output is saved as a midi file

The length of sequence can be set according to user preferences

**note** preprocessing, training and generation scripts are separate and must be run 
individually 

"""
import os
import argparse
import cPickle

import numpy as np
import tensorflow as tf  
import copy   

import midi_util
import preprocess
from model import Model, NottinghamModel
from rnn2 import DefaultConfig

# library to generate a chord sequence for conditioning the model
def i_vi_iv_v(chord_to_idx, repeats, input_dim):
    r = preprocess.Melody_Range 
    input_dim = input_dim -5

    i = np.zeros(input_dim)
    i[r + chord_to_idx['CM']] = 1
    vi = np.zeros(input_dim)
    vi[r + chord_to_idx['Am']] = 1
    iv = np.zeros(input_dim)
    iv[r + chord_to_idx['FM']] = 1
    v = np.zeros(input_dim)
    v[r + chord_to_idx['GM']] = 1

    full_seq = [i] * 16 + [vi] * 16 + [iv] * 16 + [v] * 16
    full_seq = full_seq * repeats
    
    return full_seq
"""
Uncomment one of the following to use required dataset
""" 
# Use 4/4 subset of data with duration and metrical augmented features (improved generation)
file = '/home/shaks/Desktop/Music_RNN/data/nottingham_allin_notStep_44.pickle'
config_file = 'models/44_no_Ts/numLayers_2_hidSize_200_melCoef_0p5_dropOut_0p5_inpDropOut_1_timeBatchLen_128_cellType_LSTMRUN_5_.config'

# Use complete dataset with duration and metrical augmented features (produces low quality results due to mixed signature)
#file = 'data/nottingham_allin_notStep.pickle'
#config_file = 'models/all_no_Ts/numLayers_2_hidSize_200_melCoef_0p5_dropOut_0p3_inpDropOut_1_timeBatchLen_128_cellType_LSTMRUN_4_.config'

conditioning=32
sampleLength=384
# Use random for randomly selected conditioning sequence from validation data, or use chords (doesnt work well metrical augmentation)
sample_seq ='random'    #choices = ['random', 'chords'])
time_step = 120         # midi resolution and set timestep for savin data
resolution = 480


# main program
if __name__ == '__main__':
    
    np.random.seed()      

    # open saved config file for loading trained model    
    with open(config_file, 'r') as f: 
        config = cPickle.load(f)
    if config.dataset == 'softmax':
        config.time_batch_len = 1
        config.max_time_batches = -1
        model_class = NottinghamModel
        with open(file, 'r') as f:
            pickle = cPickle.load(f)
        chord_to_idx = pickle['chord_to_idx']    
    print (config)
  
    # create the sampling model with config settings
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            sampling_model = model_class(config)
    
        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(os.path.dirname( config_file), 
            config.model_name)
        saver.restore(session, model_path)
        state = sampling_model.get_cell_zero_state(session, 1)
        
        # Create condition sequence based on user choice
        if  sample_seq == 'chords':
            # 16 - one measure, 64 - chord progression
            repeats =  sampleLength / 64
            sample_seq = i_vi_iv_v(chord_to_idx, repeats, config.input_dim)
            print ('Sampling melody using a I, VI, IV, V progression')
    
        elif  sample_seq == 'random':
            sample_index = np.random.choice(np.arange(len(pickle['valid'])))
            sample_seq = [ pickle['valid'][sample_index][i, :-5] 
                for i in range(pickle['valid'][sample_index].shape[0]) ]
          
        # create Feature Generator for augmenting structural features
        lenVec = [] 
        rev = []
        length =  sampleLength 
        lenVec = [range(length)]
        for y in lenVec:
                y = y[::-1]
                y2 = copy.deepcopy(y)
                norm = [float(i)/max(y) for i in y] # optional normalize
                y = norm
                tt = ([[item] for item in y])
                rev +=[tt]
        
        # Append the normalised counter to the conditioning sequences
        ptr = 0 
        chord1 = sample_seq[0]
        seq = [chord1]
        chord = np.append(chord1, rev[0][ptr])

        # create array for metrical features augmentation     
        mSteps = [[1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0],  
                  [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], 
                  [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
                  [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]]
        mCount = 0 
        chord = np.append(chord, mSteps[mCount])
        
        # condition the model with feature augment sequences for steps defined 
        if  conditioning > 0:
            ptr = ptr +1
            for i in range(1,  conditioning):
                seq_input = np.reshape(chord, [1, 1, config.input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                }
                state = session.run(sampling_model.final_state, feed_dict=feed)
                chord = sample_seq[i]
                seq.append(chord)
                
                # Augment and append features index forf next time step 
                if rev[0][ptr] == 0:
                    ptr = ptr
                else:
                    ptr = ptr + 1
                chord = np.hstack((chord, rev[0][ptr]))# 
                
                if mCount <15:
                    mCount +=1
                elif mCount == 15:
                    mCount = 0
                chord = np.hstack((chord, mSteps[mCount]))                   
#       # initialize samplng and MIDI saving 

        writer = midi_util.NottinghamMidiWriter(chord_to_idx, verbose=False)
        sampler = midi_util.NottinghamSampler(chord_to_idx, verbose=False)

        probb = [] #for viewing probabilities
        
        # start the generation and sampling 
        for i in range(max( sampleLength - len(seq)+ 1, 0)):
            seq_input = np.reshape(chord, [1, 1, config.input_dim])
            
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state,
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            #Calculate probabilities through sampling 
            probs = np.reshape(probs, [config.input_dim-5])
            probb.append(probs)
            chordx = sampler.sample_notes(probs)
            
            # augment features for next sampling cycle 
            chord = np.append(chordx, rev[0][ptr])
            x = rev[0][ptr]
            if float(x[0]) > 0:
                ptr = ptr + 1
            chord = np.hstack((chord, mSteps[mCount]))  
            if mCount <15:
                mCount +=1
            elif mCount == 15:
                mCount = 0
            seq.append(chordx)
        
        # repeat the last note for 1 bar to avoid abrupt ending
        seq = seq[:-2]
        for i in range(16):
            seq.append(seq[-1])
            
        # print probabilities of generated sequences
        prob_matrix = np.array(probb)
        import matplotlib.pyplot as plt
        plt.imshow(prob_matrix.T,cmap = "Reds",  interpolation='nearest')
        plt.show()
        # Save probabilities as csv
        np.savetxt("models/zzGENERATED/all_noTS/probs_s3.csv", prob_matrix, delimiter=",")
        # Write output to MIDI File 
        writer.dump_sequence_to_midi(seq, "models/zzGENERATED/all_noTS/all_noTS_s3.mid", 
            time_step=time_step, resolution=resolution)
