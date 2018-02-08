"""
Shakeel ur Rehman Raja
Final Project for MSc Data Science (2016/17). City, University of london  

StructLSTM - (Structure Augmented LSTM model)

The code is inspired by the dual softmax model developed by Yoav Zimmerman, 
at http://yoavz.com/music_rnn/. The baseline model has been developed by applying 
a few midifcations to the original code available at 
https://github.com/yoavz/music_rnn

This generation code has been used as described in Yoav, 2016 experiment. 
some minor code changes have been performed according to changes in preprocessing of data. 


The model performs the following functions
    Loads a trained model 
    Conditions the model with chords or random sequences from validation data
    samples the prbability distributrion to identify notes played
    takes the output from one timestep and inputs it to the next time step

Apart from minor adjustments to simplify the code , to additions in terms of processing have been made 
to this script. The generation from this script has been made to compare the performnce of
feature augmented models.

Feature Generator for feature Augmented Models is provided in composer_stucLSTM.py file 

**note** preprocessing, training and generation scripts are separate and must be run 
individually 


"""

import os
import argparse
import cPickle
import numpy as np
import tensorflow as tf    
import midi_util
import preprocess
from model import Model, NottinghamModel
from rnn2 import DefaultConfig

file ="data/nottingham_baseline.pickle"
config_file =    # provide location of baseline trained model.     

#generate  chords for priming the model
def generateChords(chord_to_idx, repeats, input_dim):
    r = preprocess.Melody_Range

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

if __name__ == '__main__':
    
    np.random.seed()      
    sample_seq = 'random'   # use random or chords

    with open(config_file, 'r') as f: 
        config = cPickle.load(f)
    
    if config.dataset == 'softmax':
        config.miniBatchLen = 1
        config.maxBatches = -1
        model_class = NottinghamModel
        with open(file, 'r') as f:
            pickle = cPickle.load(f)
        chord_to_idx = pickle['chord_to_idx']
    
        tStep = 120
        resolution = 480
    
    print (config)
#%%
    
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            sampling_model = model_class(config)
    
        saver = tf.train.Saver(tf.all_variables())
        model_path = os.path.join(os.path.dirname(args.config_file), 
            config.model_name)
        saver.restore(session, model_path)
    
        state = sampling_model.get_cell_zero_state(session, 1)
        if args.sample_seq == 'chords':
            # 16 - one measure, 64 - chord progression
            repeats = args.sample_length / 64
            sample_seq = generateChords(chord_to_idx, repeats, config.input_dim)
            print ('Sampling melody using a I, VI, IV, V progression')
    
        elif args.sample_seq == 'random':
            sample_index = np.random.choice(np.arange(len(pickle['valid'])))
            sample_seq = [ pickle['valid'][sample_index][i, :] 
                for i in range(pickle['valid'][sample_index].shape[0]) ]

#        length = 208 #args.sample_length 
#        x = np.array(length)
        chord = sample_seq[0]
        seq = [chord]
#        chord = np.append(chord, x)
            
        if args.conditioning > 0:
            for i in range(1, args.conditioning):
                seq_input = np.reshape(chord, [1, 1, config.input_dim])
                feed = {
                    sampling_model.seq_input: seq_input,
                    sampling_model.initial_state: state,
                }
                state = session.run(sampling_model.final_state, feed_dict=feed)
                chord = sample_seq[i]
                seq.append(chord)

        writer = midi_util.NottinghamMidiWriter(chord_to_idx, verbose=False)
        sampler = midi_util.NottinghamSampler(chord_to_idx, verbose=False)
        probb = []
#        x = x - np.array(1)
        for i in range(max(args.sample_length - len(seq), 0)):
            seq_input = np.reshape(chord, [1, 1, config.input_dim])
            print seq_input
            feed = {
                sampling_model.seq_input: seq_input,
                sampling_model.initial_state: state,
            }
            [probs, state] = session.run(
                [sampling_model.probs, sampling_model.final_state],
                feed_dict=feed)
            probs = np.reshape(probs, [config.input_dim])
            probb.append(probs)
            
            chord = sampler.sample_notes(probs)

            if config.dataset == 'softmax':
                r = preprocess.Melody_Range
                if args.sample_melody:
                    chord[r:] = 0
                    chord[r:] = sample_seq[i][r:]
                elif args.sample_harmony:
                    chord[:r] = 0
                    chord[:r] = sample_seq[i][:r]
    
            seq.append(chord)
        # repeat the last note for 1 bar to avoid abrupt ending
        seq = seq[:-2]
        for i in range(16):
            seq.append(seq[-1])
            
        prob_matrix = np.array(probb)
        import matplotlib.pyplot as plt
        #plt.figure(figsize=(16, 12))
        plt.imshow(prob_matrix.T, cmap='Reds', interpolation='nearest')
        plt.show()
        np.savetxt("models/zzGENERATED/baseline/probs_s2.csv", prob_matrix, delimiter=",")
        writer.dump_sequence_to_midi(seq, "models/zzGENERATED/baseline/baseline_s2.mid", 
            tStep=tStep, resolution=resolution)