# StructLSTM
StructLSTM - Structure augmented Long-Short Term Memory Networks for Music Generation. The code splits the complete process into three sections for sake of understanding and simplicity. Precprocessing, Training and Generation. These sections must be run separately at this stage according to instructions provided below and in the code itself. The code base is gheavily inspired by the dual softmax model developed by Yoav Zimmerman at http://yoavz.com/music_rnn/. The code uses this model as a base model with few changes in preprocessing. StructLSTM is created by augmenting extra durational and metrical information to this model to monitor the improvements. A detailed report on the experiment will be made available soon. 


Shakeel Raja

shakeelraja@hotmail.com

## REQUIREMENTS:
The code base has been developed and modified on a linux platform. Following libraries are needed in order to run the code. 

### Python 2.7

### TensorFlow 0.8 CPU

https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl

The data has been made available in the data folder as a zip file. 


### Python midi library:
git+https://github.com/vishnubob/python-midi#egg=midi


#### pip install following

### Mingus Notation Library:
pip install mingus

### matplotlib :
pip install matplotlob

### numpy :
pip install numpy


## FILES:

### preprocess.py


parses the MIDI dataset (a subset of data is included in the data folder), optionally adds the sequence counter and saves resulting one ot vectors as pickle file. Run presprocess files by setting appropriate flas for zero padding, simple counter, normalisation and metrical information augmentation. Identify locations for original train/valid split data and preprocessed data. This file uses midi_util.py for parsing MIDI files and mapping note values on scale specified as melody_range and harmony encoding is made through Mingus python library. 

### train.py

batches data according to given parameters and performs training and testing. Load the preprocessed dataset generated in preprocess.py. Based on flas set in preprocess.py, the code will zero pad the sequences to make them a multiple of 128 required for minibatching in tensorflow. Number of features augmented must be removed from labels. (Batch_data function allows removal of augmented features , see code for details). This file allows you to set grid search parameter values to automatically set up combinations, run each model defined by grid ans save the best model for each set of parameters used i.e. a 10 combintaion grid will generate 10 models with a training.txt file identifying loss information for each run. a png image of loss curves is also saved in each run along with a config file. (**note** edit model.py a to identify number of features in the model output. set output_dim to (input_dim - n , where n in number of augmented features) 

### model.py

contains the model class. Deafult is LSTM RNN. CAn be changed by setting model parameters in train.py config class.

### composer_structLSTM.py

takes in the config file from saved models and generates new sequences. Other composer files may be used to  generate from the baseline model and counter integration only. All the code from these intermediary composer files has been presented in strctLSTM composer, with comments. Identify the config file created by train.py along with the dataset that model was trained on. set umber of iterations and structural information will be calculated and augmented with each run during the conditioning and sampling stages. The output generated is saved as a MIDI file. (**note** edit model.py a to identify number of features in the model output. set output_dim to (input_dim - n , where n in number of augmented features) 




