'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''


import os
import glob
import argparse
from utils.inference import inference

def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ['true', 't', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'f', 'no', 'n']:
            return False
    return False

parser = argparse.ArgumentParser(description='Voice operated character animation')
parser.add_argument('--tf_model_fname', default='./checkpoints/gstep_134310.model', help='Path to trained sync-lip model')
parser.add_argument('--ds_fname', default='./ds_graph/deepspeech-0.5.0-models/output_graph.tflite', help='Path to trained DeepSpeech model')
parser.add_argument('--audio_fname', default='./audio/test_sentence.wav', help='Path of input speech sequence')

args = parser.parse_args()
tf_model_fname = args.tf_model_fname
ds_fname = args.ds_fname
audio_fname = args.audio_fname



inference(tf_model_fname, ds_fname, audio_fname)

