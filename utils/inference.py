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
from random import sample
import cv2
import scipy
import time
import tempfile
import numpy as np
import tensorflow as tf
from subprocess import call
from scipy.io import wavfile


from utils.audio_handler import  AudioHandler
from utils.rendering import render_mesh_helper

def process_audio(ds_path, audio, sample_rate, previous_state_c, previous_state_h):
    config = {}
    config['deepspeech_graph_fname'] = ds_path
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1
    #config['deepspeech_graph_fname'] = "./ds_graph/deepspeech-0.5.0-models/output_graph.tflite"
    tmp_audio = {'subj': {'audio': audio, 'sample_rate': sample_rate}}
    tmp_pre_state_c = {'subj': previous_state_c}
    tmp_pre_state_h = {'subj': previous_state_h}
    audio_handler = AudioHandler(config)


    audio, new_state_c, new_state_h = audio_handler.process(tmp_audio, tmp_pre_state_c, tmp_pre_state_h)

    return audio['subj']['audio'], new_state_c['subj'], new_state_h['subj']


def output_sequence_meshes(sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    mesh_out_path = os.path.join(out_path, 'meshes')
    if not os.path.exists(mesh_out_path):
        os.makedirs(mesh_out_path)

    if os.path.exists(uv_template_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
    else:
        vt, ft = None, None

    num_frames = sequence_vertices.shape[0]
    for i_frame in range(num_frames):
        out_fname = os.path.join(mesh_out_path, '%05d.obj' % i_frame)
        out_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            out_mesh.vt, out_mesh.ft = vt, ft
        if os.path.exists(texture_img_fname):
            out_mesh.set_texture_image(texture_img_fname)
        out_mesh.write_obj(out_fname)

def render_sequence_meshes(audio_fname, sequence_vertices, template, out_path, uv_template_fname='', texture_img_fname=''):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 60, (800, 800), True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (800, 800), True)

    if os.path.exists(uv_template_fname) and os.path.exists(texture_img_fname):
        uv_template = Mesh(filename=uv_template_fname)
        vt, ft = uv_template.vt, uv_template.ft
        tex_img = cv2.imread(texture_img_fname)[:,:,::-1]
    else:
        vt, ft = None, None
        tex_img = None

    num_frames = sequence_vertices.shape[0]
    center = np.mean(sequence_vertices[0], axis=0)
    for i_frame in range(num_frames):
        render_mesh = Mesh(sequence_vertices[i_frame], template.f)
        if vt is not None and ft is not None:
            render_mesh.vt, render_mesh.ft = vt, ft
        img = render_mesh_helper(render_mesh, center, tex_img=tex_img)
        writer.write(img)
    writer.release()

    video_fname = os.path.join(out_path, 'video.mp4')
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)


def inference_realtime(tf_model_fname, ds_fname, audio, sample_rate, previous_state_c, previous_state_h):


    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    #condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    #input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/ExpressionLayer/output_decoder:0')



    with tf.Session() as session:
        saver.restore(session, tf_model_fname)
        

        startTime = time.time()
        processed_audio, new_state_c, new_state_h = process_audio(ds_fname, audio, sample_rate, previous_state_c, previous_state_h)
        endTime = time.time()
        print("second usage for the audio in processing audio:", endTime - startTime)

        feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                    is_training: False,
                    }
        
        # Restore trained model
        predicted_vertices = np.reshape(session.run(output_decoder, feed_dict), [-1, 52])
        endTime = time.time()
        print("whole second usage for the audio:", endTime - startTime)
        print(predicted_vertices.shape)
        #output_sequence_meshes(predicted_vertices, template, out_path)
        #if(render_sequence):
        #    render_sequence_meshes(audio_fname, predicted_vertices, template, out_path, uv_template_fname, texture_img_fname)
    tf.reset_default_graph()
    return predicted_vertices, new_state_c, new_state_h

def inference(tf_model_fname, ds_fname, audio_fname):

    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:,0]

    

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    #condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    #input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/ExpressionLayer/output_decoder:0')

    seconds = len(audio) / sample_rate
    previous_state_c = np.zeros([1, 2048], dtype = np.float32)
    previous_state_h = np.zeros([1, 2048], dtype = np.float32)

    with tf.Session() as session:
        saver.restore(session, tf_model_fname)
        
        for i in range(int(seconds * 10)):

            startTime = time.time()
            processed_audio, new_state_c, new_state_h = process_audio(ds_fname, audio[int(i * 0.1 * sample_rate): int((i + 1) * 0.1 * sample_rate)], sample_rate, previous_state_c, previous_state_h)
            endTime = time.time()
            #print("second usage for 100ms audio in processing audio:", endTime - startTime)

            feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                        is_training: False,
                        }
            previous_state_c = new_state_c
            previous_state_h = new_state_h
            # Restore trained model
            predicted_vertices = np.reshape(session.run(output_decoder, feed_dict), [-1, 52])
            endTime = time.time()
            print("second usage for processing 100ms audio clip: ", endTime - startTime)
            print("the shape of generated blendshape numpy array: ", predicted_vertices.shape)
        #output_sequence_meshes(predicted_vertices, template, out_path)
        #if(render_sequence):
        #    render_sequence_meshes(audio_fname, predicted_vertices, template, out_path, uv_template_fname, texture_img_fname)
    tf.reset_default_graph()


def inference_interpolate_styles(tf_model_fname, ds_fname, audio_fname, template_fname, condition_weights, out_path):
    template = Mesh(filename=template_fname)

    sample_rate, audio = wavfile.read(audio_fname)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:, 0]

    processed_audio = process_audio(ds_fname, audio, sample_rate)

    # Load previously saved meta graph in the default graph
    saver = tf.train.import_meta_graph(tf_model_fname + '.meta')
    graph = tf.get_default_graph()

    speech_features = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/speech_features:0')
    condition_subject_id = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/condition_subject_id:0')
    is_training = graph.get_tensor_by_name(u'VOCA/Inputs_encoder/is_training:0')
    input_template = graph.get_tensor_by_name(u'VOCA/Inputs_decoder/template_placeholder:0')
    output_decoder = graph.get_tensor_by_name(u'VOCA/output_decoder:0')

    non_zeros = np.where(condition_weights > 0.0)[0]
    condition_weights[non_zeros] /= sum(condition_weights[non_zeros])

    num_frames = processed_audio.shape[0]
    output_vertices = np.zeros((num_frames, template.v.shape[0], template.v.shape[1]))

    with tf.Session() as session:
        # Restore trained model
        saver.restore(session, tf_model_fname)

        for condition_id in non_zeros:
            feed_dict = {speech_features: np.expand_dims(np.stack(processed_audio), -1),
                         condition_subject_id: np.repeat(condition_id, num_frames),
                         is_training: False,
                         input_template: np.repeat(template.v[np.newaxis, :, :, np.newaxis], num_frames, axis=0)}
            predicted_vertices = np.squeeze(session.run(output_decoder, feed_dict))
            output_vertices += condition_weights[condition_id] * predicted_vertices

        output_sequence_meshes(output_vertices, template, out_path)