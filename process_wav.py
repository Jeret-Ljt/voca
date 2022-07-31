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


from scipy.io import wavfile
import numpy as np
import pickle



audioDic = {}
for count in np.arange(1, 8):
    audioName = "./audio/" +  str(count) + ".wav"
    videoName = str(count) + ".mp4"
    print(videoName)
    sample_rate, audio = wavfile.read(audioName)
    if audio.ndim != 1:
        print('Audio has multiple channels, only first channel is considered')
        audio = audio[:,0]
    audioDic[videoName] = {'audio': audio, 'sample_rate': sample_rate}


pickle.dump(audioDic, open("training_data_new/raw_audio_fixed.pkl", 'wb'))

