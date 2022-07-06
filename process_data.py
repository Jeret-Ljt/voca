from re import A
import numpy as np
import os 
import pickle
import argparse
import json

parser = argparse.ArgumentParser(description='Voice operated character animation')
parser.add_argument('--path', default='./tmp/raw_blend/', help='Path to trained VOCA model')
parser.add_argument('--genJson', type = bool, default=False, help='Path to trained VOCA model')

args = parser.parse_args()
 


def main():
    if os.path.exists("./training_data_new/data_verts.npy"):
        data_vert = np.load("./training_data_new/data_verts.npy", mmap_mode='r')
        print(data_vert.shape)
        data2array = pickle.load(open("./training_data_new/subj_seq_to_idx.pkl", 'rb'), encoding='latin1')
        
        print(len(data2array["1.mp4"]) / 3600)
        print(len(data2array["2.mp4"]) / 3600)
        print(len(data2array["3.mp4"]) / 3600)
        print(len(data2array["4.mp4"]) / 3600)
        print(len(data2array["5.mp4"]) / 3600)
        print(len(data2array["6.mp4"]) / 3600)
        print(len(data2array["7.mp4"]) / 3600)



        if args.genJson == False:
            return

        nameMap = [
        "browInnerUp",  #(browInnerUp_L + browInnerUp_R) / 2
        "browDownLeft",  #browDown_L
        "eyeBlinkLeft",          #eyeBlink_L
        "eyeSquintLeft",       #"eyeSquint_L",
        "eyeWideLeft",        #"eyeWide_L",
        "eyeLookUpLeft",      #"eyeLookUp_L",
        "eyeLookOutLeft",      #"eyeLookOut_L",
        "eyeLookInLeft",        #"eyeLookIn_L",
        "eyeLookDownLeft",       #"eyeLookDown_L",
        "noseSneerLeft",     #noseSneer_L",
        "mouthUpperUpLeft",    #"mouthUpperUp_L",
        "mouthSmileLeft",    #"mouthSmile_L",
        "mouthLeft",        #"mouthLeft"
        "mouthFrownLeft", #"mouthFrown_L",
        "mouthLowerDownLeft", #"mouthLowerDown_L",
        "jawLeft", #jawLeft
        "cheekPuff", #cheekPuff
        "mouthShrugUpper", #mouthShrugUpper
        "mouthFunnel", #mouthFunnel
        "mouthRollLower", #mouthRollLower
        "jawOpen", #jawOpen
        "tongueOut", #tongueOut
        "mouthPucker", #mouthPucker
        "mouthRollUpper", #mouthRollUpper
        "jawRight",     #jawRight
        "mouthLowerDownRight",  #"mouthLowerDown_R",
        "mouthFrownRight",      #mouthFrown_R,
        "mouthRight",           #mouthRight
        "mouthSmileRight",       #"mouthSmile_R",
        "mouthUpperUpRight",     #"mouthUpperUp_R",
        "noseSneerRight",   #noseSneer_R
        "eyeLookDownRight",  #eyeLookDown_R
        "eyeLookInRight",   #eyeLookIn_R
        "eyeLookOutRight", #eyeLookOut_R
        "eyeLookUpRight", #eyeLookUp_R
        "eyeWideRight", #eyeWide_R
        "eyeSquintRight", #eyeSquint_R
        "eyeBlinkRight", #eyeBlink_R
        "browDownRight", #browDown_R
        "browOuterUpRight", #browOuterUp_R
        "jawForward",
        "mouthClose", 
        "mouthDimpleLeft",  
        "mouthDimpleRight",  
        "mouthStretchLeft",  
        "mouthStretchRight",  
        "mouthShrugLower",  
        "mouthPressLeft",  
        "mouthPressRight",  
        "browOuterUpLeft",  
        "cheekSquintLeft",  
        "cheekSquintRight"
        ] 
        print(len(nameMap))
        assert(len(nameMap) == 52)
        
        outputJson = {'version': 3.0, 'fps': 30.0, 'scene': {'timestamp': 2991.55, 'actors': [{'name': 'Alan', 'color': [45, 116, 197], 'meta': {'hasGloves': False, 'hasLeftGlove': False, 'hasRightGlove': False, 'hasBody': False, 'hasFace': True}, 'dimensions': {'totalHeight': 1.85, 'hipHeight': 0.98249}, 'face': {'faceId': '8kxpLX', 'eyeBlinkLeft': 5.92665, 'eyeLookDownLeft': 31.85337, 'eyeLookInLeft': 13.41856, 'eyeLookOutLeft': 0.0, 'eyeLookUpLeft': 0.0, 'eyeSquintLeft': 3.3543, 'eyeWideLeft': 0.0, 'eyeBlinkRight': 5.92734, 'eyeLookDownRight': 31.93685, 'eyeLookInRight': 0.0, 'eyeLookOutRight': 1.21167, 'eyeLookUpRight': 0.0, 'eyeSquintRight': 3.35427, 'eyeWideRight': 0.0, 'jawForward': 9.19123, 'jawLeft': 0.87074, 'jawRight': 0.0, 'jawOpen': 18.4327, 'mouthClose': 7.8636, 'mouthFunnel': 14.44514, 'mouthPucker': 8.23915, 'mouthLeft': 0.0, 'mouthRight': 0.45597, 'mouthSmileLeft': 4.65525, 'mouthSmileRight': 7.38398, 'mouthFrownLeft': 0.0, 'mouthFrownRight': 0.0, 'mouthDimpleLeft': 6.98382, 'mouthDimpleRight': 7.29106, 'mouthStretchLeft': 27.60838, 'mouthStretchRight': 28.04687, 'mouthRollLower': 6.62563, 'mouthRollUpper': 2.71992, 'mouthShrugLower': 12.17715, 'mouthShrugUpper': 9.36523, 'mouthPressLeft': 4.58441, 'mouthPressRight': 4.75144, 'mouthLowerDownLeft': 24.13173, 'mouthLowerDownRight': 23.33478, 'mouthUpperUpLeft': 13.73715, 'mouthUpperUpRight': 13.28578, 'browDownLeft': 0.0, 'browDownRight': 0.0, 'browInnerUp': 17.85841, 'browOuterUpLeft': 2.24519, 'browOuterUpRight': 2.24475, 'cheekPuff': 1.55678, 'cheekSquintLeft': 9.27116, 'cheekSquintRight': 10.266, 'noseSneerLeft': 18.50312, 'noseSneerRight': 16.85102, 'tongueOut': 0.0}}], 'props': []}}
         
        for frame, id in data2array['6.mp4'].items():
            if frame >= 12 * 30:
                break    
            for j in range(52):
                if j < 40:
                    outputJson['scene']['actors'][0]['face'][nameMap[j]] = float(data_vert[id][j][0])
                else:
                    outputJson['scene']['actors'][0]['face'][nameMap[j]] = 0
            outputJson['scene']['timestamp'] = frame / 30
            with open('./training_data_new/json/blendshape-'+str(frame)+".json", 'w')  as f:
                json.dump(outputJson, f)
        
        return
    
    count = 1
    index = 0
    frame = 0
    array = []
    data2array = {}

    nowTimeStamp = 0
    recordTimeStamp = 0
    lastTimeStamp = 0

    lastBlend = np.zeros(40)
    mp4Now = "xxx"

    while True:
        fileName = args.path + "debugout-" + str(count) + ".txt"
        if not os.path.exists(fileName):
            break
        
        fp=open(fileName,'r')
        lines=fp.readlines()

        for line in lines:
            if line[0] == '-':
                continue
            if line.split('.')[1][:3] == 'mp4':
                if line[1:-2] != mp4Now:
                    nowTimeStamp = 0
                    frame = 0
                    mp4Now = line[1:-2]

                    print(mp4Now)
                    data2array[mp4Now] = {}
                continue
            if line[0] == '[':
                recordBlend = np.array(line[1:-2].split(', '), dtype = np.float32)
                while nowTimeStamp < recordTimeStamp:
                    nowBlend = (recordBlend - lastBlend) / (recordTimeStamp - lastTimeStamp) * (nowTimeStamp - lastTimeStamp) +  lastBlend
                    array.append(nowBlend)
                    nowTimeStamp += 1 / 30
                    data2array[mp4Now][frame] = index
                    frame += 1
                    index += 1
                lastBlend = recordBlend
                lastTimeStamp = recordTimeStamp
            else:
                recordTimeStamp = float(line)

        count += 1

    array = np.reshape(np.array(array, dtype = np.float32), [-1, 40, 1])

    print(array)
    print(data2array)

    np.save("training_data_new/data_verts.npy", array)
    pickle.dump(data2array, open("training_data_new/subj_seq_to_idx.pkl", 'wb'))


main()