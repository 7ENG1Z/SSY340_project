
import numpy as np
import os


def loaddata(DATA_PATH):
    #give the actions lable_map
    # get actions from folder
    actions = [action for action in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, action))]
    #map the actions to index
    label_map = {label: num for num, label in enumerate(actions)}
    #load the dataset in nparray
    sequences, labels = [], []
    for action in actions:                                                 
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            video = []
            files = sorted(os.listdir(os.path.join(DATA_PATH, action, str(sequence))), key=lambda x: int(x.split('.')[0]))#changed
            for frame_file in files:
                # Check if it's not a video file
                _, ext = os.path.splitext(frame_file)
                if ext not in [".mp4", ".avi"]:
                # if ext != ".mp4":  
                    res = np.load(os.path.join(DATA_PATH, action, str(sequence), frame_file))
                    video.append(res)
            # for frame_num in range(len(os.listdir(os.path.join(DATA_PATH, action, str(sequence)))) - 1):  # -1 to avoid count the vedios
            #     res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            #     video.append(res)
            sequences.append(video)
            labels.append(label_map[action])

    X = np.array(sequences)
    Y = np.array(labels)

    print(X.shape)
    print(Y.shape)
    return X,Y,label_map




