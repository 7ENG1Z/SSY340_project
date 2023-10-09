import os
import numpy as np
from utils import*

def create_folder(DATA_PATH,actions,number_sequences):
    for action in actions: 
        # dirmax = np.max(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        for sequence in range(number_sequences):
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)


def add_folder(DATA_PATH,add_action,number_sequences):
    for sequence in range(number_sequences):
        os.makedirs(os.path.join(DATA_PATH, add_action, str(sequence)), exist_ok=True)



if "__main__" == __name__:

    create_folder(DATA_PATH,actions,number_sequences)
    #add new actions 
    # add_action = 'help'
    # add_folder(DATA_PATH,add_action,number_sequences)





