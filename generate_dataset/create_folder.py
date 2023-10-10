import os
import numpy as np
from utils import*

def create_folder(DATA_PATH,add_action,number_sequences,add=False):
    if add:
        l = len(os.listdir(os.path.join(DATA_PATH, add_action)))
        for sequence in range(l,l+number_sequences):
            os.makedirs(os.path.join(DATA_PATH, add_action, str(sequence)), exist_ok=True)
    else:        
        for sequence in range(number_sequences):
            os.makedirs(os.path.join(DATA_PATH, add_action, str(sequence)), exist_ok=True)



if "__main__" == __name__:
    for add_action in actions:
        create_folder(DATA_PATH,add_action,number_sequences)



