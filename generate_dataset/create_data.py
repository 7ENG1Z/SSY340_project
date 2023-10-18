import os
import numpy as np
from utils import*
from create_folder import create_folder

def create_action_fromcamera(DATA_PATH,action,number_sequences,add = False):
    '''
    DATA_PATH: the location of the camera collected dataset
    action:the word of the sign language
    number_sequence: number of sequences collected 
    add: add new data
    This function can create a folder for an action in path location,then it could save the keypoint positions for each frame and the video in folders of frames.
    
    
    '''
    #create the folder for the action
    if add:
        create_folder(DATA_PATH,action,number_sequences,add)
    else:
        create_folder(DATA_PATH,action,number_sequences)
    
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        break_all = False  # Add flag
        # NEW LOOP
        # Loop through actions
        for sequence in range(number_sequences):
            if break_all:  # Check the flagq
                break
            # Loop through video length aka sequence length
            if add:
                l = len(os.listdir(os.path.join(DATA_PATH, action)))
                video_path = os.path.join(DATA_PATH, action, str(l-number_sequences+sequence),f"{action}_{l-number_sequences+sequence}.mp4")
            else:
                video_path = os.path.join(DATA_PATH, action, str(sequence),f"{sequence}.mp4")


            fourcc = cv2.VideoWriter_fourcc(*'mp4v')            
            out = cv2.VideoWriter(video_path, fourcc,30, (640, 480)) # the fradd_action
            
            

            for frame_num in range(sequence_length):
 
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0 : 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    while cv2.waitKey(0) & 0xFF != ord('s'):
                        pass
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # Write the frame into the video file
                out.write(image)
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                if add:
                    l = len(os.listdir(os.path.join(DATA_PATH, action)))
                    npy_path = os.path.join(DATA_PATH, action, str(l-number_sequences+sequence), str(frame_num))
                else:
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
            
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break_all = True  # change flag
                    break
            out.release()    
                    
        cap.release()
        cv2.destroyAllWindows()

def create_action_frompicture(image_path):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        img = cv2.imread(image_path)
        image, results = mediapipe_detection(img, holistic)
        keypoints = extract_keypoints(results)
        # draw_styled_landmarks(image, results)
        # cv2.imshow('OpenCV Feed', image)
        # cv2.waitKey(5000)
        # cv2.destroyAllWindows()
        npy_path = os.path.splitext(image_path)[0]
        np.save(npy_path, keypoints)    

if "__main__" == __name__:


    action = 'go'
    add = True
    # create_action_fromcamera(DATA_PATH,action,number_sequences)
    create_action_fromcamera(DATA_PATH,action,30,add)



    