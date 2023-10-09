import os
import numpy as np
from utils import*
from create_folder import add_folder


def create_actions_fromcamera():
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        break_all = False  # Add flagthanks
        # NEW LOOP
        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            if break_all:  # Check the flag
                break
            for sequence in range(number_sequences):
                # Loop through video length aka sequence length
                if break_all:  # Check the flagq
                    break
                


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
                    
                    # NEW Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break_all = True  # change flag
                        break
                        
        cap.release()
        cv2.destroyAllWindows()

def create_action_fromcamera(action):
    
    add_folder(DATA_PATH,action,number_sequences)
    
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
            
            video_path = os.path.join(DATA_PATH, action, str(sequence),f"{action}_{sequence}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')            
            out = cv2.VideoWriter(video_path, fourcc,30, (640, 480)) # the frame size is 640x480, you may need to change this based on your webcam resolution
            
            

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
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
            
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break_all = True  # change flag
                    break
            out.release()    
                    
        cap.release()
        cv2.destroyAllWindows()


# def create_action_from_videos(action, video_files):

#     add_folder(DATA_PATH, action, len(video_files))

#     for sequence, video_file in enumerate(video_files):
#         cap = cv2.VideoCapture(video_file)

#         # 查询视频的帧率和分辨率
#         camera_fps = int(cap.get(cv2.CAP_PROP_FPS))
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         video_path = os.path.join(DATA_PATH, action, str(sequence), f"{action}_{sequence}.mp4")
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(video_path, fourcc, camera_fps, (frame_width, frame_height))

#         frame_num = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Make detections
#             image, results = mediapipe_detection(frame, holistic)

#             # Draw landmarks
#             draw_styled_landmarks(image, results)

#             # Show to screen
#             cv2.imshow('OpenCV Feed', image)
#             out.write(image)

#             # Export keypoints
#             keypoints = extract_keypoints(results)
#             npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
#             np.save(npy_path, keypoints)

#             frame_num += 1

#             # Break gracefully
#             if cv2.waitKey(10) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         out.release()
        
#     cv2.destroyAllWindows()

# # 使用
# video_files = ["path_to_video1.mp4", "path_to_video2.mp4", ...]
# create_action_from_videos('your_action_name', video_files)





if "__main__" == __name__:
    # create_actions()
    new_action = 'help'
    create_action(new_action)


    