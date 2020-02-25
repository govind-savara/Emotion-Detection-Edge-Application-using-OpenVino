import argparse
import cv2
import numpy as np
from inference import Network

INPUT_STREAM = "./resource/test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
FACE_DET_MODEL = "./models/face-detection-adas-0001.xml"
EMOTIONS_MODEL = "./models/emotions-recognition-retail-0003.xml"


EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']


def get_args():
    """
    Gets the arguments from the command line.
    """
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default='CPU')
    parser.add_argument("-ct", help=ct_desc, default=0.5)
    args = parser.parse_args()

    args.ct = float(args.ct)

    return args


def preprocessing(input_frame, height, width):
    """
    Given an input frame, height and width:
    - Resize to height and width
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    """
    frame = cv2.resize(input_frame, (width, height))
    frame = frame.transpose((2, 0, 1))
    frame = frame.reshape(1, *frame.shape)

    return frame


def infer_on_data(args, face_detection_model, emotions_model):
    """
    Handles input image, video or webcame and detect the faces.
    """

    # Handle image, video or webcam

    # create a flag for single image
    image_flag = False

    # Check if the input is a webcam
    if args.i == 'CAM':
        args.i = 0
    elif args.i.endswith('.jpg') or args.i.endswith('.bmp') or args.i.endswith('.png'):
        image_flag = True

    # Initialize the Inference Engine
    plugin_face_detection = Network()

    # Load the network models into the IE
    plugin_face_detection.load_model(face_detection_model, args.d, CPU_EXTENSION)
    net_input_shape_fcd = plugin_face_detection.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    if not image_flag:
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        out = cv2.VideoWriter('out_emotions.mp4', 0x00000021, 30, (width, height))
    else:
        out = None

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame_fcd = preprocessing(frame, net_input_shape_fcd[2], net_input_shape_fcd[3])

        # Perform inference on the frame to detect face
        plugin_face_detection.async_inference(p_frame_fcd)

        # Get the output of inference
        if plugin_face_detection.wait() == 0:
            result_fcd = plugin_face_detection.extract_output()

            # pipeline for emotion detection
            out_frame = emotion_detection(emotions_model, frame, result_fcd, args, width, height)

            # Writing out the frame, depending on image or video
            if image_flag:
                cv2.imwrite('output_emotions_img.jpg', out_frame)
            else:
                out.write(out_frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Closing the stream and any windows at the end of the application
    if not image_flag:
        out.release()

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def emotion_detection(emotions_model, frame, result, args, width, height):
    """
    Detect the emotion of the faces of a frame.
    """
    # Initialize the Inference Engine
    plugin_emotions_detection = Network()

    # Load the network models into the IE
    plugin_emotions_detection.load_model(emotions_model, args.d, CPU_EXTENSION)
    net_input_shape_ed = plugin_emotions_detection.get_input_shape()

    for box in result[0][0]:
        conf = box[2]
        if conf >= args.ct:
            # calculate the rectangle box margins
            x_min = max(int(box[3] * width), 0)
            y_min = max(int(box[4] * height), 0)
            x_max = min(int(box[5] * width), width)
            y_max = min(int(box[6] * height), height)

            # crop the image for emotion detection
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            if cropped_frame.shape[0] and cropped_frame.shape[1]:
                # Draw rectangle box on the input
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)

                print('cropped frame: ', cropped_frame.shape)
                # Preprocess the cropped image
                p_frame_ed = preprocessing(cropped_frame, net_input_shape_ed[2], net_input_shape_ed[3])

                # Perform inference on the frame to detect emotion
                plugin_emotions_detection.async_inference(p_frame_ed)

                if plugin_emotions_detection.wait() == 0:
                    result_ed = plugin_emotions_detection.extract_output()

                    # Get the emotions class
                    emotion_class_id = np.argmax(result_ed)
                    emotion_class = EMOTIONS[emotion_class_id]
                    print('emotion detected:', emotion_class)

                    # # Crate a rectangle box to display emotion text
                    # sub_img = frame[y_min:y_min+20, x_min:x_max]
                    # white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

                    # res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

                    # # Putting the image back to its position
                    # frame[y_min:y_min+20, x_min:x_max] = res

                    # Create a rectangle to display the predicted emotion
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_min + 20), (51, 255, 196), cv2.FILLED)
                    cv2.putText(frame, emotion_class, (x_min + 5, y_min + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)

    return frame


def main():
    args = get_args()
    face_detection_model = FACE_DET_MODEL
    emotions_model = EMOTIONS_MODEL
    # Detect face and predict the emotion
    infer_on_data(args, face_detection_model, emotions_model)


if __name__ == "__main__":
    main()