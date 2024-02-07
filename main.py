import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*synchronize.*")

import cv2

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the neural network
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

import easygui

# Function to open file dialog and get the selected file path
def get_image_path():
    image_path = easygui.fileopenbox(msg="Select an Image", filetypes=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"])
    return image_path

# Get the image path from the user
image_path = get_image_path()

# Check if a valid image path is selected
if image_path:
    # Read the image using OpenCV
    img = cv2.imread(image_path)

    # Display the image using OpenCV
    cv2.imshow("Selected Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Detect objects in the image
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    print(classIds, bbox)

    # Draw bounding boxes and labels
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output image
    cv2.imshow("Output", img)
    cv2.waitKey(0)
else:
    print("No image selected.")
