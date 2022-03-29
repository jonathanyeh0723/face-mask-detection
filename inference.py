import tensorflow as tf
model = tf.keras.models.load_model('face_mask_model.h5')
tf.saved_model.save(model,'ov_model_mask')

"""
(ov_2022_dev) C:\Users\Jonathan>mo --saved_model_dir ov_model_mask --input_shape [1,150,150,3]

#### Logs #####
Model Optimizer arguments:
Common parameters:
        - Path to the Input Model:      None
        - Path for generated IR:        C:\Users\Jonathan\.
        - IR output name:       saved_model
        - Log level:    ERROR
        - Batch:        Not specified, inherited from the model
        - Input layers:         Not specified, inherited from the model
        - Output layers:        Not specified, inherited from the model
        - Input shapes:         [1,150,150,3]
        - Source layout:        Not specified
        - Target layout:        Not specified
        - Layout:       Not specified
        - Mean values:  Not specified
        - Scale values:         Not specified
        - Scale factor:         Not specified
        - Precision of IR:      FP32
        - Enable fusing:        True
        - User transformations:         Not specified
        - Reverse input channels:       False
        - Enable IR generation for fixed input shape:   False
        - Use the transformations config file:  None
Advanced parameters:
        - Force the usage of legacy Frontend of Model Optimizer for model conversion into IR:   False
        - Force the usage of new Frontend of Model Optimizer for model conversion into IR:      False
TensorFlow specific parameters:
        - Input model in text protobuf format:  False
        - Path to model dump for TensorBoard:   None
        - List of shared libraries with TensorFlow custom layers implementation:        None
        - Update the configuration file with input/output node names:   None
        - Use configuration file used to generate the model with Object Detection API:  None
        - Use the config file:  None
OpenVINO runtime found in:      C:\Users\Jonathan\anaconda3\envs\ov_2022_dev\lib\site-packages\openvino
OpenVINO runtime version:       2022.1.0-7019-cdb9bec7210-releases/2022/1
Model Optimizer version:        2022.1.0-7019-cdb9bec7210-releases/2022/1
[ SUCCESS ] Generated IR version 11 model.
[ SUCCESS ] XML file: C:\Users\Jonathan\saved_model.xml
[ SUCCESS ] BIN file: C:\Users\Jonathan\saved_model.bin
[ SUCCESS ] Total execution time: 17.03 seconds.
[ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
Find more information about API v2.0 and IR v11 at https://docs.openvino.ai

(ov_2022_dev) C:\Users\Jonathan>
"""

import os
import cv2
import numpy as np
from random import choice

from openvino.preprocess import PrePostProcessor, ResizeAlgorithm
from openvino.runtime import Core, Layout, Type

from tensorflow.keras.preprocessing import image as I
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

core = Core()
model = core.read_model('C:/Users/Jonathan/saved_model.xml')

#image = cv2.imread('face_mask_test/nomask_3.jpg')

test_img_path = "face_mask_test"
test_images = os.listdir(test_img_path)

picked_img = choice(test_images)
img = I.load_img(test_img_path+'/'+picked_img, target_size=(150, 150))

#print(test_img_path)
#print(show_img_path)
show_img_path = test_img_path + '/' + picked_img
print(show_img_path)

show_img = mpimg.imread(show_img_path)

images = I.img_to_array(img)
images = np.expand_dims(images, axis=0)

compiled_model = core.compile_model(model, 'CPU')
results = compiled_model.infer_new_request({0: images})
#print(results)
#print(type(results))
#print(results.values())
inference = list(results.values())[0]

print(inference)

text0 = "Mask. We're safe...."
text1 = "No Mask! Be careful~~~~~"

if inference == 0:
    text_info = text0
    print(text_info)
else:
    text_info = text1
    print(text_info)
    
plt.axis('Off')
plt.imshow(show_img)


# load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# read image
img = cv2.imread(show_img_path)

# convert color image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face detection
faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(32, 32))

# draw the bounding box around the face
for (x, y, w, h) in faces:
    
    if inference == 0:
        color = (0, 255, 0)
        line_w = 2
    else:
        color = (0, 0, 255)
        line_w = 5
        
    cv2.rectangle(img, (x, y), (x + w, y + h), color, line_w)
    cv2.putText(img, text_info, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        
# show image detected
#cv2.namedWindow('img', cv2.WINDOW_NORMAL)  #a djustment
cv2.imshow('img', img)                     # show the image
#cv2.imwrite( "result.jpg", img )           # save the image
cv2.imwrite('input/savedImage.jpg', img)
cv2.waitKey(0)                             # press any key to wait
cv2.destroyAllWindows()                    # close
