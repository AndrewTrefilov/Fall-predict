import sys 
import argparse
import cv2
import time
import json
import numpy as np
import os


sys.path.append('/projects/openpose/build/python')

from openpose import pyopenpose as op

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="img/fall/2_crop2040-0.png", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
parser.add_argument("--image_dir", default="img/before", help="Process a directory of images.")
parser.add_argument("--no_display", default=True, help="Enable to disable the visual display.")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "/projects/openpose/models/"

 # Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Read frames on directory
imagePaths = op.get_images_on_directory(args[0].image_dir);
start = time.time()

# Process and display images
data = {}
i = 0
for imagePath in imagePaths:
    t_0 = time.time()
    name = imagePath.split('/')[-1]
    datum = op.Datum()
    imageToProcess = cv2.imread(imagePath)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    # print("Body keypoints: \n" + str(datum.poseKeypoints[0]))
    data[name] = datum.poseKeypoints
    t_1 = time.time()
    print(f"{i} Time detect:", t_1 - t_0)
    i += 1
    if not args[0].no_display:
        cv2.imshow("OpenPose 1.6.0 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(15)
        if key == 27: break

if not os.path.exists('results/'):
    os.makedirs('results/', exist_ok=True)

state = args[0].image_dir.split('/')[-1]
with open(f'results/data_{state}.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)


end = time.time()
print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")