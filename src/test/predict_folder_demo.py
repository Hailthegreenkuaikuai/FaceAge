# -----------------
# Estimate FaceAge for all the subjects in a given folder
# (this script will parse the configuration file "config_predict_folder_demo.yaml")
# -----------------

# The code and data of this repository are intended to promote transparent and reproducible research
# of the paper "Decoding biological age from face photographs using deep learning"

# All the details about the project can be found at the following webpage:
# aim.hms.harvard.edu/FaceAge

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# AIM 2022

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import sys
import time
import yaml
import argparse

import PIL
import mtcnn
import keras
import numpy as np
import pandas as pd
import tensorflow as tf

# suppress warnings/errors due to migration from TensorFlow 1.x to 2.x
# ponytail: removed disable_eager_execution — breaks MTCNN on TF 2.x
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from skimage.io import imsave, imread

print("Python version     : ", sys.version.split('\n')[0])
print("TensorFlow version : ", tf.__version__)
print("Keras version      : ", keras.__version__)
print("Numpy version      : ", np.__version__)
print("")

## ----------------------------------------

def get_face_bbox_from_image(path_to_image):
  
  """
  Use the MTCNN face detector to localise the subject's face withing the image.
  Returns the coordinates to draw bounding box enclosing the face,
  the keypoints coordinates, and the confidence associated with the prediction.
  
  Make sure the image contains only one subject for the pipeline to work as intended.

  @params:
    path_to_image - required: absolute path to the image file to be processed.
     
   """

  # sanity check
  assert os.path.exists(path_to_image)

  pat_img = imread(path_to_image)
  
  try:
    # return the MTCNN output associated with the first face found in the image
    # make sure the image contains only one subject for the pipeline to work as intended
    return mtcnn.mtcnn.MTCNN().detect_faces(pat_img)[0]
  except:
    print('ERROR: Processing error for file "%s"'%(path_to_image))
    return dict()

## ----------------------------------------

def preprocess_face(path_to_image, mtcnn_output_dict):

  """
  Preprocess a single face image for model input.
  Returns a (160, 160, 3) normalized numpy array.
  """

  # sanity check
  assert os.path.exists(path_to_image)

  pat_img = imread(path_to_image)

  # extract the bounding box from the first face
  x1, y1, width, height = mtcnn_output_dict['box']
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height

  # crop the face
  pat_face = pat_img[y1:y2, x1:x2]

  # resize cropped image to the model input size
  pat_face_pil = PIL.Image.fromarray(np.uint8(pat_face)).convert('RGB')
  pat_face = np.asarray(pat_face_pil.resize((160, 160)))

  # prep image for TF processing
  mean, std = pat_face.mean(), pat_face.std()
  pat_face = (pat_face - mean) / std
  return pat_face

## ----------------------------------------

def get_model_prediction(model, path_to_image, mtcnn_output_dict):
  """Single-image prediction (kept for backward compat)."""
  pat_face = preprocess_face(path_to_image, mtcnn_output_dict)
  pat_face_input = pat_face.reshape(1, 160, 160, 3)
  return np.squeeze(model.predict(pat_face_input, verbose=0))

## ----------------------------------------

def get_model_prediction_batch(model, face_inputs):
  """Batch prediction for a list of (160, 160, 3) images."""
  if not face_inputs:
    return np.array([])
  batch = np.stack(face_inputs, axis=0)  # shape: (N, 160, 160, 3)
  return np.squeeze(model.predict(batch, verbose=0))

## ----------------------------------------
## ----------------------------------------

def main(config):

  model_name = config["model_name"]
  base_model_path = config["base_model_path"]

  base_output_path = config["base_output_path"]

  input_folder_name = config["input_folder_name"]
  input_folder_path = config["input_folder_path"]

  input_file_list = [f for f in os.listdir(input_folder_path) if ".png" in f or ".jpg" in f]

  print("Predicting FaceAge for %g subjects at: '%s'\n"%(len(input_file_list),
                                                         input_folder_path))


  face_bbox_dict = dict()

  # FIXME: DEBUG
  # limit the number of subjects for a faster execution
  # if set to -1, run on all the hi-res UTK data (provided)
  N_SUBJECTS = -1

  # subset the file list to speed up the execution of the whole notebook
  input_file_list = input_file_list[:N_SUBJECTS] if N_SUBJECTS > 0 else input_file_list

  t = time.time()

  for idx, input_image in enumerate(input_file_list):

    subj_id = input_image.split(".")[0]

    print('(%g/%g) Running the face localization step for "%s"'%(idx + 1,
                                                                len(input_file_list),
                                                                input_image),
    end = "\r")

    path_to_image = os.path.join(input_folder_path, input_image)
    
    face_bbox_dict[subj_id] = dict()
    
    face_bbox_dict[subj_id]["path_to_image"] = path_to_image

    face_bbox_dict[subj_id]["mtcnn_output_dict"] = get_face_bbox_from_image(path_to_image)

    # solves known TF memory leaks for the MTCNN pipeline
    # (should work with all the recent versions of tensorflow)
    if not idx % 5:
      tf.keras.backend.clear_session()
      gc.collect()


  elapsed = time.time() - t
  print("\n... Done in %g seconds."%(elapsed))

  # ------------------------

  model_path = os.path.join(base_model_path, model_name)
  model = keras.models.load_model(model_path, safe_mode = False)
  print(model)
  print("")

  age_pred_dict = dict()

  t = time.time()

  # Preprocess all faces
  subj_ids = list(face_bbox_dict.keys())
  face_batch = []
  for idx, subj_id in enumerate(subj_ids):
    print('(%g/%g) Preprocessing face for "%s"'%(idx + 1, len(subj_ids), subj_id),
    end = "\r")
    path_to_image = face_bbox_dict[subj_id]["path_to_image"]
    mtcnn_output_dict = face_bbox_dict[subj_id]["mtcnn_output_dict"]
    face_batch.append(preprocess_face(path_to_image, mtcnn_output_dict))

  # Batch predict all at once
  print("\nRunning batch prediction on %d faces..." % len(face_batch))
  predictions = get_model_prediction_batch(model, face_batch)

  # Distribute results
  for subj_id, faceage in zip(subj_ids, predictions):
    age_pred_dict[subj_id] = {"faceage": float(faceage)}

  elapsed = time.time() - t
  print("... Done in %g seconds."%(elapsed))

  age_pred_df = pd.DataFrame.from_dict(age_pred_dict, orient = 'index')
  age_pred_df.reset_index(level = 0, inplace = True)
  age_pred_df.rename(columns = {"index": "subj_id"}, inplace = True)

  outfile_name = '%s_res.csv'%(input_folder_name)
  outfile_path = os.path.join(base_output_path, outfile_name) 

  print("\nSaving predictions at: '%s'... "%(outfile_path), end = "")

  age_pred_df.to_csv(outfile_path, index = False)

  print("Done.")


## ----------------------------------------
## ----------------------------------------
      
if __name__ == '__main__':

  base_conf_file_path = '.'
  
  parser = argparse.ArgumentParser(description = 'FaceAge - predict folder demo')

  parser.add_argument('--conf',
                      required = False,
                      help = 'Specify the path to the YAML configuration file containing the run details.',
                      default = "config_predict_folder_demo.yaml"
                     )

  args = parser.parse_args()

  conf_file_path = os.path.join(base_conf_file_path, args.conf)

  with open(conf_file_path) as f:
    yaml_conf = yaml.load(f, Loader = yaml.FullLoader)

  # base data directory
  base_path = yaml_conf["test"]["base_path"]

  data_folder_name = yaml_conf["test"]["data_folder_name"]

  model_name = yaml_conf["test"]["model_name"]
  models_folder_name = yaml_conf["test"]["models_folder_name"]

  input_folder_name = yaml_conf["test"]["input_folder_name"]
  outputs_folder_name = yaml_conf["test"]["outputs_folder_name"]

  base_data_path = os.path.join(base_path, data_folder_name)
  base_model_path = os.path.join(base_path, models_folder_name)
  base_output_path = os.path.join(base_path, outputs_folder_name)

  input_folder_path = os.path.join(base_data_path, input_folder_name)

  ## ----------------------------------------
  
  # dictionary to be passed to the main function
  config = dict()
  
  config["base_model_path"] = base_model_path
  # Prefer .keras format (Python 3.13 compat) over legacy .h5
  keras_path = os.path.join(base_model_path, model_name + ".keras")
  h5_path = os.path.join(base_model_path, model_name + ".h5")
  if os.path.exists(keras_path):
      config["model_name"] = model_name + ".keras"
  elif os.path.exists(h5_path):
      config["model_name"] = model_name + ".h5"
  else:
      config["model_name"] = model_name + ".h5"

  config["base_output_path"] = base_output_path
  
  config["input_folder_name"] = input_folder_name
  config["input_folder_path"] = input_folder_path
  
  main(config)
