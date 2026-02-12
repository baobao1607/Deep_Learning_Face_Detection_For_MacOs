import os
import json
import cv2 
import numpy as np
from multiprocessing import get_context, cpu_count
import time
from typing import Dict, List, Tuple, Optional

#these are global defintion for workers since we are using multiprocessing


#hold the model for each CPU worker
_DNN_NET = None
#threshhold to detect face
_DNN_CONF_THRESH = 0.5
#OpenCV required input size
_DNN_INPUT_SIZE = (300, 300)


#this function intialize the variables for each worker in pool of multiprocessing
def _init_worker(prototxt_path: str, model_path: str, conf_thresh:float):
    #declaring global variables inside this worker
    global _DNN_NET, _DNN_CONF_THRESH
    #set the threshold
    _DNN_CONF_THRESH = conf_thresh
    #load the model for each worker
    _DNN_NET = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

#this function detect if the image has a face in it or not
#it take in a image path and return a tuple contains a image path string and true/false
def _has_face_dnn(full_image_path:str) -> Tuple[str, bool]:

    #call the variable for each worker
    global _DNN_NET, _DNN_CONF_THRESH, _DNN_INPUT_SIZE

    try:
        #read the image from the image path
        img = cv2.imread(full_image_path)
        #if there is no image available
        if img is None:
            #return false (consider no face)
            return full_image_path, False

        #get the wdith and height of the image
        (h, w) = img.shape[:2]

        #initialize the blob, the blob is a tensorflow preprocessing input
        blob = cv2.dnn.blobFromImage(
            img,
            #no need for normalizing RGB since we minus the mean
            scalefactor = 1.0,
            #resizing the image to 300x300
            size = _DNN_INPUT_SIZE,
            #subtract the mean values 
            mean = (104, 177, 123),
            #no need to swap Red and Blue
            swapRB=False,
            #do not resizing
            crop = False
        )

        #set the input for the model
        _DNN_NET.setInput(blob)
        #feedforwarding to get detection result
        detections = _DNN_NET.forward()
        
        #if detection does not return anything or number of detected faces is 0, return false
        if detections is None or detections.shape[2] == 0:
            return full_image_path, False

        #go through all number of faces detected
        for i in range(detections.shape[2]):
            #get the confidence
            conf = float(detections[0,0,i,2])
            #if the confidence score is greater than the threshold
            if conf >= _DNN_CONF_THRESH:
                #return True
                return full_image_path, True
        
        #if no pass , return False
        return full_image_path, False
            
    #if image file is corrupted, return False
    except Exception:
        return full_image_path, False

#this class reads the raw_index.json file from metadata
#runs face detection in faces that does not in face_check.json
#updates if there are new faces
#return summaries with no-face list
class FaceExplorer:
    def __init__(
            self,
            #datasource dir
            raw_dir,
            #metadata dir
            metadata_dir,
            #the skeleteon for the model (layers, parameters)
            prototxt_path = "./models/deploy.prototxt",
            #weight for the model
            model_path = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel",
            #threshold
            conf_thresh = 0.5,
            #number of processor (CPU)
            processes = None,
            #multiprocessing start method for MACOS
            mp_start_method = "spawn",
            #progress log every N images
            progress_every = 500
    ):
        self.raw_dir = raw_dir
        self.metadata_dir = metadata_dir
        os.makedirs(self.metadata_dir, exist_ok= True)
        self.raw_index_file = os.path.join(self.metadata_dir, "raw_index.json")
        self.face_cache_file = os.path.join(self.metadata_dir,"face_check.json")
        self.explore_done_file = os.path.join(self.metadata_dir,"explore.done")

        self.protoxtx_path = prototxt_path
        self.model_path = model_path
        self.conf_thresh = conf_thresh

        self.processes = processes or max (1, min(cpu_count(),8))
        self.mp_start_method = mp_start_method
        self.progress_every = max(1, int(progress_every))

        #function to validate model weights and skeletons
        self.validate_model_files()
    
    #this function check if the model skeleton and weights file exists or not
    def validate_model_files(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing model weight file")
        if not os.path.exists(self.protoxtx_path):
            raise FileNotFoundError(f"Missing model skeleton files")


    #this function load the image from the json file:
    def load_raw_index(self):
        #check if the json file exists or not
        if not os.path.exists(self.raw_index_file):
            raise FileNotFoundError(f"Missing JSON file")
        
        #open the json file as read mode
        with open(self.raw_index_file,"r") as f:
            #load the data dictionary from json file
            data = json.load(f)
        
        #if images list is not in data (key is not in data)
        if "images" not in data:
            raise ValueError(f"JSON file missing images list")

        #return the list of variables
        return data["images"]
    
    #this function load the face cache form the face cache json file
    #we need this to not check already-checked faces
    #the return is a dictionary with str is relative path, bool is true or false
    def load_face_cache(self):
        #if the file does not exists, returm nothing
        if not os.path.exists(self.face_cache_file):
            return {}
        #open the face-cache file with reaf mode
        with open(self.face_cache_file, "r") as f:
            #load the data from json
            data = json.load(f)
        #if it is not a dictionary, return none
        if not isinstance(data, dict):
            return {}
        #return the dictionary
        return {k: bool(v) for k, v in data.items()}
    
    #this function update the face cache whenever we detect new updated faces
    def save_face_cache(self, cache):
        with open(self.face_cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    #this function runs face detection on faces that is not in face_check json
    #compile multi processing and DNN provided from OpenCV
    #return summary 
    def explore_faces(self):
        #get the relative path from load_raw_index
        relative_paths = self.load_raw_index()
        #load the cache
        cache = self.load_face_cache()

        #initialize a list to hold unchecked path
        unchecked_path = []
        #if path is not in cache, append to unchecked paths
        for path in relative_paths:
            if path not in cache:
                unchecked_path.append(path)

        #get the already checked lengths
        checked_len = len(relative_paths) - len(unchecked_path)

        #if all is already checked, return summary
        if len(unchecked_path) == 0:
            self._touch_done()
            return self._summary(relative_paths, cache, newly_checked = 0,already_checked = checked_len)
        
        #get the full path by append with datasouce path
        unchecked_full_path = [os.path.join(self.raw_dir, p) for p in unchecked_path]

        #initalize an object to control multiprocesing
        ctx = get_context(self.mp_start_method)
        start_time = time.time()
        processed = 0
        #use multiprocessing
        with ctx.Pool(
            #number of CPU worker
            processes = self.processes,
            #initalize each woker
            initializer= _init_worker,
            #initalize worker parameters
            initargs = (self.protoxtx_path, self.model_path, self.conf_thresh),
        ) as pool:
            #send 64 images in a chunk to workers, workers return full path + bool from dnn
            for abs_path, has_face in pool.imap_unordered(_has_face_dnn, unchecked_full_path, chunksize=64):
                #get the relative path
                relative_path = os.path.relpath(abs_path, self.raw_dir)
                #build dictionary
                cache[relative_path] = bool(has_face)
                processed += 1
                if processed % self.progress_every == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = len(unchecked_full_path) - processed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"FaceExplorer progress: {processed}/{len(unchecked_full_path)} "
                          f"({rate:.2f} img/s, ETA {eta:.1f}s)")
        
        #save the cache into json
        self.save_face_cache(cache)
        #mark for done
        self._touch_done()
        #return summary
        return self._summary(relative_paths, cache, newly_checked = len(unchecked_path), already_checked = checked_len)
    
    #this function mark if the current dataset is checked or not
    def _touch_done(self):
        with open(self.explore_done_file, "w") as f:
            f.write("done\n")

    #this function return the summary for a checked over the whole dataset
    def _summary(self, rel_paths, cache, newly_checked, already_checked):
        total = len(rel_paths)
        no_face = [p for p in rel_paths if cache.get(p, False) is False]
        has_face = total-len(no_face)

        return {
            "total_images": total,
            "has_face": has_face,
            "no_face": len(no_face),
            "newly_checked": newly_checked,
            "already_checked": already_checked,
            "no_face_paths_preview": no_face[:25],
        }







        
