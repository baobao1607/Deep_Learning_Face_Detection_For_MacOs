import os
import json
import random
import shutil
from typing import Dict, List


#this class split the raw datasource into different sets with stratified sampling
#it also make sure that the image contains the face before splitiing
#it also enable resplitting if updating image
#you dont have to split the whole dataset again
class DatasetSplitter:
    def __init__(
            self,
            raw_dir,
            dataset_dir,
            metadata_dir,
            train_ratio,
            valid_ratio,
            test_ratio,
    ):
        self.raw_dir = raw_dir
        self.dataset_dir = dataset_dir
        self.metadata_dir = metadata_dir

        #create the dataset for train, valid and test
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, "valid"), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_dir, "test"), exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        #get all cache json file 
        self.raw_index_file = os.path.join(self.metadata_dir, "raw_index.json")
        self.face_cache_file = os.path.join(self.metadata_dir, "face_check.json")
        self.split_index_file = os.path.join(self.metadata_dir, "split_index.json")
        self.split_done_file = os.path.join(self.metadata_dir, "split.done")
        
        #set the ratio for train, valid and test
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_state = 42

    #this function help load the data in the json file
    def load_json(self, path):
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
        
    #this function help save the data to json file
    def save_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent = 2)
    

    #this function load the relative path list from json
    def load_raw_index(self):
        data = self.load_json(self.raw_index_file)
        if data is None or "images" not in data:
            raise FileNotFoundError("missing JSON files")
        return data["images"]


    #this function load the dictionary with relative path + bool from face_check cache
    def load_face_cache(self):
        cache = self.load_json(self.face_cache_file)
        if isinstance(cache,dict):
            return cache
        return {}
    
    #this function load the dicionary with key is train, val, test, value is list of relativr path
    #we need to load this to see which image has been split or not
    def load_split_index(self):
        idx = self.load_json(self.split_index_file)
        if isinstance(idx, dict):
            return idx
        return {"train":[], "valid":[],"test":[]}
    
    #this function assign the group needed for stratification for 1 image
    def strat_group(self, relative_path):
        #split the relative path based on /
        parts = relative_path.split(os.sep)

        if parts[0] == "ffhq":
            return "real"
        
        if parts[0] == "fake":
            gen = parts[1].lower()
            return f"fake_{gen}"
        
        return "unknown"
    
    #this function load the image from the datasource and split it based stratification key
    def split_datasource(self):
        #set the random seed state for reproductive random
        rng = random.Random(self.random_state)
        #get all image relative path from the json file
        all_images = self.load_raw_index()
        #get the dictionary for relative path + bool
        face_cache = self.load_face_cache()
        #get the split index dictionary to compare later
        split_index = self.load_split_index()
        #guard: stop if any image appears in more than one split
        collisions = self.find_split_collisions()
        if any(v > 0 for v in collisions["counts"].values()):
            raise ValueError(f"split collisions detected: {collisions['counts']}")

        #get all the relative path from the index dictionary 
        already_split = set(split_index["train"]) | set(split_index["valid"]) | set(split_index["test"])

        #if they are not in already split or is not face then they are new images 
        new_images = [
            img for img in all_images
            if img not in already_split and face_cache.get(img, False)
        ]

        #if there are no images
        if len(new_images) == 0:
            self._touch_done()
            return self._summary(
                status="No image",
                new_images_count=0,
                total_images_count=len(all_images),
                split_index=split_index
            )


        #create a dictionary to hold group based on detectors -> variable is the relative path
        groups = {}
        #get all the relative path in new images
        for img in new_images:
            #get the key for the images
            g = self.strat_group(img)
            #append to the group didtionary
            groups.setdefault(g, []).append(img)

        #for the key, image in dictionary
        for g, imgs in groups.items():
            #shuffle the images in one key
            rng.shuffle(imgs)

            #get the len of images of this key
            n = len(imgs)
            #get the len for train
            n_train = int(n * self.train_ratio)
            #get the len for valid
            n_valid = int(n * self.valid_ratio)

            #assign the training images list 
            train_imgs = imgs[:n_train]
            #assing the valid images list 
            valid_imgs = imgs[n_train:n_train+n_valid]
            #assing the test images list 
            test_imgs  = imgs[n_train+n_valid:]

            #this inner function use to label of the image
            def label(img):
                #get the folder in the relative path
                folder = img.split(os.sep)[0].lower()
                #if folder is ffhq return 1 else 0
                return 0 if folder == "ffhq" else 1
            
            
            #get the the train labels for train, test, valid 
            #build label for all of them
            train_labels = [label(i) for i in train_imgs]
            valid_labels = [label(i) for i in valid_imgs]
            test_labels  = [label(i) for i in test_imgs]
            
            #update the split_index cache
            #so that later on we do not need to recompute it if there is new data coming in
            split_index["train"].extend(train_imgs)
            split_index["valid"].extend(valid_imgs)
            split_index["test"].extend(test_imgs)

            #copy and save the set into the right folder
            self._copy_split_set(train_imgs, train_labels, "train")
            self._copy_split_set(valid_imgs, valid_labels, "valid")
            self._copy_split_set(test_imgs, test_labels, "test")

        #save the split_index to the json file
        self.save_json(self.split_index_file, split_index)
        #mark for dont
        self._touch_done()
        
        #return summary stats
        return self._summary(
            status="Splitting done",
            new_images_count=len(new_images),
            total_images_count=len(all_images),
            split_index=split_index
        )

    #this function return the summary stat for the splitter
    #it includes status, new images being processed
    #total images counted
    #len of split_test index json file
    def _summary(self, status, new_images_count, total_images_count, split_index):
        collisions = self.find_split_collisions()
        return {
            "status": status,
            "new_split_count": new_images_count,
            "total_image": total_images_count,
            "train": len(split_index["train"]),
            "valid": len(split_index["valid"]),
            "test": len(split_index["test"]),
            "collision_counts": collisions["counts"],
        }
    
    #this function copy an images to a folder split_name, create a subdirectories based on its label and put the images in there
    def _copy_split_set(self, images, labels, split_name):
        #iterate htrough the images and labels in the same index style
        for img, label in zip(images, labels):
            #create the label folder 0 or 1
            label_folder = str(label)

            #get the source path of that image
            source_path = os.path.join(self.raw_dir, img)
            #cretate the dataset subdirectories for that immages
            dst_folder = os.path.join(self.dataset_dir, split_name,label_folder)
            #create the dataset if not exists
            os.makedirs(dst_folder, exist_ok=True)

            #build the full path
            dst = os.path.join(dst_folder, os.path.basename(img))

            #if not exists, copy the image from source path to the full path
            if not os.path.exists(dst):
                shutil.copy(source_path, dst)
    
    #this function makr for done 
    def _touch_done(self):
        with open(self.split_done_file, "w") as f:
            f.write("done\n")

    def report_distribution(self):

        split_index = self.load_split_index()
        splits = ["train", "valid", "test"]

        def get_gen(path):
            parts = path.split(os.sep)
            if parts[0] == "ffhq":
                return "real"
            if parts[0] == "fake":
                return f"fake_{parts[1].lower()}"
            return "unknown"


        # Accumulate counts: { split: { generator: count } }
        gen_counts = {s: {} for s in splits}

        for split in splits:
            for img in split_index[split]:
                gen = get_gen(img)
                gen_counts[split].setdefault(gen, 0)
                gen_counts[split][gen] += 1

        # Print table
        for split in splits:
            print(f"\n--- {split.upper()} ---")
            total_split = sum(gen_counts[split].values())
            for gen, count in sorted(gen_counts[split].items()):
                ratio = (count / total_split * 100) if total_split > 0 else 0
                print(f"{gen:28s} : {count:6d}  ({ratio:5.2f}%)")
            print(f"TOTAL {split}: {total_split}")

        print(f"\n")

        def is_real(path):
            return path.startswith("ffhq" + os.sep)

        for split in splits:
            real_count = sum(1 for img in split_index[split] if is_real(img))
            fake_count = len(split_index[split]) - real_count
            total = real_count + fake_count

            real_ratio = real_count / total * 100 if total > 0 else 0
            fake_ratio = fake_count / total * 100 if total > 0 else 0

            print(f"{split.upper():6} | real={real_count:6d} ({real_ratio:5.2f}%)   "
                f"fake={fake_count:6d} ({fake_ratio:5.2f}%)   total={total}")

    #this function check if any image path appears in more than one split
    #it returns a dict with overlap lists and counts
    def find_split_collisions(self):
        split_index = self.load_split_index()

        train_set = set(split_index.get("train", []))
        valid_set = set(split_index.get("valid", []))
        test_set = set(split_index.get("test", []))

        train_valid = sorted(train_set & valid_set)
        train_test = sorted(train_set & test_set)
        valid_test = sorted(valid_set & test_set)
        all_three = sorted(train_set & valid_set & test_set)

        return {
            "train_valid": train_valid,
            "train_test": train_test,
            "valid_test": valid_test,
            "all_three": all_three,
            "counts": {
                "train_valid": len(train_valid),
                "train_test": len(train_test),
                "valid_test": len(valid_test),
                "all_three": len(all_three),
            }
        }
