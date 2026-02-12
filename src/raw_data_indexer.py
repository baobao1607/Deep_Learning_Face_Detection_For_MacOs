import os
import json
import hashlib

#this function flag if the file is a hidden file create by MACOS system
def is_hidden(file_path):
    return os.path.basename(file_path).startswith(".")

#this class is used to scan the raw datasets
#if it detect any changes, then there is a update in our raw dataset
class RawDataIndexer:
    #this function is a constructor for our rawdataindexer class
    def __init__(self, raw_dir, metadata_dir):
        #assign directory of datasource
        self.raw_dir = raw_dir
        #assign directory of metadata
        self.metadata_dir = metadata_dir

        #create the metadata_dir if not exists
        os.makedirs(self.metadata_dir, exist_ok=True)
        #assign raw_index JSON to keep metadata about raw image
        self.index_file = os.path.join(self.metadata_dir, "raw_index.json")
        #assing raw_hash to check for data integrity
        #if we later add more image into the datasource, we can check and update our dataset
        self.hash_file = os.path.join(self.metadata_dir, "raw_hash.txt")


    #this function scan all the image path in the datasouce folder
    def scan_raw(self):
        #a list to hold all the image paths
        image_path = []

        #walk through every file, directory from the raw datasouce folrder
        for root, dirs, files in os.walk(self.raw_dir):
            #run through the files
            for file in files:
                #if it is not a hidden file, good
                if not is_hidden(file):
                    #if the extension is an image file, good
                    if file.lower().endswith((".jpg",".jped",".png")):
                        #build the full path
                        full_path = os.path.join(root, file)
                        #append the relative path to the image path
                        #full_path = image_path + raw_dir
                        image_path.append(os.path.relpath(full_path, self.raw_dir))
        
        #sort the image path alphabetically
        image_path.sort()
        #return the image path
        return image_path

    #this function compute the hash function for all image path
    #we do this so that if a new datasource appears, we can do sanity chech
    def compute_hash(self, image_paths):
        #create a hash object
        hash_md5 = hashlib.md5()

        #go through all the relative file path
        for path in image_paths:
            #compute the full_path
            full_path = os.path.join(self.raw_dir, path)

            #get the stat of the file: file name, size, etc
            stat = os.stat(full_path)
            #compute the hash input includes relative path + size of the image
            file_info = f"{path}:{stat.st_size}"
            #encode the file_info into binary
            #then put it through the hash function
            hash_md5.update(file_info.encode("utf-8"))
        
        #return the hash code
        return hash_md5.hexdigest()
    

    #this function load the previous hash if it exists 
    def load_previous_hash(self):
        #if this hash file does not exists
        if not os.path.exists(self.hash_file):
            #return nothing
            return None
        #else open the hash file with read mode
        with open(self.hash_file, "r") as f:
            #return the hash code
            #strip to remove new line
            return f.read().strip()
    
    #this function write the new hash into the hash file
    def save_hash(self, hash_code):
        #open the hash file with write model
        with open(self.hash_file, "w") as f:
            #write the hash code into the file 
            f.write(hash_code)


    #this function save the raw index JSON file
    def save_index(self, image_paths):
        #open the json file in write mode
        with open(self.index_file, "w") as f:
            #write into the json file with the format
            #image is the key, variable is the whole image paths (relative_path)
            json.dump({"images": image_paths}, f, indent=4)

    #this function count how many ffhq and fake images per generator
    def count_data_sources(self, image_paths):
        summary = {
            "total": len(image_paths),
            "ffhq": 0,
            "fake_total": 0,
            "fake_generators": {},
            "unknown": 0,
        }

        for relative_path in image_paths:
            #normalize separators to keep parsing stable
            normalized = relative_path.replace("\\", "/")
            parts = normalized.split("/")

            if len(parts) == 0:
                summary["unknown"] += 1
                continue

            root = parts[0].lower()

            if root == "ffhq":
                summary["ffhq"] += 1
            elif root == "fake":
                summary["fake_total"] += 1
                generator = parts[1].lower() if len(parts) > 1 and parts[1] else "unknown_generator"
                summary["fake_generators"][generator] = summary["fake_generators"].get(generator, 0) + 1
            else:
                summary["unknown"] += 1

        summary["fake_generators"] = dict(sorted(summary["fake_generators"].items()))
        return summary

    #this function scan the raw dataset and see if there is any changes
    def scan_and_detect(self):
        print("Scanning in progress")

        #get the image path list of the datasource
        image_path = self.scan_raw()
        #compute the new hash
        new_hash = self.compute_hash(image_path)
        #load the previous hash
        old_hash = self.load_previous_hash()

        #if there are changed
        if (new_hash != old_hash):
            print("Hash is changed, datasource is dirty")
            #save new hash
            self.save_hash(new_hash)
            #save new JSON
            self.save_index(image_path)
        else:
            print("Hash is not changed. Data is integrity")
        
        #return 
        return {
            "changed": (new_hash != old_hash),
            "len_image_paths": len(image_path),
            "hash": new_hash,
            "source_counts": self.count_data_sources(image_path),
        }
