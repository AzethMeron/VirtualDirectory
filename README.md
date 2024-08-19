# VirtualDirectory - Seamless interface for huge datasets

Simple and universal library, designed to handle datasets stored in multiple subdirectories and with hundreds of thousands of files. Typically used for machine learning, but implementation is versatile enough to support any type of data. Provides seamless interface *(all files appear to be in the same directory)*, reduces number of system calls by loading list of files into memory and allows to *(but does NOT require)* load entire dataset into RAM for quick access.

## Installation

Designed and tested for ```Python 3.12.0```. You can install this package using PIP with following command.
```
pip install git+https://github.com/AzethMeron/VirtualDirectory.git
```

## Setup

```VirtualDirectory``` introduces simple and useful hierarchy to the dataset. Within main directory *(called ```root```)* there's a number of subdirectories of arbitrary names *(names does NOT matter)*. All regular files within ```root``` are ignored, so you can store your ```csv``` or ```json``` there. Everything inside each of of subdirectories is considered ```object``` *(a file)*. Simple file structure can be seen below. **NOTE: name of any file within subdirectories must be unique for entire dataset**, overlapping filenames will result in overwriting.

```
root/
├── subdir_0/
│   └── file_0
│   └── file_1
├── subdir_X/
│   └── file_2
│   └── file_3
├── dataset.csv (ignored)
```

To create instance of ```VirtualDirectory``` you can use following recipe.

```py
from VirtualDirectory import VirtualDirectory, PillowDataManager
vd = VirtualDirectory(
  root = "dataset", # main directory of the dataset (string)
  data_manager = PillowDataManager(), # instance of a class that is used to save, load and serialize datatype used in dataset
  verbose = True, # whether object should print progress as dataset is loading (True) or not (False). Defaults to True
  load_to_memory = True, # whether entire dataset should be loaded into RAM (True) or not (False). Defaults to False
  min_subdir_num=100 # minimal amount of subdirectories used to store data. Defaults to 100
)
```

**If ```root``` doesn't exist, it will be created automatically**. If there're not enough subdirectories inside *(less than ```min_subdir_num```)*, additional ones will be created aswell. 


The most imporatnt argument required to construct ```VirtualDirectory``` is the ```data_manager```. It must implement following methods:
```py
class DataManager:
  def save(self, path, object): # saves object into a file under given path
  def load(self, path): # loads object from a file under given path
  def serialize(self, object): # serializes object into a binary string. Should include compression
  def deserialize(self, binary_string): # reads object back from binary string
```
Example that works with pillow images can be seen below. Serialization is used when dataset loads all of the data into memory: by using ```PNG``` compression we can get much smaller size than by pickling the image and compressing it with ```zlib```.

```py
import io
from PIL import Image as PilImage

class PillowDataManager:
    def save(self, path, pil_image):
        pil_image.save(path)
    def load(self, path):
        return PilImage.open(path)
    def serialize(self, pil_image):
        buffer = io.BytesIO()
        pil_image.save(buffer, format = 'PNG')
        return buffer.getvalue()
    def deserialize(self, binary_string):
        buffer = io.BytesIO(binary_string)
        return PilImage.open(buffer)
```

Loading of dataset can take a lot of time, especially if there're hundreds of thousands of tiny images. If ```load_to_memory = True```, then all subdirectories are also saved to the drive in batched form *(files ```.vdd```)* that can be loaded much faster. 

## Usage 

Once ```VirtualDirectory``` is properly set up, it will work absolutely seamlessly. It provides very simple yet useful interface:
```py
vd = VirtualDirectory(...)
vd.save(filename, object) # saves object under given filename in randomly chosen subdirectory
vd.load(filename) # loads object from given filename
vd.exists(filename) # checks if object with given filename exists within dataset
vd.save_state() # Saves all data in RAM to .vdd files. By default it happens when instance is destroyed, you can sometimes call it manually to make sure no progress is lost
len(vd) # number of files in dataset
```
You can also iterrate over files in it, but then consider the data read-only:
```py
for filename, path, object in vd:
  # filename - string, just filename
  # path - string, entire path; root/subdir/filename
  # object - instance of data stored in given file (Only if dataset is fully loaded into RAM, otherwise None)
```
**If you ever add/remove data to dataset without using 
