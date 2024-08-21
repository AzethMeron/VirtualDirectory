# VirtualDirectory - Seamless interface for huge datasets

Simple and universal library, designed to handle datasets stored in multiple subdirectories and with hundreds of thousands of files. Typically used for machine learning, but implementation is versatile enough to support any type of data. Provides seamless interface *(all files appear to be in the same directory)*, reduces number of system calls by loading list of files into memory and allows to *(but does NOT require)* load entire dataset into RAM for quick access.

---

# Table of Contents
1. [Installation](#installation)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Advanced usage](#advusage)
5. [Miscellaneous](#misc)
6. [Free-for-all scripts](#ffa)

---

## Installation <a name="installation"></a>

Designed and tested for ```Python 3.12.0```. You can install this package using PIP with following command.
```
pip install git+https://github.com/AzethMeron/VirtualDirectory.git
```

## Setup <a name="setup"></a>

**Check ```mnist_example/``` to see the setup in practice. It's simpler than it appears when described.**

```VirtualDirectory``` introduces simple and useful hierarchy to the dataset. Within main directory *(called ```root```)* there's a number of subdirectories of arbitrary names *(names does NOT matter)*. All regular files within ```root``` are ignored, so you can store your ```csv``` or ```json``` there. Everything inside each of of subdirectories is considered ```object``` *(a file)*. Simple file structure can be seen below. **NOTE: name of any file must be unique for entire dataset**, even if those are stored in different subdirectory. Overlapping filenames will result in overwriting.

```
root/
├── subdir_0/
│   └── file_0
│   └── file_1
├── subdir_X/
│   └── file_2
│   └── file_3
├── dataset.csv (ignored)
├── subdir_0.vdd
├── subdir_X.vdd
└── .anything (ignored)
```

**All files and directories with leading ```.``` are invisible to ```VirtualDirectory```.**

To create instance of ```VirtualDirectory``` you can use following recipe.

```py
from VirtualDirectory import VirtualDirectory, PillowDataManager
vd = VirtualDirectory(
  root = "dataset", # main directory of the dataset (string)
  data_manager = PillowDataManager(), # instance of a class that is used to save, load and serialize datatype used in dataset
  verbose = True, # whether object should print progress as dataset is loading (True) or not (False). Defaults to True
  load_to_memory = True, # whether entire dataset should be loaded into RAM (True) or not (False). Defaults to True
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
Example that works with pillow images can be seen below. Serialization is used when dataset loads all of the data into memory: using ```PNG``` format we can achieve best lossless compression for images, significantly better than by pickling Pillow image and compressing it with ```zlib```. 

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

## Usage <a name="usage"></a>

Once ```VirtualDirectory``` is properly set up, it will work absolutely seamlessly. It provides very simple yet useful interface:
```py
vd = VirtualDirectory(...)
vd.save(filename, object) # saves object under given filename in randomly chosen subdirectory
vd.load(filename) # loads object from given filename. Returns None if there's no such file
vd.exists(filename) # checks if object with given filename exists within dataset (True/False)
vd.remove(filename) # removes file from dataset. If there's no such file, it's silently ignored
vd.get_path(filename) # Get full path to given file. Note that editing it outside of VirtualDirectory interface may make it go out of sync. Returns None, if file doesn't exist
vd.get_list_of_files() # Get list of filenames in VirtualDirectory
vd.save_state() # Saves all data in RAM to .vdd files. Speeds up loading in the future
vd.redistribute_all_files(are_you_sure=True) # When called with are_you_sure=True, it moves all files from all subdirectories into root/.secret, then removes subdirectories and essentially recreates file structure. Allows to easily split dataset into subdirectories (batches) of similar size
len(vd) # number of files in dataset
```
You can also iterrate over files within it, but then consider the data read-only:
```py
for filename, path, binary_data in vd:
  # filename - string, just filename
  # path - string, entire path; root/subdir/filename
  # binary_data - binary (serialized) data stored in given file (Only if load_to_memory=True, otherwise it's None). Must be deserialized before usage
  #data = vd.data_manager.deserialize(binary_data) # like this
  #data = vd.load(filename) # or you can just request data from vd
```
**If you ever add/remove data without using proper interface tools, you have to create a new instance of ```VirtualDirectory```**. Fortunatelly, it does NOT require ```.vdd``` to be regenerated - class will update them on its own.

---

# Advanced usage <a name="advusage"></a>

One day, hopefully...

```py
from VirtualDirectory import VirtualDirectory, PillowDataManager
vd = VirtualDirectory(
  root, # main directory of the dataset (string)
  data_manager, # instance of a class that is used to save, load and serialize datatype used in dataset
  verbose = True, # whether object should print progress as dataset is loading (True) or not (False). Defaults to True
  load_to_memory = True, # whether entire dataset should be loaded into RAM (True) or not (False). Defaults to True
  min_subdir_num=100, # minimal amount of subdirectories used to store data. Defaults to 100
  save_on_destruction = False, # Whether state of memory should be saved automatically when object of VirtualDirectory is removed - it's known to bug out and raise exceptions
  serializer = PickleSerializer(), # Serializer used for .vdd files. Python dictionary is serialized, then compressed and saved onto drive
  compressor = ZlibCompressor(), # Compressor used for .vdd files. Python dictionary is serialized, then compressed and saved onto drive
  seed = None # Seed for random number generated contained within VirtualDirectory. I've no idea why anyone would want to set it to any specific value
)
```

```py
import pickle
class PickleSerializer:
    def serialize(self, python_object):
        return pickle.dumps(python_object)
    def deserialize(self, binary_string):
        return pickle.loads(binary_string)
```

```py
import zlib
class ZlibCompressor:
    def compress(self, binary_string):
        return zlib.compress(binary_string)
    def decompress(self, binary_string):
        return zlib.decompress(binary_string)
```

---

# Miscellaneous <a name="misc"></a>

I've included tools that convert between Pillow and OpenCV image format. Given the primary objective of this library is to be used in AI, it might be useful to some. 
```py
from VirtualDirectory import VirtualDirectory, pil_to_opencv, opencv_to_pil, PillowDataManager, OpenCVDataManager
```

```py
import numpy
import cv2
def pil_to_opencv(pil_image):
    data = pil_image.convert("RGB")
    data = numpy.array(data)
    opencv_image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return opencv_image
```

```py
import cv2
from PIL import Image as PilImage
def opencv_to_pil(opencv_image):
    data = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = PilImage.fromarray(data)
    return pil_image
```

It's possible to implement opencv's ```data_manager``` by building upon ```PillowDataManager```, however this would mean that any time image is accessed it must be converted from Pillow format *(RGB)* into OpenCV *(BGR)*, which can cause noticeable overhead, for large datasets dedicated implementation would be much better.


```py
import io
from PIL import Image as PilImage
class OpenCVDataManager:
    def save(self, path, opencv_image):
        pil_image = opencv_to_pil(opencv_image)
        pil_image.save(path)
    def load(self, path):
        pil_image = PilImage.open(path)
        return pil_to_opencv(pil_image)
    def serialize(self, opencv_image):
        buffer = io.BytesIO()
        pil_image = opencv_to_pil(opencv_image)
        pil_image.save(buffer, format = 'PNG')
        return buffer.getvalue()
    def deserialize(self, binary_string):
        buffer = io.BytesIO(binary_string)
        pil_image = PilImage.open(buffer)
        return pil_to_opencv(pil_image)
```

---

# Free-for-all scripts <a name="ffa"></a>

Scripts made by me or others, applying ```VirtualDirectory``` to practical use. No descriptions, don't ask for one. Not included in library, but useful for insight.

## Pytorch multimodal dataset, with pre-trained models for feature extraction
```py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import transformers
from VirtualDirectory import VirtualDirectory, PillowDataManager

class MultimodalDataset(Dataset):
    def __init__(self, dataframe, img_dir, augmentation, img_header, text_header, labels_header):
        self.df = dataframe
        self.img_dir = img_dir
        self.augmentation = augmentation
        self.img_header = img_header # "filename"
        self.text_header = text_header # "text"
        self.labels = labels_header # ["label_1", "label_2"]
        self.virtual_space = VirtualDirectory(root=img_dir, data_manager=PillowDataManager(), load_to_memory=True)
        self.virtual_space.save_state()

    def get_image(self, index):
        img_name = self.df.iloc[index][self.img_header]
        image = self.virtual_space.load(img_name)
        return image

    def get_text(self, index):
        return self.df.iloc[index][self.text_header]

    def get_label(self, index):
        return self.df.iloc[index][self.labels].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.get_image(idx)
        image = self.augmentation(image)
        text = self.get_text(idx)
        labels = self.get_label(idx)
        return image, text, labels

class TorchDataset(Dataset):
    def __init__(self, multimodal_dataset, image_model, text_model): # Custom model classes, needs to have "normalize" function which turns input into tensor
        self.multimodal_dataset = multimodal_dataset
        self.image_model = image_model
        self.text_model = text_model

    def __len__(self):
        return len(self.multimodal_dataset)

    def __getitem__(self, idx):
        image, text, labels = self.multimodal_dataset[idx]
        torch_image = self.image_model.normalize(image)
        torch_encodings = self.text_model.tokenize(text)
        return torch_image, torch_encodings, np.array(labels).astype(float)

class DistilRoBERTa(nn.Module):
    def __init__(self, max_tokens):
        super(DistilRoBERTa, self).__init__()
        self.max_tokens = max_tokens
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")
        self.model = transformers.AutoModel.from_pretrained("distilroberta-base")
        for param in self.model.parameters(): param.requires_grad = False

    def tokenize(self, text):
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_tokens,
            truncation=True,
            padding='max_length',  # Pad to the fixed length
            return_tensors='pt'  # Return as a PyTorch tensor
        )

    def forward(self, encoding):
        self.model.eval()
        input_ids, attention_mask = encoding['input_ids'].squeeze(1), encoding['attention_mask'].squeeze(1)
        with torch.no_grad():
            _, features = self.model(input_ids = input_ids, attention_mask = attention_mask, return_dict=False)
            return features

class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg = models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:-2])
        for param in self.vgg.parameters(): param.requires_grad = False
  
    def normalize(self, image):
        return models.VGG19_Weights.IMAGENET1K_V1.transforms()(image)
  
    def forward(self, x):
        self.vgg.eval()
        with torch.no_grad():
            x = self.vgg(x)
            return x
```
