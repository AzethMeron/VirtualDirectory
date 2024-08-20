
# Python 3.12.0
import os
import random
import zlib
import pickle
import io
import pathlib
# Non-standard
import tqdm # pip install tqdm>=4.66.5
import cv2 # pip install opencv-python>=4.10.0.84
from PIL import Image as PilImage # pip install pillow>=10.4.0
import numpy # pip install numpy>=2.1.0

def pil_to_opencv(pil_image):
    data = pil_image.convert("RGB")
    data = numpy.array(data)
    opencv_image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return opencv_image

def opencv_to_pil(opencv_image):
    data = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = PilImage.fromarray(data)
    return pil_image

class PickleSerializer:
    def serialize(self, python_object):
        return pickle.dumps(python_object)
    def deserialize(self, binary_string):
        return pickle.loads(binary_string)

class ZlibCompressor:
    def compress(self, binary_string):
        return zlib.compress(binary_string)
    def decompress(self, binary_string):
        return zlib.decompress(binary_string)

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

class VirtualDirectory:
    def __init__(self, root, data_manager, verbose = True, min_subdir_num = 100, load_to_memory = False, serializer = None, compressor = None, seed = None):
        os.makedirs( root, exist_ok = True )
        self.root = root # String
        self.load_to_memory = load_to_memory
        self.verbose = verbose
        self.random = random.Random(seed) if seed else random.Random()
        self.serializer = serializer if serializer else PickleSerializer()
        self.compressor = compressor if compressor else ZlibCompressor()
        self.data_manager = data_manager
        if self.verbose: print(f"VirtualDirectory {self.root}: Scanning for subdirectories...")
        self.subdir_list = [ subdir for subdir in os.listdir(root) if os.path.isdir( os.path.join(root, subdir) ) ] # [ subdir_0, subdir_1 ... subdir_X ]
        if len(self.subdir_list) < min_subdir_num: self.__generate_subdirs(min_subdir_num)
        if self.verbose: print(f"Found {len(self.subdir_list)} directories.")
        self.files_map = self.__build_files_map(self.subdir_list) # files_map[filename] = subdir, which means path = os.path.join(root, subdir, filename)
        self.memory = self.__load_to_memory(self.files_map) if self.load_to_memory else None
    def __generate_subdirs(self, min_subdir_num):
        subdir_set = set(self.subdir_list)
        limit = min_subdir_num - len(self.subdir_list)
        new_subdirs = [ self.__default_subdir_name(i) for i in range(0 ,min_subdir_num) if self.__default_subdir_name(i) not in subdir_set ][:limit]
        for subdir in new_subdirs:
            os.makedirs( os.path.join(self.root, subdir), exist_ok = True )
            self.subdir_list.append(subdir)
    def __default_subdir_name(self, x):
        return f"subdir_{x}"
    def __build_files_map(self, subdir_list):
        files_map = dict()
        if self.verbose: print("Scanning subdirectories for files...")
        for subdir in tqdm.tqdm(subdir_list, disable = not self.verbose, postfix = {"stage": "scanning subdirectories"}):
            files_list = os.listdir( os.path.join(self.root, subdir) )
            for filename in files_list:
                files_map[filename] = subdir
        if self.verbose: print(f"Found {len(files_map)} files.")
        return files_map
    def __load_to_memory(self, files_map):
        memory = self.__load_state(self.subdir_list)
        if self.verbose: print("Scanning directory for new or missing files...")
        memory = self.__strip_removed_files(memory, files_map)
        for filename in tqdm.tqdm(files_map.keys(), disable = not self.verbose, postfix={"stage":"loading into memory"}):
            subdir = files_map[filename]
            path = os.path.join(self.root, subdir, filename)
            if subdir not in memory: memory[subdir] = dict()
            if filename in memory[subdir]: continue
            data = self.data_manager.load(path)
            binary_data = self.data_manager.serialize(data)
            memory[subdir][filename] = binary_data
        return memory
    def __strip_removed_files(self, memory, files_map):
        for subdir in memory.keys():
            for filename in memory[subdir].keys():
                if filename in files_map and files_map[filename] == subdir: continue
                del memory[subdir][filename]
        return memory
    def __get_next_subdir(self):
        return self.random.choice(self.subdir_list)
    def load(self, filename):
        if self.exists(filename):
            subdir = self.files_map[filename]
            if self.load_to_memory:
                return self.data_manager.deserialize( self.memory[subdir][filename] )
            else:
                path = os.path.join( self.root, subdir, filename )
                return self.data_manager.load(path)
        return None
    def save(self, filename, data):
        subdir = self.files_map[filename] if self.exists(filename) else self.__get_next_subdir()
        path = os.path.join( self.root, subdir, filename )
        if self.load_to_memory and filename in self.memory[subdir]: del self.memory[subdir][filename]
        self.files_map[filename] = subdir
        self.data_manager.save(path, data)
        if self.load_to_memory: self.memory[subdir][filename] = self.data_manager.serialize(data)
    def remove(self, filename):
        if self.exists(filename):
            subdir = self.files_map[filename]
            path = os.path.join(self.root, subdir, filename)
            os.remove(path)
            del self.files_map[filename]
            if self.load_to_memory: del self.memory[subdir][filename]
    def exists(self, filename):
        return filename in self.files_map
    def save_state(self):
        if not self.load_to_memory: return
        for subdir in self.memory.keys():
            path = os.path.join( self.root, f"{subdir}.vdd")
            dump = self.serializer.serialize( self.memory[subdir] )
            dump = self.compressor.compress(dump)
            with open(path, "wb") as handle:
                handle.write(dump)
    def __load_state(self, subdir_list):
        if not self.load_to_memory: return
        memory = dict()
        for subdir in subdir_list: memory[subdir] = dict()
        dump_list = [ i for i in os.listdir(self.root) if i.endswith(".vdd") ]
        subdir_set = set(subdir_list)
        if self.verbose and len(dump_list) > 0: print(f"Found {len(dump_list)} dumpfiles, loading...")
        for dump in tqdm.tqdm(dump_list, disable = not self.verbose, postfix={"stage":"loading vdd files"}):
            p = pathlib.Path(dump)
            subdir = p.stem
            if subdir in subdir_set:
                path = os.path.join(self.root, dump)
                with open(path, "rb") as handle:
                    dump = handle.read()
                    dump = self.compressor.decompress(dump)
                    memory[subdir] = self.serializer.deserialize(dump)
        return memory
    def __len__(self):
        return len(self.files_map)
    def __del__(self):
        self.save_state()
    def __iter__(self):
        for filename in self.files_map.keys():
            subdir = self.files_map[filename]
            path = os.path.join( self.root, subdir, filename )
            data = self.memory[subdir][filename] if self.load_to_memory else None
            yield (filename, path, data)
