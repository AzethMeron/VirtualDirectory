
from VirtualDirectory import VirtualDirectory, PillowDataManager
import os

mnist = os.listdir("mnist_single_dir")
data_manager = PillowDataManager()
vd = VirtualDirectory(root="mnist_vd", data_manager=data_manager, verbose=True, min_subdir_num=10, load_to_memory=True)

for filename in mnist:
	path = os.path.join("mnist_single_dir", filename)
	image = data_manager.load(path)
	vd.save(filename, image)
