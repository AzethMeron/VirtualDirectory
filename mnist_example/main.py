
from VirtualDirectory import VirtualDirectory, PillowDataManager
import os

# This script loads all images from one directory and puts it into VirtualDirectory, building the hierarchy with subdirectories on its own.
# The same can be achieved by moving entire mnist_single_dir into mnist_vd, then by calling vd.redistribute_all_files(are_you_sure=True)

mnist = os.listdir("mnist_single_dir")
data_manager = PillowDataManager()
vd = VirtualDirectory(root="mnist_vd", data_manager=data_manager, verbose=True, min_subdir_num=10, load_to_memory=True)

for filename in mnist:
	path = os.path.join("mnist_single_dir", filename)
	image = data_manager.load(path)
	vd.save(filename, image)
vd.save_state()


"""
# Some tests
print(vd.get_list_of_files())
for file, path, binary_img in vd:
	if not vd.exists(file): print(f"File {file} exists and doesn't exist at the same time")
	if not vd.get_path(file) == path: print(f"Wrong paths: {vd.get_path(file)} vs {path}")
	vd.save(file, vd.data_manager.deserialize(binary_img))
	if not vd.load(file) == vd.data_manager.deserialize(binary_img): print("Something went wrong with images")
"""