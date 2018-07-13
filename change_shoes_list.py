import os

list_path = "photo_test.txt"
path = "/home/lhy/datasets/ShoeV2_F/{}".format(list_path)
with open(path, "r") as f:
    sketch_paths = f.read().splitlines()
new_path = "/home/lhy/datasets/ShoeV2_F/ab_{}".format(list_path)
with open(new_path, "w") as f:
    for sketch_path in sketch_paths:
        f.write("/home/lhy/datasets/ShoeV2_F/ShoeV2_photo/{}\n".format(sketch_path))
