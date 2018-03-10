import os
import numpy as np


def load_bndbox(filename):
    with open(filename, 'r') as reader:
        xml = reader.read()
    soup = BeautifulSoup(xml)
    bndbox = {}
    for tag in soup.bndbox:
        if tag.string != '':
            bndbox[tag.name] = int(tag.string)
    return bndbox
def accs_message(accs):
    if  isinstance(accs, dict):
        return 'top {}:{}'.format(tuple(accs.keys()), tuple(str(acc.avg)[:4] for acc in accs.values()))
    elif isinstance(accs, float):
        return 'acc :{%4f}'.format(accs)
def to_rgb(im):
    w,h = im.shape
    ret = np.empty((w,h,3), dtype=np.uint8)
    ret[:,:,0] = im
    ret[:,:,1] = ret[:,:,2] = ret[:,:,0]
    return ret
    
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
