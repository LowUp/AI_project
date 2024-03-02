from img2vec_pytorch import Img2Vec
from PIL import Image
from arrow import now
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from os.path import basename

from zipfile import ZipFile
import patoolib
# import pygame

#unrar folder called AiArtData.rar


img2vec = Img2Vec(cuda=True, model='resnet-18', layer='default', layer_output_size=512)

FOLDER_AI = 'train/FAKE/'
FOLDER_REAL = 'train/REAL/'

def img_to_array(tag: str, zipped_folder: str = None) -> list:
    result = []
    _IMG_COUNT = 100
    count = 0
    zip_object = ZipFile('datasets/dataset2.zip', 'r')
    zip_object.extractall()
    # zip_object.extract(zipped_folder, path='./datasets')
    for input_file in zip_object.namelist():
        if input_file.startswith(zipped_folder):
            name = basename(input_file)
            try:
                with Image.open(fp=input_file, mode='r') as image:
                    vector = img2vec.get_vec(image, tensor=True).numpy().reshape(512,)
                    result.append(pd.Series(data=[tag, name, vector], index=['tag', 'name', 'value']))
            except Exception as error:
                print("Runtime Error : ", error)
                # pass
            if count >= _IMG_COUNT:
                break # Stops processing images after 10 000 images processed. 
            count += 1
    zip_object.close()
    return result

time_start = now()
ai = img_to_array(zipped_folder=FOLDER_AI, tag='ai')
print('done encoding the AI images in {}'.format(now() - time_start))
real = img_to_array(zipped_folder=FOLDER_REAL, tag='real')
df = pd.DataFrame(data=ai + real)
print('done in {}'.format(now() - time_start))