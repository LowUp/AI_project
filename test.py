from img2vec_pytorch import Img2Vec
from PIL import Image
from arrow import now
from glob import glob
import pandas as pd
from os.path import basename

img2vec = Img2Vec(cuda=True, model='resnet-18', layer='default', layer_output_size=512)

GLOB_AI = './datasets/train/FAKE/*'
GLOB_REAL = './datasets/train/REAL/*'

def get_from_glob(arg: str, tag: str) -> list:
    result = []
    for count, input_file in enumerate(glob(pathname=arg)):
        name = basename(input_file)
        try:
            with Image.open(fp=input_file, mode='r') as image:
                vector = img2vec.get_vec(image, tensor=True).numpy().reshape(512,)
                result.append(pd.Series(data=[tag, name, vector], index=['tag', 'name', 'value']))
        except RuntimeError as error:
            print("Runtime Error : ", error)
            # pass
        print("Image processed : ", count)
    return result

time_start = now()
ai = get_from_glob(arg=GLOB_AI, tag='ai')
print('done encoding the AI images in {}'.format(now() - time_start))
real = get_from_glob(arg=GLOB_REAL, tag='real')
df = pd.DataFrame(data=ai + real)
print('done in {}'.format(now() - time_start))