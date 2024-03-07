import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

data_dir = pathlib.Path('test/').with_suffix('')

roses = list(data_dir.glob('./*/*.jpg'))
PIL.Image.open(str(roses[0]))
print(roses[0])