import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize

# 학습속도를 높이기 위해 흑백화면으로 전처리
def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe