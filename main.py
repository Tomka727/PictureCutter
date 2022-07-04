import numpy as np
from PIL import Image as Img
import config as cfg
from numpy import asarray as asar

image = Img.open(cfg.image_name)

data = asar(image)

image2 = Img.fromarray(data)

line_size = cfg.line_size
print(data.shape)
data1 = data[0:line_size:]
data2 = data[line_size:line_size*2:]
for x in range(2, int(data.shape[0]/line_size)):
    if x % 2 == 0:
        data1 = np.concatenate((data1, data[x*line_size:(x+1)*line_size:]), dtype='uint8')
    else:
        data2 = np.concatenate((data2, data[x * line_size:(x + 1) * line_size:]), dtype='uint8')
data = np.concatenate((data1, data2), dtype='uint8')

data1 = data[::, 0:line_size:]
data2 = data[::, line_size:line_size*2:]
for x in range(2, int(data.shape[1]/line_size)):
    if x % 2 == 0:
        data1 = np.concatenate((data1, data[::, x*line_size:(x+1)*line_size:]), axis=1, dtype='uint8')
    else:
        data2 = np.concatenate((data2, data[::, x * line_size:(x + 1) * line_size:]), axis=1, dtype='uint8')
data = np.concatenate((data1, data2), axis=1, dtype='uint8')

img = Img.fromarray(data, 'RGB')
img.save('my.png')
img.show()
