# Part 1 build CNN

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialize CNN
classifier = Sequential()

# Step 1 - Convolution (adding CNN layer)
# 32 feature detectors that are 3x3
# 64x64 size shape and 3 channels
classifier.add(Conv2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))
