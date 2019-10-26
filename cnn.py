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

# Step 2 - Max Pooling
# Stride of 2 - two pxs over at a time
# Takes the maximum value in the 2x2 square
# Size of original feature map is divided by 2 when applying max pooling
# Reduces time complexity and less compute intensive
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Part 3 - Flattening
# Takes all the pooled layers and flattens them into a single vector
# Becomes the input layer for an ANN
# Each number in the vector represents a feature (i.e. dog nose)
classifier.add(Flatten())