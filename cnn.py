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
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Max Pooling
# Stride of 2 - two pxs over at a time
# Takes the maximum value in the 2x2 square
# Size of original feature map is divided by 2 when applying max pooling
# Reduces time complexity and less compute intensive
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
# Takes all the pooled layers and flattens them into a single vector
# Becomes the input layer for an ANN
# Each number in the vector represents a feature (i.e. dog nose)
classifier.add(Flatten())

# Step 4 - Full Connection
# Output of 128 is randomly chosen
classifier.add(Dense(units = 128, activation='relu'))

# Output layer
classifier.add(Dense(units = 1, activation='sigmoid'))

# Compiling the CNN
# Adam is stochastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

# Rescaling pixels between 1 and 255 makes the values between 0 and 1
# Sheer range performs random transvections
# Zoom range performs random zooms
# Images will be flipped horizontally
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Target size is the size of image dims
# Class node is if it's binary or more than 2 categories
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Steps_per_epoch is the # of images per training set
# Validation steps is the # of images in the test set
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)