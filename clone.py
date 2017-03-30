
import csv
import cv2
import numpy as np

lines = []
images = []
measurements = []

data_files=["data/driving_log.csv"]

for  cfilename in data_files:
	print("Opening: ", cfilename)
	with open(cfilename) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	#CVS file has a tile line as first.
	lines.pop(0)


	for line in lines:
		source_path=line[0]
		filename = source_path.split('/')[-1]
		current_path = cfilename.split('/')[0] + '/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)


X_train =  np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

input_image_shape = (160,320,3)

model = Sequential()
#normalize and mean center the data in a labda layer
model.add(Lambda(lambda x: x / 255.0 - 0.5 , input_shape=input_image_shape))
#model.add(Flatten(input_shape=input_image_shape))

model.add(Convolution2D(6,5,5,
                 activation='relu',
                 ))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,
                 activation='relu',
                 ))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
model.add(Dense(120))
model.add(Dense(86))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))


model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=5)
model.save('model.h5')










		




