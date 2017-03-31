
import csv
import cv2
import numpy as np


#http://stackoverflow.com/questions/19068085/shift-image-content-with-opencv
Mat translateImg(Mat &img, int offsetx, int offsety){
    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(img,img,trans_mat,img.size());
    return trans_mat;
}


lines = []
images = []
measurements = []

data_files=["data/driving_log.csv"]

#camera_steering_angle_correction = [0.1,0,-0.1]
camera_steering_angle_correction = [0.22,0,-0.22]

for  cfilename in data_files:
	print("Opening: ", cfilename)
	with open(cfilename) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	#CVS file has a tile line as first.
	lines.pop(0)


	for line in lines:
		for x in range(3):
			source_path=line[x]
			filename = source_path.split('/')[-1]
			current_path = cfilename.split('/')[0] + '/IMG/' + filename
			image = cv2.imread(current_path)
			images.append(image)
			measurement = float(line[3])
			measurement = measurement + camera_steering_angle_correction[x]
			measurements.append(measurement)
			# Enrich the date set with a fliped version of the images
			images.append(cv2.flip(image,1))
			measurements.append(measurement*-1.0)



X_train =  np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Flatten, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

input_image_shape = (160,320,3)

model = Sequential()
#normalize and mean center the data in a labda layer
model.add(Lambda(lambda x: x / 255.0 - 0.5 , input_shape=input_image_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))

#model.add(Flatten(input_shape=input_image_shape))

model.add(Convolution2D(6,5,5,
                 activation='relu',
		 border_mode='valid'
                 ))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(16,5,5,
                 activation='relu',
		 border_mode='valid'
                 ))

model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))

#model.add(Convolution2D(120,1,1,
#                 activation='relu',
#		 border_mode='valid'
#                 ))


model.add(Flatten())
model.add(Dense(120))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(120))
model.add(Dense(84))
#model.add(Dense(128, activation='relu'))
#model.add(Dense(num_classes, activation='softmax'))


model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

print(model.summary())


model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=10, verbose=1)

model.save('model.h5')










		




