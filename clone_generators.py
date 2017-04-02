
import csv
import cv2
import numpy as np
import sklearn
import random


#http://stackoverflow.com/questions/19068085/shift-image-content-with-opencv
#Mat translateImg(Mat &img, int offsetx, int offsety){
#    Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
#    warpAffine(img,img,trans_mat,img.size());
#    return trans_mat;
#}


samples_total = []
measurements = []

data_files=["data/udacity/driving_log.csv"]

#camera_steering_angle_correction = [0.1,0,-0.1]
camera_steering_angle_correction = [0.0,0.2,-0.2]

for  cfilename in data_files:
	lines = []
	print("Opening: ", cfilename)
	with open(cfilename) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)
	#CVS file has a tile line as first.
	lines.pop(0)


	for line in lines:
		#print (line)
		for correction_id in range(3):
			source_path=line[correction_id]
			filename = source_path.split('/')[-1]
			current_path = cfilename.split('/')[0] + "/" + cfilename.split('/')[1] +  '/IMG/' + filename
			measurement = float(line[3])
			measurement = measurement + camera_steering_angle_correction[correction_id]

			#print ("Loading Image: " + current_path + " with angle:" + str(measurement) )
			#image = cv2.imread(current_path)
			#images.append(image)

			samples_total.append(current_path)
			measurements.append(measurement)
			
			# Enrich the date set with a fliped version of the images
			#images.append(cv2.flip(image,1))
			#measurements.append(measurement*-1.0)

	print ("Processing " + str(len(samples_total)) + " images")

#shuffle data and measurements outside
#reference: http://stackoverflow.com/questions/23289547/shuffle-two-list-at-once-with-same-order
c = list(zip(samples_total,measurements))
random.shuffle(c)
samples_total,measurements=zip(*c)


for iter in range(10):
	randval = random.randrange(0,len(samples_total))
	print (" ** file name samples: " + samples_total[randval] + " angle: " +  str(measurements[randval]))




	
#X_train =  np.array(images)
#y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Flatten, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import  train_test_split

train_samples, validation_samples,train_measurements,validation_measurements = train_test_split(samples_total, measurements,  test_size=0.2)


number_of_samples_returned_per_yield = (32)

def generator(samples,meass ,batch_size=number_of_samples_returned_per_yield):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #shuffled full list outside

	#batch size / 2 as we generate 2 samples each round.
        for offset in range(0, num_samples, int(batch_size/2)):
            batch_samples = samples[offset:offset+batch_size]
            batch_meas = meass[offset:offset+batch_size]
            images = []
            angles = []
            ang_index=0
            for batch_sample in batch_samples:
                name = batch_sample
                center_image = cv2.imread(name)
                images.append(center_image)
                angle = batch_meas[ang_index]
                ang_index = ang_index + 1
                angles.append(angle)
                #print("Processing image: " + name + " with angle: " + str(angle))


                # Enrich the date set with a fliped version of the images
                images.append(cv2.flip(center_image,1))
                angles.append(angle * -1.0)

            # images are trimed in the lambda function


            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples,train_measurements, batch_size=number_of_samples_returned_per_yield)
validation_generator = generator(validation_samples,validation_measurements, batch_size=number_of_samples_returned_per_yield)


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


#model.fit(X_train,y_train, validation_split=0.2,shuffle=True,nb_epoch=10, verbose=1)

#model.fit_generator(train_generator, 
#                    samples_per_epoch=len(train_samples), 
#                    validation_data=validation_generator,
#                    nb_val_samples=len(validation_samples), nb_epoch=5)

model.fit_generator(train_generator, 
                    steps_per_epoch=len(train_samples)/number_of_samples_returned_per_yield, 
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')










		




