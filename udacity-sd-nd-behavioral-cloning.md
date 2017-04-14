
***Behavioral Cloning***

The following is a summary of the udacity car-nd behavioral cloning project.

The project entails capturing of driving steering wheel angles wheel driving on a simulated track. The data is used to train a model which is subsequently used to "drive" a car around the track in simulation.

---

***Behavioral Cloning Project***

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results in this report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"




***Code and Data repository***

The code, models and training dataset is uploaded to a public github repo.
https://github.com/peterbarry/udacity-self-driving-ND-behavioral_cloning-car-sim/


My project includes the following files:
* model_nvidia.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* udacity-self-driving-ND-behavioral-cloning-sim.md  summarizing the results


***Methodology***

The development consisted of
 1. Capture a behavior data, by driving the car in the simulator. The saved data consists of a snapshot image of the track as viewed by 3 positions (left/center/right) and a steering wheel angle at taken at that image while driving the car. Details of the types of data captured is described below.
 2. Develop a keras model with the appropriate image processing and model which when trained will output a keras model .m5 file. The .m5 file consists of a full network description (including image processing included in the graph), and all learned network weights.
 3. The saved model file can be reloaded and run as an execution model where we can submit an image and the model predicts the steering angle. The model is executed in conjunction with the simulator. The simulator forwards a center angle image to the model executing and waits for the projected steering angle, sets the angle and moves forward. The current model was tested at 9mph.
 4. Iterate with the model and additional data gathering.

 Aside: As the data sets grew, training time increased significantly. I migrated from a CPU only training model to a GPU based model. Training times improved very significantly where iteration loops were much shorter and much more satisfactory. I had access to a GPU but found it challenging to install GPU enabled version of tensorflow for use with the car-nd provided environments. I had the latest packages for tensorflow/keras and my code as a result ended up using Keras 2.0 APIs.


***Codebase***

The model_nvidia.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


***Data Gathering***

Data enrichment:
 The data was enriched by adding
 1. Flipped versions of all images
 2. Driving the track in reverse.
 3. Gather multiple good runs of data.
 4. Gather a large number of recover data- close to edge, turn away.
 5. Add right and left camera images. The simulator provides additional image data but only steering angle data associated with the center image. When using these additional images for training and validation - we must adjust the steering angle as though it was a center image. This took considerable tuning. A number too large resulted in wide oscillations of the car from side to side of the track - too little compensation caused the car to just run off the track. There was no car dimension data provided so the value had to be obtained through trial and error- a better approach if the camera and car geometry provided was to use trigonometry to calculate the exact correction factor to use the data.


***Data Processing***
 1. Normalizing the data : by dividing the pixels by 255 and set mean to zero
 ```sh
 model.add(Lambda(lambda x: x / 255.0 - 0.5 , input_shape=input_image_shape))
 ```

 2. Croping the image to a subset of the image size to those most likly to be usefull.

 ```sh
 model.add(Lambda(lambda x: x / 255.0 - 0.5 , input_shape=input_image_shape))
 model.add(Cropping2D(cropping=((70,25),(0,0))))
 ```

 3. Though other corrections were considered eg reduce to grayscale, resize the image, add enriched data via manipulation. It was unnecessary in the end as I was able to get a model to sufficient quality by talking additional simulation data.



***Actual Model***
The project was started with the provide data set and LeNet5 model. The model worked well at the start but I could not get it to perform well towards the bridge. I added more data but no improvement. Eventually I switched to the NVidia model where it improved over LeNet5 for my data set.

The model deployed for the final results was the NVidia model.
Reference: End to End Learning for Self-Driving Cars : https://arxiv.org/pdf/1604.07316v1.pdf


| Layer (type)  | Output Shape | Param # |
| ------------- | ------------- |
|lambda_1 (Lambda)     |        (None, 160, 320, 3)   |     0      |
|cropping2d_1 (Cropping2D)    |(None, 65, 320, 3)        |0|         
|conv2d_1 (Conv2D)            |(None, 31, 158, 24)       |1824   |   
|activation_1 (Activation)    |(None, 31, 158, 24)       |0     |     
|conv2d_2 (Conv2D)            |(None, 14, 77, 36)        |21636  |    
|dropout_1 (Dropout)          |(None, 14, 77, 36)        |0     |     
|activation_2 (Activation)    |(None, 14, 77, 36)        |0     |     
|conv2d_3 (Conv2D)            |(None, 5, 37, 48)         |43248  |    
|dropout_2 (Dropout)          |(None, 5, 37, 48)         |0     |     
|activation_3 (Activation)    |(None, 5, 37, 48)         |0      |    
|conv2d_4 (Conv2D)            |(None, 3, 35, 64)         |27712  |    
|dropout_3 (Dropout)          |(None, 3, 35, 64)         |0      |    
|activation_4 (Activation)    |(None, 3, 35, 64)         |0      |    
|conv2d_5 (Conv2D)            |(None, 1, 33, 64)         |36928  |    
|dropout_4 (Dropout)          |(None, 1, 33, 64)         |0     |    
|activation_5 (Activation)    |(None, 1, 33, 64)         |0      |    
|flatten_1 (Flatten)          |(None, 2112)              |0     |     
|h1 (Dense)                   |(None, 100)               |211300 |    
|h2 (Dense)                   |(None, 50)                |5050   |    
|out (Dense)                  |(None, 1)                 |51     |    



Total params: 347,749.0
Trainable params: 347,749.0
Non-trainable params: 0.0



The following model commend has been employed to display the network.


```sh
print(model.summary())
```



My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model_nvidia.py lines 161-219)

The model includes RELU layers to introduce nonlinearity (eg code line 206), and the data is normalized in the model using a Keras lambda layer.

The fully connected layer:
```sh
model.add(Dense(100, name="h1"))
model.add(Dense(50, name="h2" ))
model.add(Dense(1, name="out"))
```

 ***Optimiser***

 The model was developed to output a target floating point steering angle. Previous networks eg Lenet was developed to categorize a digit using softmax and selecting a target match with the highest probability.
 For this model case the output node of the network as a single node with a predicated fp value. The loss function is a mean squared error.
 ```sh
 adam = optimizers.Adam(lr=learning_rate)
 model.compile(loss='mse',optimizer=adam)
```

***Model overfitting***

The model contains dropout layers in order to reduce overfitting, I used a dropout value of 0.5.
The reference paper for dropout: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf


```sh
model.add(Dropout(dropout_value))
```
The model was trained and validated on different data sets 80% training/20% validation - to ensure that the model was not overfitting.  The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
```sh
train_samples, validation_samples,train_measurements,validation_measurements = train_test_split(samples_total, measurements,  test_size=0.2)
```

***Model parameter tuning***

The model used an adam optimizer, I did update the model learning rates but returned to the default learning rate after no major change in performance.

***Epoch Selection***

In many cases the loss converged or increased after 3 runs. I typically ran for 5 epochs on a new test/data set and re-ran with a lower number of epochs once I could see the approx. are where losses had become converged.

| Epoch | loss | Val_loss  
| ------------- | ------------- |
|1    |        0.0804   |     0.0144      |
|2 | 0.0128 | 0.0144
|3 | 0.0117|  0.0116
|4| 0.0111| 0.0109
|5| 0.0110| 0.0121


***Appropriate training data***

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.


***Model Architecture and Training Strategy***


My first step was to use a convolution neural network model similar to the Lenet ... I thought this model might be appropriate because it is a generic network and performed well on traffic sign classification
To combat the overfitting, I first added more data.
Then I updated to the nvidia model, added dropout and continued to collect data for areas where the car went off the track. The process was time consuming, in many cases adding new data for a target corner/condition resulted in failing earlier in the track. The approach was to add full data sets of good runs AND new data for area where the track was particularly different (eg sandbank runoff near bridge)

I do have a concern that I'm contaminated my model in these iterations but in general I was adding more data not changing the network so I don’t believe I'm contaminating the network through the updates I make.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I added more specific data AND full track data to prevent overfitting on small areas of the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

***Final Model Architecture***

The final model architecture consisted of a convolution neural network with the layers and layer sizes shown above.

Here is a visualization of the architecture

![alt text][image1]

The model output file was genated by
```sh
plot_model(model, to_file='model.png',show_shapes=True)
```
