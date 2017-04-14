#**Behavioral Cloning**

The following is a summary of the udacity car-nd behavioral cloning project.

The project entails capturing of driving stearing wheel angles whiel driving on a simulated track. The data is used to train a model which is subsequently used to "drive" a car around the track in simulation.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results in this report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"



**Code and Data repository**

The code,models and traning dataset is uploaded to a public github repo.
https://github.com/peterbarry/udacity-self-driving-ND-behavioral_cloning-car-sim/


My project includes the following files:
* model_nvidia.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* udacity-self-driving-ND-behavioral-cloning-sim.md  summarizing the results


** Methodology  **

The development consisted of
 1. Capture a behaviour data, by driving the car in the simulator. The saved data consists of a snapshot image of the track as viewed by 3 positions (left/center/right) and a steering wheel angle at taken at that image while driving the car. Details of the types of data captured is described below.
 2. Develope a keras model with the appropriate image processing and model which when trained will output a keras model .m5 file. The .m5 file consists of a full network description (including image processing included in the graph), and all learned network weights.
 3. The saved model file can be reloaded and run as an execution model where we can submit an image and the model predicts the steering angle. The model is executed in conjunction with the simulator. The simulator forwards a center angle image to the model execuing and waits for the projected steering angle, sets the angle and moves foward. The current model was tested at 9mph.
 4. Iterate with the model and additional data gathering.

 Aside: As the data sets grew, training time increased significantly. I migrated from a CPU only traning model to a GPU based model. Trainging times improved very significantly where iteration loops were much shorter and much more satisfactory. I had access to a GPU but found it challeinging to install GPU eanbled version of tensorflow for use with the car-nd provided enviroments. I had the latest packages for tensorflow/keras and my code as a result ended up using Keras 2.0 APIs.


** Codebase  **

The model_nvidia.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy


*** Actual Model ***
####1. An appropriate model architecture has been employed

The model deployed for the final results was the nvidia model.

| Layer (type)           |       Output Shape        |       Param # |   
=================================================================
| --- | ---- | ---- |

| lambda_1 (Lambda)       |      (None, 160, 320, 3)  |      0      |  
 --


_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
activation_1 (Activation)    (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
dropout_1 (Dropout)          (None, 14, 77, 36)        0         
_________________________________________________________________
activation_2 (Activation)    (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_2 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
activation_3 (Activation)    (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
dropout_3 (Dropout)          (None, 3, 35, 64)         0         
_________________________________________________________________
activation_4 (Activation)    (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_4 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
activation_5 (Activation)    (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
h1 (Dense)                   (None, 100)               211300    
_________________________________________________________________
h2 (Dense)                   (None, 50)                5050      
_________________________________________________________________
out (Dense)                  (None, 1)                 51        
=================================================================
Total params: 347,749.0
Trainable params: 347,749.0
Non-trainable params: 0.0



The following model commend has been employed to display the network.


```sh
print(model.summary())
```



My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).






####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
