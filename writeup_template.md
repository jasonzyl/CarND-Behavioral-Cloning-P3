# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./center_lane_driving.jpg "Center Lane Driving"
[image2]: ./correcting.jpg "Correcting"
[image3]: ./flip.jpg "Flipped"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried a network based on LeNet first:
* Cropping layer
* Normalization layer
* Convolutional layer with 5x5 filter and 6 channels
* Maxpooling layer
* Convolutional layer with 5x5 filter and 16 channels
* Maxpooling layer
* Flatten
* Fully connected 11840 -> 1000
* Fully connected 1000 -> 70
* Fully connected 70 -> 1

Later I switched to using an architecture developed by Nvidia:
* Cropping layer
* Normalization layer
* Convolutional layer with 5x5 filter, 24 channels and 2x2 downsampling
* Convolutional layer with 5x5 filter, 36 channels and 2x2 downsampling
* Convolutional layer with 5x5 filter, 48 channels and 2x2 downsampling
* Convolutional layer with 3x3 filter and 64 channels
* Convolutional layer with 3x3 filter and 64 channels
* Flatten
* Fully connected 2112 -> 200
* Fully connected 200 -> 14
* Fully connected 14 -> 1

#### 2. Attempts to reduce overfitting in the model

I tried dropout with different parameters (0.5, 0.6, etc.) but it does not seem to improve the validation loss, so I did not end up using it.

The model was trained on both original images, and horizontally flipped images, so that we have more training data and hence mitigating overfitting problem.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 84).

#### 4. Appropriate training data

Here are techniques used to provide appropriate training data:
* Drive on center of the lane as much as possible
* Crop out top and bottom parts of the images
* Use left and right camera images with a correction angle, to obtain more training data
* Flip the images horizontally (also with flipped target variable), to overcome the problem that the training data mostly consists with left-turning
* Drive out of center of the lane without recording, then record the process of coming back to the center, so that the model learns how to go back after drift off accidentally

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because the task is similar to the lane finding problem we had before, and LeNet seems to work well on that one.

After that, I did a few improvements to the model such as:
* Use lambda layer to normalize the data
* Use left and right camera images for more training data
* Cropping image to get rid of unrelated parts for deciding an angle to drive with
* Flip image horizontally
* Do recovering laps

To this point, the model seems to be able to drive a full loop, although after a few loops it still went out of the lane.

Then I tried the Nvidia model, and it seems to work better: the model drives for at least 10 laps, and it seems be able to stay at the center more closely.

####2. Final Model Architecture

The final model architecture (model.py lines 66-85) consisted of a convolution neural network with the following layers and layer sizes:
* Cropping layer
* Normalization layer
* Convolutional layer with 5x5 filter, 24 channels and 2x2 downsampling
* Convolutional layer with 5x5 filter, 36 channels and 2x2 downsampling
* Convolutional layer with 5x5 filter, 48 channels and 2x2 downsampling
* Convolutional layer with 3x3 filter and 64 channels
* Convolutional layer with 3x3 filter and 64 channels
* Flatten
* Fully connected 2112 -> 200
* Fully connected 200 -> 14
* Fully connected 14 -> 1


####3. Creation of the Training Set & Training Process

To make data collection easier, I made data reading process more flexible: instead of taking one set of data (from a single `driving_log.csv`), the `read_data()` function (model.py line 15 - 28) can take multiple sets of data, and each set has a `driving_log.csv`. Therefore, each time I can record a set of data into a separate directory under `data/` instead of appending data into an existing directory.

To capture good driving behavior, I first recorded two sets of data using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center after accidentally drifting off.

![Correcting][image2]

To augment the data sat, I also flipped images and angles thinking that this would make the model generalize better: not only know when to turn left (since the track is counter-clockwise), but also know when to turn right. For example, here is an image that has then been flipped:

![Flipped][image3]

After the collection process, I had 8109 raw images in all 4 `driving_log.csv`s, and considering each data sample generates 6 data points, I had 48654 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I was using 5 epochs since it seems it does not improve (actually sometimes the validation loss got worse). I used an adam optimizer so that manually training the learning rate wasn't necessary.
