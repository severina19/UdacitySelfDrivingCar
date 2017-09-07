#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./NumOfDatTraining.png "Number of Traffic Sign of different Classes in Training Data"
[image2]: ./speed_70.png "Example of Training Data: Speed Limit 70 kmh"
[image3]: ./speed_30.png "Example of Training Data: Speed Limit 70 kmh"
[image4]: ./speed_30_gray.png "Example of Training Data: Speed Limit 70 kmh"
[image5]: ./signs/1.png "traffic sign found on internet - 1"
[image6]: ./signs/2.png "traffic sign found on internet - 2"
[image7]: ./signs/3.png "traffic sign found on internet - 3"
[image8]: ./signs/4.png "traffic sign found on internet - 4"
[image9]: ./signs/5.png  "traffic sign found on internet - 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/severina19/UdacitySelfDrivingCar/blob/master/Lesson2/Traffic_Sign_Classifier_trained.ipynb)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set. In the pickled data which was provided to us, the dataset is already devided into training, validation and testing. I find that the validation set has a bit less data, that is why I have reassined the validation and training set with a ratio of 1:4. I left the test data set as it is. After the reassignment:

* The size of training set is 31367
* The size of the validation set is 7842
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how data are distributed among different classes for the training set. We can see that the distribution is not balanced, for some classes such as class 19 which is "Dangerous curve to the left" there are only few dataset. 

![alt text][image1]

All the traffic signs are sampled to 32x32 bit, and below we can see an example of it.  
![alt text][image2]
As we can see, the resolution of 32 bit is quite low and under poor lighting it is sometimes not easy for human to classify it correctly.  

###Design and Test a Model Architecture

In the pre-processing step I converted the images into grayscale and normalized the data. I tried using RGB images for training too but the grayscale images yield better results. Using grayscale also saves computational time. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]
![alt text][image4]

I normalized the image data so that the range of distribution of the input values are the same, so that we will not have a large change of gradient for input with higher values.  

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x256    |  
| Fully connected		| outputs 1x1x256           					|
| Relu           		|                            					|
| Dropout          		|                            					|
| Fully connected		| outputs 1x1x86 .        						|
| Relu           		|                            					|
| Dropout          		|                            					|
| Fully connected		| outputs 1x1x43   							    |
| Softmax				|           									|
|						|												|
|						|												|
 

The Model has 3 convolutional layers and 3 fully connected layers. To train the model, I used the adam optimizer which was also used in the LeNet example. As for the batch size I chose 128, and epochs I chose 40. 

For the training I tried out different set of hyper parameters. For the leaning rate, I started with 1e-3 but the optimizer reached a plateau at around 75% accuracy. I also tried out learning decay starting with rate of 1e-3. I find that a constant learning rate of 1e-4 is a good value. 

My final model results were:
* training set accuracy of 0.974
* validation set accuracy of 0.965 
* test set accuracy of 0.911

At the beginning I used the exact same net as in the LeNet example with the only change of number of output from 10 to 43. The result was not very satisfying, in my opinion because the number of filter in the conv Net was too low. I increased the number and the result was better. But the result of the test set was still low probably because of overfitting, this is why I introduced a drop out probablity of 0.5 during training in the first two fully connected layer and the performance became much better. 
 
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 



I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 



