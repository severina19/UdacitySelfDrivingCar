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
[image10]: ./signs/6.png  "traffic sign found on internet - 5"
[image11]: ./signs/7.png  "traffic sign found on internet - 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/severina19/UdacitySelfDrivingCar/blob/master/Lesson2/Traffic_Sign_Classifier_trained.ipynb)

###Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic signs data set. In the pickled data which was provided to us, the dataset is already devided into training, validation and testing. I find that the validation set has a bit less data, that is why I have reassined the validation and training set with a ratio of 1:4. I left the test data set as it is. Here are the results after the reassignment:

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
| Dropout          		| with keep probablity of 0.5            		|
| Fully connected		| outputs 1x1x86 .        						|
| Relu           		|                            					|
| Dropout          		| with keep probablity of 0.5                   |
| Fully connected		| outputs 1x1x43   							    |
| Softmax				|           									|
 
The Model has 3 convolutional layers and 3 fully connected layers. To train the model, I used the adam optimizer which was also used in the LeNet example. As for the batch size I chose 128, and epochs I chose 40. Those are the values I determined through an iterative way.

For the training I tried out different set of hyper parameters. For the leaning rate, I started with 1e-3 but the optimizer reached a plateau at around 75% accuracy. I also tried out learning decay starting with rate of 1e-3. I find that a constant learning rate of 1e-4 is a good value. 

My final model results were:
* training set accuracy of 0.971
* validation set accuracy of 0.964
* test set accuracy of 0.907

At the beginning I used the exact same net as in the LeNet example with the only change of number of output from 10 to 43. The result was not very satisfying, in my opinion because the number of filter in the previous Convolutional Network was too low. I increased the number and the result was better. But the result of the test set was still not accurate due to overfitting, this is why I introduced a drop out probablity of 0.5 during training in the first two fully connected layer and the performance became much better. 

From the final result we can see that the network is able to classify traffic signs with an accuracy of 0.907 for the test set. The accuracy of the training set and validation set are around 5% higher. In my opinion, feeding in more data will increase the performance. For example, we could rotate the traffic signs and feed the result into our training, thus obtain many more data sets. 


###Test a Model on New Images

Here are seven German traffic signs that I found on the web:

![alt text][image5] 
![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9]
![alt text][image10]
![alt text][image11]

#### Results 
Here are the results of the prediction:

| Image			           |     Prediction	        			| 
|:------------------------:|:----------------------------------:| 
| Stop Sign      		   | Stop sign   						| 
| Road Work   			   | Double curve 						|
| Yield					   | Yield								|
| Right-of-way at the  next intersection	    | Right-of-way at the next intersection              |
| Vehicles over 3.5 metric tons prohibited	| Roundabout mandatory              |
| Roundabout mandatory	   | Roundabout mandatory				|
| Vehicles over 3.5 metric tons prohibited	| Vehicles over 3.5 metric tons prohibited     |


The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71%. It is not able to correctly classify the 2nd and 5th sign, both with notable watermarks on them. 

For the first image, the model is very sure that this is a stop sign (probability of nearly 1), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.999989         			| Stop sign    									| 
| 0.000006     				| Bumpy road 										|
| 0.000003					| Yield											|
| 0.000003	      			| Keep right					 				|
| 0.000003				    |Turn left ahead     							|


For the second image the result was not satisfying. The sign was actually Road Work but the classified was half certain that it is a double curve sign. The top five soft max probabilities are listed below. We can see that, although for human eye it is clear that the sign is a road work sign, the watermarks have huge inpact on the performace of the classifier. In my opinion this is because our network has never seen similar pattern during training. If signs with noise applied are also used during training, the performance could be improved.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.455007         			| Double curve    									| 
| 0.105905    				| Wild animals crossing										|
| 0.093534					| Dangerous curve to the left											|
|  0.093534	      			| Road narrows on the right					 				|
| 0.093534		    |Right-of-way at the next intersection     							| 

For the third image, the classifier has given a very confident and correct classification of almost 100% to the yield sign. And the classification of the fourth image, which is a 'Right-of-way at the next intersection' sign is also classified with 99.978% confidence. The 6th image, which is a 'Roundabout mandatory' sign was also classified correct with a confidence of 99.7%.

The fifth image which is a 'Vehicles over 3.5 metric tons prohibite'- sign was only classified by the network as the 5th candidate with 0.0016% possibility, again due to the watermarks. In the 7th Image I have used the classifier on the same sign without watermark and the result was correct with 99.8% confidence.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.986550         			| Roundabout mandatory    									| 
| 0.009567   				| Priority road									|
| 0.001562				| Pedestrians											|
|  0.001562     			| General caution				 				|
| 0.001562		    |Vehicles over 3.5 metric tons prohibited     							| 





