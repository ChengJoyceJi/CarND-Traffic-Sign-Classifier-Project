# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram1]: ./output_images/histogram1.png
[all_classes]: ./output_images/all_classes.png
[grey_scale_and_normalization]: ./output_images/grey_scale_and_normalization.png
[rotate_image]: ./output_images/rotate_image.png
[histogram2]: ./output_images/histogram2.png
[new_images.png]: ./output_images/new_images.png

[Rubric Points](https://review.udacity.com/#!/rubrics/481/view)

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of all the labels (classes) with class id and traffic sign names.
![alt text][all_classes]

Here is an exploratory visualization of the data set. It is a histogram showing how the data distributes across the 43 classes.

![alt text][histogram1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the LeNet takes a greyscale and the input image depth does not affect the rest of CNN architecture in LeNet. 

Then I normalized the image data by transforming the pixels to a number between 0 and 1 because it makes it easier for the optimizer to proceed numerically.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][grey_scale_and_normalization]

I decided to generate additional data because according to the histogram of the distribution of the 43 labels in the training set, there are around 2000 data points for some labels while there are less than 500 data points for some other labels. This big difference may cause that our neural network learns to predict those highly frequent classes to achieve higher overall accuracy, while not picking up the features that we want it to learn.

To add more data to the the data set, I rotate the less frequent images, and append the new image and its label to the training set (i.e. `X_train` and `y_train` variable) until at least 1000 data points fall into this label. The rotation degree is between -20 degrees to 20 degrees because usually the traffic sign would not be very skewed.

Here is an example of an original image and an augmented image:

![alt text][rotate_image]

Here's a visualization of the distribution of labels in the training sets with new data added:

![alt text][histogram2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   		     		| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten       		| output 400                         			|
| Fully Connected		| output 250        							|
| RELU					|												|
| Dropout   			|												|
| Fully Connected		| output 120        							|
| RELU					|												|
| Dropout   			|												|
| Fully Connected		| output 43        			    				|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I pass the inputs to the LeNet network to get the logits. 
Then applied a cross-entropy function as the loss function to compare the difference between the logits and the ground truth, and averaged the cross-entropy from all training images. 
Then I used Adam optimizer to minimize the loss function.
Finally use the minimize function on the optimizer which uses backpropgation to update the network and minimize the training loss.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.2%
* test set accuracy of 93.6%

Initially I started with the LeNet model introduced in class because it does pretty well in image classification problems.

Three most important changes that boosted my model accuracy were
- Normalization. Normalizing the image data by transforming the pixels to a number between 0 and 1 made it a lot easier for the optimizer to proceed numerically.
- Generate additional data by rotating images. Since there's a large variance in the number of images in the data set between each classes (~2000 vs. ~300), our model might tend to predict the more frequent classes to gain high validation accuracy, which is not ideal. So by rotating images by -20 degrees to 20 degrees, I generate more data points for less frequent classes and guaranteed each class has at least 1000 data points. 
- Dropout. My validation accuracy is higher than test accuracy, which means there may exist over-fitting, so I applied dropout after the activation function in each fully connected layer to prevent over-fitting.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 7 German traffic signs that I found on the web:

![alt text][new_images]

I manually re-sized the image to be 32*32*3 so that it fits in my neural net, but it reduced the resolution a lot.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (60km/h)	| Speed limit (60km/h)  	        			| 
| Keep right			| Keep right    								|
| Right-of-way at the next intersection | Right-of-way at the next intersection	|
| Roundabout mandatory  | Roundabout mandatory           				|
| Stop                  | Stop          			          			|
| General caution    	| General caution    							|
| Speed limit (70km/h)	| Speed limit (30km/h)  	        			| 


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 85.7%. This compares unfavorably to the accuracy on the test set of 93.6% because the sample size of the new image set is too small, so getting one prediction wrong makes the accuracy lower than the test set accuracy.

The model predicted the speed limit sign of 70km/h wrong. It does predict it as a speed limit sign based on its shape and having numbers inside he circle. However, it gets the number wrong (30 instead of 70). It may be hard to accurately predict the speed limit because of the reduced pixels in the image.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook.

For the third image, the model is relatively sure that this is a right-of-way at the next intersection sign (probability of 0.51), which is correct.

For the last image, the model thinks it has 0.18 probability of being a speed limit (30km/h) sign, and 0.11 probability of being a speed limit (70km/h) sign, while in reality it's a 70km/h sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.15               	| Speed limit (60km/h)  	        			| 
| 0.28               	| Keep right    								|
| 0.51               	| Right-of-way at the next intersection      	|
| 0.26               	| Roundabout mandatory           				|
| 0.15               	| Stop          			          			|
| 0.48               	| General caution    							|
| 0.18               	| Speed limit (30km/h)  	        			|

