# **Finding Lane Lines on the Road** 

## Writeup from Anja Severin

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I applied the gaussian blur to surpress noise. After that, I applied the canny transformation to detect edges with lower threshold of 200 and higher threshold of 300. Following that, I determined the region of interest using a quadrangle mask. Finally, applied hough transform to find the lanes. I set the minumum length to 20 and maximal gap to 10. 

In order to draw a single line on the left and right lanes, I have first classified the lines in two groups, one group presenting the lines of the left lane and vice versa. To do that I computed the gradient and classified the lines accordingly. To do the extrapolation of the lines, I implemented a function called extrapolateLine. In there I use the function polyfit to ding a polynominal of degree 1 (which is a line) for all the points to connect one line.


If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when we enter a curve entry and the lane becomes also curvy, the current extrapolation method using polinomial of degree one could result in larger error.

Also, if there are other type of lines entering our region of interest, for example dashed white lines signalising end of this lane/parking prohibitated, it will also have a large influence of the current pipeline


### 3. Suggest possible improvements to your pipeline


For the 1st shortcoming I mentioned, we can try to approximate with a polinomial of higher degree.
