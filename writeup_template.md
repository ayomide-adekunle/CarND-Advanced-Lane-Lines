## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/combined_binary.png "Binary Example"
[image4]: ./output_images/perspective_transform.png "Warp Example"
[image5]: ./output_images/color_fit_line.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell 6 in `result.ipynb`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `birdseye_view(img, src, dst, image_size)`, which appears 2nd code cell iof my IPython notebook .  The `birdseye_view()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points, and image size as seen above.  I chose the hardcode the source and destination points in the following manner:

```python
def gen_warp_points(image):
    # Get image shape/size
    imshape = image.shape
    
    
    corners = np.float32([[imshape[1]*0.198, imshape[0]*.968], [imshape[1]*0.457,\
                                imshape[0]*0.635], [imshape[1]*0.547, imshape[0]*0.6333],\
                          [imshape[1]*.829,imshape[0]*.958]])


    top_left = np.array([corners[0, 0], 0])
    top_right = np.array([corners[3, 0], 0])
    offset = [60, 0]
    
    
    
    # Map source points and destination points into new array
    src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_points = np.float32([corners[0] + offset, top_left + offset, top_right - offset, corners[3] - offset])

    
    return src_points, dst_points

        # Get warp points from image
    image = cv2.imread("../test_images/straight_lines2.jpg")
    src, dst = gen_warp_points(image)

    # Make a copy of the image to draw source and destination points on
    image_src_locations = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_dst_locations = image_src_locations.copy()

    # Make a polygon based on source points and draw it on the image
    src_pts = src.reshape((-1, 1, 2)).astype("int32")
    cv2.polylines(image_src_locations, [src_pts], True, (255, 0, 0), thickness=5)

    # Plot source image
    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_src_locations)
    plt.title("Source Points")

    # Make a polygon based on destination points and draw it on the image
    dst_pts = dst.reshape((-1, 1, 2)).astype("int32")
    cv2.polylines(image_dst_locations, [dst_pts], True, (0, 0, 255), thickness=15)

    # Plot destination image
    plt.subplot(1, 2, 2)
    plt.imshow(image_dst_locations)
    plt.title("Destination Points")

    # Save image
    plt.savefig('../output_images/perspective_transform.png')
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In my find_lane_pixels() function, I first used a histogram to identify where the lane lines could be. The histogram will give the left and the right lane positons. 

After this, I then set the number of sliding windows that shows where in the image the lane lines are. I slide the windows over the indices and fit these indices.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in code cell 2 with function measure_curvature_pixels(). I first convert the x and y pixel space to meters which ia applicable in the real world. I then fix a second order polynomial to pixel position in each lane lines. Finally, I used the code snippet bellow to calculate the right and left curvatures.

 ```
#calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
  ```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the code cell near the end of the jupyter notebook.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [https://youtu.be/nPkb5-6cxDs](.output_video/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The major tasks in the project is to tune the mnany parameters in the code to be able to detect and track lane properly. I try as much as possible to use parameters that will work well for all images with different resolution, so I use the image sizes to set some of my parameters.

The improvement I plan for this algoithm is to make it perfect in shaded areas.
