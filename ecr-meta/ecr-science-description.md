The plugin uses the optical flow method to compute cloud motion vectors from the Sage sky camera images.

# Science
The plugin uses the optical flow method to analyze the motion of clouds in images taken by the Sage sky camera. This information can be used to understand local weather conditions and predict short-term changes in solar irradiance. When combined with other plugins such as Cloud Cover and Weather Classification, the output of this plugin can be used to classify and track the movement of different types of clouds at various altitudes, which can be useful for meteorological studies [1].

The Farnebäck method is one approach to estimating the optical flow between two images. It assumes that the flow of pixels between two images is approximately constant within a local neighborhood [2]. The method calculates the optical flow by minimizing the sum of the squared differences between intensity values in the two images, subject to the constraint that the flow is approximately constant within a local neighborhood. The Farnebäck method is fast and simple, but it can be sensitive to noise and may not provide accurate results in the presence of large motion or changes in illumination.
# AI@Edge
The plugin uses the OpenCV calcOpticalFlowFarneback method to compute dense optical flow vectors from a pair of images taken at a user-specified time interval. The method returns an array of 2-channel floating-point images that has the same size as the input images, and each pixel of the output array stores the computed optical flow for the corresponding pixel of the input images. The optical flow is represented as a 2D vector, with the x and y components representing the flow along the x and y axes, respectively.

Currently, the plugin reads images from the sky-facing camera every 30 seconds and uses only the "Red" channel image. The mean direction and velocity of the cloud motion vectors are calculated and the results are published. The user can set the quality of the vector field, the time interval, and the RGB channels used for the computation.

# Using the code
Sensor needed: Sky Camera  
Output: Direction and Velocity of mean cloud motion.  
Input: Pair of images (30-60 sec interval)  
Image resolution: Variable.  
Inference time: 0.5 Sec  
Model loading time: NA  

# Arguments
The plugin has several arguments that can be set by the user:

   '--input'     Path to input file or camera URL.  
   '--i'             Time interval in seconds (Default 30 sec).  
   '--c'            RGB channels, 0=R, 1=G, 2=B (Default 0 i.e. Red).
   '--k'            Keep fraction of the image after cropping (Default=0.9).  
   '--q'            Quality of the motion field (Default 1). Sets averaging window, poly_n and poly_sigma. 1-turbulant: detailed motion field but noisy. 2-smooth: lesser noise and fast computation,  


# Ontology
The plugin will publish mean direction and and mean velocity of the cloud motion (cmv.mean.dir, cmv.mean.vel).

# Inference from Sage codes
To query the output from the plugin, you can do with python library 'sage_data_client':
```
import sage_data_client

# query and load data into pandas data frame
df = sage_data_client.query(
    start="-1h",
    filter={
        "name": "cmv.mean.dir",
    }
)

# print results in data frame
print(df)
```
For more information, please see [Access and use data documentation](https://sagecontinuum.org/docs/tutorials/accessing-data) and [sage_data_client](https://pypi.org/project/sage-data-client/).

# References
[1] Raut, B. A., Collis, S., Ferrier, N., Muradyan, P., Sankaran, R., Jackson, R., ... & Beckman, P. "Phase correlation on the edge for estimating cloud motion". Atmospheric Measurement Techniques Discussions, 2022, 1-18.

[2] Farnebäck, Gunnar. "Two-frame motion estimation based on polynomial expansion." _Scandinavian conference on Image analysis_. Springer, Berlin, Heidelberg, 2003.

# Credits
ecr-icon.jpg  Wheat Field with Cypresses, Vincent van Gogh (Dutch), 1889.  
https://www.metmuseum.org/art/collection/search/436535  
https://www.metmuseum.org/about-the-met/policies-and-documents/image-resources  


