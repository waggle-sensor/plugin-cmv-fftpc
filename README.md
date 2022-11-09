# Cloud Motion Detector
The plugin uses the phase correlation method to compute cloud motion vectors from the Waggle sky camera images.

## Science
Converting still cloud images into a time series of motion vectors has implications in reporting local weather conditions and short-term forecasting of solar irradiance. In association with the other plugins namely Cloud Cover and Weather Classification, the output of the plugin will help in further classification of the cloud types and their movements with heights that can be used for meteorological studies.  
Phase correlation (PC) is a computationally efficient method for estimating shifts in the cloud patterns in the subsequent images and it can be implemented without a preprocessing of the image. Therefore, it is a natural choice for estimating the cloud motion from the hemispheric camera images on the edge computing platform. The PC method is robust in presence of noise and changes in illumination, and its output is less sensitive to the choice of RGB/Gray channels. The application uses the block-based PC approach to estimate the motion [1] and the neighborhood median fluctuation method to remove the spurious vectors [2]. Sensitivity analysis showed that averaging the motion vectors computed at a shorter time interval yields stable output even when the larger block size is used [3]. 

## AI@Edge
The application works with a pair of images at the user-provided time interval. Currently, the application reads images from the sky-facing camera every 30 seconds and uses the "Red" channel image only. The sky region is divided into a 10 x 10 grid of 100 blocks. The mean of cloud motion vectors is computed and output is published. The user can set the grid size, time interval as well as select RGB channels for the computation.

## Using the code
Sensor needed: Sky Camera  
Output: U and V components of mean cloud motion.  
Input: Pair of images (30-60 sec interval)  
Image resolution: 400x400 sky region expected. This can be changed.  
Inference time: 0.5 Sec  
Model loading time: NA  

## Arguments
   '--input'     Path to input file or camera URL.  
   '--i'             Time interval in seconds (Default 30 sec).  
   '--k'            k x k image blocks used for CMV computation (Default k=20).   
   '--c'            RGB channels, 0=R, 1=G, 2=B (Default 0 i.e. Red).

## Funding
https://www.nsf.gov/awardsearch/showAward?AWD_ID=1935984

## Collaborators
Dario Dematties Reyes,
Joseph Swantek,
Neal Conrad,
Nicola Ferrier,
Pete Beckman,
Raj Sankaran,
Robert Jackson,
Scott Collis,
Sean Shahkarami,
Seongha Park,
Sergey Shemyakin,
Wolfgang Gerlach,
Yongho kim

# References
[1] J. A. Leese, C. S. Novak, and B. B. Clark, “An automated technique for obtaining cloud motion from geosynchronous satellite data using cross-correlation,” J. Appl. Meteor., vol. 10, no. 1, pp. 118–132, 1971.  
[2] J. Westerweel and F.Scarano, “Universal outlier detection for PIV data,” Experiments in Fluids, vol. 39, no. 6, pp. 1096–1100, 2005. 
[3] Raut, B. A., Collis, S., Ferrier, N., Muradyan, P., Sankaran, R., Jackson, R., ... & Beckman, P., "Phase correlation on the edge for estimating cloud motion". Atmospheric Measurement Techniques Discussions, 1-18, 2022.i

## Credits
ecr-icon.jpg  Wheat Field with Cypresses, Vincent van Gogh (Dutch), 1889.  
https://www.metmuseum.org/art/collection/search/436535  
https://www.metmuseum.org/about-the-met/policies-and-documents/image-resources  
