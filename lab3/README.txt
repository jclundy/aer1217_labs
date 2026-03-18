The file we are submitting for lab 3 is a python notebook - you can see the interative steps in figuring out the target localization
which included calibrating the threshold value and the size of the contours. After testing if hough circles might work better, 
the code continues to test the rotation and transformation functions to bring the points from a pixel to a real world coordinate.
Then, the entire batch of images is iterated through and each target location in the world frame is stored in a csv file along with the frame
index of the corresponding image. The final step is to iterate through KMeans to remove any final outliers and average the locations of 
the 6 targets.

INPUTS - photo folder and pose .csv file locations to be updated in the first cell to match your INPUTS
OUTPUTS - csv file with all target locations, and the final 6 target locations will be printed after the last cell is run