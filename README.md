# Kinect-Camera-Code
 Code and instructions for using the Kinect for facial recognition. The code recognizes faces and takes reference and target images to be sent to the facial expression model.  

## Setting up the Kinect  
To set up the Kinect to be used as a webcam there are a couple of drivers that need to be installed.  
Make sure to read the system requirements for each of these drivers. If the requirements are not met the kinect may not be recognized on your system.  

1. [Install Kinect for Windows Runtime 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44559)
2. [Install Kinect for Windows SDK](https://www.microsoft.com/en-us/download/details.aspx?id=44561)
3. Reboot Computer

## Script controls
When running pressing 'q' on your keyboard closes the script.  

## Details about the kinect_dnn script
- The script is <em>not</em> limited to only be used with the Kinect. It captures video using OpenCV's capture method, so anything that can act like a webcam on a computer is able to be used with this script.
- The script runs using OpenCV's dnn library. The model used for facial recognition is a pre trained caffemodel.  
- When the script is ran it will create a local folder name **images_sets** that stores all of the reference and target images. <em>This can be changed later to have them sent to a Google Drive or some sort of online storage.</em>
- The script operates on 2 threads. The first thread is used to display a video feed of the facial recognition. The second is to capture the reference png and 5 seconds later capture the target png. The capture thread infinetely repeats untill the video feed is closed.  
- The pngs are stored in the following file format: **Set**\_**Type**.png. 
Ex:
1_refernce.png
1_target.png
2_reference.png
2_target.png 
  
<em>Currently the script has trouble with multiple faces on screen. Looking for a solution.</em>  



