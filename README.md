# Event-Correlation
Inferencing for external driving events such as a lane change based on gaze estimation and object detection input using a dash cam and a face camera. 

## Required Files: Event Correlation

### 1.) All Epochs File
	Format: CSV with the given columns
	
	trialID	epochID	epochType	timeOfDay	baseballCap	glasses	epochBeginTimestamp	epochEndTimestamp	epochBeginFrame	epochEndFrame
  	

### 2.) All processed CSV Files Dir

	Real Values CSV:

	e.g., dgf_CMVP_0000_0000_10_130204_1926_00031_Face_real_bbox.csv
  
	Frame#		X-value for Gaze	Y-value for Gaze	Object Name

	Adjusted to Windshield Camera CSV:
	
	e.g., dgf_l2cs_CMVP_0000_0000_10_130218_1429_00079_Face_adjusted_bbox.csv
	
	Frame#	X-value for Gaze	Y-value for Gaze	Object Name
  
### 3.) Mapping CSV file for Epochs to processed 
	
	Format: dataset.csv
	
	trialID		CSV File Name (Detections and Gaze)
  

### 4.) Output directory for histogram images

For a demo, use the pre-existing folder on VTTI server: vtti/projects03/451600/Working/FromSandesh/Event Correlation Folders/

For a smaller sample set use the one provided with the repository.

### 5.) How to run

Installation: 

```
pip3 install -r requirements.txt
```


For custom folders use the arguments

```
python3 event_correlation_hist.py --out_dir='' --epochs_path='' --mapped_path='' --detections_dir=''
```


