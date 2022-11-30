# -*- coding: utf-8 -*-
"""
@author: Sandesh Jain
Organization: Virginia Tech Transportation Institute
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import collections
import pandas as pd
import seaborn as sns



REAL_BOOL = False
SHAPE = (1920, 1080)#(480, 356)
REJECT = ['airplane', 'train', 'cat', 'kite', 'sink', 'cow', 
'horse', 'toilet', 'surfboard', 'boat', 'vase', 
'dog', 'sheep', 'N/A', 'bird', 'chair', 'skateboard', 'tv']

""" Including the driver gaze outside the FOV
    Zone is now divided into 5 parts: Left OFOV, Right OFOV:
	if else into 5 sections and assign accordingly for all the event relevant 
	frames. 

"""


if REAL_BOOL:
	ZONE_SEQ = ['Left outside FOV', 'Left', 'Central', 'Right', 'Right outside FOV']
else:
	ZONE_SEQ = ['Left', 'Central', 'Right']

EVENT_DICT = {1: 'Calibration',
		2: 'Signalized intersection - left turn',
		3: 'Signalized intersection - right turn',
		4: 'Stop controlled intersection - left turn',
		5: 'Stop controlled intersection - right turn',
		6: 'Merge',
		7: 'Speed check',
		8:'Proximate vehicles',
		9: 'Radio on or off',
		10: 'Simulated cell phone',
		11: 'Looking forward',
		12: 'Other intersection',
		13: 'Other merge',
		14: 'Lane change'}
OUT_DATA_OBJ = {EVENT_DICT[key]: [] for key in EVENT_DICT}

OUT_DATA_SEC = {EVENT_DICT[key]: [] for key in EVENT_DICT}



def parse_args():
	parser = argparse.ArgumentParser(description='Visual')
	parser.add_argument('--out_dir', type=str, default='./Hist_obj/', help='Output directory')
	parser.add_argument('--epochs_path', type=str, default='./maskDynamicVehicleAllEpochsV1.csv', help='All epochs directory')
	parser.add_argument('--mapped_path', type=str, default='./dataset.csv', help='All videos mapped one-one to all the epochs')
	parser.add_argument('--detections_dir', type=str, default='./Negative Angles/hpv/', help='All video detections directory')
	args = parser.parse_args()
	return args

""" 
steps for processing:
	1.) Get the bbox csvs set
	2.) Get the epoch csv file
	3.) Get the mapped data set 
	
	For each event pass through all the videos, keep doing scan frames with 
	latest data_points. !!!Make outo and outs as global for this to agg or not.
	
	Address the non existance of a frame number: 
		sub_data_row, col = np.where(data_points == 15541) 
		a = data_points[:, 0]
		a[np.abs(a-a0).argmin())]
		
	
	Build a dictionary of event keywords > nest the video/csv > nest their 
	start_stop lists 
	
	
	# 14.98 fps
	
	All categories:
		
		Code: Value
		
		1: Calibration
	*	2: Signalized intersection - left turn
	*	3: Signalized intersection - right turn
	*	4: Stop controlled intersection - left turn
	*	5: Stop controlled intersection - right turn
	*	6: Merge
		7: Speed check
	*	8: Proximate vehicles
		9: Radio on / off
		10: Simulated cell phone
		11: Looking forward
		12: Other intersection
		13: Other merge
	*	14: Lane change


"""

def extract_info(subd, sh, out_data_obj, out_data_sec, keyword):
	for data in subd:
		if REAL_BOOL:
			frame, px, py, obj = data
		else:
			frame, px, py, obj, _ = data
		if px<0:
			sector = 'Left outside FOV'
		elif px>=0 and px<sh[1]/3:
			sector = 'Left'
		elif px>= sh[1]/3 and px<2*sh[1]/3:
			sector ='Central'
		elif px>=2*sh[1]/3 and px<sh[1]:
			sector = 'Right'
		else:
			sector = 'Right outside FOV'
		out_data_obj[keyword].append(obj)
		out_data_sec[keyword].append(sector)
	return out_data_obj, out_data_sec

def save_hist(out_data_obj, out_data_sec, keyword, out_dir = './Histograms/'):
	# airplane, train, cat, kite, sink, cow, 
	# horse, toilet, surfboard, boat, vase, 
	# dog, sheep
	reject = REJECT
	obj_list = list(filter(('N/A').__ne__, out_data_obj[keyword]))
	#[i for i in lst_fcflds if i not in RROPFields and i not in ["OBJECTID","SHAPE"]]
	obj_list = [i for i in out_data_obj[keyword] if i not in  reject]
	if obj_list:
		plt.figure()
		o = pd.Series(sorted(obj_list)).value_counts(sort=True).plot(kind='bar', color = 'black')
		axo = o
		
		axo.set_xlabel("Salient instance")
		axo.set_ylabel("Count")
		axo.set_title(keyword)
		axo.figure.savefig(out_dir+keyword+'_obj.jpg', bbox_inches='tight')
		
		s = pd.Series(out_data_sec[keyword]).value_counts(sort=False).loc[ZONE_SEQ].plot(kind='bar', color = 'black')
		plt.figure()
		axs = s
		
		axs.set_xlabel("Zone of gaze")
		axs.set_ylabel("Count")
		axs.set_title(keyword)
		axs.figure.savefig(out_dir+keyword+'_sec.jpg', bbox_inches='tight')
	





def scan_frames_hist(start_stop_list, keyword, tmp_data_points, outo, outs):
	for tups in start_stop_list:
		a0 = tups[0] #-100
		a1 = tups[1] #+100
		a = tmp_data_points[:, 0]
		a0= a[np.abs(a-a0).argmin()]
		a1= a[np.abs(a-a1).argmin()]
		sub_data_row, _ = np.where(tmp_data_points == a0)
		sub_data_row_end, _ = np.where(tmp_data_points == a1)
		
		sub_d=tmp_data_points[sub_data_row[0]:sub_data_row_end[0]]
		#fname = './CHPV_0000_0000_10_130218_1924_00088_Front/' + str(int(1)).zfill(7) + '.jpg'
		#im = plt.imread(fname)
		sh = SHAPE
		if outo:
			outo, outs = extract_info(sub_d, sh, outo, outs, keyword)
		else:
			outo, outs = extract_info(sub_d, sh, OUT_DATA_OBJ, OUT_DATA_SEC, keyword)
	#save_hist(outo, outs, keyword)
	return outo, outs


###	example: see this below

# start_stop_list = [(11746, 11781), (17167, 17329), (17626, 17687)]


# scan_frames_hist(start_stop_list, 'stop_sign')

#### Data aggregation



def build_aggregator(epoch_csv, mapped_data_csv, real_bool):
	# for all maps in mapped data take the csv path for the 
	# bbox and add to a list
	mapped_df = pd.read_csv(mapped_data_csv, header=None)
	if real_bool:
		mapped_df[:][1] = 'dgf_' + mapped_df[:][1][:-4] + '_real_bbox.csv'
	else:
		mapped_df[:][1] = 'dgf_' + mapped_df[:][1][:-4] + '_adjusted_bbox.csv'
	epochs_pd = pd.read_csv(epoch_csv)
	
	subset_trails = set(mapped_df[:][0])
	epoch_trial_dict = {x: collections.defaultdict(list) for x in range(1, 15)}
	# e.g., epoch_trial_dict = {2: {203: [(100,125), (255,400), (1000,1200)]}
	for epochs in range(len(epochs_pd)):
		if epochs_pd['trialID'][epochs] in subset_trails:
			try: 
				epoch_trial_dict[epochs_pd['epochType'][epochs]][epochs_pd['trialID'][epochs]].append((epochs_pd['epochBeginFrame'][epochs], epochs_pd['epochEndFrame'][epochs]))
			except KeyError:
				epoch_trial_dict[epochs_pd['epochType'][epochs]] = collections.defaultdict(list)
				epoch_trial_dict[epochs_pd['epochType'][epochs]][epochs_pd['trialID'][epochs]].append((epochs_pd['epochBeginFrame'][epochs], epochs_pd['epochEndFrame'][epochs]))
	return epoch_trial_dict



def save_big_hists(set_of_data, mapped_data_csv, data_path, real_bool):
	mapped_df = pd.read_csv(mapped_data_csv, header=None)
	inito, inits = 0,0
	for event in EVENT_DICT:
		print("In event: ", EVENT_DICT[event])
		
		for trial_id in set_of_data[event]:
			print("In trial: ", trial_id)
			trial_row, trial_col = np.where(mapped_df == trial_id) 
			if real_bool:
				data_points = np.genfromtxt(data_path + 'dgf_' + mapped_df[1][trial_row[0]][:-4] + '_real_bbox.csv', delimiter=',', dtype='unicode')
			else:
				data_points = np.genfromtxt(data_path + 'dgf_' + mapped_df[1][trial_row[0]][:-4] + '_adjusted_bbox.csv', delimiter=',', dtype='unicode')
			data_points = data_points.astype('object')
			data_points[:,[0,1,2]] = data_points[:,[0,1,2]].astype(np.int32)
			
			if not inito:
				outo, outs = scan_frames_hist(set_of_data[event][trial_id], EVENT_DICT[event], data_points, inito, inits)
				inito = 1
			else: 
				outo, outs = scan_frames_hist(set_of_data[event][trial_id], EVENT_DICT[event], data_points,outo, outs)
		
		#save_hist(outo, outs, EVENT_DICT[event])
	return outo, outs


# Signalized intersection - right turn
# Signalized intersection - left turn
# Stop controlled intersection - left turn
# Stop controlled intersection - right turn
# Merge
# Speed check
# Proximate vehicles
# Radio on or off
# Simulated cell phone
# Looking forward
# Other intersection
# Other merge
# Lane change





def main(args):
	sns.set(rc={'axes.facecolor':'white'})
	reject = REJECT
	#keyword = 'Merge'
	
	set_of_data = build_aggregator(args.epochs_path,
								   args.mapped_path, REAL_BOOL)
	
	OUTO, OUTS = save_big_hists(set_of_data, args.mapped_path, args.detections_dir, REAL_BOOL)
	for keyword in OUTO:
	 	#if keyword == "Lane change":
			obj_list = [i for i in OUTO[keyword] if i not in  reject]
			print(keyword)
			print("Filtered object list size: ", len(obj_list))
			print("Raw obj list size: ", len(OUTO[keyword]))
			o = sorted(obj_list) 
			if o:
	 			dfo = pd.Series(sorted(obj_list)).value_counts(sort=True)
	 			dfo_mod = pd.DataFrame(columns=["Salient Objects", "Probability"])
	 			sum_counts = 0
	 			for val, idx in zip(dfo, dfo.index):
	 				 dfo_mod.loc[len(dfo_mod)] = [idx, val]
	 				 sum_counts+=val
	 			dfo_mod["Probability"] = dfo_mod["Probability"]/sum_counts
	 			dfo_mod.sort_values("Probability")
	 			sns.set(style="white")
	 			plt.figure()
	 			ax = sns.barplot(x = "Salient Objects",y = "Probability" , data=dfo_mod, color = 'gray')
	 			ax.set_title(keyword)
	 			ax.figure.savefig(args.out_dir+ keyword+'_obj.jpg', bbox_inches='tight')
	
	
	for keyword in OUTS:
	 	#if keyword == "Lane change":
			obj_list = OUTS[keyword]
			print(keyword)
			print("Raw zone list size: ", len(OUTS[keyword]))
			o = sorted(obj_list) 
			if o:
	 			dfo = pd.Series((obj_list)).value_counts(sort=True).reindex(ZONE_SEQ)
	 			dfo = dfo.fillna(0)
	 			dfo_mod = pd.DataFrame(columns=["Zone of View", "Probability"])
	 			sum_counts = 0
	 			for val, idx in zip(dfo, dfo.index):
	 				 dfo_mod.loc[len(dfo_mod)] = [idx, val]
	 				 sum_counts+=val
	 			dfo_mod["Probability"] = dfo_mod["Probability"]/sum_counts
	 			#dfo_mod.sort_values("Probability")
	 			sns.set(style="white")
	 			plt.figure()
	 			ax = sns.barplot(x = "Zone of View",y = "Probability" , data=dfo_mod, color = 'gray')
	 			ax.set_ylim(0,1.0)
	 			
	 			ax.set_title(keyword)
	 			ax.figure.savefig(args.out_dir+ keyword+'_sec.jpg', bbox_inches='tight')
	 	


if __name__ == '__main__':

	args = parse_args()
	main(args)
