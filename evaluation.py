#!/usr/bin/env python
# Author: roman.klinger@ims.uni-stuttgart.de
# Evaluation script for Empathy shared task at WASSA 2023
# Adapted for CodaLab purposes by Orphee (orphee.declercq@ugent.be) in May 2018
# Adapted for multiple subtasks by Valentin Barriere in December 2021 (python 3), then in February 2022

from __future__ import print_function
import sys
import os
from math import sqrt

to_round = 3

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

def readFileToList(filename):
	#eprint("Reading data from",filename)
	lines=filename.readlines()
	result=[]
	for x in lines:
		result.append(x.rstrip().split('\t'))
	filename.close()
	return result

def calculatePRF(gold,prediction):
	"""
	gold/prediction list of list of emo predictions 
	"""
	# initialize counters
	labels = set(gold+prediction)
	tp = dict.fromkeys(labels, 0.0)
	fp = dict.fromkeys(labels, 0.0)
	fn = dict.fromkeys(labels, 0.0)
	precision = dict.fromkeys(labels, 0.0)
	recall = dict.fromkeys(labels, 0.0)
	f = dict.fromkeys(labels, 0.0)
	# check every element
	for g,p in zip(gold,prediction):
		# TP 
		if (g == p):
			tp[g] += 1
		else:
			fp[p] += 1
			fn[g] += 1
   # print("Label\tTP\tFP\tFN\tP\tR\tF")
	for label in labels:
		recall[label] = 0.0 if (tp[label]+fn[label]) == 0.0 else (tp[label])/(tp[label]+fn[label])
		precision[label] = 1.0 if (tp[label]+fp[label]) == 0.0 else (tp[label])/(tp[label]+fp[label])
		f[label] = 0.0 if (precision[label]+recall[label])==0 else (2*precision[label]*recall[label])/(precision[label]+recall[label])
		microrecall = (sum(tp.values()))/(sum(tp.values())+sum(fn.values()))
		microprecision = (sum(tp.values()))/(sum(tp.values())+sum(fp.values()))
		microf = 0.0 if (microprecision+microrecall)==0 else (2*microprecision*microrecall)/(microprecision+microrecall)
	# Macro average
	macrorecall = sum(recall.values())/len(recall)
	macroprecision = sum(precision.values())/len(precision)
	macroF = sum(f.values())/len(f)

	accuracy = 0
	for label in labels:
		accuracy += tp[label]

	accuracy = accuracy/len(gold)

	return round(microrecall,to_round),round(microprecision,to_round),round(microf,to_round),round(macrorecall,to_round),round(macroprecision,to_round),round(macroF,to_round),round(accuracy,to_round)

def calculatePRF_MLabel(gold,prediction):
	"""
	gold/prediction list of list of emo predictions 
	"""
	# initialize counters
	# labels = set(gold+prediction)

	gold = [k.lower().split('/') for k in gold]
	prediction = [k.lower().split('/') for k in prediction]

	from sklearn.preprocessing import MultiLabelBinarizer
	from sklearn.metrics import precision_recall_fscore_support, jaccard_score
	mlb = MultiLabelBinarizer()
	mlb.fit(gold)

	y_true = mlb.transform(gold)
	y_pred = mlb.transform(prediction)

	microprecision, microrecall, microf, s = precision_recall_fscore_support(y_true, y_pred, average='micro')
	macroprecision, macrorecall, macroF, s = precision_recall_fscore_support(y_true, y_pred, average='macro')

	accuracy = jaccard_score(y_true, y_pred, average='micro')

	return round(microrecall,to_round),round(microprecision,to_round),round(microf,to_round),round(macrorecall,to_round),round(macroprecision,to_round),round(macroF,to_round),round(accuracy,to_round)


def pearsonr(x, y):
	"""
	Calculates a Pearson correlation coefficient. 
	"""

	assert len(x) == len(y), 'Prediction and gold standard does not have the same length...'

	xm = sum(x)/len(x)
	ym = sum(y)/len(y)

	xn = [k-xm for k in x]
	yn = [k-ym for k in y]

	r = 0 
	r_den_x = 0
	r_den_y = 0
	for xn_val, yn_val in zip(xn, yn):
		r += xn_val*yn_val
		r_den_x += xn_val*xn_val
		r_den_y += yn_val*yn_val

	r_den = sqrt(r_den_x*r_den_y)

	if r_den:
		r = r / r_den
	else:
		r = 0

	# Presumably, if abs(r) > 1, then it is only some small artifact of floating
	# point arithmetic.
	r = max(min(r, 1.0), -1.0)

	return round(r,to_round)

def calculate_pearson(gold, prediction):
	"""
	gold/prediction are a list of lists [ emp pred , distress pred ]
	"""

	# converting to float
	gold = [float(k) for k in gold]
	prediction = [float(k) for k in prediction]

	return pearsonr(gold, prediction)

def calculate_metrics(golds, predictions, task1, task2, task3, task4):
	"""
	gold/prediction list of list of values : [ emp pred , distress pred , emo pred ]
	"""
	if task1:
		gold_empathy = [k[0] for k in golds]
		prediction_empathy = [k[0] for k in predictions]
		pearson_empathy = calculate_pearson(gold_empathy, prediction_empathy)

		gold_distress = [k[1] for k in golds]
		prediction_distress = [k[1] for k in predictions]
		pearson_distress = calculate_pearson(gold_distress, prediction_distress)
		avg_pearson = (pearson_empathy + pearson_distress)/2
	else:
		avg_pearson, pearson_empathy, pearson_distress = 0,0,0

	if task2:
		gold_emo = [k[2] for k in golds]
		prediction_emo = [k[2] for k in predictions]

		microrecall,microprecision,microf,macrorecall,macroprecision,macroF,accuracy = calculatePRF_MLabel(gold_emo, prediction_emo)
	else:
		microrecall,microprecision,microf,macrorecall,macroprecision,macroF,accuracy = 0,0,0,0,0,0,0

	if task3:
		gold_per=[]
		prediction_per=[]
		pearson_per=[]
		for i in range(3, 8):
			gold_per.append([k[i] for k in golds])
			prediction_per.append([k[i] for k in predictions])
			pearson_per.append(calculate_pearson(gold_per[-1], prediction_per[-1]))
			
		avg_pearson_PER = sum(pearson_per)/len(pearson_per)
	else:
		avg_pearson_PER = 0

	if task4:
		gold_iri=[]
		prediction_iri=[]
		pearson_iri=[]
		for i in range(8, len(golds[0])):
			gold_iri.append([k[i] for k in golds])
			prediction_iri.append([k[i] for k in predictions])
			pearson_iri.append(calculate_pearson(gold_iri[-1], prediction_iri[-1]))
			
		avg_pearson_IRI = sum(pearson_iri)/len(pearson_iri)
	else:
		avg_pearson_IRI = 0

	return avg_pearson, pearson_empathy, pearson_distress, microrecall, microprecision, microf, macrorecall, macroprecision, macroF, accuracy, avg_pearson_PER, avg_pearson_IRI

def calculate_metrics_CONV(golds, predictions, task5):
	"""
	gold/prediction list of list of values : [ emp pred , distress pred , emo pred ]
	"""

	if task5:
		gold_CONV=[]
		prediction_CONV=[]
		pearson_CONV=[]
		for i in range(3):
			gold_CONV.append([k[i] for k in golds])
			prediction_CONV.append([k[i] for k in predictions])
			pearson_CONV.append(calculate_pearson(gold_CONV[-1], prediction_CONV[-1]))
	else:
		pearson_CONV = [0,0,0]

	avg_pearson_CONV = sum(pearson_CONV)/len(pearson_CONV)
	pearson_CONV_EMP, pearson_CONV_EMOP, pearson_CONV_EMOI = pearson_CONV

	return avg_pearson_CONV, pearson_CONV_EMP, pearson_CONV_EMOP, pearson_CONV_EMOI

def read_file(submission_path, nb_labels=2, nb_samp=10):
	"""
	Read the tsv file
	"""
	# unzipped submission data is always in the 'res' subdirectory
	if not os.path.exists(submission_path):
		print('Could not find submission file {0}'.format(submission_path))
		predictedList_EMP = [[0]*nb_labels]*nb_samp
		task1 = False
	else:
		submission_file = open(os.path.join(submission_path))
		# The 2 first columns
		predictedList_EMP = [k[:nb_labels] for k in readFileToList(submission_file)]
		task1 = True

	return task1, predictedList_EMP

nb_labels_EMP = 2
nb_labels_EMO = 1
nb_labels_PER = 5
nb_labels_IRI = 4
nb_labels_CONV = 3

def score(input_dir, output_dir):

	# unzipped reference data is always in the 'ref' subdirectory
	truth_file = open(os.path.join(input_dir, 'ref', 'goldstandard.tsv'))
	goldList = readFileToList(truth_file)
	nb_samp = len(goldList)

	truth_file_CONV = open(os.path.join(input_dir, 'ref', 'goldstandard_CONV.tsv'))
	goldList_CONV = readFileToList(truth_file_CONV)
	nb_samp_CONV = len(goldList_CONV)

	submission_path = os.path.join(input_dir, 'res', 'predictions_EMP.tsv')
	task1, predictedList_EMP = read_file(submission_path=submission_path, nb_labels=nb_labels_EMP, nb_samp=nb_samp)

	submission_path = os.path.join(input_dir, 'res', 'predictions_EMO.tsv')
	task2, predictedList_EMO = read_file(submission_path=submission_path, nb_labels=nb_labels_EMO, nb_samp=nb_samp)
	if goldList[0][2] == 'Nolabel': task2=False

	submission_path = os.path.join(input_dir, 'res', 'predictions_PER.tsv')
	task3, predictedList_PER = read_file(submission_path=submission_path, nb_labels=nb_labels_PER, nb_samp=nb_samp)

	submission_path = os.path.join(input_dir, 'res', 'predictions_IRI.tsv')
	task4, predictedList_IRI = read_file(submission_path=submission_path, nb_labels=nb_labels_IRI, nb_samp=nb_samp)

	submission_path = os.path.join(input_dir, 'res', 'predictions_CONV.tsv')
	task5, predictedList_CONV = read_file(submission_path=submission_path, nb_labels=nb_labels_CONV, nb_samp=nb_samp_CONV)

	predictedList = [i+j+k+l for i,j,k,l in zip(predictedList_EMP, predictedList_EMO, predictedList_PER, predictedList_IRI)]

	if (len(goldList) != len(predictedList)):
		eprint("Number of labels is not aligned!")
		sys.exit(1)

	if task5 and (len(goldList_CONV) != len(predictedList_CONV)):
		eprint("Number of labels for CONV is not aligned!")
		sys.exit(1)

	avg_pearson, pearson_empathy, pearson_distress, micror, microp, microf, macror, macrop, macrof, accuracy, avg_pearson_PER, avg_pearson_IRI = calculate_metrics(goldList,predictedList, task1, task2, task3, task4)

	avg_pearson_CONV, pearson_CONV_EMP, pearson_CONV_EMOP, pearson_CONV_EMOI = calculate_metrics_CONV(goldList_CONV,predictedList_CONV, task5)

	with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
		str_to_write = ''
		# Not sure of that. Useful if the participant want to do only one subtask. Need to see if the leaderboard of the subtask does not update if there are nothing on score.txt 
		if task1:
			str_to_write += "Averaged Pearson Correlations: {0}\nEmpathy Pearson Correlation: {1}\nDistress Pearson Correlation: {2}\n".format(avg_pearson, pearson_empathy, pearson_distress)
		if task2:
			str_to_write += "Macro F1-Score: {5}\nMicro Recall: {0}\nMicro Precision: {1}\nMicro F1-Score: {2}\nMacro Recall: {3}\nMacro Precision: {4}\nMicro Jaccard: {6}\n".format(micror,microp,microf,macror,macrop,macrof,accuracy) 
		if task3:
			str_to_write += "PER Pearson Correlations: {0}\n".format(avg_pearson_PER)
		if task4:
			str_to_write += "IRI Pearson Correlations: {0}\n".format(avg_pearson_IRI)		
		if task5:
			str_to_write += "Conversation Pearson Correlations: {0}\nConversation Empathy Pearson Correlation: {1}\nConversation Emotional Polarity Pearson Correlation: {2}\nConversation Emotional Intensity Pearson Correlation: {3}\n".format(avg_pearson_CONV,pearson_CONV_EMP,pearson_CONV_EMOP,pearson_CONV_EMOI)
		output_file.write(str_to_write)

def main():
	[_, input_dir, output_dir] = sys.argv
	score(input_dir, output_dir)

if __name__ == '__main__':
	main()