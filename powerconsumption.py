from __future__ import print_function

import glob
import math
import os
import csv
import time
import paho.mqtt.client as mqtt
from math import sqrt
from numpy import split
from numpy import array
import pandas as pd
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
import tensorflow as tf
from numpy import array
import numpy


# split a univariate dataset into train/test sets
# def split_dataset(dataset):
# 	# split into standard weeks
# 	train_size = int(len(dataset) * 0.67)
# 	train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# 	return train, test

# evaluate one or more weekly forecasts against expected values
# def evaluate_forecasts(actual, predicted):
# 	scores = list()
# 	# calculate an RMSE score for each day
# 	for i in range(actual.shape[0]):
# 		# calculate mse
# 		mse = mean_squared_error(actual[i], predicted[i])
# 		# calculate rmse
# 		rmse = sqrt(mse)
# 		# store
# 		scores.append(rmse)
# 	# calculate overall RMSE
# 	s = 0
# 	for row in range(actual.shape[0]):
# 		for col in range(actual.shape[1]):
# 			s += (actual[row, col] - predicted[row, col]) ** 2
# 	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
# 	return score, scores


# summarize scores
# def summarize_scores(name, score, scores):
# 	s_scores = ', '.join(['%.1f' % s for s in scores])
# 	print('%s: [%.3f] %s' % (name, score, s_scores))


# convert history into inputs and outputs
def to_supervised(train, n_input, n_out=1):
	# flatten data
	data = train.reshape((train.shape[0] * train.shape[1], 1))
	X, y = list(), list()
	in_start = 0
	# step over the entire history one time step at a time
	for _ in range(len(data)):
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
		# ensure we have enough data for this instance
		if out_end < len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return array(X), array(y)


# train the model
def build_model(train, n_input):
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
	# define parameters
	verbose, epochs, batch_size = 0, 50, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], 1, train_y.shape[1]
	# reshape output into [samples, timesteps, features]
	train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
	# define model
	model = Sequential()
	model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(RepeatVector(n_outputs))
	model.add(LSTM(200, activation='relu', return_sequences=True))
	model.add(TimeDistributed(Dense(100, activation='relu')))
	model.add(TimeDistributed(Dense(1)))
	model.compile(loss='mse', optimizer='adam')
	# fit network
	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
	return model


# make a forecast
def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0] * data.shape[1], 1))
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
	# forecast the next week
	yhat = model.predict(input_x, verbose=0)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat

def forecast_month(model, history, n_input):
	next_month_val = 0
        a = len(history[0])
	for _ in range(30 - a):
		predict_day = forecast(model, history, n_input)
		next_month_val  += predict_day
		history = numpy.append(history, predict_day)
		history = [array(history)]		
	return next_month_val


# evaluate a single model
# def evaluate_model(train, test, n_input):
# 	# fit model
# 	model = build_model(train, n_input)
# 	# history is a list of weekly data
# 	history = [x for x in train]
# 	# walk-forward validation over each week
# 	predictions = list()
# 	for i in range(len(test)):
# 		# predict the week
# 		yhat_sequence = forecast(model, history, n_input)
# 		# store the predictions
# 		predictions.append(yhat_sequence)
# 		# get real observation and add to history for predicting the next week
# 		history.append(test[i, :])
# 	# evaluate predictions days for each week
# 	predictions = array(predictions)
# 	score, scores = evaluate_forecasts(test[:, :], predictions)
# 	return score, scores

def forecaster(dataset):
# evaluate model and get scores
	n_input = 7
	model = build_model(dataset, n_input)
	history = [x for x in dataset]
	forecaster = forecast_month(model, history, n_input)
	return forecaster


def on_message(client, userdata, message):
	temp = list()
	tempStr = message.payload.decode("utf-8")
	String = tempStr[1:len(tempStr) - 1]
	dataset = String.split(',')
        total = 0
	for i in range(len(dataset)):
                total += float(dataset[i])
		temp.append(float(dataset[i]))
	temp = [temp]
	pT = forecaster(array(temp))
	print(str(pT[0][0]+total))
	client.subscribe("forecast/update")
	client.publish("forecast/update", str(pT[0][0]+total))
########################################

broker_address="baokhoa.tk"
print("creating new instance")
client = mqtt.Client("P1") #create new instance
client.on_message=on_message #attach function to callback
print("connecting to broker")
client.connect(broker_address) #connect to broker
print("Subscribing to topic","forecast/getdata")
client.subscribe("forecast/getdata")
client.loop_start() #start the loop
time.sleep(60)
client.loop_stop()
#print("Publishing message to topic","house/bulbs/bulb1")
#client.publish("forecast/getdata","OjjjjjjjFF")
 #stop the loop



#client.subscribe("forcast/update")
#client.publish('forecast/update', forecaster(dataset))
#print(forecaster)
#score, scores = evaluate_model(train, test, n_input)
# summarize scores
#summarize_scores('lstm', score, scores)
# plot scores

