import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, merge,Conv1D
from keras.models import Model
import time
import os
import multiprocessing

def returntrans(dayreturn):
	if dayreturn>0:
		return [1,0]
	else:
		return [0,1]
def returntrans2(dayreturn):
	dayreturn[dayreturn>0] = 1
	dayreturn[dayreturn<0] = -1
	return dayreturn		

def dataprocess(stk,changeratio,pb,pettm,pvv,fr5):

	# print("\nRun %s,pid: %s" %(stk,os.getpid()))
	starttime = time.time()

	trainlen = 30
	data_dim = 1
	stkchangeratio = changeratio[stk].dropna()
	stkpb = pb[stk]
	stkpettm = pettm[stk]
	stkpvv = pvv[stk]

	frdata = fr5[stk].dropna()

	tmpdata = pd.DataFrame(index = frdata.index)
	tmpdata['changeratio'] = stkchangeratio
	# tmpdata['pb'] = stkpb
	# tmpdata['pettm'] = stkpettm
	# tmpdata['pvv'] = stkpvv

	# print('frdata '+str(len(frdata)))
	# print('tmpdata '+str(len(tmpdata)))

	if(len(tmpdata)!= len(tmpdata.dropna())):
		tmpdata.fillna(method = 'pad',inplace = True)
		tmpdata.fillna(method = 'backfill',inplace = True)
	
	# tmpscale = scale(tmpdata,axis = 1)
	
	tempresultx = []
	tempresulty = []

	
	if len(tmpdata) < trainlen : print('error'+stk)
	try:
		fl = True
		for i in range(trainlen-1,len(tmpdata)):

			temp = np.array(tmpdata[i+1-trainlen:i+1])
			temp = np.transpose(temp)
			# temp = scale(temp,axis = 1)
			temp = temp.reshape(1,trainlen,1)
			temp2 = np.array(returntrans(frdata.ix[i]))
			temp2 = temp2.reshape(1,2,1)

			if fl:
				tempresultx = temp
				tempresulty = temp2

				fl = False
			else:
				tempresultx = np.concatenate((tempresultx,temp),axis = 0)
				tempresulty = np.concatenate((tempresulty,temp2),axis = 0)

			# tempresulty = np.append(tempresulty,returntrans(frdata.ix[i]))

		# print(stk+ ' OK')
		return tempresultx,tempresulty	
	except:
		print('error'+stk)
	endtime = time.time()
	# print(str(endtime - starttime ) + ' elapsed')
	

if __name__ == "__main__":

	fr5 = pd.read_csv('./factor/fr5.csv',encoding = 'GBK',index_col = 0,parse_dates = [0])
	# fr20 = pd.read_csv('./factor/fr20.csv',encoding = 'GBK',index_col = 1,parse_dates = [1])
	# fr5 = fr20[fr20.columns[1:]]

	changeratio = pd.read_csv('./factor/changeratio.csv',encoding = 'GBK',index_col = 1,parse_dates = [1])
	changeratio = changeratio[changeratio.columns[1:]]
	pb = pd.read_csv('./factor/pb.csv',encoding = 'GBK',index_col = 0,parse_dates = [0])
	pettm = pd.read_csv('./factor/pettm.csv',encoding = 'GBK',index_col = 0,parse_dates = [0])
	pvv = pd.read_csv('./factor/pvv.csv',encoding = 'GBK',index_col = 1,parse_dates = [1])
	pvv = pvv[pvv.columns[1:]]
	pvv.fillna(method = 'pad',inplace = 'True')
	# data.index = list(map(pd.to_datetime, list(map(dateutil.parser.parse, data.date))))


	allstk = pd.read_csv('./allstk.csv',dtype = {'stk':str})#原来是int型 修正了为str 补零
	allstk = allstk.ix[:,1]
	stklist = list(allstk.values)
	stklistlen = len(stklist)

	trainx = np.array([])
	trainy = np.array([])
	testx = np.array([])
	testy = np.array([])
	trainresult = []
	samplelist = random.sample(range(stklistlen),600)
	trainlen = 30
	timesteps = trainlen
	data_dim = 1

	##muti thread
	pool=multiprocessing.Pool(4)

	for j in samplelist[0:10]:
		
		stk = stklist[j]

		##multi threads
		trainresult.append(pool.apply_async(dataprocess, (stk,changeratio,pb,pettm,pvv,fr5)))

	pool.close()
	pool.join()

	resx = []
	resy = []
	for res in trainresult:
		(tempx,tempy) = res.get()
		resx.append(tempx)
		resy.append(tempy)
	trainx = np.concatenate(resx,axis = 0)
	trainy = np.concatenate(resy,axis = 0)



	trainresult = []
	pool=multiprocessing.Pool(4)

	for j in samplelist[10:11]:
		stk = stklist[j]

		##sigle thread
		# tempresultx,tempresulty=dataprocess(stk,changeratio,pb,pettm,pvv,fr5)
		# if j==samplelist[5]:
		# 	testx = tempresultx
		# 	testy = tempresulty
		# else:
		# 	testx= np.concatenate((testx,tempresultx),axis = 0)
		# 	testy = np.concatenate((testy,tempresulty),axis = 0)

		###multi thread
		trainresult.append(pool.apply_async(dataprocess, (stk,changeratio,pb,pettm,pvv,fr5)))

	pool.close()
	pool.join()

	resx = []
	resy = []
	for res in trainresult:
		(tempx,tempy) = res.get()
		resx.append(tempx)
		resy.append(tempy)
	testx = np.concatenate(resx,axis = 0)
	testy = np.concatenate(resy,axis = 0)

	print(trainx.shape)
	print(trainy.shape)
	print(testx.shape)
	print(testy.shape)

	# expected input data shape: (batch_size, timesteps, data_dim)
	model = Sequential()
	model.add(Conv1D(40, 5,input_shape=(None,timesteps,1)))  # returns a sequence of vectors of dimension 100
	model.add(Dense(40))  # returns a sequence of vectors of dimension 100
	model.add(Dense(20))  # return a single vector of dimension 32
	model.add(Dense(5))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='mean_squared_error',
	              optimizer='SGD',
	              metrics=['accuracy'])

	model.fit(trainx, trainy,
	          batch_size=1, epochs=5)
