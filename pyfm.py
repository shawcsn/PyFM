import sys
import argparse
from data import *
from model import *
from sgd import *

parser = argparse.ArgumentParser(description='parsing parameters of fm')
parser.add_argument('-task', dest='task', default='c', help='r=regression, c=binary classification')
parser.add_argument('-train', dest='train', help='filename for training data')
parser.add_argument('-test', dest='test', help='filename for test data')
parser.add_argument('-f', dest='factor', type=int, default=8, help='number of factors')
parser.add_argument('-lr', dest='learn_rate', type=float, default=0.1, help='learn_rate for SGD; default=0.1')
parser.add_argument('-r', dest='regular', default='0,0,0', help="regularization parameters; default='0,0,0'")
parser.add_argument('-iter', dest='iter', type=int, default=100, help='number of iterations; default=100')

results = parser.parse_args()
#print results
if results.train==None or results.test==None:
	raise ValueError('train_file=%s, test_file=%s' %(results.train,results.test))

def main():	
	train=Data()
	train.load(results.train)
	test=Data()
	test.load(results.test)	
	max_fea_num=max(train.feature_num,test.feature_num)	
	fm_model=FM_model(max_fea_num,results.factor,results.regular)
	fm_learn=SGD(results.task,train,test,fm_model,results.iter,results.learn_rate)
	fm_learn.learn()

if __name__=="__main__":
	#starttime = datetime.datetime.now()
	main()

