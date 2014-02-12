import numpy as np
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix

class Data:
	def __init__(self):
		self.X= csr_matrix([])
		self.Y= np.array([])
		self.feature_num=0
		self.ins_num=0
		self.min_target=0
		self.max_target=0
		self.value_num=0

	def load(self,filename):
		self.X,self.Y=load_svmlight_file(filename)
		self.min_target=min(self.Y)
		self.max_target=max(self.Y)
		self.ins_num=self.X.shape[0]
		self.feature_num=self.X.shape[1]
		self.value_num=self.X.nnz
		#print Y.shape
		#shuffle data		

	def debug(self):
		print 'instances: %d, features: %d, nonzero elements: %d' %(self.ins_num,self.feature_num,self.value_num)




