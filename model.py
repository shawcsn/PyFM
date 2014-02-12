import numpy as np
import math

class FM_model:
	def __init__(self,fea_num,factor_num,regular):
		self.init_mean = 0
		self.init_stdev = 0.01
		self.w0=0.0
		self.w=np.zeros(fea_num)	
		self.factor_num=factor_num	
		seed=28
		np.random.seed(seed)
		self.v = np.random.normal(scale=self.init_stdev,size=(factor_num, fea_num))
		#self.v = np.ones((factor_num, fea_num))
		#self.v = self.v*-0.01
		reg=regular.split(',')
		self.reg0 = float(reg[0])
		self.regw = float(reg[1])
		self.regv = float(reg[2])
		self.sum = np.zeros(factor_num)
		self.sum_sqr = np.zeros(factor_num)

	def predict(self,x):
		result=0.0
		result+=self.w0
		for i in x.indices:
			result+=self.w[i]*x[0,i]
		for f in range(self.factor_num):
			self.sum[f]=0.0
			self.sum_sqr[f]=0.0
			for j in x.indices:
				d=self.v[f,j]*x[0,j]
				self.sum[f]+=d
				self.sum_sqr[f]+=d*d
			result+=0.5*(self.sum[f]*self.sum[f]-self.sum_sqr[f])
		return result

	def eval(self,test,task):
		eval_sum=0
		for i in range(test.ins_num):
			x=test.X[i]
			y=test.Y[i]
			p=self.predict(x)
			if task=='c':
				if (p>=0 and y>=0) or (p<0 and y<0):
					eval_sum+=1
		return eval_sum/float(test.ins_num)

	def loss(self,train,task):
		loss_sum=0.0
		for i in range(train.ins_num):
			x=train.X[i]
			y=train.Y[i]
			p=self.predict(x)
			if task=='c':
				loss_sum+=-math.log((1.0/(1.0+math.exp(-y*p))),math.e)
			if task=='r':
				loss_sum+=(y-p)*(y-p)
		return loss_sum		
		
	def savemodel(self):
		modelfile=open('model','w')
		#modelv=open('v_py','w')
		modelfile.write(str(self.w0)+'\n')
		for i in self.w:
			modelfile.write(str(i)+'\n')
		for row in self.v:
			v_str= '\t'.join([str(j) for j in row])
			modelfile.write(v_str+'\n')

	def debug(self):
		print 'w0=%f' %self.w0
		print 'w=',self.w
		print 'v=',self.v
		print 'reg0=%f' %self.reg0
		print 'regw=%f' %self.regw
		print 'regv=%f' %self.regv