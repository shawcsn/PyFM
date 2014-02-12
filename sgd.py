import math

class SGD:

	def __init__(self,task,train,test,model,iter_num,learn_rate):
		self.train=train
		self.test=test
		self.model=model
		self.iter_num=iter_num
		self.task=task
		self.learn_rate=learn_rate

	def learn(self):
		#print self.train.Y.shape
		if self.task=='c':
			self.sgd_classification()
		self.model.savemodel()

	def sgd_classification(self):		
		for epoch in range(self.iter_num):
			for i in range(self.train.ins_num):
				#self.model.savemodel()
				x=self.train.X[i]
				y=self.train.Y[i]
				p=self.model.predict(x)				
				mult=y*((1.0/(1.0+math.exp(-y*p)))-1)
				self.model.w0-=self.learn_rate*(mult+self.model.w0*self.model.reg0)
				for i in x.indices:
					gradw=mult*x[0,i]
					self.model.w[i]-=self.learn_rate*(gradw+self.model.w[i]*self.model.regw)
				for f in range(self.model.factor_num):
					for j in x.indices:
						grad=mult*x[0,j]*(self.model.sum[f]-self.model.v[f,j]*x[0,j])
						self.model.v[f,j]-=self.learn_rate*(grad+self.model.v[f,j]*self.model.regv)
			#print "%d iteration loss: %f" %(epoch,self.model.loss(self.train,self.task))			
			print "iter:%d: train: %f,  test: %f" %(epoch,self.model.eval(self.train,self.task),self.model.eval(self.test,self.task))

		