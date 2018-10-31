import pandas as pd
import numpy as np
from scipy import sparse
import tensorflow as tf


"""

########################################################
BAYESIAN PERSONALIZED RANKING USING MATRIX FACTORIZATION
########################################################


self.data_path:   path to feedback self.data (We use online retail data which contains invoice 
                  details of various products purchased by various users)

batch_size:       Batch size for training. The paper reccommends a stochastic gradient
                  descent approach with bootstrap sampling. Hence default batch_size 
                  is left as 1.

epochs:           "Abandoning the idea of full cycles through the data is especially
                  useful in our case as the number of examples is very large and for
                  convergence often a fraction of a full cycle is sufficient"             
                  Therefore a safe number would be = number of training examples ie (u,i,j) triplets.

lr:               learning rate for gradient descent optimization.

k:                dimensionality/rank of the approximation for Matrix Factorization.
                
reg:              a list containing regularization parameters for W, Hi and Hj respectively
                  eg:[0.02, 0.03, 0.03].
                
model_name:       name using which the model is saved.

threshold:        to stop training and save model if it reaches a certain threshold AUC value 
                  for the test data. 
                 
seed:             seed for random uniform initializer for W, H_i and H_j         

"""


class BPR():
    
    def __init__(self,data_path="online_retail.xlsx",batch_size=1,
                 lr=0.01,k=15,reg=[0.01, 0.01, 0.01], model_name="model",threshold=0.95,seed=1234,):
        
        self.data_path = data_path
        data_path
        self.batch_size = batch_size
        self.lr = lr
        self.k = k
        [self.lamda_W,self.lambda_Hi,self.lambda_Hj] = reg
        self.model_name = model_name
        self.threshold=threshold
        self.seed=seed
        
        
        self.data, self.customers,self.stockcodes = self.preprocess()
        self.S_master, self.S_train, self.test_set, self.complement,self.epochs = self.create_splits()
        
        self.model_graph = tf.Graph()
        self.build_graph()
        self.session = tf.Session(graph=self.model_graph)
        
 
    def create_splits(self):
        
        S_master=sparse.csr_matrix((len(self.customers), len(self.stockcodes)), dtype=np.int8)
        for i in range(len(self.data)):
            S_master[self.data.loc[i,"customerid"],self.data.loc[i,"stockcode"]]=1
        
        S_train=S_master.copy()  
        
        test_set=[]
        complement = {}
        epochs=0

        for p in range(S_master.shape[0]):
            
             candidates = S_master[p,:].nonzero()[1]
             comp=[i for i in range(len(self.stockcodes)) if i not in candidates]
             remove=np.asscalar(np.random.choice(candidates,1))     
             S_train[p,remove]=0
             test_set.append([p,remove])
             complement[p]=comp
             epochs=epochs+(len(candidates)-1)*(S_master.shape[0]-len(candidates)+1)
             
        return S_master,S_train,test_set,complement,epochs     
           
            
    
    def preprocess(self):
        
        data=pd.read_excel(self.data_path)
        data.columns=[str(i).lower() for i in data.columns]
        data=data[[not str(i).startswith('C') for i in data["invoiceno"]]]
        data["customerid"] = data["customerid"].fillna(-1).astype('int32')
        data=data[data["customerid"]!=-1].reset_index(drop=True)
        stockcode_values = data["stockcode"].astype('str')
        stockcodes = sorted(set(stockcode_values))
        stockcodes = {c: i for (i, c) in enumerate(stockcodes)}
        data["stockcode"] = stockcode_values.map(stockcodes).astype('int32')
        customers = sorted(set(data["customerid"]))
        customers = {c: i for (i, c) in enumerate(customers)}
        data["customerid"] = data["customerid"].map(customers)

        return data, customers, stockcodes



    def sampler(self,S,batch_size=1):
       
        counter=batch_size
        u,i,j = [],[],[]
      
        while(counter!=0):
          user=np.random.randint(0,S.shape[0])
        
          ones=list(S[user,:].nonzero()[1])
        
      
          if(len(ones)!=0):
            u.append(user)
            i.append(np.asscalar(np.random.choice(ones,1)))
         
            while(True):
                val=np.random.randint(0,len(self.stockcodes))
                if val not in ones:
                    break
            j.append(val)  
            counter=counter-1
          
        return u,i,j
        
    def build_graph(self):
      
     tf.reset_default_graph() 
     
     with self.model_graph.as_default(): 
        
        self.uu = tf.placeholder(tf.int32, shape=[None])
        self.ii = tf.placeholder(tf.int32, shape=[None])
        self.jj = tf.placeholder(tf.int32, shape=[None])
        
        with tf.variable_scope("parameters", reuse=False):
            
            self.W=tf.get_variable("user_features",shape=(len(self.customers),self.k),
                              initializer=tf.random_normal_initializer(0, 0.1,seed=self.seed),trainable=True)
            self.Hi=tf.get_variable("item_i",shape=(len(self.stockcodes),self.k),
                               initializer=tf.random_normal_initializer(0, 0.1,seed=(self.seed+1)),trainable=True)
            self.Hj=tf.get_variable("item_j",shape=(len(self.stockcodes),self.k),
                               initializer=tf.random_normal_initializer(0, 0.1,seed=(self.seed-1)),trainable=True)
            
 
        self.embed_u = tf.nn.embedding_lookup(self.W,self.uu,name="W")
        self.embed_i = tf.nn.embedding_lookup(self.Hi,self.ii,name="Hi")
        self.embed_j = tf.nn.embedding_lookup(self.Hj,self.jj,name="Hj")
        
        
        
        self.X_uij = tf.reduce_sum(tf.multiply(self.embed_u, (self.embed_i - self.embed_j)),1, keep_dims=True)

        self.prob=tf.sigmoid(self.X_uij)
        
        self.reg_W  =  tf.reduce_mean(self.embed_u*self.embed_u)
        self.reg_Hi =  tf.reduce_mean(self.embed_i*self.embed_i) 
        self.reg_Hj =  tf.reduce_mean(self.embed_j*self.embed_j)
        
        self.user_auc = tf.reduce_mean(tf.to_float(self.X_uij > 0))
        
        
        self.loss = tf.reduce_mean(tf.log(self.prob))
        self.reg  = self.lamda_W*self.reg_W  +  self.lambda_Hi*self.reg_Hi + self.lambda_Hj*self.reg_Hj
        self.total_loss = -self.loss + self.reg
        
        self.optim=tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.step = self.optim.minimize(self.total_loss)
        self.saver = tf.train.Saver()        
        self.init=tf.global_variables_initializer()
    def train(self):
            
          self.session.run(self.init)  
          
          for i in range(self.epochs):
            
            a,b,c = self.sampler(self.S_train,self.batch_size)
            feed_dict = {self.uu:a , self.ii:b , self.jj:c }  
            loss_,_ = self.session.run((self.total_loss,self.step),feed_dict=feed_dict)
            total_auc = 0
            
            
            if i%10000==0:
                for user in self.complement.keys():    
                    a = np.tile(user,len(self.complement[user]))
                    b = np.tile(self.test_set[user][1],len(self.complement[user])) 
                    c = self.complement[user]
                    feed_dict = {self.uu:a , self.ii:b , self.jj:c}
                    total_auc=total_auc + self.session.run(self.user_auc,feed_dict)
                
                total_auc=total_auc/len(self.customers)
                
                print("Epoch:{} Train_Loss:{} AUC:{}".format(i,loss_,total_auc))
            
            if(total_auc>=self.threshold):
                self.save(i)
                print("threshold reached, model has been saved.")
                break
          
          if(total_auc<self.threshold):
                self.save(i)
                print("training is complete, threshold not reached but model has been saved ")  
    
    def save(self,epoch=0):        
          
         save_path = self.saver.save(self.session, "./models/{}-{}.ckpt".format(self.model_name,epoch))
         print("Model saved in path: %s" % save_path)
              
     
    def evaluate(self,user,i,j):
        
         prob=self.session.run(self.prob, feed_dict={self.uu:[user] ,self.ii:[i] ,self.jj:[j]})
         print("The probability that the user {} prefers item {} over item {} is {}".format(user,i,j,np.asscalar(prob)))
          
    
    def close(self):
         
         tf.reset_default_graph()
         self.session.close()

        
        
           

        
        
        