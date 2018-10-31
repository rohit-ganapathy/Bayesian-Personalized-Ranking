# Bayesian Personalized Ranking using Matrix Factorization

A recomender model used to recommend items to a user based on his purchase history. ALthough there exists models based on Matrix Factorization used for item reccommendation/prediction , BPR incorporates item ranking into the optimization criterion and training data points . Furthermore the decomposed matrix is learned using a stochastic gradient descent approach using bootstrap sampling which proves to work better than standard gradient descent.

This implementation is based on the following paper : 
Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." Proceedings of the twenty-fifth conference on uncertainty in artificial intelligence. AUAI Press, 2009.
https://arxiv.org/pdf/1205.2618

### Getting Started
 * create BPR object
 * classs method train() to begin training  
 * class method evaluate() to evaluate model for a particular (user, item_i, item_j) triplet
 * use method close() to close tf session after inferencing is done 

### Dataset

Online Retail Data Set:

""This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.""

Daqing Chen, Sai Liang Sain, and Kun Guo, Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining, Journal of Database Marketing and Customer Strategy Management, Vol. 19, No. 3, pp. 197â€“208, 2012 (Published online before print: 27 August 2012. doi: 10.1057/dbm.2012.17).

### Dependencies

* python 3+
* numpy
* pandas
* scipy
* tensorflow==1.9.0


 
### Additional Notes

* Convergence metric used is AUC (Area Under the curve) on the test data. the paper describes creation of the test data and evaluation methodology in section 6.2

