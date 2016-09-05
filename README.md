# Sentiment-Analysis-on-IMDb-Reviews
Sentiment Analysis using supervised classification of Doc2Vec Embeddings on the IMDb Review Dataset. Achieved an accuracy of 88.8% on the unseen test set.

For creating the 300 dimensional embeddings using Doc2Vec, I used DBOW-mode with negative sampling on phrase collocated reviews. To improve accuracy, I can probably concatenate DM and DBOW vectors, try hierarchial sampling, etc.

The dataset can be found here: http://ai.stanford.edu/~amaas/data/sentiment/

**Modeled in Python and R**

###Supervised Classification:

I tried a simple Logistic Regression classifier, which usually works quite well with Doc2Vec vectors. 

(A lambda search for an elastic-net mixing parameter of 0.5 yielded an optimal value of 2e-4.)

![Logistic Regression Model](https://github.com/sgrvinod/Sentiment-Analysis-on-IMDb-Reviews/blob/master/logmodel.png?raw=true)

Using a more complex classifier (such as a neural-net) yields only a very small increase in AUC; small enough that it's often not worth the trouble. 

A 3 hidden layer x 100 neuron NN  with Rectifier Activation with Dropout (dropout ratio=0.5) only did slightly better:

![Deep Learning Model](https://github.com/sgrvinod/Sentiment-Analysis-on-IMDb-Reviews/blob/master/dlmodel.png?raw=true)
