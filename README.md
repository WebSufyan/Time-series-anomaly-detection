# Time-series-anomaly-detection
Detecting abnormal hearbeats that is leading to Congestive Heart Failure for a patient using GRU autoencoder

this dataset is a univariate time series that contains 5000 rows, each row has 140 columns. Each row represents one heartbeat
in total we have 5000 heartbeats that was captured from a patient that has severe congestive heart failure.

Every single heartbeat holds it's class that represents whether the heartbeat is normal or abnormal.
there are 5 classes in this dataset:

1.Normal (N)
2.R-on-T Premature Ventricular Contraction (R-on-T PVC)
3.Premature Ventricular Contraction (PVC)
4.Supra-ventricular Premature or Ectopic Beat (SP or EB)
5.Unclassified Beat (UB).

for this project we will consider first type as normal, and other types all as abnormal


i excluded other types and left only one type in dataset which is normal class.
Built and trained a GRU autoencoder only on the normal hearbeats.
the autoencoder model is a dense artificial neural network takes in the normal heartbeats samples and tries to compress it into a very low dimension representation,
and based on that representaion it tries to reconstruct it and make it like the original input, 
and this result will be compared to the original result and calculate the cost function. 
so at the end of training the model will be able to recognize what normal heartbeats looks like. later when giving it an alien heartbeat that was not trained on,
it will try to classify it as abnormal.

how can we acheive all of this and do the classification?

Using the cost function of model, normal samples will output low cost function, while abnormal samples will output a slightly higher cost function. and this depends on 
a threshold 
that should be decided based on training and evaluating the cost fucntion of the model.




this dataset is from :  "Anthony Bagnall, Jason Lines, William Vickers and Eamonn Keogh, The UEA & UCR Time Series Classification Repository, www.timeseriesclassification.com"
