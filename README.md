# EEG-Alcohol-ML-prediction
This project is based on sklearn machine learning models and a real-life training dataset to find possible ways to find if a person is alcoholic by reading their EEG signals.  
#### Dataset resource: [Kaggle](https://www.kaggle.com/datasets/nnair25/Alcoholics?resource=download), [Original](https://archive.ics.uci.edu/dataset/121/eeg+database).
## Data Set Information:
This data arises from a large study to examine EEG correlates of genetic predisposition to alcoholism. It contains measurements from 64 electrodes placed on subject's scalps which were sampled at 256 Hz (3.9-msec epoch) for 1 second.

There were two groups of subjects: alcoholic and control. Each subject was exposed to either a single stimulus (S1) or to two stimuli (S1 and S2) which were pictures of objects chosen from the 1980 Snodgrass and Vanderwart picture set. When two stimuli were shown, they were presented in either a matched condition where S1 was identical to S2 or in a non-matched condition where S1 differed from S2.
##### Attribute Information  
Each trial is stored in its own file and will appear in the following format.  
| trial number | sensor position | sample num | sensor value | subject identifier | matching condition | channel | name | time |  
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

__Trail number__: the number order the record. E.g., 30 means this is the 30th trial for this participant.  

__sensor position__: The position of where the EEG signal is taken using 10-20 system. 64 in total. E.g., FP1, AFZ, POZ, etc.  

> Each electrode placement site has a letter to identify the lobe, or area of the brain it is reading from: pre-frontal (Fp), frontal (F), temporal (T), parietal (P), occipital (O), and central (C).  
>
> Even-numbered electrodes (2,4,6,8) refer to electrode placement on the right side of the head, whereas odd numbers (1,3,5,7) refer to those on the left; this applies to both EEG and EOG (electrooculogram measurements of eyes) electrodes, as well as ECG (electrocardiography measurements of the heart) electrode placement.  

__sample num__: the number order of when the signal is read during the trial. E.g., 128 means this is the 128th signal in this trial. Each trial has 256 sample numbers as the EEG frequency is 256 Hz.  

__sensor value__: the magnitude of EEG signal.  

__subject identifier__: 'a' for 'alcoholic', or 'c' for 'control'. Participants with 'a' identifier are alcoholic, and same applies to 'c'.

__matching condition__: which condition the participant was facing. S1 for a single image shown, S2 non-match for showing two different images, and S2 match for showing two identical images.  

__channel__: digitized category of sensor position. From 0 to 63.  

__name__: "name" of the participant, in a series of letters and numbers. E.g., co2a0000364.  

__time__: the specific timestamp of when the signal is read. From 0 to 1s.  

## Data processing
Here is a screenshot of a part of the dataset:  
![data preview](preview.png)  
The major problem here is how do we convert the raw data into a proper training and testing sets for our models.  
__Notice how each row is simply one signal data point in a trial for a participant.__  
 Intuitively, it is extremely hard to classify if a person is alcoholic by looking at one single EEG signal data in 1/256 second. It means that training with this unprocessed data will lead to huge bias and low accuracy. Thus, we want each record to have a full set of signal data of the entire trial for each participant.
This is how I processed the data:  
1. Read all csv files and append them to the dataframe.  
2. Drop necessary columns, such as 'time', 'sensor position', and the automatically generated first id column. The reason we could drop these is that they essentially work the same as 'sample number' and 'channel' in sorting and grouping in later steps.  
3. Convert 'subject identifier' and 'matching condition' to integer type.
4. Sort the dataframe by 'subject identifier', 'trial number', 'sample number', and 'name'.
5. Drop the 'sample number' column, because after sorting, all the rows are ordered by sample number already. We do not need it for the grouping later.
6. Group all the signal data points by 'subject identifier', 'trial number', 'matching condition', and 'channel', append to the dataframe. Each cell should have a list with 256 elements in the new column.
7. Split the data to X and y for model training.
8. Convert each of the sensor value list to a new dataframe, and concat to the original one, and remove the original list column. The dataframe should have 258 columns(matching condition, channel, and 256 sensor values).
9. Spilt X and y to training set and test set. Process X with standard scaler and PCA.
10. The data pre-processing should be done.

    ## Models
The machine learning models we use here are scikit-learn models. The models I imported and used are: Logistic Regression Classifier, K-Neighbors Classifier(KNN), Decision Tree Classifier(DTC), Stochastic Gradient Descent Classifier(SGD), Support Vector Machines Classifier(SVC), and Multiple-layer Perceptron Classifier(MLP). All the hyperparameters are set to default.

##Result
In terms of accuracy, MLP > KNN > SVC > DTC > Logistic Regression > SGD. MLP and KNN has outstanding accuracy with over 93% of the predictions being correct, while SGD has 59% accuracy that is slight higher than a random guess(50%).    
