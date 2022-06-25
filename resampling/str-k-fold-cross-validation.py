def StratiefKFoldCrossValidation(dataX,dataY): #split the dataset into train and test using k-5 fold cross validation
  skf = StratifiedKFold(n_splits=5, random_state = None, shuffle=True) 

  for train_index, test_index in skf.split(dataX , dataY):
        X_crosstrain, X_crosstest = dataX[train_index], dataX[test_index] 
        Y_crosstrain, Y_crosstest = dataY[train_index], dataY[test_index]
        return X_crosstrain,X_crosstest,Y_crosstrain,Y_crosstest

X_berttrain,X_berttest,Y_berttrain,Y_berttest = StratiefKFoldCrossValidation(concat, Y_new)
