def SVMOversample(X_data,Y_data): #Using Smote obtain a 50-50 balanced dataset 

  sm = SVMSMOTE(random_state = 2)
  X_train_res, Y_train_res = sm.fit_resample(X_data, Y_data.ravel())
  return X_train_res, Y_train_res

X_struct, Y_new = SVMOversample(X_sample,Y_sample)
