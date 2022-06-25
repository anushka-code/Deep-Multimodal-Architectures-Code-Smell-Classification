def PrePro(last_column, dataframe):
  dataframe.rename(columns = {last_column :'is_code_smell'}, inplace = True) #rename column
  dataframe['is_code_smell'] = dataframe["is_code_smell"].astype(int) #change boolean labels to int labels
  Y_part = dataframe.iloc[:,-1:]
  X_part = dataframe.iloc[:,:56]
  X_part = X_part.replace(to_replace =["?"], value = np.nan) #replace non existing values with null
  X_part = X_part.astype(float) #change datatype of features of X as float
  return X_part,Y_part

X_lp, Y_lp = PrePro('is_long_parameters_list',df_lp)
X_ss, Y_ss = PrePro('is_switch_statements',df_ss)

def MeanforNaN(dataframe):   #function to fill null spaces with column mean 
  column_means = dataframe.mean()
  dataframe = dataframe.fillna(column_means)
  return dataframe

X_lp = MeanforNaN(X_lp)
X_ss = MeanforNaN(X_ss)

def ConCat(df1,df2): #concatenate code smell datasets
  code_smells = [df1,df2]
  joint = pd.concat(code_smells)
  return joint

X_train = ConCat(X_lp,X_ss)
Y_train = ConCat(Y_lp,Y_ss)

def Normalize(dataframe): #apply MinMax normalisation to fit the values between 0 to 1
  scaler = MinMaxScaler()
  model = scaler.fit(dataframe)
  scaled_data = model.transform(dataframe)
  return scaled_data

X_sample = Normalize(X_train)
Y_sample = Y_train.to_numpy(dtype='int64', copy='True')
