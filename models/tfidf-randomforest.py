def get_numeric_data(x):
    return [record[:-2].astype(float) for record in x] #gets the numeric data which is all the columns except last column

def get_text_data(x):
    return [record[-1] for record in x] # gets the text data which is the last column of the dataset

transformer_numeric = FunctionTransformer(get_numeric_data) #transformer_numeric when transformed on some X will apply get_numeric_data on X
transformer_text = FunctionTransformer(get_text_data)

pipeline = Pipeline([
    ('features', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', transformer_numeric)
            ])),
             ('text_features', Pipeline([
                ('selector', transformer_text),
                ('vec', TfidfVectorizer(analyzer='word'))
            ]))
         ])),
    ('clf', RandomForestClassifier())
])

param_grid = {'clf__n_estimators': np.linspace(1, 100, 42, dtype=int),
              'clf__min_samples_split': [2],
              'clf__min_samples_leaf': [1],
              'clf__max_features': [8],
              'clf__max_depth': [None],
              'clf__criterion': ['entropy'],
              'clf__bootstrap': [False]}

scoring = {'Accuracy': 'accuracy', 'F1': 'f1_macro'}
refit = 'F1'

def ML_codesmell(X_tr, Y_tr, X_ts, Y_ts):
  rf_model = GridSearchCV(pipeline, param_grid=param_grid, scoring=scoring, refit=refit, n_jobs=-1, return_train_score=True, verbose=1)
  rf_model.fit(X_tr, Y_tr)
  model_score = rf_model.score(X_ts, Y_ts)
  y_pred_test = rf_model.predict(X_ts)
  accuracy = accuracy_score(Y_ts, y_pred_test)
  classification = classification_report(Y_ts, y_pred_test)
  return model_score, accuracy, classification
