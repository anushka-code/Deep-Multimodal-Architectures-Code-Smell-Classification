height, width, depth = 917, 56, 1
input_shape=(width,depth)
input_struct = Input(shape=input_shape)
layer_1 = Convolution1D(filters=32, kernel_size=3, activation='relu')(input_struct)
layer_2 = Convolution1D(filters=64, kernel_size=3, activation='relu')(layer_1)
layer_3 = Convolution1D(filters=64, kernel_size=3, activation='relu')(layer_2)
flatten_cnn = Flatten()(layer_3)

model_left = Model(input_struct, flatten_cnn)

EMBEDDING_DIM = 100
input_sem = Input(shape=(28))
embedding_layer = Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length= X_seman_train.shape[1],
                    weights = [gensim_weight_matrix],trainable = False)(input_sem)
layer_1 = Bidirectional(LSTM(100,return_sequences=True))(embedding_layer)
layer_2 = Bidirectional(LSTM(200,return_sequences=True))(layer_1)
layer_3 = Bidirectional(LSTM(100,return_sequences=False))(layer_2)
flatten_bilstm = Flatten()(layer_3)

model_right = Model(input_sem, flatten_bilstm)

merge_layer = concatenate([flatten_cnn, flatten_bilstm])
dense_1 = Dense(500, activation='relu')(merge_layer)
X = Dropout(0.5)(dense_1)
dense_2 = Dense(100, activation='relu')(X)
X = Dropout(0.5)(dense_2)
dense3 = Dense(20, activation='relu')(X)
dense_final = Dense(1, activation='sigmoid')(dense3)

merged_model = Model([input_struct, input_sem], dense_final)

merged_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
