token = Tokenizer()
token.fit_on_texts(vocab_words)
vocab_size = len(token.word_index) + 1
print(vocab_size)

encoded_text_train = token.texts_to_sequences(word_groups_train)
encoded_text_test = token.texts_to_sequences(word_groups_test)

X_seman_train = pad_sequences(encoded_text_train, maxlen = 28, padding = 'pre') #max number of words is 28
X_seman_test =  pad_sequences(encoded_text_test, maxlen = 28, padding = 'pre')
print(X_seman_train)

import gensim.downloader as api
glove_gensim  = api.load('glove-wiki-gigaword-100') #100 dimensional

vector_size = 100 
gensim_weight_matrix = np.zeros((1300 ,vector_size)) 
gensim_weight_matrix.shape

for word, index in token.word_index.items():
    if index < vocab_size: 
        if word in glove_gensim.wv.vocab:
            gensim_weight_matrix[index] = glove_gensim[word]
        else:
            gensim_weight_matrix[index] = np.zeros(100)
