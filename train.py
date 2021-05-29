import pandas as pd 
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Read data from files
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

#remove missing values
fake.dropna()
true.dropna()

#Set label for news 
true["label"]=1
fake["label"]=0

#Separate test data from train data
train_fake_data = fake[0:22200]
train_true_data = true[0:20200]
test_fake_data = fake[22200:23482]
test_true_data = true[20200:21420]

# Concating:
data_train = pd.concat([train_true_data, train_fake_data], axis=0)
data_test = pd.concat([test_true_data, test_fake_data], axis=0)

#Save files
data_train.to_csv("data_train.csv")
data_test.to_csv("data_test.csv")

#Prepare train and test sentances 
train_sentences = data_train['title'].tolist()
test_sentences = data_test['title'].tolist()

vocab_size = 5000
embedding_dim = 16
max_length = 500
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

#Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(train_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type)

testing_sequences = tokenizer.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(300, dropout=0.3, recurrent_dropout=0.3)
))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
#Compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
			  
#add callback
filepath = "model.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#train model
data_model = model.fit(training_padded, data_train['label'], epochs=1, validation_data=(testing_padded, data_test['label']), callbacks=[callbacks_list])
data_model.save();