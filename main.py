import numpy as np
import re
import math
import random
from collections import defaultdict
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.utils import *
from keras.preprocessing import sequence
from keras.optimizers import Adam

random.seed(3)
np.random.seed(3)
tf.set_random_seed(3)

def tokenize(data):
    return [x.strip().lower() for x in re.split('(\W+)', data) if x.strip()]

def get_embeddings(word_dict, word_embeddings="wiki.en.vec", emb_size=300):
    word_vectors = np.random.uniform(-0.1, 0.1, (len(word_dict), emb_size))

    f = open(word_embeddings, mode='r', encoding='utf-8')
    vec = {}
    for line in f:
        line = line.split()
        vec[line[0]] = np.array([float(x) for x in line[-300:]])
    f.close()    	

    for key in word_dict:
        low = key.lower()
        if low in vec:
            word_vectors[word_dict[key]] = vec[low]
    return word_vectors

def cosine_distance(x1,x2):
    sumxx, sumxy, sumyy = 0,0,0
    for i in range(len(x1)):
        x, y = x1[i], x2[i]
        sumxx += x*x
        sumxy += x*y
        sumyy += y*y
    return 1 - sumxy/math.sqrt(sumxx*sumyy)        

def euclidean_distance(x, y):   
    return np.sqrt(np.sum((x - y) ** 2))

def top_n(d, n):
    dct = defaultdict(list) 
    for k, v in d.items():
        dct[v].append(k)      
    return sorted(dct.items())[-n:][::-1]  

def map_to_id(data, vocab):
	return [vocab[word] if word in vocab else 1 for word in data]

testSet = []
with open('ETS.tsv','r') as f:
    for line in f:
        line = line.strip().split('\t')
        testSet.append(line)

data_X = []
data_Y = []

word_dict = {}
word_dict['PAD'] = len(word_dict) #padding = 0
word_dict['UNK'] = len(word_dict) #unknown token = 1
word_dict['BOS'] = len(word_dict) #begin of sentence = 2
word_dict['EOS'] = len(word_dict) #end of sentence = 3
for line in testSet:
    words =  line[2].strip().split('|')
    #print(gold_test)
    gold_label = []
    sentence = []
    sentence.append('BOS')
    for word in words:
        term = re.sub('[^A-Za-z0-9]+', '', word)
        term, score = term[:-1].lower(), term[-1].lower()
        if term == '':
            continue
        sentence.append(term)
        if score != str(0):
            gold_label.append(term)
    sentence.append('EOS')
    #print(sentence)
    for word in sentence:
        if word not in word_dict:
            word_dict[word] = len(word_dict)
    #print(gold_label)
    sentence = map_to_id(sentence, word_dict)
    #print(len(sentence))
    if len(sentence) > 30:
        sentence = sentence[:30]
    gold_label = map_to_id(gold_label, word_dict)
    data_X.append(sentence)
    data_Y.append(gold_label)

index2word = {v: k for k, v in word_dict.items()}
data_X = sequence.pad_sequences(data_X, maxlen = 30)

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size = 0.2)

print('Found %s texts in training sets' % len(train_X))
print('Found %s texts in test sets' % len(test_X))

batch_size = 64
max_len = 30
emb_dim = 300
latent_dim = 64
intermediate_dim = 256
epsilon_std = 1.0
kl_weight = 0.01
NB_WORDS = len(word_dict)
word_embeddings = get_embeddings(word_dict)

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

x = Input(shape=(max_len,))
x_embed = Embedding(NB_WORDS, emb_dim, weights=[word_embeddings],
                            input_length=max_len, trainable=False)(x)
h = Bidirectional(LSTM(intermediate_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(x_embed)
h = Attention(max_len)(h)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0],latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# we instantiate these layers separately so as to reuse them later
repeated_context = RepeatVector(max_len)
decoder_h = LSTM(intermediate_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
decoder_mean = Dense(NB_WORDS, activation='linear') #softmax is applied in the seq2seqloss by tf #TimeDistributed()
h_decoded = decoder_h(repeated_context(z))
x_decoded_mean = decoder_mean(h_decoded)


# placeholder loss
def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, x, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        self.target_weights = tf.ones(shape=(tf.shape(x)[0], max_len))
        #self.target_weights = xent_weight
    def vae_loss(self, x, x_decoded_mean):
        #xent_loss = K.sum(metrics.categorical_crossentropy(x, x_decoded_mean), axis=-1)
        labels = tf.cast(x, tf.int32)
        xent_loss = K.sum(tf.contrib.seq2seq.sequence_loss(x_decoded_mean, labels, 
                                                     weights=self.target_weights,
                                                     average_across_timesteps=False,
                                                     average_across_batch=False), axis=-1)#,
                                                     #softmax_loss_function=softmax_loss_f), axis=-1)#,
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        xent_loss = K.mean(xent_loss)
        kl_loss = K.mean(kl_loss)
        return K.mean(xent_loss + kl_weight * kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # we don't use this output, but it has to have the correct shape:
        return K.ones_like(x)
    

loss_layer = CustomVariationalLayer(x)([x, x_decoded_mean])
vae = Model(x, [loss_layer])
optimizer = Adam(lr=0.01)
vae.compile(optimizer=optimizer, loss=[zero_loss])
#vae.summary()

encoder = Model(x, z_mean, name='encoder')

# build a generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,), name='z_sampling')
_h_decoded = decoder_h(repeated_context(decoder_input))
_x_decoded_mean = decoder_mean(_h_decoded)
_x_decoded_mean = Activation('softmax')(_x_decoded_mean)

generator = Model(decoder_input, _x_decoded_mean, name='decoder')
#generator.summary()

checkpoint = ModelCheckpoint("vae_lstm.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

lr_sched = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=1, cooldown=1, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

vae.fit(train_X, train_X,
                epochs=50,
                shuffle=True,
                validation_data=(test_X, test_X),
                batch_size = batch_size,
                callbacks=[lr_sched])

#vae.load_weights('vae_lstm.h5')
vae.save('vae_lstm100epoch.h5')

preds = encoder.predict(train_X)
original = preds[0]
original_sentence = [index2word[w] for w in train_X[0] if index2word[w] not in ['PAD','EOS','BOS']]
print(' '.join(original_sentence))
for i in range(5):
    score = euclidean_distance(original, preds[i+1])
    sentence = [index2word[w] for w in train_X[i+1] if index2word[w] not in ['PAD','EOS','BOS']]
    print(score, ' '.join(sentence))

def print_latent_sentence(sent_vect):
    sent_vect = np.reshape(sent_vect,[1,latent_dim])
    sent_reconstructed = generator.predict(sent_vect)
    sent_reconstructed = np.reshape(sent_reconstructed,[max_len,NB_WORDS])
    reconstructed_indexes = np.apply_along_axis(np.argmax, 1, sent_reconstructed)
    word_list = list(np.vectorize(index2word.get)(reconstructed_indexes))
    w_list = [w for w in word_list if w not in ['PAD']]
    print(' '.join(w_list))

print_latent_sentence(preds[0])


