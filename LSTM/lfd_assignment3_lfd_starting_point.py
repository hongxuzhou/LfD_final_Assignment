#!/usr/bin/env python

'''TODO: add high-level description of this Python script'''

import random as python_random
import json
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.initializers import Constant
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='train.txt', type=str,
                        help="Input file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", type=str, default='dev.txt',
                        help="Separate dev set to read in (default dev.txt)")
    parser.add_argument("-t", "--test_file", type=str,
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-e", "--embeddings", default='/Users/hongxuzhou/Desktop/lfd_final_glove/converted_json/glove_twitter_100d.json', type=str,
                        help="Embedding file we are using default twitter glove 100d")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Read in review data set and returns docs and labels'''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            # Split on whitespace and take the text after the label
            tokens = line.strip().split()
            
            # exact the label which is the last token in the line
            label = tokens[-1]
            
            # Extract the text which is the rest of the tokens
            text = " ".join(tokens[:-1])
            
            # Skip potential empty lines
            if not text or not label:
                continue
            
            # Append to lists
            documents.append(text)
            
            # 2-class problem: offensive vs non-offensive
            labels.append(label)
            
    return documents, labels


def read_embeddings(embeddings_file):
    '''Read in word embeddings from file and save as numpy array'''
    embeddings = json.load(open(embeddings_file, 'r'))
    return {word: np.array(embeddings[word]) for word in embeddings}


def get_emb_matrix(voc, emb):
    '''Get embedding matrix given vocab and the embeddings'''
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def create_model(Y_train, emb_matrix):
    '''Create the Keras model to use'''
    # Define settings, you might want to create cmd line args for them
    learning_rate = 0.001
    loss_function = 'binary_crossentropy' # Changed from 'categorical_crossentropy' because of 2-class problem
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Changed from SGD to Adam
    
    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    
    #num_labels = len(set(Y_train)) <- this is removed by claude 
    
    # Now build the model
    model = Sequential()
    
    # Embedding layer -- keeping embeddings frozen (trainable=False)
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix),trainable=False))
    
    # Add LSTM layers with dropout
    model.add(LSTM(units = 100,
                   return_sequences = True, # Return full sequence for next layer
                   dropout = 0.2, # Dropout on inputs 
                   recurrent_dropout = 0.2,)) # Dropout on recurrent connections
    
    # Add another LSTM layer
    model.add(LSTM(units = 50,
                   dropout = 0.2,
                   recurrent_dropout = 0.2,))
    
    # Dense layer -- Ultimately, end with dense layer with softmax
    model.add(Dense(units=50, activation="softmax")) # Changed from num_labels to 50
    model.add(tf.keras.layers.Dropout(0.2)) # Added dropout layer
    
    # Output layer
    model.add(Dense(units=1, activation="sigmoid"))  
    
    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, 
                  optimizer=optim, 
                  metrics=['accuracy'])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev):
    '''Train the model here. Note the different settings you can experiment with!'''
    # Potentially change these to cmd line args again
    # And yes, don't be afraid to experiment!
    verbose = 1
    batch_size = 32 # 16 -> 32
    epochs = 10 # 50 -> 10, start low
    
    # Create callback to stop training early if no improvement
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_model.h5", 
                                                       save_best_only=True,
                                                       monitor='val_accuracy',) # Use val_accuracy to monitor
    
    # Early stopping callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,
                                                         monitor='val_accuracy',
                                                         restore_best_weights = True)
    
    
    # Finally fit the model to our data
    history = model.fit(
        X_train, Y_train, 
        verbose=verbose, 
        epochs=epochs, 
        callbacks=[checkpoint_cb, early_stopping_cb], 
        batch_size=batch_size, 
        validation_data=(X_dev, Y_dev))
    
    # Print final accuracy for the model (clearer overview)
    test_set_predict(model, X_dev, Y_dev, "dev") # This is not shown in Claude code, may need to be deleted if runs into error
    
    return model, history


def test_set_predict(model, X_test, Y_test, ident):
    '''Do predictions and measure accuracy on our own test set (that we split off train)'''
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred_binary = (Y_pred > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred_binary)
    
    # Calculate F1 score
    f1 = f1_score(Y_test, Y_pred_binary, average='binary')
    
    # Print out the results
    print(f'Results on {ident} set:')
    print(f"Accuracy on {ident} set: {accuracy:.4f}")
    print(f"Macro F1 score on {ident} set: {f1:.4f}")
    print('\nClassification Report:')
    print(classification_report(Y_test, Y_pred_binary)) 
    
    return accuracy, f1


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    
    print('Reading embeddings...')
    embeddings = read_embeddings(args.embeddings)

    # Transform words to indices using a vectorizer
    print("Vectorizing data...")
    vectorizer = TextVectorization(standardize='lower_and_strip_punctuation', 
                                   output_sequence_length=50) # Needs to adjust based on test results
    
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    
    # Get embedding matrix
    emb_matrix = get_emb_matrix(voc, embeddings)

    # Transform string labels to one-hot encodings (binary)
    label_dict = {'NOT': 0, 'OFF': 1}
    Y_train =  np.array([label_dict[y] for y in Y_train])  # NOT Use encoder.classes_ to find mapping back
    Y_dev = np.array([label_dict[y] for y in Y_dev])
    
    # Vectorize text data
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Create model
    #model = create_model(Y_train, emb_matrix) <- Claude removed it 

    # Transform input to vectorized input
    # Create and train model
    print('Creating model...')
    model = create_model(Y_train, emb_matrix)
    print('Training model...')
    model, history = train_model(model, X_train_vect, Y_train, X_dev_vect, Y_dev)

    # Train the model
    #model = train_model(model, X_train_vect, Y_train_bin, X_dev_vect, Y_dev_bin) <- clause removed it

    # Do predictions on specified test set -- test if specified
    if args.test_file:
        # Read in test set and vectorize
        print('Predicting on test set...')
        X_test, Y_test = read_corpus(args.test_file)
        Y_test = np.array([label_dict[y] for y in Y_test])
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        
        # Finally do the predictions
        test_set_predict(model, X_test_vect, Y_test, "test")

if __name__ == '__main__':
    main()
