#!/usr/bin/env python

'''
This script perform and evaluates text classification using multiple machine learning algorithms (NB, DT, RF, KNN, SVC, LinearSVC).
In order to further evaluate them, it is also possible to specify feature sets and hyperparameters with which to experiment.

@arg -t, --train_file: specifies the train file to learn from
@arg -d, --dev_file: specifies the dev file to evaluate on
@arg -s, --sentiment: specifies whether sentiment analysis is performed
@arg -tf, --tfidf: specifies whether to use TfidfVectorizer
@arg -a, --algorithm: specifies the Machine Learning algorithm to use
@arg -avg, --average: specifies the averaging technique used in evaluation
@arg -f, --feature: specifies the features to be used in data preprocessing 
@arg -cn, --char_ngram: specifies the number of characters to use in character-level ngrams
@arg -wn, --word_ngram: specifies the number of words to use in word-level ngrams
@arg --alpha: specifies the alpha hyperparameter for Naive Bayes
@arg --max_depth: specifies the hyperparameter controlling maximum depth of a Decision Tree
@arg --n_estimators: specifies the hyperparameter controlling number of trees for Random Forest
@arg --criterion: specifies the hyperparameter controlling the function to measure quality of a split for Random Forest
@arg --n_neighbors: specifies the hyperparameter controlling number of neighbours for KNN
@arg --weights: specifies the hyperparameter controlling weight function used in prediction for KNN
@arg --p: specifies the hyperparameter controlling power using he Minkowski metric in KNN
@arg --C: specifies the regularization hyperparameter for SVC
@arg --kernel: specifies the hyperparameter controlling kernel type to be used in SVC
@arg --C_linear: specifies the regularization hyperparameter for linear SVC
'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import spacy
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
import functools


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='datasets/train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='datasets/dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-a", "--algorithm", default='naive_bayes', type=str,
                        help="Machine Learning Algorithm to use. Options are: naive_bayes, decision_tree, random_forest, knn, svc, svc_linear")
    parser.add_argument("-avg", "--average", default='weighted', type=str,
                        help="Averaging technique to use in evaluation. Options are: binary, micro, macro, weighted, samples")
    parser.add_argument("-f", "--feature", default='identity', type=str,
                        help="Feature(s) used for tokenizer. Options are: identity, stemming, lemmatizing, ner, pos, or any COMMA-SEPARATED combination of them")
    parser.add_argument("-cn", "--char_ngram", default=0, type=int,
                        help = "Use character n-gram. Please specify the maximum n-gram size") 
    parser.add_argument("-wn", "--word_ngram", default=1, type=int,
                        help = "Use word n-gram. Please specify the maximum n-gram size")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="Alpha hyperparameter for Naive Bayes")
    parser.add_argument("--max_depth", default=20, type=int,
                        help="Hyperparameter controlling maximum depth of a Decision Tree")
    parser.add_argument("--n_estimators", default=500, type=int,
                        help="Hyperparameter controlling number of trees for Random Forest")
    parser.add_argument("--criterion", default='entropy', type=str,
                        help="Hyperparameter controlling the function to measure quality of a split for Random Forest. Options are: gini, entropy, log_loss")
    parser.add_argument("--n_neighbors", default=35, type=int,
                        help="Hyperparameter controlling number of neighbours for KNN")
    parser.add_argument("--weights", default="distance", type=str,
                        help="Hyperparameter controlling weight function used in prediction for KNN. Options are: uniform, distance")
    parser.add_argument("--p", default=1, type=int,
                        help="Hyperparameter controlling power using he Minkowski metric in KNN")
    parser.add_argument("--C", default=1, type=float,
                        help="Regularization hyperparameter for SVC")
    parser.add_argument("--kernel", default="rbf", type=str,
                        help="Hyperparameter controlling kernel type to be used in SVC. Options are: linear, poly, rbf, sigmoid, precomputed")
    parser.add_argument("--C_linear", default=0.1, type=float,
                        help="Regularization hyperparameter for linear SVC")
    parser.add_argument
    args = parser.parse_args()
    return args

def read_corpus(corpus_file, use_sentiment, use_char_ngrams=False): # Not use_sentiment, but should be offensive tag
    '''
    This function reads the corpus file and converts the textual data into a format more suitable for classification tasks later on

    @param corpus_file: input file consisting of textual data
    @param use_sentiment: boolean indicating whether sentiment will need to be used
    @return: the documents (list of tokens) and
            labels (target labels for each document, this can be either sentiment labels or category labels)
    '''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            tokens = line.strip().split()
            documents.append(tokens[0]) # Text is the first element in the line
            labels.append(tokens[1:]) # Labels are the second of the elements in the line
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp

def stemming(inp):
    '''Applies stemming to the input'''
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in inp]

def lemmatizing(inp):
    '''Applies lemmatization to the input'''
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in inp]

def remove_punct(inp):
    '''Removes punctuation from the input'''
    return [token for token in inp if token not in string.punctuation]

def ner(inp):
    '''Performs Named Entity Recognition on the input'''
    doc = nlp(' '.join(inp)) # feed input as string to spacy
    outp = []
    for token in doc:
        if token.ent_type_:
            if token.ent_iob == 3: #count only the beginning of an entity to avoid repetition of tags
                outp.append(token.ent_type_)
        else:
            outp.append(token)

    return outp

def pos(inp):
    '''Replaces each token by its POS tag'''
    doc = nlp(' '.join(inp))
    return [token.tag_ for token in doc]

def combine_features(feats,text):
    '''Combines features by applying them one after the other, in the order in which they are input'''
    for feat in feats:
        text = get_features(feat)(text)
    
    return text


# Return the classifier indicated by the input arguments
def get_classifier(algorithm):
    '''
    This function reads the algorithm given in the input parameters and returns the corresponding classifier
    
    @param algorithm: name of the machine learning algorithm as indicated in the input parameters
    @return: the classifier corresponding to the inputted algorithm
    @raise ValueError: raises an exception when the inputted algorithm can not be matched to a classifier
    '''
    # Naive bayes implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    if algorithm == 'naive_bayes':
        return MultinomialNB(alpha=args.alpha)
    # Decision tree implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    if algorithm == 'decision_tree':
        return DecisionTreeClassifier(max_depth=args.max_depth)
    # Decision tree implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    if algorithm == 'random_forest':
        return RandomForestClassifier(n_estimators=args.n_estimators, criterion=args.criterion)
    # K Nearest Neighbours implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    if algorithm == 'knn':
        return KNeighborsClassifier(n_neighbors=args.n_neighbors, weights=args.weights, p=args.p)
    # Support Vector Classification implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    if algorithm == 'svc':
        return SVC(C=args.C, kernel=args.kernel)
    # Linear Support Vector Classification implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    if algorithm == 'linear_svc':
        return LinearSVC(C=args.C_linear)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    

def get_features(feature, use_char_ngrams = False):
    '''
    This function reads the feature(s) given in the input parameters and returns the corresponding function
    
    @param feature: name of the feature as indicated in the input parameters
    @return: the function corresponding to the inputted feature
    @raise ValueError: raises an exception when the inputted feature can not be matched to a function
    '''
    # Added
    if use_char_ngrams:
        return identity 
    
    if ',' in feature: #multiple features
        return functools.partial(combine_features,args.feature.split(','))
    if feature=="identity":
        return identity
    if feature=="stemming":
        return stemming
    if feature=="lemmatizing":
        return lemmatizing
    if feature=="ner":
        return ner
    if feature=="pos":
        return pos
    else:
        raise ValueError(f"Unknown feature: {feature}")
    
def get_ngrams(char,word):
    '''
    This function reads the char_ngrams and word_ngrams parameters given in the input and
    returns their corresponding values for the analyzer and ngram_range vectorizer arguments

    @param char: maximum n value for character-level ngrams
    @param word: maximum n value for word-level ngrams
    @raise ValueError: raises an exception when both types of ngrams are used at the same time
    @raise ValueError: raises an exception when an invalid 
    '''
    # Using default values, no changes
    if char==0 and word==1:
        return "word",(1,1)

    # Both character- and word-level ngrams cannot be used at the same time
    if char!=0 and word!=1:
        raise ValueError(f"Cannot use char_ngram and word_ngram at the same time")
    
    if char>0:
        return "char",(1,char)
    if word>1:
        return "word",(1,word)
    else:
        raise ValueError(f"Invalid maximum n-gram size. Please check it is a positive number.")



if __name__ == "__main__":
    # Parse the input arguments
    args = create_arg_parser()

    # Load the train and test datasets
    X_train, Y_train = read_corpus(args.train_file) # Adjustment: to make the sets take charcter ngram parameters
    X_test, Y_test = read_corpus(args.dev_file)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.

    chosen_tokenizer = get_features(args.feature)
    ngram_level, ngram_range = get_ngrams(args.char_ngram, args.word_ngram)
    
    # NER does not work on n-grams (with n>1)
    if 'ner' in args.feature and ngram_range[1]>1:
        raise ValueError(f"Named Entity Recognition only works with unigrams")

    # Initialise spacy
    nlp = spacy.load("en_core_web_sm")

    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=chosen_tokenizer, analyzer=ngram_level, ngram_range=ngram_range)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=chosen_tokenizer, analyzer=ngram_level, ngram_range=ngram_range)

    # Get the classifier that was given in the input arguments
    chosen_classifier = get_classifier(args.algorithm)

    # Get the averaging method for multi-class classification
    chosen_average = args.average

    # Create a pipeline by combining the chosen vectorizer and classifier
    classifier = Pipeline([('vec', vec), ('cls', chosen_classifier)])

    # Train the model using the training set
    classifier.fit(X_train, Y_train)

    # Let the model make predictions on the test set
    Y_pred = classifier.predict(X_test)

    # General metrics

    # Evaluate the predictions that were made by comparing them to the ground truth, apply several metrics from the sklearn library for this:
    # Accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    acc = accuracy_score(Y_test, Y_pred)
    print(f"General accuracy: {acc}")

    # Precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    precision = precision_score(Y_test, Y_pred, average=chosen_average)
    print(f"General precision score: {precision}")

    # Recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    recall = recall_score(Y_test, Y_pred, average=chosen_average)
    print(f"General recall score: {recall}")

    # F1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    f1 = f1_score(Y_test, Y_pred, average=chosen_average)
    print(f"General f1 score: {f1}\n")

    # Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    confusion = confusion_matrix(Y_test, Y_pred)
    print(f"Confusion matrix:\n{confusion}")  

    print("\nPer-class scores")
    #Classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    class_rep = classification_report(Y_test,Y_pred)
    print(class_rep)