import nltk

from train_data import sample_tweets, test_tweets

word_features = []
tweets = []


for (words, sentiment) in sample_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3] 
    tweets.append((words_filtered, sentiment))


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
      all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document):
    word_features = get_word_features(get_words_in_tweets(tweets))
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

tweet = "Your song is annoying"
print "Tweet: is", classifier.classify(extract_features(tweet.split()))
print 'accuracy:', nltk.classify.util.accuracy(classifier, nltk.classify.apply_features(extract_features, test_tweets))
