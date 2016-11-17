import nltk

from train_data import test_data_slack
from train_data import training_data_slack

word_features = []
messages = []

MINIMUM_MESSAGE_LENGTH_FOR_ANALYSIS = 3


def structure_training_data(training_data):
    """
        Structures training data for analysis.
        Ignores messages whose length is less than MINIMUM_MESSAGE_LENGTH_FOR_ANALYSIS.

    :param training_data: Training data for slack messages.
    :return: Returns structured training data.
    """
    for (words, sentiment) in training_data:
        words_filtered = [e.lower() for e in words.split() if len(e) >= MINIMUM_MESSAGE_LENGTH_FOR_ANALYSIS]
        messages.append((words_filtered, sentiment))

    return training_data


def get_words_in_messages(messages):
    """
        Tokenizes messages and adds words to a list.

    :param messages: Messages that need to be tokenized.
    :return: List of words in the messages.
    """
    all_words = []
    for (words, sentiment) in messages:
      all_words.extend(words)

    return all_words


def get_word_features(wordlist):
    """
        Calculates the frequency distribution of words in the word list.

    :param wordlist:
    :return:
    """
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document):
    """
        Creates a dictionary with each word in the message to be classified as
        key and boolean variable as the value depending on whether the word existed in
        the training data or not

    :param document: list of words in the message to be classified.

    :return: returns a dictionary stating if words in training data are present
             in the message to be classified.
    """
    word_features = get_word_features(get_words_in_messages(messages))
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)

    return features


if __name__ == "__main__":

    structure_training_data(training_data_slack)
    training_set = nltk.classify.apply_features(extract_features, messages)
    classifier = nltk.NaiveBayesClassifier.train(training_set)

    message = "I shall see what needs to be done"
    print "Message: is", classifier.classify(extract_features(message.split()))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, nltk.classify.apply_features(extract_features, test_data_slack))
