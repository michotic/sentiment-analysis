# Sentiment Analysis
# By Michael Taylor
#
# Based on code from Dr. Paul Cook, UNB
#

import re
import sys

# Do not use the following libraries for your code
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# Calculates product of all values within a list
# Expects a list containing nothing but numerical values
def product(some_list):
    output = 1
    for value in some_list:
        output = output * value
    return output


# A simple tokenizer. Applies case folding
def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search("\w", t):
            # t contains at least 1 alphanumeric character
            t = re.sub("^\W*", "", t)  # trim leading non-alphanumeric chars
            t = re.sub("\W*$", "", t)  # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens


# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, key=lambda x: klass_freqs[x])[0]

    def classify(self, test_instance):
        return self.mfc


# A logistic regression baseline
class LogReg:
    def __init__(self, texts, klasses):
        self.train(texts, klasses)

    def train(self, train_texts, train_klasses):
        # sklearn provides functionality for tokenizing text and
        # extracting features from it. This uses the tokenize function
        # defined above for tokenization (as opposed to sklearn's
        # default tokenization) so the results can be more easily
        # compared with those using NB.
        # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        self.count_vectorizer = CountVectorizer(analyzer=tokenize)
        # train_counts will be a DxV matrix where D is the number of
        # training documents and V is the number of types in the
        # training documents. Each cell in the matrix indicates the
        # frequency (count) of a type in a document.
        self.train_counts = self.count_vectorizer.fit_transform(train_texts)
        # Train a logistic regression classifier on the training
        # data. A wide range of options are available. This does
        # something similar to what we saw in class, i.e., multinomial
        # logistic regression (multi_class='multinomial') using
        # stochastic average gradient descent (solver='sag') with L2
        # regularization (penalty='l2'). The maximum number of
        # iterations is set to 1000 (max_iter=1000) to allow the model
        # to converge. The random_state is set to 0 (an arbitrarily
        # chosen number) to help ensure results are consistent from
        # run to run.
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        self.lr = LogisticRegression(
            multi_class="multinomial",
            solver="sag",
            penalty="l2",
            max_iter=1000,
            random_state=0,
        )
        self.clf = self.lr.fit(self.train_counts, train_klasses)

    def classify(self, test_instance):
        # Transform the test documents into a DxV matrix, similar to
        # that for the training documents, where D is the number of
        # test documents, and V is the number of types in the training
        # documents.
        # test_counts = self.count_vectorizer.transform(test_texts)
        test_count = self.count_vectorizer.transform([test_instance])
        # Predict the class for each test document
        # results = self.clf.predict(test_counts)
        return self.clf.predict(test_count)[0]


##Implement the lexicon-based baseline
# You may change the parameters to each function
class Lexicon:
    def __init__(self, positive_words, negative_words):
        self.positive = positive_words
        self.negative = negative_words

    def train(self):
        pass

    def classify(self, test_instance):
        tokens = tokenize(test_instance)
        polarity = 0
        for word in tokens:
            if word in self.positive:
                polarity += 1
            elif word in self.negative:
                polarity -= 1

        if polarity == 0:
            return "neutral"
        elif polarity > 0:
            return "positive"
        elif polarity < 0:
            return "negative"
        return "Something went wrong in Lexicon.classify(...) if you see this!"


##Implement the multinomial Naive Bayes model with smoothing
class NaiveBayes:
    def __init__(self, texts, klasses):
        self.data = {
            "words": {},  # Dict that will be filled with word data after training
            "docs": {  # Dict containing data about the dataset and its contained documents and labels
                "n_total": len(texts),
                "n_pos": klasses.count("positive"),
                "n_neu": klasses.count("neutral"),
                "n_neg": klasses.count("negative"),
                "pr_pos": klasses.count("positive") / len(texts),
                "pr_neu": klasses.count("neutral") / len(texts),
                "pr_neg": klasses.count("negative") / len(texts),
            },
        }
        self.train(texts, klasses)
        pass

    def train(self, texts, klasses):
        # Variables for tracking number of occurrences
        n_types = 0
        n_tokens = {"n_total": 0, "n_pos": 0, "n_neu": 0, "n_neg": 0}
        words = {}  # Dict that will store data for each word/type found in the dataset
        # Go through all docs / tweets, and for each doc...
        for i in range(self.data["docs"]["n_total"]):
            # Grab the tokens found in the phrase and the corresponding label.
            tokens = tokenize(texts[i])
            label = klasses[i]
            # Create key 'n_label' to access values: 'n_pos', 'n_neu', and 'n_neg' (# of label occurrences
            n_label = ""
            if label == "positive":
                n_label = "n_pos"
            elif label == "neutral":
                n_label = "n_neu"
            elif label == "negative":
                n_label = "n_neg"
            """ 
                'n_label' is a key used in the following ways: 
                       -> "words[word][n_label]"
                       -> "n_tokens[n_label]"
    
                The values accessed by 'n_label' represent the # of occurrences for each label in the set of documents.
            """
            # Go through all tokens and for each word (represented by 'word')...
            for word in tokens:
                # If it's a new word, add it to 'words' (a dict for word-related values)
                if word not in words.keys():
                    words[word] = {
                        "n_total": 0,
                        "n_pos": 0,
                        "n_neu": 0,
                        "n_neg": 0,  # Word occurrences (total and with labels)
                        "pr_pos": 0,
                        "pr_neu": 0,
                        "pr_neg": 0,  # Probabilities for each word given the label
                        "label": label,
                    }
                    n_types = len(
                        words.keys()
                    )  # Update the number of types when our vocabulary expands.
                # Update the token counts
                n_tokens[n_label] += 1
                n_tokens["n_total"] += 1
                # Update the 'words[word]' dict with the word's new number of occurrences (in all docs + all docs with label)
                words[word][n_label] += 1
                words[word]["n_total"] += 1
                # Calculate probabilities of word occurring with each label using Bayes Theorem
                pr_ = {
                    "positive": (words[word]["n_pos"] + 1)
                    / (n_tokens["n_pos"] + n_types),
                    "neutral": (words[word]["n_neu"] + 1)
                    / (n_tokens["n_neu"] + n_types),
                    "negative": (words[word]["n_neg"] + 1)
                    / (n_tokens["n_neg"] + n_types),
                }
                # Update 'words[word]' with new probabilities
                words[word]["pr_pos"] = pr_["positive"]
                words[word]["pr_neu"] = pr_["neutral"]
                words[word]["pr_neg"] = pr_["negative"]
        # Assign our filled words dict to the NaiveBayes instance
        self.data["words"] = words

    def classify(self, test_instance):
        # Grab tokens from test phrase
        tokens = tokenize(test_instance)
        # Create dict that will store the probabilties of each label for all tokens
        pr_words = {"pos": [], "neu": [], "neg": []}
        # For every word
        for word in tokens:
            # If we have it in our dataset, add the probabilities to 'pr_words'
            if word in self.data["words"].keys():
                pr_words["pos"].append(self.data["words"][word]["pr_pos"])
                pr_words["neu"].append(self.data["words"][word]["pr_neu"])
                pr_words["neg"].append(self.data["words"][word]["pr_neg"])
        # Calculate probabilities of each label for the test instance
        pr_pos = self.data["docs"]["pr_pos"] * product(pr_words["pos"])
        pr_neu = self.data["docs"]["pr_neu"] * product(pr_words["neu"])
        pr_neg = self.data["docs"]["pr_neg"] * product(pr_words["neg"])
        pr_labels = {"positive": pr_pos, "neutral": pr_neu, "negative": pr_neg}
        # Return the label with the highest probability
        prediction = max(pr_labels, key=pr_labels.get)
        return prediction


##Implement the binarized multinomial Naive Bayes model with smoothing
class BinaryNaiveBayes:
    def __init__(self, texts, klasses):
        self.data = {
            "words": {},  # Dict that will be filled with word data after training
            "docs": {  # Dict containing data about the dataset and its contained documents and labels
                "n_total": len(texts),
                "n_pos": klasses.count("positive"),
                "n_neu": klasses.count("neutral"),
                "n_neg": klasses.count("negative"),
                "pr_pos": klasses.count("positive") / len(texts),
                "pr_neu": klasses.count("neutral") / len(texts),
                "pr_neg": klasses.count("negative") / len(texts),
            },
        }
        self.train(texts, klasses)
        pass

    def train(self, texts, klasses):
        # Variables for tracking number of occurrences
        n_types = 0
        n_tokens = {"n_total": 0, "n_pos": 0, "n_neu": 0, "n_neg": 0}
        words = {}  # Dict that will store data for each word/type found in the dataset
        # Go through all docs / tweets, and for each doc...
        for i in range(self.data["docs"]["n_total"]):
            #  Grab the tokens & types found in the phrase and the corresponding label.
            tokens = tokenize(texts[i])
            types = list(set(tokens))
            label = klasses[i]
            # Create key 'n_label' to access values: 'n_pos', 'n_neu', and 'n_neg' (# of label occurrences
            n_label = ""
            if label == "positive":
                n_label = "n_pos"
            elif label == "neutral":
                n_label = "n_neu"
            elif label == "negative":
                n_label = "n_neg"
            """ 
                'n_label' is a key used in the following ways: 
                       -> "words[word][n_label]"
                       -> "n_tokens[n_label]"

                The values accessed by 'n_label' represent the # of occurrences for each label in the set of documents.
            """
            # Go through all types (binarized) and for each word (represented by 'word')...
            for word in types:
                # If it's a new word, add it to 'words' (a dict for word-related values)
                if word not in words.keys():
                    # Create dict
                    words[word] = {
                        "n_total": 0,
                        "n_pos": 0,
                        "n_neu": 0,
                        "n_neg": 0,  # Word occurrences (total and with labels)
                        "pr_pos": 0,
                        "pr_neu": 0,
                        "pr_neg": 0,  # Probabilities for each word given the label
                        "label": label,
                    }
                    # Update the number of types when our vocabulary expands.
                    n_types = len(words.keys())

                # Update the token counts
                n_tokens[n_label] += 1
                n_tokens["n_total"] += 1
                # Update the 'words[word]' dict with the word's new number of occurrences (in all docs + all docs with label)
                words[word][n_label] += 1
                words[word]["n_total"] += 1
                # Calculate probabilities of word occurring with each label using Bayes Theorem
                pr_ = {
                    "positive": (words[word]["n_pos"] + 1.0)
                    / (n_tokens["n_pos"] + n_types),
                    "neutral": (words[word]["n_neu"] + 1.0)
                    / (n_tokens["n_neu"] + n_types),
                    "negative": (words[word]["n_neg"] + 1.0)
                    / (n_tokens["n_neg"] + n_types),
                }
                # Update 'words[word]' with new probabilities
                words[word]["pr_pos"] = pr_["positive"]
                words[word]["pr_neu"] = pr_["neutral"]
                words[word]["pr_neg"] = pr_["negative"]
        # Assign our filled words dict to the NaiveBayes instance
        self.data["words"] = words

    def classify(self, test_instance):
        # Grab tokens from test phrase
        tokens = tokenize(test_instance)
        types = list(set(tokens))
        # Create dict that will store the probabilties of each label for all tokens
        pr_words = {"pos": [], "neu": [], "neg": []}
        # For every word
        for word in types:
            # If we have it in our dataset, add the probabilities to 'pr_words'
            if word in self.data["words"].keys():
                pr_words["pos"].append(self.data["words"][word]["pr_pos"])
                pr_words["neu"].append(self.data["words"][word]["pr_neu"])
                pr_words["neg"].append(self.data["words"][word]["pr_neg"])
        # Calculate probabilities of each label for the test instance
        pr_pos = self.data["docs"]["pr_pos"] * product(pr_words["pos"])
        pr_neu = self.data["docs"]["pr_neu"] * product(pr_words["neu"])
        pr_neg = self.data["docs"]["pr_neg"] * product(pr_words["neg"])
        pr_labels = {"positive": pr_pos, "neutral": pr_neu, "negative": pr_neg}
        # Return the label with the highest probability
        prediction = max(pr_labels, key=pr_labels.get)
        return prediction


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or
    # 'nbbin'
    method = sys.argv[1]

    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]

    train_texts = [x.strip() for x in open(train_texts_fname, encoding="utf8")]
    train_klasses = [x.strip() for x in open(train_klasses_fname, encoding="utf8")]
    test_texts = [x.strip() for x in open(test_texts_fname, encoding="utf8")]

    pos_words = [x.strip() for x in open("pos-words.txt", encoding="utf8")]
    neg_words = [x.strip() for x in open("neg-words.txt", encoding="utf8")]

    # Check which method is being asked to implement form user
    if method == "baseline":
        classifier = Baseline(train_klasses)

    elif method == "lr":
        # Use sklearn's implementation of logistic regression
        classifier = LogReg(train_texts, train_klasses)

    elif method == "lexicon":
        # Use baseline lexicon model
        classifier = Lexicon(pos_words, neg_words)

    elif method == "nb":
        # Use naive bayes model
        classifier = NaiveBayes(train_texts, train_klasses)

    elif method == "nbbin":
        # Used binarized naive bayes model
        classifier = BinaryNaiveBayes(train_texts, train_klasses)

    # Run the classify method for each instance
    results = [classifier.classify(x) for x in test_texts]

    # Create output file at given output file name
    # Store predictions in output file
    outFile = sys.argv[5]
    out = open(outFile, "w", encoding="utf-8")
    for r in results:
        out.write(r + "\n")
    out.close()
