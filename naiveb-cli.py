import sys
import argparse
import nltk
import numpy

class NaiveBayesClassifier:

    def __init__(self):
        """
        Initializes an untrained Naive-Bayes classifier.
        """
        self.vocabulary_size = 0
        self.seen_tokens = []
        self.class_dict = {}
        self.document_dict = {}

    def train(self, training_file_path):
        """
        Trains the model on the given training file. Checks each line of the 
        input for its content and final token label. Classifies each token by its label.
        Parameters: 
            training_file_path (str): File path of the training .txt file.
        Returns:
            Nothing.
        """
        # Read and split the training data.
        f = open(training_file_path, "r")
        contents = f.read()
        lines = contents.splitlines()
        f.close()

        for line in lines:
            # Tokenize and take the label.
            tokens = nltk.tokenize.word_tokenize(line)
            label = tokens[-1]

            if label not in self.class_dict:
                # We have never seen this label before.
                self.class_dict[label] = {}

            if label not in self.document_dict:
                self.document_dict[label] = 0

            for token in tokens:
                # Count the token in every way that it needs to be counted.
                if token not in self.seen_tokens:
                    self.seen_tokens.append(token)
                    self.vocabulary_size += 1

                if token in self.class_dict[label]:
                    self.class_dict[label][token] += 1
                else:
                    self.class_dict[label][token] = 1
                
            self.document_dict[label] += 1

    def score(self, text):
        """
        Scores the given text, using this trained classifier. Uses Laplace Smoothing by default.
        This function chooses to take the product of the probabilities, as opposed to using sum of logs.
        Parameters:
            text (str): The text being scored against the classifier.
        Returns:
            Dictionary of the text's score in each known class.
        """
        # Tokenize the text.
        tokens = nltk.tokenize.word_tokenize(text)
        score_dict = {}

        for key in self.class_dict:
            # Calculate the prior.
            prior = self.document_dict[key] / len(self.document_dict)
            word_probs = []
            for token in tokens:
                # Are we familiar with this token in any class?
                if token not in self.seen_tokens:
                    continue

                # Are we familiar with this token in this class?
                if token in self.class_dict[key]:
                    this_prob = ((self.class_dict[key][token] + 1) / (self.vocabulary_size + self.document_dict[key]))
                    word_probs.append(this_prob)
                else:
                    this_prob = (1 / (self.vocabulary_size + self.document_dict[key]))
                    word_probs.append(this_prob)

            # Calculate the score for this class.
            class_score = numpy.prod(word_probs) * prior
            score_dict[key] = class_score

        return score_dict

def main():
    # Set up the argument parser.
    parser = argparse.ArgumentParser(description="Trains a Naive-Bayes classifier and scores a sentence against it.")
    parser.add_argument('training_file', metavar='training_file', type=str, help="The file path of the training file. ")
    args = parser.parse_args()

    # Initialize and train the classifier.
    training_file_path = args.training_file
    nbc = NaiveBayesClassifier()
    nbc.train(training_file_path)

    # Prompt for a sentence to score.
    to_score = input("Enter a string for the classifier to score: ")
    to_score = to_score.lower()

    # Score the sentence.
    score_dict = nbc.score(to_score)

    for key in score_dict:
        print(str(key) + ": " + str(score_dict[key]))

if __name__ == "__main__":
    main()
            
            