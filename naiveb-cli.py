import sys
import argparse
import nltk

class NaiveBayesClassifier:

    def __init__(self):
        """
        Initializes an untrained Naive-Bayes classifier.
        """
        self.token_count = 0
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
        """
        # Read and split the training data.
        f = open(training_file_path, "r")
        contents = f.read()
        lines = contents.splitlines()
        f.close()

        for line in lines:
            # Tokenize and take the label.
            tokens = nltk.tokenize(line)
            label = tokens[-1]

            if label not in self.class_dict:
                # We have never seen this label before.
                self.class_dict[label] = {}

            if label not in self.document_dict:
                self.document_dict[label] = 0

            for token in tokens:
                # Count the token in every way that it needs to be counted.
                self.token_count += 1

                if token not in self.seen_tokens:
                    self.seen_tokens.append(token)
                    self.vocabulary_size += 1

                if token in self.class_dict[label]:
                    self.class_dict[label][token] += 1
                else:
                    self.class_dict[label][token] = 1
                
            self.document_dict[label] += 1

def main():
    # Set up the argument parser.
    parser = argparse.ArgumentParser(description="Trains a Naive-Bayes classifier and scores a sentence against it.")
    parser.add_argument('training_file', metavar='training_file', type=str, help="The file path of the training file. ")
    args = parser.parse_args()

    # Initialize and train the classifier.
    training_file_path = args.training_file
    nbc = NaiveBayesClassifier()
    nbc.train(training_file_path)

if __name__ == "__main__":
    main()
            
            