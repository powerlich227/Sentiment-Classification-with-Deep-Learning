import tarfile
import sys
import os
import io
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

reload(sys)
sys.setdefaultencoding('ISO-8859-1')


# Task 1: Load the data
# For this task you will load the data, create a vocabulary and encode the reviews with integers

def read_file(path_to_dataset, path_to_data):
    """
    :param path_to_dataset: a path to the tar file (dataset)
    :param path_to_data: a path to the extracted files
    :return: two lists, one containing the movie reviews and another containing the corresponding labels
    """

    if not tarfile.is_tarfile(path_to_dataset):
        sys.exit("Input path is not a tar file")
    dirent = tarfile.open(path_to_dataset)
    dirent.extractall()
    dirent.close()

    data = []
    labels = []
    if not os.path.isdir(path_to_data):
        sys.exit("Input path is not a directory")
    for label in os.listdir(path_to_data):
        path = os.path.join(path_to_data, label)
        if os.path.isdir(path):
            for filename in os.listdir(path):
                labels.append(label)
                filepath = os.path.join(path, filename)
                if os.path.isfile(filepath):
                    try:
                        reader = io.open(filepath, encoding='ISO-8859-1')
                        text = reader.read()
                        text = text.strip('\n')
                        # print text
                        data.append(text)
                    except IOError:
                        sys.exit("Cannot read file")
    # print data
    # print labels
    return data, labels


def preprocess(text):
    """
    :param text: list of sentences or movie reviews
    :return: a dict of all tokens you encounter in the dataset. i.e. the vocabulary of the dataset
    Associate each token with a unique integer
    """
    if type(text) is not list:
        sys.exit("Please provide a list to the method")
    vocab = {}
    number = 1
    for review in text:
        word = review.strip().split()
        # print word
        for token in word:
            if token not in vocab:
                vocab[token] = number
                number += 1
    # print vocab
    return vocab


def encode_review(vocab, text):
    """
    :param vocab: the vocabulary dictionary you obtained from the previous method
    :param text: list of movie reviews obtained from the previous method
    :return: encoded reviews
    """

    if type(vocab) is not dict or type(text) is not list:
        sys.exit("Please provide a list to the method")

    data = []
    for review in text:
        word = review.strip().split()
        for token in word:
            if token in vocab:
                word[word.index(token)] = vocab[token]
        data.append(word)

    # print data
    return data


def encode_labels(labels):  # Note this method is optional (if you have not integer-encoded the labels)
    """
    :param labels: list of labels associated with the reviews
    :return: encoded labels
    """

    if type(labels) is not list:
        sys.exit("Please provide a list to the method")
    for label in labels:
        if label == "neg":
            labels[labels.index(label)] = 0
        if label == "pos":
            labels[labels.index(label)] = 1

    # print labels
    return labels


def pad_zeros(encoded_reviews, seq_length=200):
    """
    :param encoded_reviews: integer-encoded reviews obtained from the previous method
    :param seq_length: maximum allowed sequence length for the review
    :return: encoded reviews after padding zeros
    """

    if type(encoded_reviews) is not list:
        sys.exit("Please provide a list to the method")

    for encoded_review in encoded_reviews:
        if len(encoded_review) > seq_length:
            seq_length = len(encoded_review)
    for encoded_review in encoded_reviews:
        pad_zero = [0] * (seq_length - len(encoded_review))
        encoded_reviews[encoded_reviews.index(encoded_review)].extend(pad_zero)

    # print encoded_reviews
    # print len(encoded_reviews)
    # print seq_length
    return encoded_reviews


# Task 2: Load the pre-trained embedding vectors
# For this task you will load the pre-trained embedding vectors from Word2Vec

def load_embedding_file(embedding_file, token_dict):
    """
    :param embedding_file: path to the embedding file
    :param token_dict: token-integer mapping dict obtained from previous step
    :return: embedding dict: embedding vector-integer mapping
    """

    if not os.path.isfile(embedding_file):
        sys.exit("Input embedding path is not a file")
    if type(token_dict) is not dict:
        sys.exit("Input a dictionary!")

    embedding_dict = {}
    with io.open(embedding_file, encoding="ISO-8859-1") as vec_file:
        lines = vec_file.readlines()
        first_line = lines[0].strip().split()
        first_line = map(int, first_line)
        tensor_size = first_line[1]
        zero_pad = [0] * tensor_size
        zero_pad_tensor = torch.FloatTensor(zero_pad)

        embedding_dict[0] = zero_pad_tensor
        for token in token_dict:
            embedding_dict[token_dict[token]] = zero_pad_tensor

        del lines[0]
        for line in lines:
            vector = line.strip().split()
            for token in token_dict:
                if token == vector[0]:
                    vector.remove(token)
                    vector = map(float, vector)
                    vector_tensor = torch.FloatTensor(vector)
                    # print token_dict[token]
                    embedding_dict[token_dict[token]] = vector_tensor

    # print embedding_dict
    return embedding_dict, tensor_size


# Task 3: Create a TensorDataset and DataLoader

def create_data_loader(encoded_reviews, labels, batch_size=32):
    """
    :param encoded_reviews: zero-padded integer-encoded reviews
    :param labels: integer-encoded labels
    :param batch_size: batch size for training
    :return: DataLoader object
    """

    if type(encoded_reviews) is not list or type(labels) is not list:
        sys.exit("Please provide a list to the method")

    encoded_reviews_tensor = torch.LongTensor(encoded_reviews)
    # print encoded_reviews_tensor
    # print encoded_reviews_tensor.size()
    labels_tensor = torch.LongTensor(labels)
    # print labels_tensor
    # print labels_tensor.size()
    dataset = TensorDataset(encoded_reviews_tensor, labels_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=2,
    )

    # print loader.dataset
    # print dataset.tensors[0].size()
    return loader


# Task 4: Define the Baseline model here

# This is the baseline model that contains an embedding layer and an fcn for classification
def create_embedding_layer(input_words, emb_dict, pad_data):
    x, y = input_words.size()
    emb_layer = torch.zeros(x, y, embedding_dim)
    for j in range(x):
        for k in range(y):
            emb_layer[j][k] = emb_dict[pad_data[j][k]]
    # print emb_layer
    return emb_layer


class BaseSentiment(nn.Module):
    def __init__(self):
        super(BaseSentiment, self).__init__()
        self.emb_layer = None
        self.fcn = nn.Linear(300, 2)

    def forward(self, input_words, emb_dict, pad_data):
        self.emb_layer = create_embedding_layer(input_words, emb_dict, pad_data)
        out = self.fcn(self.emb_layer)
        # print out.shape
        # print out
        return out[:, -1]


# Task 5: Define the RNN model here

# This model contains an embedding layer, an rnn and an fcn for classification
class RNNSentiment(nn.Module):
    def __init__(self, layer, bi):
        super(RNNSentiment, self).__init__()
        self.emb_layer = None
        self.rnn = nn.RNN(300, 300, num_layers=layer, bidirectional=bi)
        if bi:
            self.fcn = nn.Linear(600, 2)
        else:
            self.fcn = nn.Linear(300, 2)

    def forward(self, input_words, emb_dict, pad_data):
        self.emb_layer = create_embedding_layer(input_words, emb_dict, pad_data)
        out, _ = self.rnn(self.emb_layer)
        out = self.fcn(out)
        return out[:, -1]


# Task 6: Define the self-attention model here

# This model contains an embedding layer, self-attention and an fcn for classification
class AttentionSentiment(nn.Module):
    def __init__(self):
        super(AttentionSentiment, self).__init__()
        self.emb_layer = None
        self.attn = nn.MultiheadAttention(300, num_heads=1)
        self.fcn = nn.Linear(300, 2)

    def forward(self, input_words, emb_dict, pad_data):
        self.emb_layer = create_embedding_layer(input_words, emb_dict, pad_data)
        out = self.attn(self.emb_layer, self.emb_layer, self.emb_layer)[0]
        out = self.fcn(out)
        return out[:, -1]


"""
ALL METHODS AND CLASSES HAVE BEEN DEFINED! TIME TO START EXECUTION!!
"""

# Task 7: Start model training and testing
if __name__ == "__main__":

    tarfile_path = sys.argv[1]
    train_file_path = tarfile_path[:-7] + '/train'
    test_file_path = tarfile_path[:-7] + '/test'
    # Instantiate all hyper-parameters and objects here
    # train_data, train_labels = read_file('movie_reviews.tar.gz', 'test/train')
    # test_data, test_labels = read_file('movie_reviews.tar.gz', 'test/test')
    train_data, train_labels = read_file(tarfile_path, train_file_path)
    test_data, test_labels = read_file(tarfile_path, test_file_path)

    train_vocabulary = preprocess(train_data)
    test_vocabulary = preprocess(test_data)

    train_encoded_data = encode_review(train_vocabulary, train_data)
    test_encoded_data = encode_review(test_vocabulary, test_data)

    train_padded_data = pad_zeros(train_encoded_data)
    test_padded_data = pad_zeros(test_encoded_data)

    train_encoded_labels = encode_labels(train_labels)
    test_encoded_labels = encode_labels(test_labels)

    train_data_loader = create_data_loader(train_padded_data, train_encoded_labels)
    test_data_loader = create_data_loader(test_padded_data, test_encoded_labels)

    vec_path = sys.argv[2]
    # train_embedding_dict, embedding_dim = load_embedding_file('test.vec', train_vocabulary)
    # test_embedding_dict, _ = load_embedding_file('test.vec', test_vocabulary)
    train_embedding_dict, embedding_dim = load_embedding_file(vec_path, train_vocabulary)
    test_embedding_dict, _ = load_embedding_file(vec_path, test_vocabulary)

    # Define loss here
    criterion = nn.CrossEntropyLoss()
    model = sys.argv[3]

    correct = 0
    total = 0

    if model == "base":
        lrate = float(sys.argv[4])
        iterations = int(sys.argv[5])
        base_sentiment = BaseSentiment()
        # attention_sentiment = AttentionSentiment()

        # Define optimizer here
        optimizer_base = optim.SGD(base_sentiment.parameters(), lr=lrate, momentum=0.9)

        # Training starts!!
        base_sentiment.train()
        for _ in range(iterations):
            for _, data in enumerate(train_data_loader):
                # get the input
                inputs, labels = data
                # wrap time in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer_base.zero_grad()

                # forward + backward + optimize
                output1 = base_sentiment(inputs, train_embedding_dict, train_padded_data)
                loss = criterion(output1, labels)
                loss.backward()
                optimizer_base.step()

            print (loss.data)

        # Testing starts!!
        for _, data in enumerate(test_data_loader):
            # get the input
            inputs, labels = data

            # wrap time in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            output1 = base_sentiment(inputs, test_embedding_dict, test_padded_data)
            _, predicted = torch.max(output1.data, 1)
            # print output1.data
            # print predicted
            # print labels
            total += labels.size(0)
            correct += (predicted == labels).sum()

    if model == "rnn":
        layer = int(sys.argv[4])
        bi_dir = bool(int(sys.argv[5]))
        lrate = float(sys.argv[6])
        iterations = int(sys.argv[7])
        rnn_sentiment = RNNSentiment(layer, bi_dir)
        optimizer_rnn = optim.SGD(rnn_sentiment.parameters(), lr=lrate, momentum=0.9)

        # Training starts!!
        rnn_sentiment.train()
        for _ in range(iterations):
            for _, data in enumerate(train_data_loader):
                # get the input
                inputs, labels = data
                # wrap time in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer_rnn.zero_grad()

                # forward + backward + optimize
                output2 = rnn_sentiment(inputs, train_embedding_dict, train_padded_data)
                loss = criterion(output2, labels)
                loss.backward()
                optimizer_rnn.step()

            print(loss.data)

        # Testing starts!!
        for _, data in enumerate(test_data_loader):
            # get the input
            inputs, labels = data

            # wrap time in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            output2 = rnn_sentiment(inputs, test_embedding_dict, test_padded_data)
            # torch.FloatTensor.abs_(output2)
            _, predicted = torch.max(output2, 1)
            # print output2
            # print predicted
            # print labels
            total += labels.size(0)
            correct += (predicted == labels).sum()

    if model == "att":
        lrate = float(sys.argv[4])
        iterations = int(sys.argv[5])
        att_sentiment = AttentionSentiment()
        optimizer_att = optim.SGD(att_sentiment.parameters(), lr=lrate, momentum=0.9)

        att_sentiment.train()
        for _ in range(iterations):
            for _, data in enumerate(train_data_loader):
                # get the input
                inputs, labels = data
                # wrap time in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer_att.zero_grad()
                output3 = att_sentiment(inputs, train_embedding_dict, train_padded_data)
                loss = criterion(output3, labels)
                loss.backward()
                optimizer_att.step()

            print (loss.data)

        # Testing starts!!
        for _, data in enumerate(test_data_loader):
            # get the input
            inputs, labels = data

            # wrap time in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            output3 = att_sentiment(inputs, test_embedding_dict, test_padded_data)
            _, predicted = torch.max(output3, 1)
            # print output3
            # print predicted
            # print labels
            total += labels.size(0)
            print labels.size()
            correct += (predicted == labels).sum()

    # print(total)
    # print(correct)
    print('Accuracy of the deep learning on the test samples: %d %%' % (100 * correct / total))
