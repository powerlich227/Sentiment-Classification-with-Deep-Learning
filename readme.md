In the case of baseline model,
For example, type python project1.py test.tar.gz test.vec base 0.001 5,
where test.tar.gz denote the training and test file, test.vec denote the embedding dictionary.
base denote baseline model, 0.001 denote learning rate and 5 denote epoch times

In the case of rnn model,
For example, type python project1.py test.tar.gz test.vec rnn 1 0 0.001 5,
where test.tar.gz denote the training and test file, test.vec denote the embedding dictionary.
rnn denote rnn model, 1 denote the number of rnn layer, 0/1 denote bidirectional(1) or not(0), 0.001 denote learning rate and 5 denote epoch times

In the case of self-attention model,
For example, type python project1.py test.tar.gz test.vec att 0.001 5,
where test.tar.gz denote the training and test file, test.vec denote the embedding dictionary.
att denote self-attention model, 0.001 denote learning rate and 5 denote epoch times

It will return the loss and accuracy as below:
tensor(0.6887)
tensor(0.7047)
tensor(0.6993)
tensor(0.6992)
tensor(0.6940)
Accuracy of the deep learning on the test samples: 52 %

The zip file also include test.tar.gz and test.vec for quick test.

The tarfilename should be same as father directory name.
test.tar.gz
test/train
test/test
test/train/pos
test/train/neg
test/test/pos
test/test/neg

All codes successfully run in MacOS 10.14.6 and python 2.7.10