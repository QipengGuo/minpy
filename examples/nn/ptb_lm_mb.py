import numpy.random as RNG
import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *

def RNNModel(Model):
    def __init__(self, H_DIM, vocab_size, EMB_DIM):
        super(RNNModel, self).__init__(loss='softmax_loss')

        self._embedding = Embedding(voab_size, EMB_DIM) # need to implement 
        self._rnn = RNN(H_DIM, 'tanh')
        self._linear = FullyConnected(num_hidden=H_DIM)

    def forward(self, data, mode='training'):
        embs = self._embedding(data)
        N, length, D = embs

        hm1 = None
        hs = []
    
        for i in xrange(length):
            hm1 = self._rnn(embs[:, i, :], hm1)
            hs.append(hm1)

        hs = np.stack(hs, axis=1)
        return self._linear(hs.reshape((N*length, D))).reshape((N,length,D))


if __name__ = '__main__':

    vocab_size= 10000
    seq_len = 35
    num_samples = 5000
    train_X = RNG.randint(vocab_size, size=(num_samples, seq_len))
    train_Y = np.concatenate([train_X[:, 1:], np.zeros((num_samples, 1))], axis=1)
    test_X = RNG.randint(vocab_size, size=(num_samples, seq_len))
    test_Y = np.concatenate([test_X[:, 1:], np.zeros((num_samples, 1))], axis=1)

    batch_size = 64

    from minpy.nn.io import NDArrayIter
    train_data_iter = NDArrayIter(train_X, train_Y, batch_size=batch_size, shuffle=True)
    test_data_iter = NDArrayIter([est_X, test_Y, batch_size=batch_size, shuffle=False)

    model = RNNModel(128, vocab_size, 200)
    updater = Updater(model, update_rule='rmsprop', learning_rate=0.001)


    iter_num = 0
    for ep in xrange(50):
        for batch in train_data_iter:
            iter_num += 1

            data, labels = unpack_batch(batch)
            grad_dict, loss = model.grad_and_loss(data, labels)
            updater(grad_dict)

            if iter_num % 100 ==0:
                print 'Iteration {0:} loss {1:.4f}'.format(iter_num, loss)

        test_data_iter.reset()
        errs, samples = 0, 0
        for batch in test_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data, 'inference')
            predictions = np.argmax(scores, axis=2)
            errs += np.count_nonzeros(predictions - labels)
            samples += data.shape[0]

        print 'epoch {0:} valid set error {1:.4f}'.format(ep, 1.0*errs/samples)



