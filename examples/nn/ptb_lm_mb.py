import numpy.random as RNG
import numpy as NP
import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *
class LMModel(Model):
    def __init__(self, vocab_size=10000, H_DIM =200, EMB_DIM=200):
        super(LMModel, self).__init__(loss='softmax_loss') # softmax don't support index target 

        self._embedding = Embedding(input_dim=vocab_size, output_dim=EMB_DIM) 
        self._rnn = RNN(H_DIM, 'tanh')
        self._linear = FullyConnected(num_hidden=H_DIM)

    def forward(self, data, mode='training'):
        embs = self._embedding(data)
        N, length, D = embs.shape

        hm1 = None
        hs = []
    
        for i in xrange(length):
            hm1 = self._rnn(embs[:, i, :], hm1)
            hs.append(hm1)

        hs = np.stack(hs, axis=1)
        return self._linear(hs.reshape((N*length, D))).reshape((N,length,D))

unpack_batch = lambda batch : (batch.data[0], batch.label[0])

if __name__ == '__main__':

    vocab_size= 10000
    seq_len = 35
    num_samples = 5000
    train_X = RNG.randint(vocab_size, size=(num_samples, seq_len))
    train_Y = NP.concatenate([train_X[:, 1:], NP.zeros((num_samples, 1))], axis=1) # raise an error if we use minpy.numpy
    test_X = RNG.randint(vocab_size, size=(num_samples, seq_len))
    test_Y = NP.concatenate([test_X[:, 1:], NP.zeros((num_samples, 1))], axis=1)

    batch_size = 64

    from minpy.nn.io import NDArrayIter
    train_data_iter = NDArrayIter(train_X, train_Y, batch_size=batch_size, shuffle=True)
    test_data_iter = NDArrayIter(test_X, test_Y, batch_size=batch_size, shuffle=False)

    model = LMModel(vocab_size=vocab_size, H_DIM=200, EMB_DIM=200)
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



