import numpy.random as RNG
import numpy as NP
import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.context import set_context, gpu
import h5py
import os
import time 

set_context(gpu(1)) # set the global context with gpu

def softmax_crossentropy(x, y):
    EPSI = 1e-6
    batch_size, seq_len, prob_dim = x.shape
    x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
    y = y.reshape((y.shape[0]*y.shape[1],))

    #print x.shape, y.shape
    # x should be (batch, prob)
    # y should be (batch, )

    x_dev = x - np.max(x, axis=1, keepdims=True) # minpy doesn't support x.max()
    sm = x_dev - np.log(EPSI+np.sum(np.exp(x_dev), axis=1, keepdims=True))
    ids = np.arange(0, y.shape[0])*seq_len+ y
    ce = -np.sum(sm.reshape((sm.shape[0]*sm.shape[1],))[ids])/(1.0*y.shape[0])  # minpy doesn't support -1 in shape inference
    return ce

class LMModel(Model):
    def __init__(self, vocab_size=10000, H_DIM =200, EMB_DIM=200):
        super(LMModel, self).__init__(loss=softmax_crossentropy) # softmax don't support index target 

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

unpack_batch = lambda batch : (batch.data[0].asnumpy(), batch.label[0].asnumpy())

if __name__ == '__main__':

    vocab_size= 10000
    seq_len = 35
    if os.path.exists('ptb_train.h5') and os.path.exists('ptb_valid.h5'):
        print 'Load PTB data'
        f = h5py.File('ptb_train.h5', 'r')
        train_X = f['word'][:].reshape((-1, seq_len))
        train_Y = f['Yword'][:].reshape((-1, seq_len))
        f.close()
        f = h5py.File('ptb_valid.h5', 'r')
        test_X = f['word'][:].reshape((-1, seq_len))
        test_Y = f['Yword'][:].reshape((-1, seq_len))
        f.close()
    else:
        print 'Gen Random data'
        num_samples = 5000
        train_X = RNG.randint(vocab_size, size=(num_samples, seq_len))
        train_Y = NP.concatenate([train_X[:, 1:], NP.zeros((num_samples, 1))], axis=1) # raise an error if we use minpy.numpy
        test_X = RNG.randint(vocab_size, size=(num_samples, seq_len))
        test_Y = NP.concatenate([test_X[:, 1:], NP.zeros((num_samples, 1))], axis=1)

    batch_size = 20

    from minpy.nn.io import NDArrayIter
    train_data_iter = NDArrayIter(train_X, train_Y, batch_size=batch_size, shuffle=True)
    test_data_iter = NDArrayIter(test_X, test_Y, batch_size=batch_size, shuffle=False)

    model = LMModel(vocab_size=vocab_size, H_DIM=200, EMB_DIM=200)
    updater = Updater(model, update_rule='sgd', learning_rate=0.001)


    mt = time.time()
    iter_num = 0
    for ep in xrange(50):
        train_data_iter.reset()
        for batch in train_data_iter:
            iter_num += 1

            data, labels = unpack_batch(batch)
            loss = model(data, labels=labels)
            #grad_dict, loss = model.grad_and_loss(data, labels)
            grad_dict = model.backward()
            updater(grad_dict)

            if iter_num % 10 ==0:
                print 'Iteration {0:} Time {1:.2f}, loss {2:.4f}'.format(iter_num, time.time()-mt, float(loss.asnumpy()))
                mt = time.time()

        test_data_iter.reset()
        errs, samples = 0, 0
        for batch in test_data_iter:
            data, labels = unpack_batch(batch)
            scores = model.forward(data, 'inference')
            predictions = np.argmax(scores, axis=2)
            errs += np.count_nonzero(predictions - labels)
            samples += data.shape[0]

        print 'epoch {0:} valid set error {1:.4f}'.format(ep, 1.0*errs/samples)



