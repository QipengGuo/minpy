import numpy.random as RNG
import numpy as NP
import minpy.numpy as np
from minpy.nn.model_builder import *
from minpy.nn.modules import *
from minpy.context import set_context, gpu

set_context(gpu(0)) # set the global context with gpu

def softmax_crossentropy(x, y):
    x = x.reshape((x.shape[0]*x.shape[1], -1))
    y = y.reshape((-1,))

    # x should be (batch, prob)
    # y should be (batch, )

    x_dev = x - np.max(x, axis=1, keepdims=True) # minpy doesn't support x.max()
    sm = x_dev - np.log(np.sum(np.exp(x_dev), axis=1, keepdims=True))
    ids = np.arange(0, y.shape[0])*x.shape[1] + y
    ce = -np.sum(sm.reshape((sm.shape[0]*sm.shape[1],))[ids])/(1.0*y.shape[0])  # minpy doesn't support -1 in shape inference
    return ce

class RegModel(Model):
    def __init__(self, H_DIM=200, OUT_DIM=10):
        super(RegModel, self).__init__(loss=['l2_loss', softmax_crossentropy])

        self._linear1 = FullyConnected(num_hidden=H_DIM)
        self._linear2 = FullyConnected(num_hidden=OUT_DIM)
        self._linear3 = FullyConnected(num_hidden=OUT_DIM)

    def forward(self, data, mode='training'):
        h = relu(self._linear1(data))
        h2 = self._linear2(h)

        return h

    def forward2(self, data, mode='training'):
        h = relu(self._linear1(data))
        h2 = self._linear3(h)

        return h


unpack_batch = lambda batch : (batch.data[0], batch.label[0])

if __name__ == '__main__':

    num_samples = 5000
    IN_DIM = 10
    H_DIM = 200
    OUT_DIM = 10
    train_X = RNG.random((num_samples, IN_DIM))
    train_Y1 = train_X**2
    train_Y2 = train_X**0.5

    test_X = RNG.random((num_samples, IN_DIM))
    test_Y1 = test_X**2
    test_Y2 = test_X**0.5

    batch_size = 64

    from minpy.nn.io import NDArrayIter
    train_data_iter1 = NDArrayIter(train_X, train_Y1, batch_size=batch_size, shuffle=True)
    train_data_iter2 = NDArrayIter(train_X, train_Y2, batch_size=batch_size, shuffle=True)
    test_data_iter1 = NDArrayIter(test_X, test_Y1, batch_size=batch_size, shuffle=False)
    test_data_iter2 = NDArrayIter(test_X, test_Y2, batch_size=batch_size, shuffle=False)

    model = RegModel(H_DIM=H_DIM, OUT_DIM=OUT_DIM)
    updater1 = Updater(model, update_rule='rmsprop', learning_rate=0.001)
    updater2 = Updater(model, update_rule='rmsprop', learning_rate=0.001)

    print type(model)

    iter_num = 0
    for ep in xrange(50):
        for i in xrange(num_samples/batch_size):
            iter_num += 1
            if iter_num%2==0:
                data, labels = unpack_batch(train_data_iter1.next())
                grad_dict, loss = model.grad_and_loss(data, labels, f=model.forward)
                updater1(grad_dict)
            else:
                data, labels = unpack_batch(train_data_iter2.next())
                grad_dict, loss = model.grad_and_loss(data, labels, f=model.forward2)
                updater2(grad_dict)

            if iter_num % 100 ==0:
                print 'Iteration {0:} loss {1:.4f}'.format(iter_num, loss)

        test_data_iter.reset()
        errs1, samples1, errs2, samples2 = 0, 0, 0, 0
        for i in xrange(num_sampels/batch_size):
            data, labels = unpack_batch(test_data_iter1.next())
            scores = model.forward(data, 'inference', f=model.forward)
            predictions = np.argmax(scores, axis=2)
            errs1 += np.count_nonzeros(predictions - labels)
            samples1 += data.shape[0]
            data, labels = unpack_batch(test_data_iter2.next())
            scores = model.forward(data, 'inference', f=model.forward2)
            predictions = np.argmax(scores, axis=2)
            errs2 += np.count_nonzeros(predictions - labels)
            samples2 += data.shape[0]

        print 'epoch {0:} valid set error 1 {1:.4f} error 2 {2:.4f}'.format(ep, 1.0*errs1/samples1, 1.0*errs2/samples2)



