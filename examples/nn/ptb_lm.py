import minpy.numpy as np
from minpy.context import set_context, gpu
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from txt_data_final import * # data pre-processing 
import argparse


set_context(gpu(0)) # set the global context with gpu

def softmax_crossentropy(x, y):
    # x should be (batch, prob)
    # y should be (batch, )

    x_dev = x - np.max(x, axis=1, keepdims=True) # minpy doesn't support x.max()
    sm = x_dev - np.log(np.sum(np.exp(x_dev), axis=1, keepdims=True))
    ids = np.arange(0, y.shape[0])*x.shape[1] + y
    ce = -np.sum(sm.reshape((sm.shape[0]*sm.shape[1],))[ids])/(1.0*y.shape[0])  # minpy doesn't support -1 in shape inference
    return ce

class LM_RNN(ModelBase):
    def __init__(self, batch_size=64, WORD_DIM=10000):
        super(LM_RNN, self).__init__()
        self.WORD_DIM = WORD_DIM # vocabulary size
        self.WORD_EMB_DIM = 200 # embedding dim
        self.HID_DIM = 400 # GRU hidden size

        self.add_param(name='W_Emb', shape=(self.WORD_DIM,self.WORD_EMB_DIM))\
            .add_param(name='W_GRU_h', shape=(self.WORD_EMB_DIM, self.HID_DIM))\
            .add_param(name='U_GRU_h', shape=(self.HID_DIM, self.HID_DIM))\
            .add_param(name='b_GRU_h', shape=(self.HID_DIM,))\
            .add_param(name='W_GRU_g', shape=(self.WORD_EMB_DIM, self.HID_DIM*2))\
            .add_param(name='U_GRU_g', shape=(self.HID_DIM, self.HID_DIM*2))\
            .add_param(name='b_GRU_g', shape=(self.HID_DIM*2,))\
            .add_param(name='W_Softmax', shape=(self.HID_DIM, self.WORD_DIM))\
            .add_param(name='b_Softmax', shape=(self.WORD_DIM,))

    def one_step(self, x, prev_h):
        h = layers.gru_step(x, prev_h, self.params['W_GRU_g'], self.params['U_GRU_g'], self.params['b_GRU_g'], self.params['W_GRU_h'], self.params['U_GRU_h'], self.params['b_GRU_h'])
        return h

    def forward(self, X, mode):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        X_emb = self.params['W_Emb'][X]
        hm1 = np.zeros((batch_size, self.HID_DIM))
        hs = []
        for t in xrange(seq_len):
            hm1 = self.one_step(X_emb[:,t,:], hm1)
            hs.append(hm1)
        hs = np.stack(hs, axis=1).reshape((batch_size*seq_len, self.HID_DIM))
        pred_out = layers.affine(hs, self.params['W_Softmax'], self.params['b_Softmax'])
        return pred_out.reshape((batch_size, seq_len, self.WORD_DIM))
    # the func check_accuracy in nn/solver.py may raise an error because it reduce the prob dim with axis 1 but not the last dim


    def loss(self, predict, y):
        return softmax_crossentropy(predict.reshape((predict.shape[0]*predict.shape[1], predict.shape[2])), y.reshape((y.shape[0]*y.shape[1],))) 


def get_data(opts, test=False, post_name='.keep50kr'):
    return txt_data(opts.data_name, batch_size = opts.batch_size, test=test, post_name=post_name)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='data/ptb')
#    parser.add_argument('--train')
#    parser.add_argument('--test')
#    parser.add_argument('--fname', type=str)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    
    Dataset = get_data(args)
    train_word = Dataset.train_word.reshape((-1, 35))
    train_Yword = Dataset.train_Yword.reshape((-1, 35))
    test_word = Dataset.test_word.reshape((-1, 35))
    test_Yword = Dataset.test_Yword.reshape((-1, 35))

    train_dataiter = NDArrayIter(train_word, train_Yword, batch_size=64, shuffle=True)
    test_dataiter = NDArrayIter(test_word, test_Yword, batch_size=64, shuffle=False)

    model = LM_RNN(batch_size=64, WORD_DIM=Dataset.w_dim+1)
    solver = Solver(model, train_dataiter, test_dataiter, num_epochs=2, init_rule='xavier', update_rule='adam', print_every=20)
    solver.init()
    solver.train()
