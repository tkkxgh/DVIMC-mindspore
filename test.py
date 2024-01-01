import numpy as np

data = np.array([0.2, 0.5, 0.2], dtype=np.float32)
label = np.array([1, 0], dtype=np.float32)
label_pt = np.array([0], dtype=np.float32)

import mindspore as ms
from mindspore.common.initializer import initializer, Zero
class Net2(ms.nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.dense = ms.nn.Dense(3, 2)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        if isinstance(cell, ms.nn.Dense):
            cell.weight.set_data(initializer(Zero(), cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        return self.dense(x)

net2 = Net2()
loss_fn = ms.nn.CrossEntropyLoss()

def forward_fn(data, label):
    logits = net2(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ms.grad(forward_fn, grad_position=None, weights=net2.trainable_params(), has_aux=True)
grads = grad_fn(ms.Tensor(data), ms.Tensor(label))
print(grads)
# Before clip out:
# ((Tensor(shape=[2, 3], dtype=Float32, value=
# [[-1.00000001e-01, -2.50000000e-01, -1.00000001e-01],
#  [ 1.00000001e-01,  2.50000000e-01,  1.00000001e-01]]), Tensor(shape=[2], dtype=Float32, value= [-5.00000000e-01,  5.00000000e-01])), (Tensor(shape=[2], dtype=Float32, value= [ 0.00000000e+00,  0.00000000e+00]),))
grads = ms.ops.clip_by_value(grads, clip_value_min=-0.1, clip_value_max=0.1)
print(grads)