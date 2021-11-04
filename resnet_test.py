import mxnet as mx
import numpy as np
import memonger
import os
import struct
import gzip
from models import resnet

os.environ['MXNET_MEMORY_OPT'] = '1'
value = os.environ.get('MXNET_MEMORY_OPT')
print("MXNET_MEMORY_OPT = %s" % (value))

def save_debug_str(sym, type_dict=None, **kwargs):
    with open("log_resnet.txt", "w") as f:
        texec = sym.simple_bind(ctx=mx.gpu(), grad_req='write', type_dict=type_dict, **kwargs)
        f.write(texec.debug_str())

# 数据集
def read_data(label_url, image_url):
    with gzip.open(mx.test_utils.download(label_url)) as flbl:
        struct.unpack(">II", flbl.read(8))
        label = np.frombuffer(flbl.read(), dtype=np.int8)
    with gzip.open(mx.test_utils.download(image_url), 'rb') as fimg:
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
        image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32)/255
        image = np.resize(image, (1024, 3, 256, 256))
    return (label, image)

path = 'http://data.mxnet.io/data/mnist/'
(train_lbl, train_img) = read_data(path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
mnist =  {'train_data':train_img, 'train_label':train_lbl}

batch_size = 32
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)

# 构建网络
net = resnet.get_symbol(10, 50, "3,256,256")
# mx.viz.print_summary(net, {'data':(32,3,256,256),})

# call memory optimizer to search possible memory plan.
dshape = (batch_size, 3, 256, 256)
net_planned = memonger.search_plan(net, data=dshape)
# save_debug_str(net_planned, data=dshape)

old_cost = memonger.get_cost(net, data=dshape)
new_cost = memonger.get_cost(net_planned, data=dshape)

print('Old feature map cost=%d MB' % old_cost)
print('New feature map cost=%d MB' % new_cost)

ctx = mx.gpu(0)

# model = mx.mod.Module(net, context=ctx)
executor = net.simple_bind(ctx=ctx, grad_req='write', data=dshape)

# initialize the weights
for r in executor.arg_arrays:
    r[:] = np.random.randn(*r.shape)*0.02

cnt = 0
for batch in train_data:
    executor.arg_dict['data'] = batch.data[0]
    executor.arg_dict['softmax_label'] = batch.label[0]
    executor.forward()
    executor.backward()
    # update params
    for pname, W, G in zip(net.list_arguments(), executor.arg_arrays, executor.grad_arrays):
        # Don't update inputs
        if pname in ['data', 'softmax_label']:
            continue
        W[:] = W - G * .001
    cnt = cnt+1

mx.nd.waitall()
print("done for traning %d batches."%(cnt))
