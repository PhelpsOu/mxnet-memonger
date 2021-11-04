import mxnet as mx
import numpy as np
import memonger
import os
import struct
import gzip
from models import vggnet

#os.environ['MXNET_MEMORY_OPT'] = '1'
value = os.environ.get('MXNET_MEMORY_OPT')
print("MXNET_MEMORY_OPT = %s" % (value))

def save_debug_str(sym, type_dict=None, **kwargs):
    with open("log_vgg.txt", "w") as f:
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
        image = np.resize(image, (640, 3, 256, 256))
    return (label, image)

path = 'http://data.mxnet.io/data/mnist/'
(train_lbl, train_img) = read_data(path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz')
mnist =  {'train_data':train_img, 'train_label':train_lbl}

batch_size = 32
train_data = mx.io.NDArrayIter(mnist["train_data"], mnist["train_label"], batch_size, shuffle=True)

# 构建网络
net = vggnet.get_symbol(10, num_layers=11)
# mx.viz.print_summary(net, {'data':(32,3,256,256),})

# call memory optimizer to search possible memory plan.
dshape = (batch_size, 3, 256, 256)
net_planned = memonger.search_plan(net, data=dshape)
save_debug_str(net_planned, data=dshape)

old_cost = memonger.get_cost(net, data=dshape)
new_cost = memonger.get_cost(net_planned, data=dshape)

print('Old feature map cost=%d MB' % old_cost)
print('New feature map cost=%d MB' % new_cost)

ctx = [mx.gpu(0)]

# model = mx.mod.Module(net, context=ctx)
model = mx.mod.Module(net_planned, context=ctx)

model.bind(train_data.provide_data, train_data.provide_label)
model.init_params()
model.init_optimizer()

for batch in train_data:
    model.forward_backward(batch)
    model.update()
