# MXNet Memory Monger

This project contains a 150 lines of python script to give sublinear memory plans of deep neural networks.
This allows you to trade computation for memory and get sublinear memory cost,
so you can train bigger/deeper nets with limited resources.

The algorithm is described in the paper

[Training Deep Nets with Sublinear Memory Cost](xx).

## How to Use

This code is based on [MXNet](https://github.com/dmlc/mxnet), a lightweight, flexible and efficient framework for deep learning.

- Configure your network as you normally will do using symbolic API
- Give hint to the allocator about the possible places that can be critical points
  - Set attribute ```mirror_stage='True'```, see [example_resnet.py](example_resnet.py#L25)
- Call ```memonger.search_plan``` to get an symbolic graph with memory plan.

```python
import mxnet as mx
import memonger

# configure your network
net = my_symbol()

# call memory optimizer to search possible memory plan.
net_planned = memonger.search_plan(net)

# use as normal
model = mx.FeedForward(net, ...)
model.fit(...)
```

## Write your Own Memory Optimizer

MXNet's symbolic graph support attribute to give hint on whether (mirror attribute) a result
can be recomputed or not. You can choose to re-compute instead of remembering a result
for less memory consumption. To set output of a symbol to be re-computable, use
```python
sym._set_attr(force_mirroring='True')
```

mxnet-memonger actually use the same way to do memory planning. You can simply write your own memory
allocator by setting the force_mirroring attribute in a smart way.