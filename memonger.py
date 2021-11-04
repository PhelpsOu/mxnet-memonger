import mxnet as mx
import math

debug = True

def prod(shape):
    """Get product of the shape.
    """
    ret = 1
    for s in shape:
        ret *= s
    return ret


def is_param(name):
    """Quick script to check if name is a parameter.
    """
    if name == 'data':
        return False
    if name.endswith('weight'):
        return True
    if name.endswith('bias'):
        return True
    if name.endswith('beta'):
        return True
    if name.endswith('gamma'):
        return True
    return False


def make_mirror_plan(sym, threshold, plan_info=None, **kwargs):
    """Memory allocation planner with a given threshold.

    The user can pass in a network configuration,
    a threshold that limits memory per block.
    And input shape configurations.

    Parameters
    ----------
    sym : symbol
        Input configuration of symbols.
        The user need to pre-mark the attribute "mirror_stage" on the nodes
        that can be book-kept as stage

        The algorithm will decide whether to disbale mirror on the stage nodes.

    threshold: integer
        A tuning parameter to tune the approximate size of each stage blocks

    plan_info: dict, optional
        Used to hold plan information.

    **kwargs:
        The arguments to infer shape.

    Returns
    -------
    alloc_sym: symbol
        A symbol with force mirror tagged on the nodes for better allocation.
    """
    threshold = threshold << 20
    sym = sym.__copy__()
    internals = sym.get_internals()
    _, out_shapes, _ = internals.infer_shape(**kwargs)
    shape_dict = list(zip(internals.list_outputs(), out_shapes))
    total_size = 0
    param_size = 0
    local_size = 0
    save_size = 0
    max_size = 0
    last_stage = ''
    stage_decision = ''

    for idx, item in enumerate(shape_dict):
        sb = internals[idx]
        name, shape = item
        if is_param(name):
            # param size 指参数大小，即输入的arg_maps
            param_size += prod(shape) * 4
            continue
        else:
            # total_size 指所有中间临时变量的大小
            # local_size <= total_size，指进行Recompute的Block占用的大小
            total_size += prod(shape) * 4
            local_size += prod(shape) * 4
            # 将所有中间结果标记mirror
            sb._set_attr(force_mirroring='True')

        # 只对网络结构中显式指定mirror_stage的节点进行考察，从中选出可能的checkpoint
        if sb.attr('mirror_stage') is not None:
            stage = sb.attr('mirror_stage')
            if stage == 'True' or stage != last_stage:
                if local_size > threshold:
                    # Block大小大于阈值，将其设置为checkpoint
                    # save_size为所有checkpoint加起来的大小，max_size为最大Block的大小
                    save_size += prod(shape) * 4
                    max_size = max(max_size, local_size)
                    local_size = 0
                    stage_decision = 'False'
                    sb._set_attr(force_mirroring='False')
                    if(debug):
                        print(" - Select Ckpt: %s"%(name))
                        print(" - Save size: %dMB, max_size: %dMB"%(save_size>>20, max_size>>20))
                else:
                    stage_decision = 'True'
                    pass
                last_stage = stage
            # 最后一个checkpoint
            elif stage == last_stage and stage_decision == 'False':
                save_size += prod(shape) * 4
                sb._set_attr(force_mirroring='False')
                if(debug):
                    print(" - Select Ckpt: %s"%(name))
                    print(" - Save size: %d, max_size: %d"%(save_size, max_size))
    if(debug):
        print("param_size: %dMB"%(param_size>>20))
        print("total_temp_size: %dMB"%(total_size>>20))
    if plan_info is not None:
        plan_info['max_size'] = max_size
        plan_info['save_size'] = save_size
    return sym


def get_cost(sym, type_dict=None, **kwargs):
    """Get the cost of the current symbolic plan by running bind on CPU.

    sym : Symbolic Variable

    """
    texec = sym.simple_bind(ctx=mx.gpu(),
                            grad_req='write',
                            type_dict=type_dict,
                            **kwargs)
    return int(texec.debug_str().split('\n')[-3].split()[1])


def search_plan(sym, ntrial=5, type_dict=None, **kwargs):
    """Quickly heurestic search over possible plans to find good memory plan.

    Parameters
    ----------
    sym : symbolic
       Symbolic configurations

    ntrial: integer
       Additional grid search steps
    """
    history = []
    threshold = 0
    min_threshold = None
    min_cost = None
    nbegin = 3

    # nbegin次
    for k in range(nbegin):
        info = {}
        sym = make_mirror_plan(sym, threshold=threshold, plan_info=info, **kwargs)
        cost = get_cost(sym, type_dict, **kwargs)
        save_size = info['save_size'] >> 20
        local_size = info['max_size'] >> 20
        # save_size为所有checkpoint加起来的大小，max_size为最大Block的大小
        # save_size越大，说明ckpt越多，单个Block就越小，local_size(max_size)就越小，二者成反比
        # 我们的目标是让save——size和local_size都尽量小，用guess来衡量
        guess = int(math.sqrt(save_size * local_size / 2))
        if min_cost is None or min_cost > cost:
            min_cost = cost
        # min_threshold 表示历次搜索最大Block中的最小值
        if min_threshold is None or local_size < min_threshold:
            min_threshold = local_size
        if(debug):
            print ("Search threshold=%d MB, cost=%d MB" % (threshold, cost))
            print('-----------------------------------------------------')  
        history.append((cost, threshold, sym))
        # guess是历史搜索中做得最好的（值最小的），所以超过threhold的就不考虑
        threshold = guess

    # max_threshold = sqrt(save_size * local_size)
    max_threshold = threshold * math.sqrt(2)
    # step = (sqrt(save_size*local_size) - local_size) / ntrial
    step = int((max_threshold - min_threshold) / ntrial)
    step = abs(step)
    threshold = min_threshold + step
    if step > 0:
        # 额外的ntrial次grid搜索
        for k in range(ntrial):
            sym = make_mirror_plan(sym, threshold=threshold, plan_info=info, **kwargs)
            cost = get_cost(sym, type_dict, **kwargs)
            if(debug):
                print ("Grid Search: threshold=%d MB, cost=%d MB" % (threshold, cost))
                print('-----------------------------------------------------')
            history.append((cost, threshold, sym))
            threshold += step

    history.sort(key = lambda x: x[0])
    cost, threshold, sym = history[0]
    print('Find best plan with threshold=%d, cost=%d MB' % (threshold, cost))
    return sym
