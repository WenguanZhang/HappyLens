from .optim import Merit, MeritZ

class Generation_Prime(Merit):
    def __init__(self, **kwargs):
        super(Generation_Prime, self).__init__(**kwargs)
    
    def params_lr(self, lr, scale=10.):
        param_list = []
        # for generation
        for name, params in self.sys.system.named_parameters():
            print(name)
            if name.endswith('roc'):
                param_list.append({'params': params, 'lr': lr / scale})
            elif name.endswith('thick'):
                param_list.append({'params': params, 'lr': lr * scale})
            elif name.endswith('conic'):
                param_list.append({'params': params, 'lr': lr})
            elif name.endswith('g1'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif name.endswith('g2'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif 'ai' in name:
                param_list.append({'params': params, 'lr': lr * scale ** -(int(name.split('.ai')[-1]) - 1)})
            elif 'qi' in name:
                param_list.append({'params': params, 'lr': lr})
        return param_list
    
class Generation_Zoom(MeritZ):
    def __init__(self, **kwargs):
        super(Generation_Zoom, self).__init__(**kwargs)
    
    def params_lr(self, lr, scale=10.):
        param_list = []
        # for generation
        for name, params in self.sys.system.named_parameters():
            print(name)
            if name.endswith('roc'):
                param_list.append({'params': params, 'lr': lr / scale})
            elif name.endswith('thick'):
                param_list.append({'params': params, 'lr': lr * scale})
            elif name.endswith('conic'):
                param_list.append({'params': params, 'lr': lr})
            elif name.endswith('g1'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif name.endswith('g2'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif 'ai' in name:
                param_list.append({'params': params, 'lr': lr * scale ** -(int(name.split('.ai')[-1]) - 1)})
            elif 'qi' in name:
                param_list.append({'params': params, 'lr': lr})
        return param_list