import torch
import torch.nn as nn
import nets
import os


UNAVAILABLE_MODELS = {
    'DWDN': 'https://github.com/dongjxjx/dwdn',
    'CDWDN': 'https://github.com/dongjxjx/dwdn',
    'MIMOUNET': 'https://github.com/chosj95/mimo-unet',
    'MIMOUNET+F': 'https://github.com/chosj95/mimo-unet',
    'DEEPSN': 'https://github.com/pandazcx/DeepSN-Net',
    'DEEPSN+F': 'https://github.com/pandazcx/DeepSN-Net',
}

MODEL_NAMES = (
    'SRCNN',
    'RESTORMER',
    'FSNET',
    'FSNET+F',
    'DIFF',
    *UNAVAILABLE_MODELS,
)


def _raise_unavailable_model(model_name):
    upstream_url = UNAVAILABLE_MODELS[model_name]
    raise RuntimeError(
        f'{model_name} is not distributed with HappyLens because its upstream '
        'implementation does not provide an explicit redistribution license. '
        f'Upstream: {upstream_url}'
    )

class Model(nn.Module):
    def __init__(self, model_name:str):
        super(Model, self).__init__()
        
        match model_name:
            case 'DWDN':
                _raise_unavailable_model(model_name)
            case 'SRCNN':
                self.model = nets.srcnn.model.SRCNN()
                self.loss_fuc = nets.srcnn.loss.Loss()
            case 'CDWDN':
                _raise_unavailable_model(model_name)
            case 'MIMOUNET':
                _raise_unavailable_model(model_name)
            case 'MIMOUNET+F':
                _raise_unavailable_model(model_name)
            case 'RESTORMER':
                self.model = nets.restormer.model.RESTORMER()
                self.loss_fuc = nets.restormer.loss.Loss()
            case 'DEEPSN':
                _raise_unavailable_model(model_name)
            case 'DEEPSN+F':
                _raise_unavailable_model(model_name)
            case 'FSNET':
                self.model = nets.fsnet.model.FSNET()
                self.loss_fuc = nets.fsnet.loss.Loss()
            case 'FSNET+F':
                self.model = nets.fsnet.model.FSNET(field_code=True)
                self.loss_fuc = nets.fsnet.loss.Loss()
            case 'DIFF':
                self.time_steps = 100
                self.model = nets.diff.model.DocDiff(time_steps=self.time_steps)
                self.loss_fuc = nets.diff.loss.Loss(time_steps=self.time_steps)
            case _:
                valid_names = ', '.join(MODEL_NAMES)
                raise ValueError(
                    f'Unknown model name {model_name!r}. '
                    f'Choose one of: {valid_names}'
                )
    
    def forward(self, *args):
        return self.model(*args)
    
    def loss(self, *args):
        return self.loss_fuc(*args)
    
    def save(self, pth, filename):
        filename = f'model_{filename}.pt'
        torch.save({'model': self.state_dict()}, os.path.join(pth, filename))
    
    def load(self, pth, device):
        self.load_state_dict(torch.load(pth, map_location=torch.device(device))['model'])


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
