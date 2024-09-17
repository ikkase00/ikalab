import torch
import torch.nn as nn
import inspect
import copy

no = False
yes = True


class Ops:
    @staticmethod
    def repkey(d: dict, new, old):
        dII = {}
        for key, val in d.items():
            if key == old:
                dII[new] = val
            else:
                dII[key] = val
        return dII

    @staticmethod
    def inskey(d: dict, base_key, new, new_val):
        dII = {}
        done = no
        for key, val in d.items():
            dII[key] = val
            if key == base_key and not done:
                dII[new] = new_val
                done = yes
        return dII

    @staticmethod
    def sort(key):
        parts = key.split('_')
        if len(parts) > 1:
            try:
                numeric_part = int(*parts[:-1])
            except ValueError:
                numeric_part = parts[:-1]
            idk = (*parts[-1], numeric_part)
        else:
            try:
                numeric_part = int(*parts[:])
            except ValueError:
                numeric_part = parts[:]
            idk = ('z', numeric_part)
        return idk

    @staticmethod
    def mad(seq_layers, device):
        class Chanpanmod(nn.Module):
            def __init__(self):
                super(Chanpanmod, self).__init__()

                self.layers = seq_layers
                self.device = device

            def forward(self, c):
                return self.layers(c)

        return Chanpanmod().to(device)


def chanpanlays(model: torch.nn.Module, ratio: tuple, conv_bias: bool, devices: list):
    """
    NOTE: This function does NOT move any layer to any device!!!Use chanpanzee if you want to create an entire model
    :param model: your model
    :param ratio: the performance ratio between your devices
    :param conv_bias: enable convolutional bias or not
    :param devices: device names, in a list of the same order of the ratio
    :return: two blocks of layers in sequential format
    """

    assert len(devices) == 2, 'chanpanmod supports at most two devices 🫨'

    cu0 = torch.device(devices[0])
    cu1 = torch.device(devices[1])
    tot_rat = sum(ratio)
    cu0_rat = ratio[0] / tot_rat
    cu1_rat = ratio[1] / tot_rat

    paras = {}
    namaes = []
    for namae, lay in model.named_children():
        lame = type(lay).__name__
        namaes.append(namae)
        what = {
            'lame': lame,
        }
        sig = inspect.signature(type(lay))
        args = sig.parameters.keys()
        for arg in args:
            try:
                what[arg] = getattr(lay, arg)
            except AttributeError as _:
                pass
        paras[namae] = what
    able = ['Conv', 'BatchNorm']
    acts = ['ReLU', 'LeakyReLU', 'Tanh']
    for namae in namaes:

        assert (namae[-1] != 'a') and (namae[-1] != 'b') and (
                '_' not in namae), ('do nooot end your layer names with a or b or contain _ as the function '
                                    'will blow up 🫠')

        layer = paras[namae]
        lame = layer['lame']
        base = copy.deepcopy(layer)
        if able[0] in lame:
            # try:
            layer['in_channels'] = int(int(layer['in_channels']) * cu0_rat)
            layer['out_channels'] = int(int(layer['out_channels']) * cu0_rat)
            layer['bias'] = conv_bias
            layer['device'] = cu0
            new_aame = f'{namae}_a'

            new_bame = f'{namae}_b'
            base['in_channels'] = int(int(base['in_channels']) * cu1_rat)
            base['out_channels'] = int(int(base['out_channels']) * cu1_rat)
            base['bias'] = conv_bias
            base['device'] = cu1
            paras = Ops.repkey(paras, new_aame, namae)
            paras = Ops.inskey(paras, new_aame, new_bame, base)
        elif able[1] in lame:
            # try:
            layer['num_features'] = int(int(layer['num_features']) * cu0_rat)
            new_aame = f'{namae}_a'

            new_bame = f'{namae}_b'
            base['num_features'] = int(int(base['num_features']) * cu1_rat)
            paras = Ops.repkey(paras, new_aame, namae)
            paras = Ops.inskey(paras, new_aame, new_bame, base)
        else:
            new_aame = f'{namae}_a'
            new_bame = f'{namae}_b'
            paras = Ops.repkey(paras, new_aame, namae)
            paras = Ops.inskey(paras, new_aame, new_bame, base)

    # good_paras = dict(sorted(paras.items(), key=lambda name: key_ops.sort(name[0])))
    aod = []
    bod = []
    for key, pars in paras.items():
        determinant = key[-1]
        lass = getattr(nn, pars.pop('lame'))
        l = lass(**pars)
        if determinant == 'b':
            bod.append(l)
        elif determinant == 'a':
            aod.append(l)

    aod = nn.Sequential(*aod)
    bod = nn.Sequential(*bod)

    return aod, bod


def chanpanzee(model: torch.nn.Module, data: torch.tensor, ratio: tuple, conv_bias: bool, devices: list):
    """
    :param model: your model
    :param data: the input tensor, expected to have a shape of B, C, W, H
    :param ratio: the performance ratio between your devices
    :param conv_bias: enable convolutional bias
    :param devices: device names, in a list of the same order of the ratio
    :return: a list of models + a list of data ordered and correspond to your devices
    """

    rat_tot = sum(ratio)
    rat_a, rat_b = ratio[0] / rat_tot, ratio[1] / rat_tot
    main = data.size(1)
    siza, size_b = int(main * rat_a), int(main * rat_b)
    data, datab = torch.split(data, [siza, size_b], 1)

    aod, bod = chanpanlays(model, ratio, conv_bias, devices)
    aom, bom = Ops.mad(aod, devices[0]), Ops.mad(bod, devices[1])
    data, datab = data.to(devices[0]), datab.to(devices[1])

    dom = [aom, bom]
    dat = [data, datab]

    raise Warning('You might need to adjust the data type of weights or data')

    return dom, dat
