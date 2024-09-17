import torch
from pathlib import Path
root = str(Path(__file__).resolve().parent)

def get_weights(model_name: str, project: str, device, casual = False):
    """
    :param model_name: model name (from doms)
    :param project: the name of the module you imported your model from
    :return: loaded weights, you can load it directly into your model without calling torch.load()
    """
    project_list = ['conditional_num']
    project, model_name = project.lower(), model_name.lower()
    assert (project in project_list), f'no project named >{project}<'

    if not casual:
        parent = root.replace('/mod_related/wws', '/weiiights')
        path = f'{parent}/{project}/{model_name}.pth'
    else:
        path = f'../weiiights/{project}/{model_name}.pth'
    print(f'loading from {path} ...')

    try:
        weights = torch.load(path, map_location = device, weights_only = True)
    except FileNotFoundError as _:
        raise FileNotFoundError(f'no model named >{model_name}<')
    except RuntimeError as _:
        print('weight dismatch...check again`')

    return weights 
    
