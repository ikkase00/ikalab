import torch


def get_weights(model_name: str, project: str):
    """
    :param model_name: model name (from doms)
    :param project: the module name you import your model from
    :return: loaded weights, you can load it directly into your model without calling torch.load()
    """
    project_list = ['conditional_num']
    project, model_name = project.lower(), model_name.lower()
    assert (project in project_list), f'no project named >{project}<'

    parent = '/mnt/linusat/ikalab/doog/weiiights'
    path = f'{parent}/{project}/{model_name}.pth'

    try:
        weights = torch.load(path)
    except FileNotFoundError as _:
        raise FileNotFoundError(f'no model named >{model_name}<')

    return weights
    
