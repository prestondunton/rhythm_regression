import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import warnings

def grid_search_experiment(train_func, param_options, train_args=None, return_model='best', model_ranking_metric=None, model_ranking_mode='max', exceptions='warn'):
    """
    Arguments
    train_func (function): A training function that takes **params and returns a dictionary of metrics and a model
    param_options (dictionary of {str: list of any}): A list of options to grid and pass to train_func
    train_args (dict): A dictionary of arguments to pass to train_func
    return_model ([best, all, none]): Which model(s) to return
    model_ranking_metric (str): A key into the returned metrics dictionary from train_func that is used when return_model = 'best'
    model_ranking_mode ([min, max]): Whether to minimize or maximize the model_ranking_metric when return_model = 'best
    exceptions (['raise', 'warn', 'silence']): What to do with exceptions

    Returns
    results (pd.DataFrame): A dataframe with grided parameters and returned metrics
    models (list) OR best_model (any): Returned if return_model is not 'none'
    """
    
    if return_model not in ['best', 'all', 'none']:
        raise ValueError(f'return_model must be one of [best, all, none].  Got value \'{return_model}\'')

    if return_model == 'best':
        if model_ranking_metric is None:
            raise ValueError('You must specify a model_ranking_metric in order to return the best model.')

        if model_ranking_mode not in ['min', 'max']:
            raise ValueError(f'model_ranking_mode must be one of [min, max].  Got mode \'{model_ranking_mode}\'')

    if exceptions not in ['raise', 'warn', 'silence']:
        raise ValueError(f'excpetions parameter must be one of [raise, warn, silence]. Got value \'{exceptions}\'')


    grid = ParameterGrid(param_options)
    metrics_dicts = []
    metrics_keys = None

    models = []
    best_model = None
    best_model_metric = float('inf') if model_ranking_mode == 'min' else float('-inf')

    for params in tqdm(grid):
        try:
            if train_args is None:
                metrics, model = train_func(**params)
            else:
                metrics, model = train_func(**train_args, **params)

            if return_model == 'best':
                if model_ranking_metric not in metrics.keys():
                    raise ValueError(f'model_ranking_metric \'{model_ranking_metric}\' is not in the returned metrics from train_func: {metrics.keys()}')
                if model_ranking_mode == 'min':
                    if metrics[model_ranking_metric] < best_model_metric:
                        best_model = model
                        best_model_metric = metrics[model_ranking_metric]
                else:
                    if metrics[model_ranking_metric] > best_model_metric:
                        best_model = model
                        best_model_metric = metrics[model_ranking_metric]

            elif return_model == 'all':
                models.append(model)

            metrics_dicts.append(params | metrics) # merge the dictionaries with |
            if metrics_keys is None:
                metrics_keys = metrics.keys()

            models.append(model)

        except Exception as e:
            if exceptions == 'raise':
                raise e
            else:
                if exceptions == 'warn':
                    warnings.warn(f'Encountered exception with parameters {params}: {e}')
                if metrics_keys is None:
                    metrics_dicts.append({}) 
                else:
                    metrics_dicts.append({key: np.nan for key in metrics_keys})

    results = pd.DataFrame(metrics_dicts)
    results.index.name = 'model_id'

    if return_model == 'best':
        return results, best_model
    elif return_model == 'all':
        return results, models
    else:
        return results



def print_verbose(message, threshold, level):
    if level >= threshold:
        print(message)