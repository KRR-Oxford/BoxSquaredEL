import wandb
import sys
from baseline_sweep import main

baseline_config = {
    'method': 'bayes',
    'name': 'PLACEHOLDER',
    'metric': {
        'name': 'surrogate',
        'goal': 'minimize'
    },
    'command': ['python', 'baseline_sweep.py'],
    'parameters': {
        'dataset': {
            'value': 'PLACEHOLDER'
        },
        'task': {
            'value': 'PLACEHOLDER'
        },
        'model': {
            'value': 'PLACEHOLDER'
        },
        'lr': {
            'values': [0.001, 0.005, 0.01]
        },
        'margin': {
            'max': 0.5,
            'min': 0.0
        },
        'embedding_dim': {
            'values': [50, 100, 200]
        }
    }
}

boxsqel_config = {
    "method": "bayes",
    'name': 'PLACEHOLDER',
    "metric": {
        "goal": "minimize",
        "name": "surrogate"
    },
    "parameters": {
        "dataset": {
            "value": "PLACEHOLDER"
        },
        "lr": {
            "values": [
                0.001,
                0.005,
                0.01
            ]
        },
        "lr_schedule": {
            "values": [
                2000,
                10000
            ]
        },
        "margin": {
            "max": 0.5,
            "min": 0.0
        },
        "neg_dist": {
            "distribution": "uniform",
            "max": 10.0,
            "min": 1.0
        },
        "num_neg": {
            "max": 5,
            "min": 1
        },
        "reg_factor": {
            "max": 0.5,
            "min": 0.0
        },
        "task": {
            "value": "PLACEHOLDER"
        },
        'model': {
            'value': 'PLACEHOLDER'
        },
    }
}

boxel_config = {
    'method': 'grid',
    'name': 'PLACEHOLDER',
    'metric': {
        'name': 'surrogate',
        'goal': 'minimize'
    },
    'parameters': {
        'dataset': {
            'value': 'PLACEHOLDER'
        },
        'task': {
            'value': 'PLACEHOLDER'
        },
        'model': {
            'value': 'PLACEHOLDER'
        },
        'lr': {
            'values': [0.001, 0.005, 0.01]
        },
        'embedding_dim': {
            'values': [25, 50, 100, 200]
        }
    }
}

config = boxel_config
model = 'boxel'
task = 'prediction'
for dataset in ['GALEN', 'GO', 'ANATOMY']:
    print(f'Starting sweep {model}-{dataset}-{task}')
    config['name'] = f'{model}-{dataset}-{task}'
    config['parameters']['dataset']['value'] = dataset
    config['parameters']['task']['value'] = task
    config['parameters']['model']['value'] = model
    sweep_id = wandb.sweep(sweep=config, project='el-baselines')
    print(f'Starting agent')
    wandb.agent(sweep_id, function=main)
    print(f'{model}-{dataset} sweep finished')

# for model in ['elem', 'emelpp', 'elbe']:
#     for dataset in ['GALEN', 'GO', 'ANATOMY']:
#         task = 'inferences'
#         print(f'Starting sweep {model}-{dataset}-{task}')
#         config['name'] = f'{model}-{dataset}-{task}'
#         config['parameters']['dataset']['value'] = dataset
#         config['parameters']['task']['value'] = task
#         config['parameters']['model']['value'] = model
#         sweep_id = wandb.sweep(sweep=config, project='el-baselines')
#         print(f'Starting agent')
#         wandb.agent(sweep_id, function=main, count=20)
#         print(f'{model}-{dataset} sweep finished')
