
model_configs = {
    'GALEN': {
        'prediction': {
            'boxsqel': {
                'margin': 0.05,
                'disjoint_dist': 2,
                'reg_factor': 0.05,
                'lr': 5e-3,
                'scheduler_steps': [2000],
                'epochs': 5000
            }
        }
    }
}