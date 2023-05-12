import json
from hw2.experiments import cnn_experiment

print("starting experiment")
params_file = "./results/optuna_K32_64_L1.json"
seed = 42
with open(params_file, "r") as f:
    param_dict = json.load(f)
hp_optim = {'betas':(param_dict['beta1'], param_dict['beta2'])}
param_dict['reg']=param_dict['weight_decay']
param_dict['hidden_dims']=[param_dict['hidden_dims_val']]*param_dict['hidden_dims_num']
entries_to_remove = ('beta1', 'beta2', 'weight_decay', 'hidden_dims_val', 'hidden_dims_num', 'value')
for key in entries_to_remove:
    param_dict.pop(key, None)
for l in [8,16,32]:
    name = "exp1_4"
    print(name)
    # param_dict['pool_every'] = l//4 if l > 4 else 1 
    param_dict['pool_every'] = l//4
    cnn_experiment(
        name, seed=seed, bs_train=128, epochs=100, early_stopping=5,
        filters_per_layer=[32], layers_per_block=l, **param_dict, optimizer='Adam',hp_optim=hp_optim,
        model_type='resnet'
    )

