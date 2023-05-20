import json
from hw2.experiments import cnn_experiment

print("starting experiments")
params_file = "./results/optuna_resnet_Adam_L2_K64-128-256.json"
seed = 42
with open(params_file, "r") as f:
    param_dict = json.load(f)['config']
hp_optim = {'betas':(param_dict['beta1'], param_dict['beta2'])}
param_dict['reg']=param_dict['weight_decay']
param_dict['hidden_dims']=[param_dict['hidden_dims_val']]*param_dict['hidden_dims_num']
entries_to_remove = ('beta1', 'beta2', 'weight_decay', 'hidden_dims_val', 'hidden_dims_num', 'value', 'layers_per_block', 'filters_per_layer')
for key in entries_to_remove:
    param_dict.pop(key, None)
print("params", param_dict, hp_optim)
# for k in [32,64]:
#     for l,p in zip([2,4,8,16], [1,1,2,4]):
#         name = "exp1_1"
#         print(name)
#         param_dict['pool_every'] = p
#         cnn_experiment(
#             name, seed=seed, bs_train=128, epochs=100, early_stopping=10, filters_per_layer=[k], layers_per_block=l, optimizer='Adam',hp_optim=hp_optim,
#             model_type='cnn',**param_dict)
# for k in [32,64,128]:
#     for l,p in zip([2,4,8], [1,1,2]):
#         name = "exp1_2"
#         print(name)
#         param_dict['pool_every'] = p
#         cnn_experiment(
#             name, seed=seed, bs_train=128, epochs=100, early_stopping=10, filters_per_layer=[k], layers_per_block=l, optimizer='Adam',hp_optim=hp_optim,
#             model_type='cnn',**param_dict)

# for l,p in zip([2,3,4], [1,2,2]):
#     name = "exp1_3"
#     print(name)
#     param_dict['pool_every'] = p
#     cnn_experiment(
#         name, seed=seed, bs_train=128, epochs=100, early_stopping=10, filters_per_layer=[64,128], layers_per_block=l, optimizer='Adam',hp_optim=hp_optim,
#         model_type='cnn',**param_dict)
    
for l,p in zip([8,16,32], [2,4,8]):
    name = "exp1_4"
    print(name)
    # param_dict['pool_every'] = l//4 if l > 4 else 1 
    param_dict['pool_every'] = p
    cnn_experiment(
        name, seed=seed, bs_train=128, epochs=100, early_stopping=10, filters_per_layer=[32], layers_per_block=l, optimizer='Adam',hp_optim=hp_optim,
        model_type='resnet',**param_dict)


for l,p in zip([2,4,8], [2,3,8]):
    name = "exp1_4"
    print(name)
    # param_dict['pool_every'] = l//4 if l > 4 else 1 
    param_dict['pool_every'] = p
    cnn_experiment(
        name, seed=seed, bs_train=128, epochs=100, early_stopping=10, filters_per_layer=[64,128,256], layers_per_block=l, optimizer='Adam',hp_optim=hp_optim,
        model_type='resnet',**param_dict)

