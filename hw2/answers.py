r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


1.A - the shape of the input is 64X1024. the shape of the output is 64X512. the shape of the Jacobian is therefore 64X512x1024 
<br>
1.B - since this is linear layer, each output element $y_i$ is dot product the $i^{th}$ row of $W$ with x.so for each sample the jacobian $\frac{\partial{y}}{\partial{x}}$ is just the weight matrix. generaly, we cannot assume that the weight matrix is sparse therefore in general the Jacobian would not be sparse, altough there are  cases when using L1 regularization to induce some sparsity of the weight matrix. in this case the jacobian might be sparse. 
<br>
1.C - using the chain rule we can calculate the Jacobian vector product instead of fully materialize the Jacobian - since the Jacobian $J$ in this case is $W$  and we need to calculate $\delta X = J \delta Y$ we can just multiply the weight matrix with $\delta Y$ to get the result : $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}\frac{\partial Y}{\partial X} = W^T*\delta Y 
$ 
<br>
2.A - now we need to calcualte $\frac{\partial{y}}{\partial{W}}$ we take the derivative of each of the 512 elements $Y_i$ with respect to all 1024X512 elements $W_{ij}$. so for each sample we have 512X1024X512 elements and in total 64X512X1024X512 elements in the full Jacobian. 
<br>
2.B - since each element $Y_i$ is a linear combination of the i'th row of $W$, $\frac{\partial{y_i}}{\partial{W_{j,k}}} = 0$ for $i \neq j$. this means that the Jacobian is sparse (non zero only for the $i^{th}$ row of $W$) and is essentially the gradient of each output elements with respect to the corresponding weight row. 
<br>
2.C - we again don't need to materialize the Jacobian. we can again use the chain rule and multiply product between the input tensor with the scalar loss $\delta Y$

"""

part1_q2 = r"""
**Your answer:**


backprop is just a way for efficient calculations of gradients and therefore is it not the only way to perform decent-based optimization and it is not required in a decent-based training. for example, we saw in the tutorial that it is possible to use Forward mode AD instead. (there are also other methods. for example - https://arxiv.org/abs/2202.08587). however, the use of chaine-rule, computational graphs and automatic differentiation makes backprop the method that gives the best trade-off of efficiency and accuracy in a scenario of heavy computations and it is the most used optimization method in the field of deep learning  
```


"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd ,lr, reg = 0.1, 0.1, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.1,
        0.045,
        0.003,
        0.0007,
        0.001,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0.1,
        0.003,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======

    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. The difference between the graph without dropout and the graphs with dropout corresponds to what we expected: since dropout is a kind of regularization, it is understood that without dropout we will get the best performance for the training set, but they will not necessarily reflect the performance for the test set, which can suffer from overfitting. At the same time, too strong regularization can also hurt the performance for the training set, because it limits the freedom of the model. In this case, too strong dropout will prevent the different neurons from dividing "roles" between them and learning a complex pattern. 

    Indeed, we see, as expected, that the no-dropout performance is best for the training set (and deteriorates as the dropout increases), while for the test set the change is more complex and non-monotonic: the performance for the test set improves slightly with moderate dropout, but at least for the measure of accuracy deteriorates for too high dropout.

    As an example, we can see very clearly how the gap between the loss graph of dropout=0 and the loss graph of dropout=0.4 is getting wider throughout the training both for the trust set and for the test set, but in opposite directions: for the training set the loss of dropout=0.4 is higher, and for the test set the loss of dropout=0 is higher.

2. For the training set, the expected pattern was obtained: low-dropout led to better performance according to both indices than high-dropout setting - because dropout is a regularization that limits the model. 

    But for the test set a more complex pattern was obtained: the loss of low-dropout was initially lower but it gradually increased and became higher than the loss of high-dropout (which decreased moderately); But the accuracy of low-dropout remains higher than that of high-dropout throughout the entire training process. It is also surprising to note that the loss of low-dropout increased slightly while its accuracy also increased. According to our interpretation, this pattern reflects the fact that a low-dropout model becomes more variable between examples, i.e. although the average probability for the true label for example was too low to predict correctly, the variance was high enough so that there are relatively high amount of correct predictions. This pattern increased during training (i.e. the average loss increased as well as the its variance) and in general was higher for the low-dropout then the high-dropout, which had lower average loss but also lower loss variance, and therefore lower accuracy.
    
"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible, because the accuracy for each example is boolean while the loss is continuous. So, for example, it is possible that for one example in the dataset the value of $\hat{y}$ of the correct $y$ will increase by infinitesimal, smallest as needed for it to be the maximum, while for all the other examples the $\hat{y}$ of the correct $y$ will decrease a lot but still stay the maximum. In this case, the loss will increase while the accuracy will increase too - because only the predicted label of one example change, and it change to the correct one.

It cam happen in cases where the model improves but the variance of the values in the outbut is reduced - so the scores of the correct labels are gatting closer to the smallest value that needed to be the predicted labels.
"""

part2_q3 = r"""
**Your answer:**
1. backpropagation is the algorithm that calculates the gradients of all the parameters in the model. Gradient Descent is the algorithm that updates the parameters of the model according to their gradients, and for this purpose it can use the backpropagation algorithm, in the case of a neural network.

2. In GD, the algorithm uses each iteration in the entire training set $X$, and calculates the error and gradients based on it. In contrast, in SGD, the algorithm in each iteration randomly selects a one example $x \in X$, and calculates the error and the gradients based on it.

3. First SGD is has lower time/space complaxcity than GD, becaus it calculate the gradient only for one example each time.
Second, since SGD considers a different example at random each time, it can get out of "traps": local minima points that GD cannot get out of.

4. 
    1. Yes, it is:
$\frac{\partial L_1}{\partial \Theta} + \frac{\partial L_2}{\partial \Theta} + ... + \frac{\partial L_k}{\partial \Theta} = \frac{\partial (L_1 + L_2 + ... + L_k)}{\partial \Theta} = \frac{\partial (\sum_{x \in X_1}L(x) + \sum_{x \in X_2}L(x) + ... + \sum_{x \in X_k}L(x) )}{\partial \Theta} = \frac{\partial \sum_{x \in X} L(x)}{\partial \Theta} = \frac{\partial L}{\partial \Theta}$
As needed.
    2. The computer still need to save the losses of all the examples, to compute the gradient at the end.

"""

part2_q4 = r"""
1. 

A. **pdeudocode:**

$lastResult \gets X$ 

$lastGrad \gets \vec{1}$ 

**For $i \gets 1$  to  $n$:**

$\quad lastGrad \gets lastGrad \cdot f_{i}.derivative(lastResults)$

$\quad lastResults \gets f_{i}(lastResults)$

**EndFor**

return $lastGrad$

**End**

memory complexity = $O(1)$

B. **pdeudocode:**

$results = []$

$reslts[0] \gets X$

**For $i \gets 1$  to  $n$:**

$\quad results[i] \gets f_{i}(results[i-1])$

**EndFor**

$lastGrad \gets \vec{1}$ 

**For $i \gets n-1$  downto  $0$:**

$\quad lastGrad \gets lastGrad \cdot f_{i+1}.derivative(results[i])$

**EndFor**

return $lastGrad$

**End**

If we asume that the results are already given, then:

memory complexity = $O(1)$

else:

memory complexity = $O(n)$


2. This technique is based on the assumption that all functions are executed in a queue and not in parallel. It is this assumption that allows us to remember at any moment only a fixed number of gradients. In particular, there is only one input. So, in principle, if we use this technique when there are functions that are executed in parallel, we are not guaranteed to be able to run this algorithm in $O(1)$ memory complexity. However, we can sometimes save memory to some extent.

3. This technique can help us in cases where it is not necessary to remember all the gradients. This can happen, for example, when we only perform finetuning for a limited number of layers, without updating all the other layers. Alternatively, we can use an algorithm to update each layer separately with low memory complexity - but the time complexity in this case would be very high.



"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 2  # number of layers (not including output)
    hidden_dims = 1024  # number of output dimensions for each hidden layer
    activation = "relu"  # activation function to apply after each hidden layer
    out_activation = "softmax"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.1, 0.001, 0.1  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


1. given the loss graph, the optimization error is not high- we can see that the training loss decrease smoothly until it reaches a plateau. that implies that the at the end of the optimization proccess the gradients are small and therefore the optimization error is small. there is a possibility that we reached local minima but since we reached high accuracy and we see a good decision boundary we think this is not the case.
2. looking at the test loss we conclude that the genralization error os a little bit high. compare to the train loss , the test loss is much more noisy, and don't have a good "decrease" shape. although there is no overfitting (the test loss is not raising), we can say that the generalization error is higher than the optimization error.
3. looking at the decision boundry plot, we can say that the approximation error is not high. the model is able to create the non linear shape that separates the classes and therefore it is able to approximate the real boundary of the dataset. since the approximation
error comprised from optimization error and generalization error, we 
can see from the graphs that both contribute relatively the same to the approxiamtion error (the distance from the test accuracy to the train accuracy is ~ the distance from the train accuracy to 100).  

"""

part3_q2 = r"""
**Your answer:**


for the model we trained at the beginning of the notebook, we expect the validation set to have more FNR than FPR. this is because we can see that the model's decision boundary is over-estimating the area of class 0 (regions that the model marked as class 0 and has points from class 1) i.e the probabilty for the model to classify sample with label 1 as 0 is higher than the opposite case (classifiying sample with label 0 as 1). since those mistakes are False negatives we assuming that the FNR would be higher than FPR. In general, if you know exactily how the data was generated, we can estimate the FNR and FPR based on the model decision boundary.  

"""

part3_q3 = r"""
**Your answer:**

the "optimal" point on the ROC curve may not be the best choice in this situation, as the "naive" ROC curve do not take into account the costs and risks associated with false positive and false negative classifications.
    
In scenario 1, where the disease leads to non-lethal symptoms, the cost of a false positive classification (i.e. diagnosing a healthy patient as sick) is high , as it results in a  unnecessary expensive and risky confirmation test. However, the cost of a false negative classification (i.e. failing to diagnose a sick patient) is low, as the symptoms are easy to detect and not dangerous. Therefore, in this scenario, it may be better to choose a classification threshold that maximizes sensitivity (i.e. minimizing false positive) even if it comes at the cost of increased false negative.

In scenario 2, where the disease may lead to high risk of death if not diagnosed early, the cost of a false negative classification is very high, as it may result in delayed treatment and death. Therefore, it may be better to choose a classification threshold that maximizes specificity (i.e. minimizing false positives) even if it comes at the cost of increased false negatives.


"""


part3_q4 = r"""
**Your answer:**


1. analyzing the results by column, we see that for depth=1 the width that gave the best results is the lowest one (width=2), for depth=2, the lowest (width=2) and the highest (width=8) gave the same best results and for depth=4 the best one was the model with the highest width (8). only for the last column (depth=4) we see consistency in which increasing the width leads to better results. this implies that for shallow networks, adding more parameters doese not necceseraly improve the performence.
2. for fixed width and varied depth we see much more consistency - always the best model was the one with the highest number of layers and except one case (width=2) increasing the number of layers always leads to increasing the test accuracy. this implies that adding more layers is more efficient in capturing complex features than adding more paramters (width) for a specific layer.
3. depth=4, width=8 had better results then depth=1 width=32. this is another proof to what have been written before - for a fixed number of total  parameters,  adding more layers is better than adding width for less layers. the idea behind it is that adding more layers incerase the non-linearity of the model while adding width gives a better approximation per layer. we can think of each layer as a linear function followed by non linearity. increasing the number of layers, increases the non-linearity of the model much more than increasing the width of a single layer and therefore gives better results.
4. the optimal thresould did not improved the reuslts on the test set compared to the validation set. the reason for that is that the optimal threshold can be sensitive to specific dataset. choosing the optimal thershold based on the validation set wouldn't necceseraly be optimal for the test set. it might be that the test set samples distribute differently and that the (true) optimal threshold for the test set would be different.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = torch.nn.CrossEntropyLoss()  # One of the torch.nn losses
    lr, weight_decay, momentum = 0.03, 0.01, 0.01  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1. in General, the 1x1 convolution reduce the number of parameters in the bottleneck block. in our case direct calculation gives:
for the regular block we have two layers with kernel 3x3 and 256 input and output channels that gives (including bias): $(3*3*256+1)*256*2=1180160$  for the bottleneck we have 3 layers. number of parameters is: $(1*1*256+1)*64+ (3*3*64+1)*64 + (1*1*64 + 1)*256=70016$ for the bottleneck. We see that also the bottleneck block has more layers, it has much fewer parameters in total

2. The bottleneck block requires fewer floating point operations than the regular block. This is because the 1x1 convolutions reduce and then increase the number of channels, allowing for more efficient computation of the subsequent 3x3 convolution. Therefore, the bottleneck block requires fewer computations and has lower computational complexity than the regular block.

3. Spatially within Feature Maps: both blocks has convolutions with the same kernel size (3x3) but the regular block has 2 3X3 convolutions while the bottleneck block has only one. this means that the regular block has more spatial resolution and can capture more complex features within the image.
<br>
Across Feature Maps: from ine hand, the bottleneck block has fewer channels and therefore catches more coarse (high level) features. from the other hand, it has more layers within the block so it can capture more features in total.
"""
# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


in the first experiment we tested the effect of varying depth with fixed channels in a CNN model. it can be seen from the
 graphs that for small number of layers (2,4) increasing the depth results in better accuracy. this is because more
  layers (with fixed number of channels) gives more spatial resolution and ability to learn more complex features within the image.
  <br>
when the number of layers keeps growing,
 (above 4 in our case) the loss and accuracy are constant, meaning the gradients are zero. this is vanishing gradients issue that cause 
 the larger models to become non-trainable. vanishing gradient is a problem of large models - since propagating the graidents through many layers
causes the gradients to become very small and eventually zero. this is because the gradients are multiplied by the weights in each layer, and if the weights are smaller than 1, the gradients will become smaller and smaller.
    one way to solve this problem is by using batch normalization, which normalizes the input to each layer to have zero mean and unit variance. this allows the gradients to flow through the network without vanishing.
    another way is by using skip connections, which allow the gradients to flow directly to the lower layers without passing through the upper layers. this also allows the gradients to flow without vanishing.

"""

part5_q2 = r"""
**Your answer:**


in the second experiment we tested the effect of varying width with fixed depth in a CNN model. it can be seen from the
graphs that for fixed depth, increasing the width results in a minor increase in accuracy (compared to the change we saw in exp1_1
when we changed the depth). for l=2 the difference between 
different widths is almost negligble, and for l=4 the difference is a little bit bigger. The effect of increasing the width is
minor because the number of layers is fixed and therefore the spatial resolution is fixed. this means that the model can't learn more
complex features, and therefore the effect of increasing the width is minor. moving from l=2 to l=4 reuslts in better 
preformence for all $K$ (i.e model with l=4 and k=32 is better than model with l=2 and k=128). this implies that depth is more important
than number of channels.  in addition, we can see that for l=8 we get vanishing. the performence of the best models (l=4) were similiar in both experiments.
gradients, similiar to what we saw in exp_1_1.  
"""

part5_q3 = r"""
**Your answer:**


in the third experiment we tested the effect of varying depth and width in a CNN model. it can be seen from the graphs 
that model with more than 4 layers in total in non_trainable in accordance to what we saw in exp1_1 and exp1_2.
that leaves us with only 1 trainable model - $l=2$ $k=[64,128]$ which perfmored similiar to models from exp1_1 and exp1_2. with l=4
(which has the same total number of layers)
"""

part5_q4 = r"""
**Your answer:**

in experiment 1.4 we tested the effect of skip connections by using Resent.
in experiment 1.1 and 1.3 we saw that models with more than 4 convolution layers in total faced vanishing
of the gradients and became non-trainable. using skip connections (in 1.4) the phenomena no longer occur,
all the model were trainable and we can train much deeper networks. we therefore conclude that the residual block,
together with batchnorm, enables more uniform flow of the graidents and prevent them from vanising or exploding.
regarding performance - as the hyperparameter space is very large (many parameters that spans a large range of values)
and the models can be sensitive to the choice of hyperparameters, in order to fully utilize the models one needs an
efficient method to tune them. we used optuna package which provides an optimization framework for efficient tuning 
large number of hyper-parameters. we run multiple optimization experiments for a seleceted architectures and small number of epochs
for both resnet and cnn and based on the results wechose the parameters for the actual experiment.
on this specific dataset and with the given architectures, we see that the resnet model with the lowerst number of layers converged first. this implies that the dataset is relatively simple and can be learned by simple network. we also see that  
CNN outperform Resnet (altough the difference is not high and might change with more carefull tuning of hyper-parameters). however, on different CV tasks, usually the depth that Resnet allows was proved to lead to better results than shallow networks.       
"""
# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


in generall, the model did not performed well on those pictures. in the first image there are three dolphins.
the model located on of the bounding box correctly, but the other two bounding boxes are not accurate. the labels is also
not accurate -  the model interperted the scene
as persons on a surfboard, and therefore labeled the dolphins as "person" or "surfboard". this implies a bias in the dataset
the model was trained on - the model was trained on a dataset with many images of persons on surfboards, and therefore when it
see an object above the water it interpert it as a person on a surfboard.
in the second image there are three dogs and one cat close to each other. here, the model 
located 2 bounding boxes over cat and dog together and labeled them as a cat.
possible reasons for the poor performance are: small number of classes - even when the model located the bounding box correctly,
it labeled the object not correctly. this is because the model was trained on a dataset with limited number of
classes (the basic model was trained on COCO dataset which has 80 classes).
another reason is occulusion - in the second image the cat is occluded by the dog, and the model failed to locate it. 
<br>
to resolve those issues we can: 
<br>
1. train the model on a dataset with more classes (e.g imagenet) and fine tune it on our dataset.
<br>
2. train the model on a dataset with more variability per class - many instances of the same class in different poses.
<br>
3. train the model with different size of bounding boxes to allow the model better seperate between close objects.
<br>
4. change the number of bounding boxes per grid cell to allow the model to locate more objects in the same grid cell. 


"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**

the model shows different drawbacks in each one of the images. the first image shows many many little ducks on the road together with 
people and cars. the model catches the people baut fail to catch any of the ducks. this might be because they are very small and occluded objects (they masking each other and looks like one big object). it also don't recognize any of the cars at the background possibly because of hard light conditions. the second image shows a cow licking a cat. this time the model accurately puts bounding boxes and classify the cat but he misclassify the cow as a dog. this might be because of model bias - this scene is not normal and probably most of the images the model was trained on with some animal licking a cat was of dog or another cat. therefore, when the model see a cat licked by another animal, it mistakenly interpert it as a dog (or a cat) where's here its a cow. the third image is very interesting - we see a bear in a camping playing with a stove. the model gives two, almost identically, bounding boxes around the bear one labeld as a bear and the other one (with lower probabilty) as a cat. it is possible that two factors cause the confusion - 
<br>
1. the bear's head is down. this is a form of deformation that changes the shape of the bear and makes it harder to classify
<br>
2. this is another unusual scene (bear playing with a stove) and the classification of cat might relates to model bias (cat's are more common to play with humans stuff)
<br>
the last image is of man and dog in the dark. the model do not recognize the man and recognize the dog as a horse. this is because the dark ilumination blur most of the features the helps the model recognize the objects. 

"""

part6_bonus = r"""
**Your answer:**

we did two types of image augmentations - first we did general, automatic augmentation using torchvision autoaugment. this function uses a general augmentation policy defined by IMAGENET dataset. as not all images had brightness issues, and autoaugment change the brightness, we apply this only to images that we suspected to have benefit from brightness adjustment. next, we applied geometrical augmentation on all images. this is did using test time augmentation (tta) funcionality inside YOLOv5. to do that we created a custom YOLO class which overrided the forward_augment method that takes care of the tta proccess. we can see that the results improved a lot by those augmentations - the cars that were not indentified are now identified very well. also the person in the dark now has bounding box (but with the wrong class). another example that was improved is the cat and the dogs - the model now locates the bounding box accurately and classiifed 2 of the animals correctly. to summarize - using simple augmenatations like autoaugment and tta the predictions can improved significantly with zero effort.   

"""