## fashion_mnist experiment

play with [fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist)

also experiment with the idea of [https://arxiv.org/abs/1703.00810](https://arxiv.org/abs/1703.00810), calculating mutual information between hidden layers and input/output

## Model I tried

#### 1. Random Forest 
As a quick baseline, accuracy on test set about 87%

#### 2. LeNet5,with slightly modified. 

This basic model have a pretty good performance(99%+ accuracy) on original MNIST,  so let's see how well it perform in fashion-MNIST
The outcome is roughly 92-93% on test set

#### 3. mini-ResNet 

based on the idea of ResNet, I build a mini one (6 CNN layers + 2 fc layers).


As I run the experiment on my Macbook, so can not try out deep model due to the computation limit.

## usage

`python3 unpack_data.py` for unpacking data to numpy array

`python3 random_forest.py`

`python3 runner.py` (choose the model you want in the code)