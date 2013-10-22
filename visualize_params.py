#mnist_train_params
from nnet_toolkit import select_funcs as sf;

#A handy name for this run. The data file will be given this name as
#<resultsdir><simname><version>.h5py
simname = 'visualize_noselectfunc'
version = '1.1'
img_dir = '../vis_images/'

img_width=720
img_height=360

learning_rate = .1
training_epochs = 3000

forget_epoch = 1000

num_hidden = 32
activation_function='tanh'
num_selected_neurons=16
select_func = sf.most_negative_select_func

#uncomment below to enable hidden layer
num_hidden2 = 32;
activation_function2='tanh'
num_selected_neurons2=100
select_func2 = None

activation_function_final='tanh'

#if this is false then we output to the screen instead
dump_to_file=False

frameskip=100

num_classes = 12
examples_per_class = 25

data_x_min = -1.0
data_x_max = 1.0
data_y_min = -1.0
data_y_max = 1.0

axis_x_min = -2.0
axis_x_max = 2.0
axis_y_min = -2.0
axis_y_max = 2.0

random_seed = 4;

spread = 0.05