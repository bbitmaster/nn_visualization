# -*- coding: utf-8 -*-
import sys

from nnet_toolkit import nnet

import numpy as np
import matplotlib.cm as cm # used for the color pallete
import image_plotter;
from nnet_toolkit import select_funcs as sf;
from autoconvert import autoconvert

#Get the parameters file from the command line
#use mnist_train__forget_params.py by default (no argument given)
if(len(sys.argv) > 1):
    params_file = sys.argv[1]
else:
    params_file = 'visualize_params.py'

p = {}
execfile(params_file,p)

#grab extra parameters from command line
for i in range(2,len(sys.argv)):
    (k,v) = sys.argv[i].split('=')
    v = autoconvert(v)
    if(v == 'minabs'):
        v = sf.minabs_select_func
    elif(v == 'maxabs'):
        v = sf.maxabs_select_func
    elif(v == 'most_negative'):
        v = sf.most_negative_select_func
    elif(v == 'most_positive'):
        v = sf.most_positive_select_func
    elif(v == 'minabs_normalized'):
        v = sf.minabs_normalized_select_func
    elif(v == 'maxabs_normalized'):
        v = sf.maxabs_normalized_select_func
    elif(v == 'most_negative_normalized'):
        v = sf.most_negative_normalized_select_func
    elif(v == 'most_positive_normalized'):
        v = sf.most_positive_normalized_select_func
    p[k] = v
    print(str(k) + ":" + str(v))
np.random.seed(p['random_seed']);

dump_to_file = p['dump_to_file'];
dump_path=p['img_dir']
x_axis = [p['data_x_min'],p['data_x_max']];
y_axis = [p['data_y_min'],p['data_y_max']];

#the axis for the view
vx_axis = [p['axis_x_min'],p['axis_x_max']];
vy_axis = [p['axis_y_min'],p['axis_y_max']];

num_classes = p['num_classes']
examples_per_class = p['examples_per_class'];
spread = p['spread']

img_width = p['img_width'];
img_height = p['img_height'];

frameskip = p['frameskip']

num_hidden = p['num_hidden']

layers = [];
layers.append(nnet.layer(2))
layers.append(nnet.layer(p['num_hidden'],p['activation_function'],select_func=p['select_func'],select_func_params=p['num_selected_neurons'],dropout=p['dropout']))

#Add 2nd and 3rd hidden layers if there are parameters indicating that we should
if(p.has_key('num_hidden2')):
    layers.append(nnet.layer(p['num_hidden2'],p['activation_function2'],select_func=p['select_func2'],select_func_params=p['num_selected_neurons2'],dropout=p['dropout2']))
if(p.has_key('num_hidden3')):
    layers.append(nnet.layer(p['num_hidden3'],p['activation_function3'],select_func=p['select_func3'],select_func_params=p['num_selected_neurons3'],dropout=p['dropout3']))
layers.append(nnet.layer(num_classes,p['activation_function_final']))

learning_rate = p['learning_rate']


#generate random classes
sample_data = np.zeros([2,num_classes*examples_per_class]);
class_data = np.ones([num_classes,num_classes*examples_per_class])*-1.0;

for i in range(num_classes):
    #get random center
    center_x = (np.random.rand(1) * (x_axis[1] - x_axis[0])) + x_axis[0];
    center_y = (np.random.rand(1) * (y_axis[1] - y_axis[0])) + y_axis[0];
    c = (np.random.randn(2,examples_per_class)*spread);
    c[0,:] += center_x;
    c[1,] += center_y;
    sample_data[0:2,i*examples_per_class:(i+1)*examples_per_class] = c;
    class_data[i,i*examples_per_class:(i+1)*examples_per_class] = 1.0;
    
#build a pallete
pal = cm.rainbow(np.linspace(0,1,num_classes));
pal = pal[0:num_classes,0:3]

plt = image_plotter.image_plotter();
plt.initImg(img_width,img_height);
plt.setAxis(vx_axis[0],vx_axis[1],vy_axis[0],vy_axis[1])
c=pal[np.argmax(class_data,0),:]

if(not dump_to_file):
    plt.show()

net = nnet.net(layers,learning_rate)
epoch = 1;

while(epoch < p['total_epochs']):
    net.input = sample_data;
    net.feed_forward();
    net.error = net.output - class_data
    neterror = net.error
    net_classes = net.output
    percent_miss = 1.0 - sum(np.equal(np.argmax(net_classes,0),np.argmax(class_data,0)))/float(num_classes*examples_per_class)
    if epoch < p['training_epochs']:
        net.back_propagate()
        net.update_weights();
    
    if epoch == p['training_epochs']:
        net.train = False;
        
    output_string = "epoch: " + str(epoch) + " percent: " + str(percent_miss) + " MSE: " + str(np.sum(neterror**2))
    #if we're dumping to a file, then plot everything
    #if we aren't dumping to a file then only plot at every n'th frame where n is frameskip
    if(dump_to_file or epoch%frameskip == 0):
        xv, yv = np.meshgrid(np.linspace(vx_axis[0],vx_axis[1],img_width),np.linspace(vy_axis[0],vy_axis[1],img_height))
        xv = np.reshape((xv),(img_height*img_width))
        yv = np.reshape((yv),(img_height*img_width))
        net.input = np.vstack((xv,yv))
        net.feed_forward()
        img_data=pal[np.argmax(net.output,0),:]
        img_data = img_data.reshape((img_height,img_width,3))
        plt.setImg(img_data)
        
        #draw incorrect as black, correct as white
        correct_pal = np.array([[0,0,0],[1,1,1]])
        correct = np.equal(np.argmax(net_classes,0),np.argmax(class_data,0))
        c = correct_pal[np.int32(correct),:]
        
        plt.drawPoint(sample_data[0,:],sample_data[1,:],size=1,color=c)

        #draw dot-product = 0 lines
        x1 = np.zeros(num_hidden);
        y1 = np.zeros(num_hidden);
        x2 = np.zeros(num_hidden);
        y2 = np.zeros(num_hidden);
        
        for i in range(num_hidden):
            #get m and b for y = mx + b
            m = -net.layer[0].weights[i,0]/net.layer[0].weights[i,1];
            b = net.layer[0].weights[i,2]/net.layer[0].weights[i,1] 
            #if slope is large, compute x = (y - b)/m
            if(np.abs(m) > 1):
                y1[i] = vy_axis[0];
                x1[i] = (vy_axis[0] - b)/m
                y2[i] = vy_axis[1];
                x2[i] = (vy_axis[1] - b)/m
            else:
                x1[i] = vx_axis[0];
                y1[i] = m*vx_axis[0] + b;
                x2[i] = vx_axis[1];
                y2[i] = m*vx_axis[1] + b;
        plt.drawLine(x1,x2,y1,y2,color=(0,0,0))
        plt.drawTextImg(1,1,"epoch: " + str(epoch),color=(0,0,0))
        plt.drawTextImg(120,1,"percent: " + str(percent_miss),color=(0,0,0))
        plt.drawTextImg(350,1,"MSE: " + str(np.sum(neterror**2)),color=(0,0,0))
        
            
#    x1 = np.tile(-5,num_hidden)
 #   y1 = -net.layer[0].weights[0:num_hidden,0]/net.layer[0].weights[0:num_hidden,1]*(-5) - net.layer[0].weights[0:num_hidden,2]/net.layer[0].weights[0:num_hidden,1]    
    
 #   x2 = np.tile(5,num_hidden)
 #   y2 = -net.layer[0].weights[0:num_hidden,0]/net.layer[0].weights[0:num_hidden,1]*(5) - net.layer[0].weights[0:num_hidden,2]/net.layer[0].weights[0:num_hidden,1]

    

    keypress = plt.processEvents();
    if(keypress == 27):
        break
    if(keypress == 97):
        showprogress = False;
        print("Not showing Progress")
    if(keypress == 115):
        showprogress = True
        print("Showing Progress")
    if(keypress == 122): #'z'
        net.train = False
        print("Dropout Off")
    if(keypress == 120): #'x'
        net.train = True
        print("Dropout On")

    print(output_string)

    if(dump_to_file):
        plt.save_plot(dump_path + "nn_dump" + str(epoch) + ".png")
    else:
        if(epoch%frameskip == 0):
            plt.update()
    epoch = epoch + 1;
    
plt.hide()