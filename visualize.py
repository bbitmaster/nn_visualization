# -*- coding: utf-8 -*-
from nnet_toolkit import nnet

import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.cm as cm # I like the rainbow color palette
import image_plotter;

dump_to_file = False;
dump_path="""E:\\school\\python\\data\\"""
x_axis = [-1.0,1.0];
y_axis = [-1.0,1.0];

#the axis for the view
vx_axis = [-2,2];
vy_axis = [-2,2];

num_classes = 12;
examples_per_class = 25;
spread = 0.5;

img_width = 720;
img_height = 360;

frameskip = 5000

#nn parameters
num_hidden = 128
layers = [nnet.layer(2),nnet.layer(128,'tanh',n_active_count=64),nnet.layer(256,'tanh',n_active_count=128),nnet.layer(num_classes,'linear')];
learning_rate = 0.1;
np.random.seed(5);

#generate random classes
sample_data = np.zeros([2,num_classes*examples_per_class]);
class_data = np.zeros([num_classes,num_classes*examples_per_class]);

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
showprogress = False;
epoch = 1;

while(1):
    net.input = sample_data;
    net.feed_forward();
    net.error = net.output - class_data
    neterror = net.error
    net_classes = net.output
    percent_miss = 1.0 - sum(np.equal(np.argmax(net_classes,0),np.argmax(class_data,0)))/float(num_classes*examples_per_class)
    net.back_propagate()
    net.update_weights();
    output_string = "epoch: " + str(epoch) + " percent: " + str(percent_miss) + " MSE: " + str(np.sum(neterror**2))
    if(showprogress or epoch%frameskip == 0):
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
        #plt.drawLine(x1,x2,y1,y2,color=(0,0,0))
        plt.drawText(vx_axis[0],vy_axis[0],output_string)
            
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
        if(showprogress or epoch%frameskip == 0):
            plt.update()
    epoch = epoch + 1;
    
plt.hide()