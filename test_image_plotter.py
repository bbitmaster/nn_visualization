
import image_plotter
import numpy as np

Plt = image_plotter.image_plotter();
Plt.initImg()
Plt.setAxis(-200,200,-200,200)
x1 = np.random.randint(-100,200,100)
y1 = np.random.randint(-200,200,100)
x2 = np.random.randint(-200,200,100)
y2 = np.random.randint(-200,200,100)
color = np.random.rand(100,3)

#Plt.drawLine(x1,y1,x2,y2,color=color);
x = np.random.randint(-200,200,100)
y = np.random.randint(-200,200,100)
Plt.drawPoint(x,y,size=1,color=color)
Plt.show()

while(1):
    if(Plt.processEvents() == 27):
        break
Plt.hide()