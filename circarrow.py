from __future__ import division #Uncomment for python2.7
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, RegularPolygon
import numpy as np
from numpy import radians as rad

def drawCirc(ax,radius,centX,centY,angle_,theta2_, ccw = True, color_='black'):
    #========Line
    arc = Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=1,color=color_)
    ax.add_patch(arc)


    #========Create the arrow head
    if ccw:
        headX = centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
        headY = centY+(radius/2)*np.sin(rad(theta2_+angle_))

    else:
        headX = centX+(radius/2)*np.cos(rad(angle_)) #Do trig to determine end position
        headY = centY+(radius/2)*np.sin(rad(angle_))


    ax.add_patch(                    #Create triangle as arrow head
        RegularPolygon(
            (headX, headY),          # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            rad(angle_+theta2_),     # orientation
            color=color_
        )
    )
    ax.set_xlim([centX-radius,centY+radius]) and ax.set_ylim([centY-radius,centY+radius]) 
    # Make sure you keep the axes scaled or else arrow will distort
