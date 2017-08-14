#!/usr/bin/env python
"""

http://www.hannahfry.co.uk/blog/2011/11/16/bezier-curves

"""

import scipy as sp
import pylab as plt


plt.ion()
plt.close()

#### Inputs

#A list of P0's and P3's. Must be the same length

origins = [
             [1,0],
             [-0.5, sp.sqrt(3)/2], 
             [-0.5,-sp.sqrt(3)/2]
          ]

destinations = [[0,0],[0,0],[0,0]]

#The angle the control point will make with the green line
blue_angle = sp.pi/6
red_angle = sp.pi/4

#And the lengths of the lines (as a fraction of the length of the green one)
blue_len = 1./5
red_len = 1./3

### Workings

#Generate the figure
fig = plt.figure()
ax = fig.add_subplot(111)
ax.hold(True)

#Setup the parameterisation
t = sp.linspace(0,1,100)

#Read in the origin & destination points
for i in xrange(len(origins)):
    POx,POy = origins[i][0], origins[i][1]
    P3x,P3y = destinations[i][0], destinations[i][1]

#Add those to the axes
    ax.plot(POx,POy, 'ob')
    ax.plot(P3x,P3y, 'or')
    ax.plot((POx,P3x),(POy,P3y), 'g')

#Work out r and theta (as if based at P3)
    r = ((POx-P3x)**2 + (POy-P3y)**2)**0.5
    theta = sp.arctan2((POy-P3y),(POx-P3x))

#Find the relevant angles for the control points
    aO =theta + blue_angle+ sp.pi
    aD = theta - red_angle

#Work out the control points
    P1x, P1y = POx+ blue_len*r*sp.cos(aO), POy + blue_len*r*sp.sin(aO)
    P2x, P2y = P3x+ red_len*r*sp.cos(aD), P3y + red_len*r*sp.sin(aD)

#Plot the control points and their vectors
    ax.plot((P3x,P2x),(P3y,P2y), 'r')
    ax.plot((POx,P1x),(POy,P1y), 'b')
    ax.plot(P1x, P1y, 'ob')
    ax.plot(P2x, P2y, 'or')

#Use the Bezier formula
    Bx = (1-t)**3*POx + 3*(1-t)**2*t*P1x + 3*(1-t)*t**2*P2x + t**3*P3x
    By = (1-t)**3*POy + 3*(1-t)**2*t*P1y + 3*(1-t)*t**2*P2y + t**3*P3y

#Plot the Bezier curve
    ax.plot(Bx, By, 'k')
pass

#Save it
#plt.savefig('totally-awesome-bezier.png')
plt.show()
#Bosch.
