"""
Modified on Feb 20 2020
@author: lbg@dongseo.ac.kr
"""

import re
import pygame
from sys import exit
import numpy as np
from scipy import interpolate

width = 800
height = 600
pygame.init()
screen = pygame.display.set_mode((width, height), 0, 32)

pygame.display.set_caption("ImagePolylineMouseButton")
  
# Define the colors we will use in RGB format
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE =  (  0,   0, 255)
GREEN = (  0, 255,   0)
RED =   (255,   0,   0)

tangents = []
tangents.append([1,1])
tangents.append([1,0])
tangents.append([0,-1])
tangents.append([-1,-1])
pts = [] 
pts.append([330.,440.])
pts.append([130.,140.])
pts.append([350.,320.])
pts.append([550.,450.])

tangents = np.asarray(tangents)
pts = np.asarray(pts)

screen.fill(WHITE)

# https://kite.com/python/docs/pygame.Surface.blit
clock= pygame.time.Clock()

def drawPoint(pt, color='GREEN', thick=3):
    pygame.draw.circle(screen, color, pt, thick)

def drawLine(pt0, pt1, color='GREEN', thick=3):
    x = pt0[0]
    y = pt0[1]
    x1 = pt0[0]
    y1 = pt0[1]
    x2 = pt1[0]
    y2 = pt1[1]

    dx = abs(x2-x1)
    dy = abs(y2-y1)
    gradient = dy/dx

    if gradient >1:
        dx,dy = dy,dx
        x , y = y , x
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    p = 2 * dy - dx

    for k in range(dx):
        x = x+1 if x < x2 else x-1
        if p>0:
            y = y+1 if y < y2 else y -1
            p = p+2*(dy-dx)
        else:
            p = p+2*dy

        if gradient < 1:
            drawPoint([x, y], color, thick)
        else:
            drawPoint([y, x], color, thick)
    # drawPoint((100,100), color,  thick)
    # drawPoint(pt0, color, thick)
    # drawPoint(pt1, color, thick)


def sampleCubicSplinesWithDerivative(points, tangents, resolution):
    resolution = float(resolution)
    points = np.asarray(points)
    nPoints, dim = points.shape

    dp = np.diff(points, axis=0)              
    dp = np.linalg.norm(dp, axis=1)           
    d = np.cumsum(dp)                       
    d = np.hstack([[0],d])                  
    l = d[-1]                                
    nSamples = int(l/resolution)           
    s,r = np.linspace(0,l,nSamples,retstep=True) 

    assert(len(points) == len(tangents))
    data = np.empty([nPoints, dim], dtype=object)
    for i,p in enumerate(points):
        t = tangents[i]
        assert(t is None or len(t)==dim)
        fuse = list(zip(p,t) if t is not None else zip(p,))
        data[i,:] = fuse

    # Compute splines per dimension separately.
    samples = np.zeros([nSamples, dim])
    for i in range(dim):
        poly = interpolate.BPoly.from_derivatives(d, data[:,i])
        samples[:,i] = poly(s)
    return samples

scale = 1.
tangents1 = np.dot(tangents, scale*np.eye(2))
samples1 = sampleCubicSplinesWithDerivative(pts, tangents1, 0.2)

done = False
margin = 6

for pt in pts:
    pygame.draw.rect(screen, BLUE, (pt[0]-margin, pt[1]-margin, 2*margin, 2*margin), 5)

for pt in samples1:
    drawPoint(pt)
    # drawLine(pts[i,i], pts[i+1,i+1], color,thick)

while not done:   
    time_passed = clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            pressed = -1            
        elif event.type == pygame.MOUSEBUTTONUP:
            pressed = 1            
        elif event.type == pygame.QUIT:
            done = True
        else:
            pressed = 0

    button1, button2, button3 = pygame.mouse.get_pressed()
    pygame.display.flip()

pygame.quit()

