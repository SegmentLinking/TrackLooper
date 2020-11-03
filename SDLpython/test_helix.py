import numpy as np
import math
from sdlmath import get_track_point, construct_helix_from_points


#TRUTH
#pt = 1.1325262784957886
#eta = -1.224777340888977
#phi = 1.841907024383545
#vx = -0.000797238782979548
#vy = -0.0006373987998813391
#vz = 3.2855265140533447

pt = 1.2663558721542358
eta = 0.45801877975463867
phi = -1.9742246866226196
vx = 0.001989313866943121
vy = -0.0007590867462567985
vz = 5.124274253845215
charge = 1


xs,ys,zs,rs = get_track_point(pt,eta,phi,vx,vy,vz,charge,t = 1)
print(xs,ys,zs)


#pt = 1
#xs = -4.818972663825464
#ys = 27.865702430922006
#zs = -37.264586548075904
#vx = 0
#vy = 0
#vz = 0

#zs = -37.2642637125735
#xs = -14.216115063345738
#ys = 24.446095053032607

#zs = -39.76510480634263
#xs = -3.3384174266571005
#ys = 23.796874020953346

center, phi, t, lam = construct_helix_from_points(pt,vx,vy,vz,xs,ys,zs,charge)
print("From solver")
print("center = ",center)
print("phi=",phi)
print("lambda = ",lam)
print("t = ",t)

print(get_track_point(pt,lam,phi,vx,vy,vz,charge,t = t ))
