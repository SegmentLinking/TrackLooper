#############
### USAGE ###
#############
# If run manually, expects a file called "tour.txt"
# Format should look like: 
#     1 100 120
#     2 -30 90
#     3 60 123
#     4 42 420
#     5 5 666
# First column is the unique index for the following 2-tuple coordinate
# Simulated annealing will be run over this data and a local minimum will be
# printed out (in the same format). Additionally, final path length change will
# be reported.
#
# If imported, prepare a list of 3-tuples (first element is unique index, next
# two are coordinates). Pass this into simAnneal(list), which returns the
# optimized path as the 2nd element of the list (preserving indices, so
# original path can always be retrieved afterwards. The first element of the
# returned list is a 2-tuple of the original and final path lengths.

import math, random

dDict = { }
def dist(p1, p2):
    return math.sqrt( (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 ) # 0-th is index

def distLookup(p1, p2):
    return dDict[(p1[0], p2[0])] # look up precached distances by index 2-tuple

def cost(points):
    s = distLookup(points[0], points[-1])
    for i in range(len(points)-1):
        s += distLookup(points[i], points[i+1])
    return s

def cacheDists(points):
    global dDict
    for i,p1 in enumerate(points):
        for j,p2 in enumerate(points):
            dDict[(p1[0], p2[0])] = dist(p1,p2)

def simAnneal(points):

    cacheDists(points)

    nPoints = len(points)
    minTour = points[:] # [:] for "deep" copy
    origTour = points[:]
    newTour = points[:]
    minTourCost = cost(minTour)

    T = 20.0
    for it in range(60):
        T -= 0.3
        nSweeps = 50*it

        for isweep in range(nSweeps):
            newTour = origTour[:]

            r1, r2 = random.randint(0, nPoints-1), random.randint(0, nPoints-1)

            tmp = newTour[r1]
            newTour[r1] = newTour[r2]
            newTour[r2] = tmp

            energy = cost(newTour) - cost(origTour)

            if(energy < 0 or random.random() < math.exp(-1.0*energy/T)):
                origTour = newTour

                if(cost(origTour) < minTourCost):
                    minTour = origTour[:]
                    minTourCost = cost(minTour)


    print "Path improvement from %f to %f" % ( cost(points), minTourCost )
    return ([cost(points), minTourCost], minTour)


if __name__ == "__main__":
    data = open("tour.txt", "r").read().strip().split("\n")
    data = [ [int(e.split()[0]), float(e.split()[1]), float(e.split()[2])] for e in data]

    out = simAnneal(data)[1]
    print '\n'*3
    for pt in out:
        print pt[0], pt[1], pt[2]

