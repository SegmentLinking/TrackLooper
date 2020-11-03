#!/bin/env python

import ROOT as r
import numpy as n
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation as animation
import math

def get_centroid(points):

    return n.sum(points, 1) / points.shape[1]

def find_4_corners(points, normal_vec):
    """
    The algorithm first find the point that is furthest from the centroid. (First point)
    Then draw a straight line going through centroid to first point.
    Find the second point that is furthest from the straight line (Second point)
    The first and the second point is very unlikely that they are diagonal from each other. (Assumption)
    The assumption may break if the centroid is reallly off from true center.
    But even in such case might work out.
    Then find the mid-point between first and second point and find the point furthest away. (Third point)
    Then find the fourth-point with maximal sum distance from all three points.
    """

    # Centroid point
    centroid = get_centroid(points)

    # Rotation matrix to put the plane flat on a x-y plane and consider distance only in flat projection
    rotation = rotation_matrix_from_vectors(normal_vec, n.array([0, 0, 1]))

    # Rotate the centroid point
    centroid_rotated = rotation.dot(centroid)

    # Rotate the data points
    points_rotated = rotation.dot(points)

    points_rotated_back = rotation.T.dot(points_rotated)

    # print(points)
    # print(points_rotated)
    # print(points_rotated_back)

    # Find first_point
    dmax = 0
    first_point_index = -1
    for index, point_rotated in enumerate(points_rotated.T):

        # distance vector
        dvec_in_2d =(point_rotated[0:2] - centroid_rotated[0:2])

        # distance
        d_in_2d = math.sqrt(dvec_in_2d.dot(dvec_in_2d))

        # If distance larger save
        if d_in_2d > dmax:
            dmax = d_in_2d
            first_point_index = index

    first_point_rotated = points_rotated.T[first_point_index]

    # print(first_point_index)

    # Find the second point via finding furthest point from the line formed by centroid to the first point
    dmax = 0
    second_point_index = -1
    for index, point_rotated in enumerate(points_rotated.T):

        # Distance between line to point
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        fp_x = first_point_rotated[0]
        fp_y = first_point_rotated[1]
        cn_x = centroid_rotated[0]
        cn_y = centroid_rotated[1]

        numer = abs((fp_y - cn_y) * point_rotated[0] - (fp_x - cn_x) * point_rotated[1] + fp_x * cn_y - fp_y * cn_x)
        denom = math.sqrt((fp_y - cn_y)**2 + (fp_x - cn_x)**2)

        d_point_line = numer / denom

        if d_point_line > dmax:
            dmax = d_point_line
            second_point_index = index

    second_point_rotated = points_rotated.T[second_point_index]

    # print(second_point_index)

    # Find the third point via finding the furthes point from the mid point between first and second point
    mid_point = (points_rotated.T[first_point_index][0:2] + points_rotated.T[second_point_index][0:2]) / 2.
    dmax = 0
    third_point_index = -1
    for index, point_rotated in enumerate(points_rotated.T):
        dvec = mid_point - point_rotated[0:2]
        d = math.sqrt(dvec.dot(dvec))
        if d > dmax:
            dmax = d
            third_point_index = index

    third_point_rotated = points_rotated.T[third_point_index]

    # print(third_point_index)

    # Find the last point
    dmax = 0
    fourth_point_index = -1
    for index, point_rotated in enumerate(points_rotated.T):
        d1vec = first_point_rotated[0:2] - point_rotated[0:2]
        d2vec = second_point_rotated[0:2] - point_rotated[0:2]
        d3vec = third_point_rotated[0:2] - point_rotated[0:2]
        d = math.sqrt(d1vec.dot(d1vec)) + math.sqrt(d2vec.dot(d2vec)) + math.sqrt(d3vec.dot(d3vec))
        if d > dmax:
            dmax = d
            fourth_point_index = index

    fourth_point_rotated = points_rotated.T[fourth_point_index]

    # print(fourth_point_index)

    # Sanity check that there are no duplicate points
    idxs = [first_point_index, second_point_index, third_point_index, fourth_point_index]
    if len(list(set(idxs))) != 4:
        print("SEVERE WARNING: Duplicate points")
        print(idxs)
        sys.exit(-1)

    # Get the 4 corner points with z = 0
    first_point_rotated[2] = centroid_rotated[2]
    second_point_rotated[2] = centroid_rotated[2]
    third_point_rotated[2] = centroid_rotated[2]
    fourth_point_rotated[2] = centroid_rotated[2]

    # print first_point_rotated
    # print second_point_rotated
    # print third_point_rotated
    # print fourth_point_rotated
    # print centroid_rotated

    first_point = rotation.T.dot(first_point_rotated)
    second_point = rotation.T.dot(second_point_rotated)
    third_point = rotation.T.dot(third_point_rotated)
    fourth_point = rotation.T.dot(fourth_point_rotated)

    first_point_centered = first_point_rotated - centroid_rotated
    second_point_centered = second_point_rotated - centroid_rotated
    third_point_centered = third_point_rotated - centroid_rotated
    fourth_point_centered = fourth_point_rotated - centroid_rotated

    first_point_phi = n.arctan2(first_point_centered[1], first_point_centered[0])
    second_point_phi = n.arctan2(second_point_centered[1], second_point_centered[0])
    third_point_phi = n.arctan2(third_point_centered[1], third_point_centered[0])
    fourth_point_phi = n.arctan2(fourth_point_centered[1], fourth_point_centered[0])

    phis = [first_point_phi, second_point_phi, third_point_phi, fourth_point_phi]

    corner_points = [first_point, second_point, third_point, fourth_point]

    corner_points_ordered = [ point for _, point in sorted(zip(phis, corner_points)) ]
    corner_phis_ordered = [ phi for phi, point in sorted(zip(phis, corner_points)) ]

    # print(corner_points)
    # print(corner_phis_ordered)
    # print(corner_points_ordered)

    return corner_points_ordered

def guess_4_corner(points, normal_vec, isEndcap, isFlat, isPS):

    norm_vec = normal_vec
    # isEndcap = True if abs(normal_vec[2] - 1) < 0.001 else False
    # isFlat = True if abs(math.sqrt(norm_vec[0]**2 + norm_vec[1]**2) - 1) < 0.001 else False

    centroid = get_centroid(points)

    if isEndcap:
        # Modify norm vector to a priori value
        norm_vec = n.array([0, 0, 1])
        # Compute phi from centroid
        phi = n.arctan2(centroid[1], centroid[0])
        para_vec = n.array([math.cos(phi), math.sin(phi), 0])
        perp_vec = n.cross(para_vec, norm_vec)

        length = 2.5 if isPS else 5.0

        para_vec = para_vec * length
        perp_vec = perp_vec * 5.0

        # Then use the para/perp vectors to move around 4 corners
        return [centroid + para_vec + perp_vec, centroid + para_vec - perp_vec, centroid - para_vec - perp_vec, centroid - para_vec + perp_vec]

    elif isFlat:
        phi = n.arctan2(centroid[1], centroid[0])
        norm_vec = n.array([math.cos(phi), math.sin(phi), 0])
        para_vec = n.array([0, 0, 1])
        perp_vec = n.cross(para_vec, norm_vec)

        length = 2.5 if isPS else 5.0

        para_vec = para_vec * length
        perp_vec = perp_vec * 5.0

        # Then use the para/perp vectors to move around 4 corners
        return [centroid + para_vec + perp_vec, centroid + para_vec - perp_vec, centroid - para_vec - perp_vec, centroid - para_vec + perp_vec]
    else: # isTilted
        r = math.sqrt(norm_vec[0]**2 + norm_vec[1]**2)
        z = norm_vec[2]
        sign = 1
        if centroid[2] > 0:
            sign = -1
        # print(r, z, norm_vec, r/z)
        theta = n.arccos(r)
        phi = n.arctan2(centroid[1], centroid[0])
        para_vec = n.array([math.cos(phi) * math.sin(theta), math.sin(phi) * math.sin(theta), sign * math.cos(theta)])
        perp_vec = n.cross(para_vec, norm_vec)
        para_vec = para_vec * 2.5 # 2.5 cm is half of the module size in z-direction (parallel)
        perp_vec = perp_vec * 5 # 5 cm is half of the module size in the phi-direction (perp)
        # Then use the para/perp vectors to move around 4 corners
        return [centroid + para_vec + perp_vec, centroid + para_vec - perp_vec, centroid - para_vec - perp_vec, centroid - para_vec + perp_vec]

def get_4_corner_grid(corner_points):

    X = n.zeros((2, 2))
    Y = n.zeros((2, 2))
    Z = n.zeros((2, 2))

    X[0][0] = corner_points[0][0]
    X[0][1] = corner_points[1][0]
    X[1][1] = corner_points[2][0]
    X[1][0] = corner_points[3][0]

    Y[0][0] = corner_points[0][1]
    Y[0][1] = corner_points[1][1]
    Y[1][1] = corner_points[2][1]
    Y[1][0] = corner_points[3][1]

    Z[0][0] = corner_points[0][2]
    Z[0][1] = corner_points[1][2]
    Z[1][1] = corner_points[2][2]
    Z[1][0] = corner_points[3][2]

    return X, Y, Z

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / n.linalg.norm(vec1)).reshape(3), (vec2 / n.linalg.norm(vec2)).reshape(3)
    v = n.cross(a, b)
    c = n.dot(a, b)
    s = n.linalg.norm(v)
    kmat = n.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = n.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def find_normal_vector_from_points(points):
    """
    Find the normal vector fitting the series of 3D points that consist a plane
    Take points, subtract centroid, perform singular value decomposition, the left singular vector is the normal vector
    https://math.stackexchange.com/a/99317
    """

    # Centroid
    centroid = get_centroid(points)
    # print(centroid)

    #
    # now find the best-fitting plane for the test points
    #

    maxm = points.shape[1]
    if maxm > 150:
        maxm = 150
    points_ = n.zeros((3, maxm))
    for ihit, point in enumerate(points.T):
        if ihit >= maxm:
            continue
        points_[0][ihit] = point[0]
        points_[1][ihit] = point[1]
        points_[2][ihit] = point[2]

    # subtract out the centroid
    points_wrt_centroid = n.transpose(n.transpose(points_) - centroid)

    # singular value decomposition
    svd = n.transpose(n.linalg.svd(points_wrt_centroid))

    # print svd
    # svd[1][2] # least singular value

    # the corresponding left singular vector is the normal vector of the best-fitting plane
    normal_vec = n.transpose(svd[0])[2]

    return normal_vec

def draw_all_hits():

    xs = []
    ys = []
    zs = []
    for event in t:

        # Number of points = m
        m = int(len(event.x))

        # Read the points data for this module
        points = n.zeros((3, m))
        for ihit, (x, y, z) in enumerate(zip(event.x, event.y, event.z)):
            points[0][ihit] = x
            points[1][ihit] = y
            points[2][ihit] = z
            # print(x, y, z)

        # The plane's normal vector fitted to the points
        normal_vec = find_normal_vector_from_points(points)

        points_4_corner = find_4_corners(points, normal_vec)


    # Plot all the points
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='b')

    plt.show()
    plt.savefig("allhits.pdf")

def save_planes(t):

    fout = r.TFile("data/phase2.root", "recreate")
    c = r.TCanvas()
    c.SetCanvasSize(800, 800)

    geom_data = dict()

    h = r.TH3D("h", "", 1, -300, 300, 1, -300, 300, 1, -300, 300)
    h.Draw("AXIS")

    lines = []

    for index, event in enumerate(t):

        if index % 100 == 0:
            print index

        save_plane(event, lines, geom_data)

    import json

    points_out = open("data/phase2.txt", "w")
    points_out.write(json.dumps(geom_data, sort_keys=True, indent=4, separators=(',', ': ')))

    c.Write()
    fout.Close()

def save_plane(event, lines, fileout):

    # # if not ((event.side == 1 or event.side == 2) and event.subdet == 5):
    # # if not ((event.side == 2) and event.subdet == 5 and event.layer == 1):
    # if not (event.side == 3 and event.subdet == 5):
    # # if not (event.subdet == 4 and event.isPS == 1):
    #     return

    # Number of points = m
    m = int(len(event.x))

    # Read the points data for this module
    points = n.zeros((3, m))
    for ihit, (x, y, z) in enumerate(zip(event.x, event.y, event.z)):
        points[0][ihit] = x
        points[1][ihit] = y
        points[2][ihit] = z
        # print(x, y, z)

    # Obtain centroid
    centroid = get_centroid(points)
    phi = n.arctan2(centroid[1], centroid[0])
    # if phi > 0 and phi < 2.1:
    #     return

    if event.subdet == 5 and event.side != 3:

        # The plane's normal vector fitted to the points
        normal_vec = find_normal_vector_from_points(points)

    else:

        normal_vec = n.zeros(3)

    isEndcap = True if event.subdet == 4 else False
    isFlat = True if (event.subdet == 5 and event.side == 3) else False
    isPS = event.isPS

    # Get the most likely point of the 4 corners
    # points_4_corner = find_4_corners(points, normal_vec)
    points_4_corner = guess_4_corner(points, normal_vec, isEndcap, isFlat, isPS)

    # write to txt file
    corners = []
    for i in [0, 1, 2, 3]:
        corners.append([points_4_corner[i][2], points_4_corner[i][0], points_4_corner[i][1]])
    # fileout.write("{}: [{}],\n".format(event.detId, ",".join(corners)))
    fileout[event.detId] = corners

    # Draw TPolyLine3D
    line = r.TPolyLine3D()
    # line.SetLineColorAlpha(2, 0.2)
    if ((event.side == 1 or event.side == 2) and event.subdet == 5):
        line.SetLineColor(2)
    elif (event.subdet == 4):
        line.SetLineColor(4)
    else:
        line.SetLineColor(1)
    for i in [0, 1, 2, 3, 0]:
        line.SetNextPoint(points_4_corner[i][2], points_4_corner[i][0], points_4_corner[i][1])
    line.Draw()
    lines.append(line)

def draw_planes(t):

    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    for index, event in enumerate(t):

        draw_plane(event, ax)

    plt.show()
    plt.savefig("test.pdf")

def draw_plane(event, ax):

    # Number of points = m
    m = int(len(event.x))

    # Read the points data for this module
    points = n.zeros((3, m))
    for ihit, (x, y, z) in enumerate(zip(event.x, event.y, event.z)):
        points[0][ihit] = x
        points[1][ihit] = y
        points[2][ihit] = z
        # print(x, y, z)

    # The plane's normal vector fitted to the points
    normal_vec = find_normal_vector_from_points(points)

    # Get the most likely point of the 4 corners
    points_4_corner = find_4_corners(points, normal_vec)

    # Get the plane grid points
    Z, X, Y = get_4_corner_grid(points_4_corner)

    # draw grid
    ax.plot_wireframe(X,Y,Z, color='k')

def fit_plane(event):

    # print(event.detId)
    # if event.detId != 438043654:
    #     continue

    # Number of points = m
    m = int(len(event.x))

    # Read the points data for this module
    points = n.zeros((3, m))
    for ihit, (x, y, z) in enumerate(zip(event.x, event.y, event.z)):
        points[0][ihit] = x
        points[1][ihit] = y
        points[2][ihit] = z
        # print(x, y, z)

    # The plane's normal vector fitted to the points
    normal_vec = find_normal_vector_from_points(points)

    # Get the most likely point of the 4 corners
    points_4_corner = find_4_corners(points, normal_vec)

    # Get 4 corner point coordinates to plot scatter plot
    xs = []
    ys = []
    zs = []
    for point in points_4_corner:
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])

    # Plot all the points
    fig = plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='b')

    #-------------------------------------------------------------
    # view_direction = n.zeros(3)
    # view_direction[0] = -0.3
    # view_direction[1] = -0.3
    # view_direction[2] = 1

    # Rotate for better viewing angle
    # R = rotation_matrix_from_vectors(normal_vec, view_direction)
    # new_normal_vec = n.dot(R, normal_vec)
    # normal_vec = new_normal_vec
    #-------------------------------------------------------------

    # Centroid point
    point_vec = get_centroid(points)

    X, Y, Z = get_4_corner_grid(points_4_corner)
    ax.plot_wireframe(X,Y,Z, color='k')

    # # Depending on how the normal vector is oriented compute mesh points differently
    # # This is because the mesh points computed with an plane equation needs to be computed wrt to axis that doesn't contain zero normal vector
    # if normal_vec[2] < normal_vec[1] and normal_vec[2] < normal_vec[0]:
    #     # plot plane
    #     xlim = ax.get_xlim()
    #     zlim = ax.get_zlim()
    #     X,Z = n.meshgrid(n.arange(xlim[0], xlim[1]),
    #                      n.arange(zlim[0], zlim[1]))
    #     Y = n.zeros(X.shape)
    #     for r in range(X.shape[0]):
    #         for c in range(X.shape[1]):
    #             Y[r,c] = (normal_vec[0] * (X[r,c] - point_vec[0]) + normal_vec[2] * (Z[r,c] - point_vec[2]) - normal_vec[1] * point_vec[1]) / (-normal_vec[1])
    #     # print(X)
    #     # print(Y)
    #     # print(Z)
    #     ax.plot_wireframe(X,Y,Z, color='k')
    # else:
    #     # plot plane
    #     xlim = ax.get_xlim()
    #     ylim = ax.get_ylim()
    #     X,Y = n.meshgrid(n.arange(xlim[0], xlim[1]),
    #                      n.arange(ylim[0], ylim[1]))
    #     Z = n.zeros(X.shape)
    #     for r in range(X.shape[0]):
    #         for c in range(X.shape[1]):
    #             Z[r,c] = (normal_vec[0] * (X[r,c] - point_vec[0]) + normal_vec[1] * (Y[r,c] - point_vec[1]) - normal_vec[2] * point_vec[2]) / (-normal_vec[2])
    #     # print(X)
    #     # print(Y)
    #     # print(Z)
    #     ax.plot_wireframe(X,Y,Z, color='k')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    plt.savefig("test.pdf")

    def rotate(angle):
        ax.view_init(azim=angle)

    print("Making animation")
    rot_animation = animation.FuncAnimation(fig, rotate, frames=n.arange(0, 362, 2), interval=10)
    rot_animation.save('rotation.gif', dpi=30, writer='imagemagick')

def fit_all_planes(t):

    for event in t:
        if event.detId != 438043654:
            continue
        fit_plane(event)
        break

def write_centroids(t):

    f = open("centroids.txt", "w")

    for index, event in enumerate(t):

        if not (event.subdet == 4):
            continue 

        # Number of points = m
        m = int(len(event.x))

        # Read the points data for this module
        points = n.zeros((3, m))
        for ihit, (x, y, z) in enumerate(zip(event.x, event.y, event.z)):
            points[0][ihit] = x
            points[1][ihit] = y
            points[2][ihit] = z
            # print(x, y, z)

        # Obtain centroid
        centroid = get_centroid(points)

        # phi
        phi = n.arctan2(centroid[1], centroid[0])
        r = math.sqrt(centroid[0]**2 + centroid[1]**2)
        z = centroid[2]

        f.write("{} {} {} {}\n".format(event.detId, r, phi, z))


if __name__ == "__main__":

    f = r.TFile("data/all_sim_hits_2020_0428.root")
    t = f.Get("tree")

    save_planes(t)

    # write_centroids(t)

