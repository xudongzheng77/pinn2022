import os
from os import linesep
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import torch.nn as nn

# torch.set_default_dtype(torch.float64)
mytype = torch.float

def update_surface(C, info):
    '''
    Dimensions

    info.surf.xs:  BATCH_SIZE X NSEC X Number_of_Nodes X 1
    info.surf.xs0: BATCH_SIZE X NSEC X Number_of_Nodes X 1
    info.eigen_vx: BATCH_SIZE X NSEC X Number_of_Nodes X NMODE
	[:,info.corres_s,:]
    C:             BATCH_SIZE X NSEC X NMODE X 1

    '''

    info.surf.xs = info.surf.xs0 + torch.matmul(info.eigen_vx[:, :, info.corres_s, :], C)
    info.surf.ys = info.surf.ys0 + torch.matmul(info.eigen_vy[:, :, info.corres_s, :], C)
    info.surf.zs = info.surf.zs0 + torch.matmul(info.eigen_vz[:, :, info.corres_s, :], C)

    return info


# discretize the glottal channel and return the y coordinates
def discretize_glottal_channel(device, info, surf, batch_size, nsec, i_ymin, i_ymax):

    '''
    Dimensions

    ymax:   BATCH_SIZE X 1
    steps:  BATCH_SIZE X 1
    ysec:   BATCH_SIZE X NSEC

    '''

    if i_ymax == i_ymin:
        ymax = torch.amax(info.surf.ys[:, 0, surf.k, :], dim=(1, 2)).reshape(batch_size, 1)
        ymin = torch.amin(info.surf.ys[:, 0, surf.k, :], dim=(1, 2)).reshape(batch_size, 1)
	    
        ymax = ymax - (ymax - ymin) * i_ymin/100.0  # % reduction to make segmentation robust
        ymin = ymin + (ymax - ymin) * i_ymin/100.0  # % reduction to make segmentation robust
    else:
        ymax = info.surf.ys[:, 0, i_ymax, :].reshape(batch_size, 1)
        ymin = info.surf.ys[:, 0, i_ymin, :].reshape(batch_size, 1)

        ymax = ymax - 0.02
        ymin = ymin + 0.02		


    steps = (1.0/(nsec-1))*(ymax-ymin)

    ysec = steps*torch.arange(nsec, device=device) + ymin

    del ymin
    del ymax
    del steps

    return ysec


def find_intersection_coordinates(device, info, surf, batch_size, nsec, y_intersect):

    '''
    Dimensions

    y_intersect:      BATCH_SIZE X NSEC X 1
    y_intersect_rep:  BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3
    y:                BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3
    y12:              BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 1
    yicat:            BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3

    '''

    e = surf.e_medial

    x = info.surf.xs[:, :, e].reshape(batch_size, nsec, e.size(0), e.size(1))
    y = info.surf.ys[:, :, e].reshape(batch_size, nsec, e.size(0), e.size(1))
    z = info.surf.zs[:, :, e].reshape(batch_size, nsec, e.size(0), e.size(1))

    y_intersect = y_intersect.reshape(batch_size, nsec, 1)
    y_intersect_rep = y_intersect.repeat(1, 1, e.size(0)*3)
    y_intersect_rep = y_intersect_rep.reshape(batch_size, nsec, e.size(0), 3)

    del e

    ### form the matrix to determine which two-pairs of points should be considered for intersection
    y_diff = y - y_intersect_rep
    y12 = vector_multiplication(y_diff, 0, 1)
    y23 = vector_multiplication(y_diff, 1, 2)
    y13 = vector_multiplication(y_diff, 0, 2)
    ycat = torch.cat((y12, y23, y13), 3)
    logical = ycat < 0

    del y12
    del y23
    del y13
    del ycat
    del y_diff

    ### find the intersection between two-pairs of points of elements and the intersection plane
    xi12 = linear_interpolation(device, y_intersect_rep[:, :, :, 0], y[:, :, :, 0], y[:, :, :, 1], x[:, :, :, 0], x[:, :, :, 1])
    zi12 = linear_interpolation(device, y_intersect_rep[:, :, :, 0], y[:, :, :, 0], y[:, :, :, 1], z[:, :, :, 0], z[:, :, :, 1])
    xi23 = linear_interpolation(device, y_intersect_rep[:, :, :, 0], y[:, :, :, 1], y[:, :, :, 2], x[:, :, :, 1], x[:, :, :, 2])
    zi23 = linear_interpolation(device, y_intersect_rep[:, :, :, 0], y[:, :, :, 1], y[:, :, :, 2], z[:, :, :, 1], z[:, :, :, 2])
    xi13 = linear_interpolation(device, y_intersect_rep[:, :, :, 0], y[:, :, :, 0], y[:, :, :, 2], x[:, :, :, 0], x[:, :, :, 2])
    zi13 = linear_interpolation(device, y_intersect_rep[:, :, :, 0], y[:, :, :, 0], y[:, :, :, 2], z[:, :, :, 0], z[:, :, :, 2])
    xicat = torch.cat((xi12, xi23, xi13), 3)
    zicat = torch.cat((zi12, zi23, zi13), 3)

    del x
    del y
    del z
    del xi12
    del zi12
    del xi23
    del zi23
    del xi13
    del zi13
    del y_intersect_rep

    return logical, xicat, zicat


def find_intersection_with_midline(device, logical, x, z, midline):

    logicalF = ~logical
    x[logicalF] = midline
    del logicalF

    midline_tensor = torch.ones_like(x[:, :, :, 0])*midline
    zim12 = linear_interpolation(device, midline_tensor, x[:, :, :, 0], x[:, :, :, 1], z[:, :, :, 0], z[:, :, :, 1])
    zim23 = linear_interpolation(device, midline_tensor, x[:, :, :, 1], x[:, :, :, 2], z[:, :, :, 1], z[:, :, :, 2])
    zim13 = linear_interpolation(device, midline_tensor, x[:, :, :, 0], x[:, :, :, 2], z[:, :, :, 0], z[:, :, :, 2])
    zim = torch.cat((zim12, zim23, zim13), 3)
    del midline_tensor
    del zim12
    del zim23
    del zim13

    ### form the matrix to determine which two-pairs of points should be considered for intersection
    x_diff = x - midline
    del x

    x12 = vector_multiplication(x_diff, 0, 1)
    x23 = vector_multiplication(x_diff, 1, 2)
    x13 = vector_multiplication(x_diff, 0, 2)
    del x_diff
    xcat = torch.cat((x12, x23, x13), 3)
    del x12
    del x23
    del x13

    logical_zim = xcat < 0
    del xcat

    return logical_zim, zim


def compute_glottis_area(logical, x, z, logical_zim, zim, midline, flag_right):

    '''
    Dimensions

    x:              BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3
    a:              BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf
    area:           BATCH_SIZE X NSEC
    area_min_value: BATCH_SIZE X 1
    area_min_index: BATCH_SIZE X 1
    '''

    xi = x
    logicalF = ~logical
    xi[logicalF] = midline
    zi = z * logical
    del logicalF
    del logical

    xi[torch.isnan(xi)] = midline
    zi[torch.isnan(zi)] = 0.0

    if flag_right:
        dx = midline - xi
    else:
        dx = xi - midline

    logical_dx = dx > 0

    a_m1ed1 = 0.5 * (zim[:, :, :, 0] - zi[:, :, :, 0]) * dx[:, :, :, 0] * logical_zim[:, :, :, 0] * logical_dx[:, :, :, 0]
    a_m1ed2 = 0.5 * (zim[:, :, :, 0] - zi[:, :, :, 1]) * dx[:, :, :, 1] * logical_zim[:, :, :, 0] * logical_dx[:, :, :, 1]
    a_m2ed2 = 0.5 * (zim[:, :, :, 1] - zi[:, :, :, 1]) * dx[:, :, :, 1] * logical_zim[:, :, :, 1] * logical_dx[:, :, :, 1]
    a_m2ed3 = 0.5 * (zim[:, :, :, 1] - zi[:, :, :, 2]) * dx[:, :, :, 2] * logical_zim[:, :, :, 1] * logical_dx[:, :, :, 2]
    a_m3ed1 = 0.5 * (zim[:, :, :, 2] - zi[:, :, :, 0]) * dx[:, :, :, 0] * logical_zim[:, :, :, 2] * logical_dx[:, :, :, 0]
    a_m3ed3 = 0.5 * (zim[:, :, :, 2] - zi[:, :, :, 2]) * dx[:, :, :, 2] * logical_zim[:, :, :, 2] * logical_dx[:, :, :, 2]
    del dx
    del zim
    del logical_dx
    del logical_zim

    a_m_all = a_m1ed1 + a_m1ed2 + a_m2ed2 + a_m2ed3 + a_m3ed1 + a_m3ed3
    logical_am = a_m_all > 0
    del a_m1ed1
    del a_m1ed2
    del a_m2ed2
    del a_m2ed3
    del a_m3ed1
    del a_m3ed3

    # calculate glottal area generated by all elements
    if flag_right:
        log_pos_x = xi > midline
    else:
        log_pos_x = xi < midline
    pos_x = xi[log_pos_x]
    xi[log_pos_x] = midline

    xi[torch.isnan(xi)] = midline
    zi[torch.isnan(zi)] = 0.0


    zv, zindices = torch.sort(zi, 3)

    h = zv[:, :, :, 2] - zv[:, :, :, 1]
    del zv

    if flag_right:
        a = 0.5 * h * (3*midline - (xi[:, :, :, 0] + xi[:, :, :, 1] + xi[:, :, :, 2]))
    else:
        a = 0.5 * h * (-3 * midline + (xi[:, :, :, 0] + xi[:, :, :, 1] + xi[:, :, :, 2]))

    del h
    del xi
    del zi
	
    a[logical_am] = a_m_all[logical_am]
    del a_m_all
    del logical_am

    area = torch.sum(a, 2)
    del a
    area = area * 2         # to add the other vocal fold (symmetric assumption)

    logical = area > 1.0e-6 # set a limit for closing
    area = area*logical
    del logical

    area_min_value = torch.amin(area, dim=1, keepdim=True)
    area_min_index = torch.argmin(area, dim=1, keepdim=True)

    return area, area_min_value, area_min_index, pos_x, log_pos_x


def get_minimum_area_shape(device, logical, x, z, y_sec_MinIndex, batch_size, midline, z0Interp, z1Interp, nInterp):

    '''
    Dimensions

    logical:        BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3
    x:              BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3
    idx:            BATCH_SIZE
    xi_min_area:    BATCH_SIZE X (Number_of_Elements_in_Med_Surf * 3)
    zInterp:        BATCH_SIZE X nInterp
    area_min_index: BATCH_SIZE X 1
    '''

    z_out = -0.1

    logicalF = ~logical
    x[logicalF] = midline
    z[logicalF] = z_out

    idx = torch.linspace(0, batch_size - 1, batch_size)
    idx = idx.to(dtype=torch.int64)

    log_min_area = logical[idx, y_sec_MinIndex[0:batch_size, 0], :, :]
    xi_min_area = x[idx, y_sec_MinIndex[0:batch_size, 0], :, :]
    del x
    zi_min_area = z[idx, y_sec_MinIndex[0:batch_size, 0], :, :]
    del z
    del y_sec_MinIndex

    log_min_area = log_min_area.reshape(log_min_area.size(0), log_min_area.size(1)*log_min_area.size(2))
    xi_min_area = xi_min_area.reshape(xi_min_area.size(0), xi_min_area.size(1)*xi_min_area.size(2))
    zi_min_area = zi_min_area.reshape(zi_min_area.size(0), zi_min_area.size(1)*zi_min_area.size(2))

    zi_min_area, index = torch.sort(zi_min_area, 1)
    xi_min_area = smart_sort(xi_min_area, index)
    log_min_area = smart_sort(log_min_area, index)

    zmaxi, index = torch.max(zi_min_area, 1)
    zmaxi = zmaxi.reshape(batch_size, 1)
    logical_zn = zi_min_area == z_out
    zi_min_area[logical_zn] = 1000.0
    zmini, index = torch.min(zi_min_area, 1)
    zmini = zmini.reshape(batch_size, 1)
    zi_min_area[logical_zn] = z_out
    zin_min_area_norml = (zi_min_area - zmini)/(zmaxi - zmini)
    del zmaxi
    del zmini
    zin_min_area_norml = zin_min_area_norml*(z1Interp-z0Interp) + z0Interp

    zInterp = torch.linspace(z0Interp, z1Interp, nInterp)
    zInterp = (zInterp.repeat(1, batch_size)).reshape(batch_size, nInterp)

    lo, hi, w = interp_coef_batch(device, zi_min_area, zInterp, idx)

    xInterp = interp_batch(xi_min_area, lo, hi, w, idx)

    return log_min_area, xi_min_area, zin_min_area_norml, xInterp, zInterp


def smart_sort(a, index):
    d1, d2 = a.size()
    a_sort = a[
               torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
               index.flatten()
              ].view(d1, d2)
    return a_sort


def interp_coef_batch(device, x0, x, idx):

    # find the indices into the original array
    a = torch.searchsorted(x0.to(device), x.to(device), right=True)
    b = torch.ones(a.size(1))*x0.size(1) - 1

    hi = torch.minimum(b.to(device), a.to(device))
    lo = torch.maximum(b.to(device)*0, hi.to(device) - 1)

    hi = hi.to(dtype=torch.int64)
    lo = lo.to(dtype=torch.int64)

    # calculate the distance within the range
    d_left = x.to(device) - x0[:, lo][idx, idx, :].to(device)
    d_right = x0[:, hi][idx, idx, :].to(device) - x.to(device)
    d_total = d_left + d_right
    logical = ~d_total.eq(0)	
    # weights are the proportional distance
    w = torch.Tensor(d_total.size()).to(device)
    w[logical] = d_right[logical] / d_total[logical]
    # correction if we're outside the range of the array
    w[~logical] = 0.		
    #w[w.isnan()] = 0.0
    # return the information contained by the projection matrices
    return lo, hi, w


def interp_batch(y0, lo, hi, w, idx):

    return w * y0[:, lo][idx, idx, :] + (1 - w) * y0[:, hi][idx, idx, :]


def linear_interpolation(device, xi, x1, x2, y1, y2):

    # xi, x1, x2, y1, y2 = xi.to(device), x1.to(device), x2.to(device), y1.to(device), y2.to(device)

    diff_x = x2 - x1
    logical = ~diff_x.eq(0)
    del diff_x
    yi = torch.Tensor(xi.size()).to(device)

    yi[logical] = (xi[logical] - x1[logical]) / (x2[logical] - x1[logical]) * (y2[logical] - y1[logical]) + y1[logical]
    yi[~logical] = 0.
    del logical
    #yi = (xi - x1) / (x2 - x1) * (y2 - y1) + y1
    yi = yi.reshape(yi.size(0), yi.size(1), yi.size(2), 1)

    return yi


def vector_multiplication(mat, i, j):

    vij = mat[:, :, :, i]*mat[:, :, :, j]
    vij = vij.reshape(vij.size(0), vij.size(1), vij.size(2), 1)

    return vij


def discretize_channel_Ant_Pos(device, z0Interp, z1Interp, nInterp, batch_size):

    '''
    Dimensions
    zInterp: nInterp
    zsec:    BATCH_SIZE X nInterp

    '''

    offset = 1e-4
    zInterp = torch.linspace(z0Interp, z1Interp, nInterp)
    zInterp[0] = zInterp[0] + offset
    zInterp[-1] = zInterp[-1] - offset
    zsec = (zInterp.repeat(1, batch_size)).reshape(batch_size, nInterp)

    return zsec.to(device)


def find_intersection_coordinates_Zplane(device, info, surf, batch_size, nsec, z_intersect):

    '''
    Dimensions

    z_intersect:      BATCH_SIZE X NSEC X 1
    z_intersect_rep:  BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3
    z:                BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3
    z12:              BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 1
    zicat:            BATCH_SIZE X NSEC X Number_of_Elements_in_Med_Surf X 3

    '''

    e = surf.e_medial

    x = info.surf.xs[:, :, e].reshape(batch_size, nsec, e.size(0), e.size(1))
    y = info.surf.ys[:, :, e].reshape(batch_size, nsec, e.size(0), e.size(1))
    z = info.surf.zs[:, :, e].reshape(batch_size, nsec, e.size(0), e.size(1))

    z_intersect = z_intersect.reshape(batch_size, nsec, 1)
    z_intersect_rep = z_intersect.repeat(1, 1, e.size(0)*3)
    z_intersect_rep = z_intersect_rep.reshape(batch_size, nsec, e.size(0), 3)
    del z_intersect
    del e
    ### form the matrix to determine which two-pairs of points should be considered for intersection
    z_diff = z - z_intersect_rep
    z12 = vector_multiplication(z_diff, 0, 1)
    z23 = vector_multiplication(z_diff, 1, 2)
    z13 = vector_multiplication(z_diff, 0, 2)
    del z_diff
    zcat = torch.cat((z12, z23, z13), 3)
    del z12
    del z23
    del z13
    logical = zcat < 0
    del zcat

    ### find the intersection between two-pairs of points of elements and the intersection plane
    xi12 = linear_interpolation(device, z_intersect_rep[:, :, :, 0], z[:, :, :, 0], z[:, :, :, 1], x[:, :, :, 0], x[:, :, :, 1])
    xi23 = linear_interpolation(device, z_intersect_rep[:, :, :, 0], z[:, :, :, 1], z[:, :, :, 2], x[:, :, :, 1], x[:, :, :, 2])
    xi13 = linear_interpolation(device, z_intersect_rep[:, :, :, 0], z[:, :, :, 0], z[:, :, :, 2], x[:, :, :, 0], x[:, :, :, 2])
    xicat = torch.cat((xi12, xi23, xi13), 3).to(device)
    
    del xi23
    del xi13
    del x

    yi12 = linear_interpolation(device, z_intersect_rep[:, :, :, 0], z[:, :, :, 0], z[:, :, :, 1], y[:, :, :, 0], y[:, :, :, 1])
    yi23 = linear_interpolation(device, z_intersect_rep[:, :, :, 0], z[:, :, :, 1], z[:, :, :, 2], y[:, :, :, 1], y[:, :, :, 2])
    yi13 = linear_interpolation(device, z_intersect_rep[:, :, :, 0], z[:, :, :, 0], z[:, :, :, 2], y[:, :, :, 0], y[:, :, :, 2])
    yicat = torch.cat((yi12, yi23, yi13), 3)
    del yi12
    del yi23
    del yi13
    del y
    del z_intersect_rep
    del z

    return logical, xicat, yicat


def get_channel_shape_ZPlane(device, logical, x, y, batch_size, z0Interp, z1Interp, nInterp, flag_right):

    if flag_right:
        x_out = -0.1
    else:
        x_out = 100.0

    xi = x
    logicalF = ~logical
    xi[logicalF] = x_out
    yi = y * logical

    d1, d2, d3, d4 = xi.size()
    xi = xi.reshape(d1, d2, d3 * d4)
    if flag_right:
        xProjection, index = torch.max(xi, 2)
    else:
        xProjection, index = torch.min(xi, 2)
    zInterp = torch.linspace(z0Interp, z1Interp, nInterp)
    zProjection = (zInterp.repeat(1, batch_size)).reshape(batch_size, nInterp)

    return xi.to(device), yi.to(device), xProjection.to(device), zProjection.to(device)

def get_projection_shape(device, C, info, surf, parameters, batch_size):

    info = update_surface(C, info)

    ysec = discretize_glottal_channel(device, info, surf,  batch_size, parameters.nsec, parameters.i_ymin, parameters.i_ymax)

    logical, xicat, zicat = find_intersection_coordinates(device, info, surf, batch_size, parameters.nsec, y_intersect=ysec)

    logical_zim, zim = find_intersection_with_midline(device, logical, xicat, zicat, parameters.midline)

    area, area_min_value, area_min_index, pos_x, log_pos_x = compute_glottis_area(logical, xicat, zicat, logical_zim, zim, parameters.midline, parameters.flag_right)

    mat = ysec[:, area_min_index[:, 0]]
    ysec_min_value = (torch.diagonal(mat)).reshape(batch_size, 1)

    xicat[log_pos_x] = pos_x



    zsec = discretize_channel_Ant_Pos(device, z0Interp=parameters.zmin,
                                      z1Interp=parameters.zmax,
                                      nInterp=parameters.nInterp,
                                      batch_size=batch_size)

    logical_zp, xicat_zp, yicat_zp = find_intersection_coordinates_Zplane(device, info, surf, batch_size,
                                                                          nsec=parameters.nsec,
                                                                          z_intersect=zsec)

    xInterpAll, yInterpAll, xProjection, zProjection = get_channel_shape_ZPlane(device, logical_zp, xicat_zp, yicat_zp,
                                                                                batch_size=batch_size,
                                                                                z0Interp=parameters.zmin,
                                                                                z1Interp=parameters.zmax,
                                                                                nInterp=parameters.nInterp,
                                                                                flag_right=parameters.flag_right)


    info.ysec = ysec

    info.area = area
    info.area_min_value = area_min_value
    info.area_min_index = area_min_index
    info.xInterpAll = xInterpAll
    info.yInterpAll = yInterpAll

    info.xInterp = xProjection
    info.zInterp = zProjection
	
    return info

