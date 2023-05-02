import os
from os import linesep
import torch
import numpy as np
import torch.nn as nn

cwd = os.getcwd()

# torch.set_default_dtype(torch.float64)
mytype = torch.float


def calculate_flowrate(info):

    Q = (torch.sqrt(2*info.Psub/info.rho) * info.area_min_value) 
    info.Q = Q

    return info
	

def calculate_pressure_on_sections(device, info, batch_size, nsec):

    '''
    ALL:  BATCH_SIZE X NSEC X 1

    '''

    Q = info.Q
    Area = info.area
    AminIndex = info.area_min_index
    AreaMin = info.area_min_value

    Q = Q.repeat(1, nsec).reshape(batch_size, nsec, 1)
    Area = Area.reshape(batch_size, nsec, 1)
    Area0 = Area[:, 0, 0].reshape(batch_size, 1)
    Area0 = Area0.repeat(1, nsec).reshape(batch_size, nsec, 1)
    AreaMin = AreaMin.repeat(1, nsec).reshape(batch_size, nsec, 1)
    Psub = info.Psub
    Psub = Psub.repeat(1, nsec).reshape(batch_size, nsec, 1)
    rho = info.rho
    rho = rho.repeat(1, nsec).reshape(batch_size, nsec, 1)

    logical = Area > 0.
    Psection = torch.Tensor(logical.size()).to(device)
    Psection[logical] = Psub[logical] - rho[logical]*torch.pow(Q[logical], 2)/2 * (1.0/Area[logical]**2 - 1.0/Area0[logical]**2) # * 1e-4
    Psection[~logical] = 0.

    del Q
    del Area
    del Area0
    del Psub
    del rho

    ### set the pressure equal to zero for sections upper than minumum area
    AminIndex = AminIndex.repeat(1, nsec).reshape(batch_size, nsec, 1)
    idx_all = torch.linspace(0, nsec - 1, nsec, device=device)
    idx_all = idx_all.repeat(batch_size).reshape(batch_size, nsec, 1)
    logical = idx_all > AminIndex
    del AminIndex
    del idx_all

    Psection[logical] = 0.

    info.Psection = Psection

    return info


def calculate_pressure_on_surface(device, info, batch_size):


    '''
    Dimensions

    yInterp:  BATCH_SIZE X Number_of_Nodes
    p:        BATCH_SIZE X NSEC
    '''

    yInterp = info.surf.ys[:, 0, :, 0]
    p = info.Psection
    p = p.reshape(p.size(0), p.size(1))

    idx = torch.linspace(0, batch_size - 1, batch_size)
    idx = idx.to(dtype=torch.int64)

    ysort, index = torch.sort(yInterp, 1)    # sort to get rid of non-contigious tensor
    del yInterp

    lo, hi, w = interp_coef_batch(device, info.ysec, ysort, idx)
    del ysort

    pInterp = interp_batch(p, lo, hi, w, idx)
    del p
    del idx

    pInterpUnsorted = pInterp.gather(1, index.argsort(1))
    del index

    # put the pressure above/below maximum/minimum Y section equal to zero
    ymin = info.ysec[:, 0].reshape(batch_size,1).repeat(1, info.surf.npt_s).reshape(batch_size, info.surf.npt_s)
    ymax = info.ysec[:, -1].reshape(batch_size,1).repeat(1, info.surf.npt_s).reshape(batch_size, info.surf.npt_s)
    logicalF = (info.surf.ys[:, 0, :, 0] < ymin) | (info.surf.ys[:, 0, :, 0] > ymax)
    del ymin
    del ymax
	
    pInterpUnsorted[logicalF] = 0.0
    del logicalF

    info.pInterpSurf = pInterpUnsorted
    return info


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


def calculate_triangle_area_and_normal_vector(info):

    '''
    Dimensions

    ele:     Number_of_Elements X 3
    x:       BATCH_SIZE X Number_of_Nodes
    xe:      BATCH_SIZE X Number_of_Elements X 3
    xAB:     BATCH_SIZE X Number_of_Elements
    Area:    BATCH_SIZE X Number_of_Elements
    ni:      BATCH_SIZE X Number_of_Elements
    normal:  BATCH_SIZE X Number_of_Elements X 3
    '''


    ele = info.surf.e_medial
    x = info.surf.xs[:, 0, :, 0]
    y = info.surf.ys[:, 0, :, 0]
    z = info.surf.zs[:, 0, :, 0]

    xe = x[:, ele]
    ye = y[:, ele]
    ze = z[:, ele]
    del x
    del y
    del z
    del ele

    xAB = xe[:, :, 1] - xe[:, :, 0]
    yAB = ye[:, :, 1] - ye[:, :, 0]
    zAB = ze[:, :, 1] - ze[:, :, 0]

    xAC = xe[:, :, 2] - xe[:, :, 0]
    yAC = ye[:, :, 2] - ye[:, :, 0]
    zAC = ze[:, :, 2] - ze[:, :, 0]
    del xe
    del ye
    del ze

    Area = 0.5*torch.sqrt(torch.pow(yAB*zAC-zAB*yAC, 2) +
                          torch.pow(zAB*xAC-xAB*zAC, 2) +
                          torch.pow(xAB*yAC-yAB*xAC, 2))

    info.surf.TrianglesArea = Area

    ni = yAB*zAC - zAB*yAC
    nj = zAB*xAC - xAB*zAC
    nk = xAB*yAC - yAB*xAC
    del xAB
    del yAB
    del zAB 
    del xAC
    del yAC
    del zAC

    ni = ni.reshape(ni.size(0), ni.size(1), 1)
    nj = nj.reshape(nj.size(0), nj.size(1), 1)
    nk = nk.reshape(nk.size(0), nk.size(1), 1)

    nmag = torch.sqrt(torch.pow(ni, 2) +
                      torch.pow(nj, 2) +
                      torch.pow(nk, 2))

    ni = ni / nmag
    nj = nj / nmag
    nk = nk / nmag

    noraml = torch.cat((ni, nj, nk), 2).reshape(ni.size(0), ni.size(1), 3)
    del ni
    del nj
    del nk

    info.surf.TrianglesNormal = noraml


    return info


def calculate_equivalent_surface_load(device, info, batch_size):


    Pres = info.pInterpSurf
    npt = Pres.size(1)
    Ele = info.surf.e_medial
    nele = Ele.size(0)
    Area = info.surf.TrianglesArea
    Area = Area.reshape(batch_size, nele, 1)
    normal = info.surf.TrianglesNormal

    ElePres = Pres[:, Ele]
    del Pres

    EqElePres = torch.zeros(batch_size, nele, 3, device=device)

    EqElePres[:, :, 0] = ElePres[:, :, 0] / 6.0 + ElePres[:, :, 1] / 12.0 + ElePres[:, :, 2] / 12.0
    EqElePres[:, :, 1] = ElePres[:, :, 0] / 12.0 + ElePres[:, :, 1] / 6.0 + ElePres[:, :, 2] / 12.0
    EqElePres[:, :, 2] = ElePres[:, :, 0] / 12.0 + ElePres[:, :, 1] / 12.0 + ElePres[:, :, 2] / 6.0
    del ElePres

    EqEleLoad = EqElePres * Area 
    del Area
    del EqElePres

    EqEleLoadX = EqEleLoad * (normal[:, :, 0].reshape(batch_size, nele, 1))
    EqEleLoadY = EqEleLoad * (normal[:, :, 1].reshape(batch_size, nele, 1))
    EqEleLoadZ = EqEleLoad * (normal[:, :, 2].reshape(batch_size, nele, 1))
    del normal
    del EqEleLoad

    EqEleLoadXRep = EqEleLoadX.repeat(1, npt, 1).reshape(batch_size, npt, nele, 3)
    del EqEleLoadX
    EqEleLoadYRep = EqEleLoadY.repeat(1, npt, 1).reshape(batch_size, npt, nele, 3)
    del EqEleLoadY
    EqEleLoadZRep = EqEleLoadZ.repeat(1, npt, 1).reshape(batch_size, npt, nele, 3)
    del EqEleLoadZ

    idx = torch.linspace(0, npt - 1, npt, device=device)
    idx = idx.to(dtype=torch.int64)
    idx = idx.repeat(batch_size).reshape(batch_size, npt, 1, 1)

    logical = Ele == idx
    del Ele

    EqEleLoadXRepLog = logical * EqEleLoadXRep
    del EqEleLoadXRep
    EqEleLoadYRepLog = logical * EqEleLoadYRep
    del EqEleLoadYRep
    EqEleLoadZRepLog = logical * EqEleLoadZRep
    del EqEleLoadZRep
    del logical

    EqEleLoadXAll = torch.sum(torch.sum(EqEleLoadXRepLog, 3), 2)
    del EqEleLoadXRepLog
    EqEleLoadYAll = torch.sum(torch.sum(EqEleLoadYRepLog, 3), 2)
    del EqEleLoadYRepLog
    EqEleLoadZAll = torch.sum(torch.sum(EqEleLoadZRepLog, 3), 2)
    del EqEleLoadZRepLog

    EqEleLoadXAll = EqEleLoadXAll.reshape(batch_size, npt, 1)
    EqEleLoadYAll = EqEleLoadYAll.reshape(batch_size, npt, 1)
    EqEleLoadZAll = EqEleLoadZAll.reshape(batch_size, npt, 1)

    EquivalentSurfLoad = torch.cat((EqEleLoadXAll, EqEleLoadYAll, EqEleLoadZAll), 2)
    del EqEleLoadXAll
    del EqEleLoadYAll
    del EqEleLoadZAll

    info.EquivalentSurfLoad = EquivalentSurfLoad

    return info


def calculate_equivalent_volume_load(device, info, batch_size):

    '''
    Dimensions

    ESL: BATCH_SIZE X Number_of_Nodes X 3
    EVL: BATCH_SIZE X Number_of_Volume_Nodes X 3

    '''

    ESL = info.EquivalentSurfLoad
    npVol = info.eigen_vx.size(2)  # number of volume points
    EVL = torch.zeros(batch_size, npVol, 3, device=device)
    EVL[:, info.corres_s, :] = ESL
    del ESL

    info.EquivalentVolLoad = EVL

    return info


def calculate_force(info, batch_size):

    '''
    Dimensions

    idx:   BATCH_SIZE
    eigx:  BATCH_SIZE X Number_of_Volume_Nodes X NMODE
    EVLx:  BATCH_SIZE X Number_of_Volume_Nodes
    Fx:    BATCH_SIZE X NMODE
    F:     BATCH_SIZE X NMODE
    '''

    idx = torch.linspace(0, batch_size - 1, batch_size)
    idx = idx.to(dtype=torch.int64)

    eigx = info.eigen_vx[:, 0, :, :]
    eigy = info.eigen_vy[:, 0, :, :]
    eigz = info.eigen_vz[:, 0, :, :]

    EVLx = info.EquivalentVolLoad[:, :, 0]
    EVLy = info.EquivalentVolLoad[:, :, 1]
    EVLz = info.EquivalentVolLoad[:, :, 2]

    Fx = torch.matmul(EVLx, eigx)[idx, idx, :]
    Fy = torch.matmul(EVLy, eigy)[idx, idx, :]
    Fz = torch.matmul(EVLz, eigz)[idx, idx, :]
    del eigx
    del eigy
    del eigz
    del EVLx
    del EVLy
    del EVLz
    del idx

    F = Fx + Fy + Fz
    del Fx
    del Fy
    del Fz

    info.ProjectedForce = F

    return info


def flow_solver(device, info, parameters):

    info = calculate_flowrate(info)
	
    info = calculate_pressure_on_sections(device, info, parameters.batch_size, parameters.nsec)

    info = calculate_pressure_on_surface(device, info, parameters.batch_size)

    info = calculate_triangle_area_and_normal_vector(info)

    info = calculate_equivalent_surface_load(device, info, parameters.batch_size)

    info = calculate_equivalent_volume_load(device, info, parameters.batch_size)

    info = calculate_force(info, parameters.batch_size)
	
	
    return 	info.ProjectedForce, info.Q


