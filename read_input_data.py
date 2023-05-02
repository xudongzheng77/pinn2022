import numpy as np
import math
import os
import torch
from numba import jit

#@jit(nopython=True)
def read_surface_mesh(InputDir):
    inpdir = InputDir+'/surface.dat'
    inpdir = os.linesep.join([s for s in inpdir.splitlines() if s])
    with open(inpdir, 'r') as f:
        line = f.readline()
        numlist = line.strip().split()
        npt_s = int(numlist[0])
        nel_s = int(numlist[1])
        nnode_s = int(numlist[2])

        xs = torch.zeros(npt_s, 1)
        ys = torch.zeros(npt_s, 1)
        zs = torch.zeros(npt_s, 1)
        xs0 = torch.zeros(npt_s, 1)
        ys0 = torch.zeros(npt_s, 1)
        zs0 = torch.zeros(npt_s, 1)
        ps = torch.zeros(npt_s, 1)

        xc = torch.zeros(npt_s, 1)
        yc = torch.zeros(npt_s, 1)
        zc = torch.zeros(npt_s, 1)
        eles = torch.zeros((nel_s, nnode_s+1), dtype=int)  # 1 extra slot for boundary flag
        flag_exclude = torch.zeros((nel_s, 1), dtype=int)    #set exclude flag, only the medial surface needed for FSI
        flag_node = torch.zeros((nel_s, 1), dtype=int)

        count_line = 0
        for line in f.readlines():
            #print(line)
            if(count_line > 0 and count_line < npt_s+1):
                coordlist = line.strip().split()
                ipt = count_line - 1
                xs[ipt] = float(coordlist[0])
                ys[ipt] = float(coordlist[1])
                zs[ipt] = float(coordlist[2])
                xs0[ipt] = float(coordlist[0])
                ys0[ipt] = float(coordlist[1])
                zs0[ipt] = float(coordlist[2])
            if(count_line >= npt_s+1):
                indexlist = line.strip().split()
                iel = count_line - npt_s - 1
                for j in range(nnode_s+1):
                    eles[iel, j] = int(indexlist[j])
                if(eles[iel, nnode_s]==0):
                    flag_exclude[iel] = 1
            count_line += 1
    f.close()


    for i in range(nel_s):
        ind = eles[i, -1]
        for j in range(nnode_s):
            ipt = eles[i,j] - 1
            flag_node[ipt] = ind

    surf = Parameters(npt_s=npt_s, nel_s=nel_s, nnode_s=nnode_s, xs0=xs0, ys0=ys0, zs0=zs0,
        xs=xs, ys=ys, zs=zs, ps=ps, eles=eles, flag_exclude=flag_exclude, flag_node=flag_node, xc=xc, yc=yc, zc=zc)
    return surf


#@jit(nopython=True)
def read_volume_mesh(InputDir):
    inpdir = InputDir+'/volume.dat'
    inpdir = os.linesep.join([s for s in inpdir.splitlines() if s])
    with open(inpdir, 'r') as f:
        line = f.readline()
        numlist = line.strip().split()
        npt_v = int(numlist[0])
        nel_v = int(numlist[1])
        nnode_v = int(numlist[2])

        xv = np.zeros(npt_v)
        yv = np.zeros(npt_v)
        zv = np.zeros(npt_v)
        ele = np.zeros((nel_v, nnode_v), dtype=int)

        count_line = 0
        for line in f.readlines():
            #print(line)
            if(count_line > 0 and count_line < npt_v+1):    #second line is \n
                coordlist = line.strip().split()
                ipt = count_line - 1
                xv[ipt] = float(coordlist[0])
                yv[ipt] = float(coordlist[1])
                zv[ipt] = float(coordlist[2])
            if(count_line >= npt_v+1):
                indexlist = line.strip().split()
                iel = count_line - npt_v - 1
                for j in range(nnode_v):
                    ele[iel, j] = int(indexlist[j])
            count_line += 1
    f.close()
    vol = Parameters(npt_v=npt_v, nel_v=nel_v, nnode_v=nnode_v,
        xv=xv, yv=yv, zv=zv, ele=ele)
    return vol


#@jit(nopython=True)
def build_loading_surface(surf, vol):

    corres_v = np.zeros(vol.npt_v, dtype=int)
    corres_s = np.zeros(surf.npt_s, dtype=int)
    iload = np.zeros(surf.npt_s, dtype=np.int32)
    eps = 5.0e-5

    k = 0
    for ipts in range(surf.npt_s):
        mindis = 1.0e10
        for iptv in range(vol.npt_v):
            dis = math.sqrt((surf.xs[ipts] - vol.xv[iptv])**2 + (surf.ys[ipts] - vol.yv[iptv])**2 + (surf.zs[ipts] - vol.zv[iptv])**2)
            if(dis < mindis):
                mindis = dis
                ii = iptv
        if(mindis < eps):
            corres_s[ipts] = ii
            corres_v[ii] = ipts
            flag = True
            # exclude based on element boundary flag
            exclude_stat = 0
            for iels in range(surf.nel_s):
                for inode in range(surf.nnode_s):
                    if(surf.eles[iels,inode] - 1 == ipts):
                        exclude_stat = surf.flag_exclude[iels]
                        break
                if(exclude_stat==1):
                    flag = False
                    break
            if(flag):
                iload[k] = ii + 1
                k += 1
        else:
            print('fail to find the correspondence for point', ipts)
            print('min distance', mindis)
    nload = k
    soliddir = 'solid/loading_surface.dat'
    with open(soliddir, 'w') as f:
        for ipts in range(nload):
            f.write('%6d \n' % iload[ipts])
    f.close()

    return nload, iload, corres_v, corres_s


class Parameters(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def initialization(max_epochs, batch_size, num_batch, learning_rate, nmode, InputNMode, nsec, zmin, zmax, nz, midline, Psub, rho, alpha, beta, nInterp, z0Interp, z1Interp, InputDir, data_index_start, data_index_end, data_step, num_shape_data, flag_right, i_ymin, i_ymax):
    # store all of the parameters (flow and model hyper-parameters) in "parameters" structure
    parameters = Parameters(max_epochs=max_epochs, batch_size = batch_size, num_batch = num_batch, learning_rate = learning_rate, nmode = nmode, InputNMode = InputNMode, 
                            nsec = nsec, nz = nz, zmax = zmax, zmin = zmin, midline = midline, Psub = Psub, rho = rho, alpha=alpha, beta=beta,
                            nInterp = nInterp, z0Interp = z0Interp, z1Interp = z1Interp,
                            InputDir = InputDir,
                            data_index_start=data_index_start, data_index_end=data_index_end, data_step=data_step,
                            num_shape_data=num_shape_data, flag_right=flag_right, i_ymin=i_ymin, i_ymax=i_ymax)
    return parameters


def eigen(InputDir, npt_v, nmode):
    inpdir = InputDir+'/'+'eigen.dat'
    inpdir = os.linesep.join([s for s in inpdir.splitlines() if s])

    frequency = torch.zeros(nmode)
    eigen_vx = torch.zeros((npt_v, nmode))
    eigen_vy = torch.zeros((npt_v, nmode))
    eigen_vz = torch.zeros((npt_v, nmode))
    C1 = torch.zeros(nmode)
    C2 = torch.zeros(nmode)

    with open(inpdir, 'r') as f:
        
        for imode in range(nmode):
            line = f.readline()
            line = f.readline()
            freq = line.strip().split()
            
            frequency[imode] = float(freq[2])
            C1[imode] = float(freq[3])
            C2[imode] = float(freq[4])
            
            for ipt  in range(npt_v):
                line = f.readline()
                disp = line.strip().split()
                eigen_vx[ipt, imode] = float(disp[2])
                eigen_vy[ipt, imode] = float(disp[3])
                eigen_vz[ipt, imode] = float(disp[4])
    
    return frequency, eigen_vx, eigen_vy, eigen_vz, C1, C2

#@jit(nopython=True)
def read_measurements(InputDir):
    inpdir = InputDir+'/measurements.dat'
    inpdir = os.linesep.join([s for s in inpdir.splitlines() if s])
    print('read measurements from', inpdir)
    with open(inpdir, 'r') as f:
        line = f.readline()
        numlist = line.strip().split()
        nt = int(numlist[0])
        npt = int(numlist[1])
        print('number of time snapshot', nt)
        print('number of points in a snapshot', npt)
        ind = torch.zeros(nt,dtype=torch.int64)
        tm = torch.zeros(nt)
        xm = torch.zeros((nt, npt))
        zm = torch.zeros((nt, npt))
        
        it = 0
        for it in range(nt):
            ip = 0
            line = f.readline()
            txt_part = line.strip().split(',')
            ind[it] = int(txt_part[0].split('=')[1])
            tm[it] = float(txt_part[1].split('=')[1])
            for ip in range(npt):
                line = f.readline()
                coordlist = line.strip().split()
                xm[it, ip] = float(coordlist[1]) 
                zm[it, ip] = float(coordlist[0])
                ip = ip + 1
            it = it + 1

    f.close()
 
    return ind-1, tm, xm, zm

def read_corresponding_table(InputDir):
    inpdir = InputDir + '/' + 's_v_mapping.dat'
    with open(inpdir, 'r') as f:
        lines = f.readlines()

    corres_s = torch.zeros(len(lines))
    for i in range(len(lines)):
        corres_s[i] = int(lines[i])

    corres_s = corres_s.to(dtype=torch.int64)

    return corres_s
	
