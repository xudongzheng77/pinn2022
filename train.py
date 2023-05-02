import os
import sys
import matplotlib.pyplot as plt
import torch
from torch.utils import data

from flow_solver import *
from update_surface_shape import *

cwd = os.getcwd()

def build_dt(n,device):
    re = torch.zeros([n,n],device=device)
    re[0,:3] = torch.tensor([-1.5,2,-0.5],device=device)
    lre = -0.5*torch.eye(n-2,device=device)
    hre = 0.5*torch.eye(n-2,device=device)
    re[1:n-1,:n-2] = lre
    re[1:n-1,2:] += hre
    re[-1,-3:] = torch.tensor([0.5,-2,1.5],device=device)
    return re

def build_dtt(n,device):
    re = torch.zeros([n,n],device=device)
    re[0,:4] = torch.tensor([2,-5,4,-1],device=device)
    eye = torch.eye(n-2,device=device)
    re[1:n-1,:n-2] += eye
    re[1:n-1,1:n-1] += -2*eye
    re[1:n-1,2:] += eye
    re[-1,-4:] = torch.tensor([-1,4,-5,2],device=device)
    return re

def train(device, model, data_loader, info, surf, C1,C2, parameters):
    
    Mt = build_dt(info.tm.size(0),device)
    Mtt = build_dtt(info.tm.size(0),device)
    dt = data_loader.dataset.dt
    dt2 = dt*dt
    
    weight1 = 1e4
    weight2 = 1e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=2000,cooldown=2000,verbose=True,min_lr=5e-5)
    criteria = torch.nn.MSELoss()
    losses = []
    best_loss = 1e8

    def write_information(epoch, tmodel, loss, loss_shape, loss_PINN, dataset,info,lr, CPRED_all, CTPRED_all, CTTPRED_all, LHS_all, RHS_all, M2D_all, FRT_all):
        if epoch%1 == 0:
            print('epoch: {0:7d}\tloss: {1:10.3e}\tdloss: {2:10.3e}\tcloss: {3:10.3e}'\
                .format(epoch, loss, loss_shape, loss_PINN))
            sys.stdout.flush()

        c_file = open('outputs/c_history.dat', 'a')
        ct_file = open('outputs/ct_history.dat', 'a')
        ctt_file = open('outputs/ctt_history.dat', 'a')
        lhs_file = open('outputs/lhs_history.dat', 'a') ### left-hand-side	
        rhs_file = open('outputs/rhs_history.dat', 'a')	### right-hand-side	
        M2D_file = open('outputs/M2D_history.dat', 'a')	### 2D shape
        FRT_file = open('outputs/FRT_history.dat', 'a')	### flow rate 

        debug = CPRED_all
        np.savetxt(c_file, debug.cpu().detach().numpy())
        debug = torch.tensor([], device=device)

        debug = CTPRED_all
        np.savetxt(ct_file, debug.cpu().detach().numpy())
        debug = torch.tensor([], device=device)

        debug = CTTPRED_all
        np.savetxt(ctt_file, debug.cpu().detach().numpy())
        debug = torch.tensor([], device=device)
						
        debug = LHS_all
        np.savetxt(lhs_file, debug.cpu().detach().numpy())
        debug = torch.tensor([], device=device)
		
        debug = RHS_all 
        np.savetxt(rhs_file, debug.cpu().detach().numpy())
        debug = torch.tensor([], device=device)

        debug = M2D_all 
        np.savetxt(M2D_file, debug.cpu().detach().numpy())
        debug = torch.tensor([], device=device)

        debug = FRT_all 
        np.savetxt(FRT_file, debug.cpu().detach().numpy())
        debug = torch.tensor([], device=device)
		
        c_file.close()
        ct_file.close()
        ctt_file.close()
        lhs_file.close()
        rhs_file.close()
        M2D_file.close()
        FRT_file.close()
		
        tmodel.eval()
        normdc= tmodel(dataset.normdx).squeeze()
        pred_c = normdc*C2+C1
        crep = c.unsqueeze(1).repeat([1, parameters.nsec,1]).unsqueeze(-1)
        info = get_projection_shape(device, crep, info, surf, parameters, parameters.batch_size)

                          

    for epoch in range(parameters.max_epochs):
        avgloss = 0
        returnpinnloss = 0
        returnshapeloss = 0
        counter = 0
		
        CPRED_all = torch.tensor([], device=device)
        CTPRED_all = torch.tensor([], device=device)
        CTTPRED_all = torch.tensor([], device=device)
        LHS_all = torch.tensor([], device=device)
        RHS_all = torch.tensor([], device=device)
        M2D_all = torch.tensor([], device=device)
        FRT_all = torch.tensor([], device=device)
        for normdx,batch_t,label_x in data_loader:
            batch_t.requires_grad = True
            normdx.requires_grad = True	
            optimizer.zero_grad()
            
            c = model(normdx).squeeze()
            c = c*C2+C1
            crep = c.unsqueeze(1).repeat([1, parameters.nsec,1]).unsqueeze(-1)
            info = get_projection_shape(device, crep, info, surf, parameters, parameters.batch_size)
            pred_2D = info.xInterp
            loss_shape = criteria(pred_2D, label_x[0])
            # Calculate PINN loss
            F, Q = flow_solver(device, info, parameters)
            c_t = torch.matmul(Mt,c)/dt
            c_tt = torch.matmul(Mtt,c)/dt2
            
            LHS = c_tt + (info.alpha + info.beta*torch.pow(info.omega, 2)) * c_t + torch.pow(info.omega, 2) * c
            resP = (LHS - F)

            CPRED_all = torch.cat((CPRED_all, torch.cat((batch_t.reshape(batch_t.size(1),1), c), 1)), 0)				
            CTPRED_all = torch.cat((CTPRED_all, torch.cat((batch_t.reshape(batch_t.size(1),1), c_t), 1)), 0)				
            CTTPRED_all = torch.cat((CTTPRED_all, torch.cat((batch_t.reshape(batch_t.size(1),1), c_tt), 1)), 0)				
            LHS_all = torch.cat((LHS_all, torch.cat((batch_t.reshape(batch_t.size(1),1), LHS), 1)), 0)
            RHS_all = torch.cat((RHS_all, torch.cat((batch_t.reshape(batch_t.size(1),1), F), 1)), 0)
            M2D_all = torch.cat((M2D_all, torch.cat((pred_2D, label_x[0]), 1)), 0)
            FRT_all = torch.cat((FRT_all, Q), 0)

            loss_PINN = criteria(resP, torch.zeros_like(resP))

            loss =  weight1*loss_shape + loss_PINN*weight2
            loss.backward()
            optimizer.step()
            
            avgloss += loss.item()
            returnpinnloss += loss_PINN.item()
            returnshapeloss += loss_shape.item()
            counter += 1

        avgloss,returnshapeloss, returnpinnloss = avgloss/counter,returnshapeloss/counter, returnpinnloss/counter#
        

        losses.append(avgloss)
        
        write_information(epoch, model, avgloss, returnshapeloss, 
            returnpinnloss, data_loader.dataset,info,optimizer.param_groups[0]['lr'], 
            CPRED_all, CTPRED_all, CTTPRED_all, LHS_all, RHS_all, M2D_all, FRT_all)
        scheduler.step(avgloss)
        model.train()

        if avgloss < best_loss:
            torch.save(model, cwd + "/best_model.pth")
            best_loss = avgloss
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avgloss,
        'scheduler_state_dict': scheduler.state_dict(),
    },'last_model.pth')
    

		




	


