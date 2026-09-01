import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from .surface import Sphere, OBJECT, IMAGE, Asphere, Qcon, Qbfs
from .utils import fit_get_mat_id, glass_catalog, glass_catalog_params, plastic_catalog, plastic_catalog_params, nv_to_g1_g2, limit_var, generate_normalized_numbers

class Delano(nn.Module):
    
    def __init__(self, structure, sys_num, target_fov, target_effl, target_fno, target_totr, target_bfl, stop_pos, mat_type, dist_min=None):
        super(Delano, self).__init__()
        
        self.structure = structure.split('|')
        self.num_points = len(self.structure) # number of elements
        self.sys_num = sys_num
        
        # target attributes
        self.target_fov = torch.tensor(target_fov)
        self.scale = target_effl
        self.target_effl = torch.tensor(1.)
        
        self.target_fno = torch.tensor(target_fno)
        self.target_enpd = self.target_effl / self.target_fno
        
        self.target_totr = torch.tensor(target_totr) / self.scale
        self.target_bfl = torch.tensor(target_bfl) / self.scale
        
        # opt attributes
        self.stop_pos = stop_pos # after the ith element
        self.Q = self.linv()
        self.img_h = torch.tan(torch.deg2rad(self.target_fov)) * self.target_effl
        self.y_ybar_init()
        self.generate()
        
        mat_type = mat_type.split('|')
        c, mu = [], []
        for i in mat_type:
            if i == 'K':
                mu.append(1.52 ** -1)
                c.append(64.17 ** -1)
            elif i == 'F':
                mu.append(1.65 ** -1)
                c.append(33.85 ** -1)
            else:
                mu.append((1.52 ** -1 + 1.65 ** -1) / 2)
                c.append((64.17 ** -1 + 33.85 ** -1) / 2)
        self.c = torch.tensor(c).repeat(self.sys_num, 1)
        self.mu = torch.tensor(mu).repeat(self.sys_num, 1)
        
        # phy attributes
        if dist_min == None:
            self.dist_min = (self.target_totr - self.target_bfl) / (self.num_points - 1) * 0.5
        else:
            self.dist_min = dist_min / self.scale
        
        # optimization
        self.tol_iters = 1000
        self.ins_tol_iters = 20
        self.iters = 0

    def linv(self):
        """
        Calculate Lagrange (or optical) invariant of system.
        Q = n y ubar - n ybar u
        """
        y = self.target_enpd / 2
        u = torch.tan(torch.deg2rad(self.target_fov))
        n = 1.0
        return n*y*u
    
    def y_ybar_init(self):
        """
        Initialize y and ybar
        """
        self._y = torch.sort(torch.rand(self.sys_num, self.num_points - 1), dim=-1)[0].flip(-1) * self.target_enpd / 2 # [sys, elems]
        self._y = nn.Parameter(self._y)
        self.y = torch.cat([self.target_enpd[..., None, None].repeat(self.sys_num, 1) / 2, self._y, torch.tensor(0.)[..., None, None].repeat(self.sys_num, 1)], dim=-1)
        
        if self.stop_pos > 0:
            self._ybar = torch.sort(torch.cat([
                -torch.rand(self.sys_num, self.stop_pos), torch.rand(self.sys_num, self.num_points - self.stop_pos)
            ], dim=-1), dim=-1)[0] * self.img_h # [sys, elems]
        elif self.stop_pos == 0:
            self._ybar = torch.sort(torch.rand(self.sys_num, self.num_points), dim=-1)[0] * self.img_h
        self._ybar = nn.Parameter(self._ybar)
        self.ybar = torch.cat([self._ybar, self.img_h[..., None, None].repeat(self.sys_num, 1)], dim=-1)
    
    def update_y_ybar(self):
        self.y = torch.cat([self.target_enpd[..., None, None].repeat(self.sys_num, 1) / 2, self._y, torch.tensor(0.)[..., None, None].repeat(self.sys_num, 1)], dim=-1)
        self.ybar = torch.cat([self._ybar, self.img_h[..., None, None].repeat(self.sys_num, 1)], dim=-1)
        
    def generate(self):
        """
        Generate effective focal lengths and distances of initial optical elements.
        """
        # Calculate the distances among elements
        elem_dist = (self.y[:, :-1] * self.ybar[:, 1:] - self.ybar[:, :-1] * self.y[:, 1:]) / self.Q
        # Calculate the effective focal lengths of each elements
        elem_effl = torch.zeros([self.sys_num, self.num_points])
        elem_effl[:, 1:] = (self.Q * elem_dist[:, :-1] * elem_dist[:, 1:]) / (
            (self.y[:, 1:-1] - self.y[:, :-2]) * (self.ybar[:, 2:] - self.ybar[:, 1:-1]) - 
            (self.y[:, 2:] - self.y[:, 1:-1]) * (self.ybar[:, 1:-1] - self.ybar[:, :-2]))
        elem_effl[:, 0] = self.y[:, 0] * elem_dist[:, 0] / (self.y[:, 0] - self.y[:, 1])
        # Calculate the radius of each elements
        elem_radius = torch.abs(self.y) + torch.abs(self.ybar)
        
        self.elem_effl = elem_effl 
        self.elem_dist = elem_dist
        self.elem_radius = elem_radius
        self.elem_fno = elem_effl.abs() / (elem_radius[:, 0:-1] * 2)
        
    def calc_seidel(self, opt_y_ybar=True):
        u, _ = self.calc_u_ubar()
        u = u[:, :-1] if opt_y_ybar else u[:, :-1].detach()
        y = self.y[:, :-1] if opt_y_ybar else self.y[:, :-1].detach()
        ybar = self.ybar[:, :-1] if opt_y_ybar else self.ybar[:, :-1].detach()
        phi = self.elem_effl ** (-1) if opt_y_ybar else self.elem_effl.detach() ** (-1)
        
        SIV = self.Q ** 2 * self.mu * phi
        CI = y ** 2 * phi * self.c
        CII = y * ybar * phi * self.c
        return SIV, CI, CII
    
    def calc_u_ubar(self):
        """
        Calculate u and ubar (represent in tan).
        """
        u = torch.zeros(self.sys_num, self.num_points + 1)
        ubar = torch.zeros(self.sys_num, self.num_points + 1)
        u[:, 0] = torch.tan(torch.deg2rad(torch.tensor(0.)))
        ubar[:, 0] = torch.tan(torch.deg2rad(self.target_fov))
        
        for i in range(self.num_points):
            u[:, i+1] = u[:, i] - self.y[:, i] / self.elem_effl[:, i]
            ubar[:, i+1] = ubar[:, i] - self.ybar[:, i] / self.elem_effl[:, i]
        return u, ubar
    
    def print_info(self, sys_id=0):
        """
        Print y, ybar, effl and dist of each elements.
        """
        print('Delano Diagram')
        print('Units         :   Millimeters')
        print('Field-of-View :   {:f} degree'.format(self.target_fov))
        print('Total Length  :   {:f} mm'.format(self.elem_dist[sys_id].sum()))
        print('Effective Focal Length  :   {:f} mm'.format(self.effl()[sys_id]))
        print('Entrance Pupil Diameter :   {:f} mm'.format(self.target_enpd))
        
        ####################################### Systems #######################################
        itv = 3 # interval between rows        
        print('Surf' + ' ' * 4 + 'Y-Bar' +
                       ' ' * itv + ' ' * 12 + 'Y' +
                       ' ' * itv + ' ' * 16 + 'EFFL' +
                       ' ' * itv + ' ' * 13 + 'Distance' +
                       ' ' * itv + ' ' * 9 + 'F')
        for idx in range(self.num_points + 1):
            msg = '{}'.format(idx+1)
            msg += ' ' * itv * 2 + '{:>17.10E}'.format(self.ybar[sys_id, idx].item()) + ' ' * itv + '{:>17.10E}'.format(self.y[sys_id, idx].item())
            if idx != self.num_points:
                msg += (' ' * itv + '{:>17.10E}'.format(self.elem_effl[sys_id, idx].item()) +
                        ' ' * itv + '{:>17.10E}'.format(self.elem_dist[sys_id, idx].item()) +
                        ' ' * itv + '{:>17.10E}'.format((self.elem_fno[sys_id, idx]).item()))
            print(msg)
        
        ####################################### Seidel #######################################
        print('Seidel Coefficients')
        SIV, CI, CII = self.calc_seidel(opt_y_ybar=False)
        itv = 3 # interval between rows
        print('Surf' + ' ' * 4 + 'SIV' +
                       ' ' * itv + ' ' * 14 + 'CI' +
                       ' ' * itv + ' ' * 15 + 'CII')
        for idx in range(self.num_points):
            msg = '{}'.format(idx+1) + ' ' * itv
            msg += (' ' * itv + '{:>17.10E}'.format(SIV[sys_id, idx].item()) +
                    ' ' * itv + '{:>17.10E}'.format(CI[sys_id, idx].item()) +
                    ' ' * itv + '{:>17.10E}'.format(CII[sys_id, idx].item()))
            print(msg)
        
        ####################################### Others #######################################
        print('n, v')
        itv = 3
        print('Surf' + ' ' * 4 + 'n' +
                       ' ' * itv + ' ' * 16 + 'v')
        for idx in range(self.num_points):
            msg = '{}'.format(idx+1) + ' ' * itv
            msg += (' ' * itv + '{:>17.10E}'.format((self.mu[sys_id, idx] ** -1).item()) + ' ' * itv + '{:>17.10E}'.format((self.c[sys_id, idx] ** -1).item()))
            print(msg)
            
    def plot_y_ybar(self, sys_id=0):
        lim = max(self.y[sys_id, 0], self.ybar[sys_id, -1]).cpu().detach().numpy() * 1.25
        plt.plot(self.ybar[sys_id].cpu().detach().numpy(), self.y[sys_id].cpu().detach().numpy(), color='k')
        plt.scatter(self.ybar[sys_id].cpu().detach().numpy(), self.y[sys_id].cpu().detach().numpy(), marker='*', color='k')
        plt.axis('square')
        plt.xlim([-lim, lim])
        plt.ylim([-lim, lim])
        plt.xlabel('ybar/mm', fontsize=14)
        plt.ylabel('y/mm', fontsize=14)
        plt.grid()
        plt.tight_layout()
        
    def effl(self):
        """
        Calculate the effective focal length of the system.
        """
        OF_ybar = self.ybar[:, -1]
        OF_y = self.y[:, -1]
        
        k_PM = (self.y[:, -2] - self.y[:, -1]) / (self.ybar[:, -2] - self.ybar[:, -1])
        OP_ybar = self.y[:, 0] / k_PM + self.ybar[:, -1]
        OP_y = self.y[:, 0]
        
        effl = ((OF_ybar * OP_y) - (OF_y * OP_ybar)) / self.Q
        return effl
    
    def calc_stop(self):
        """
        Seems the stop surface is not so important in Delano diagram.
        """
        if self.stop_pos == 0:
            y_stop = self.y[:, self.stop_pos]
        else:
            y_stop = (self.y[:, self.stop_pos-1] * self.ybar[:, self.stop_pos] - self.y[:, self.stop_pos] * self.ybar[:, self.stop_pos-1]) / (self.ybar[:, self.stop_pos] - self.ybar[:, self.stop_pos-1])
            
        ybar_stop = torch.tensor(0.).repeat(self.sys_num)
        # return the distance between the stop and the next element.
        dist_sp = -(self.y[:, self.stop_pos] * ybar_stop - self.ybar[:, self.stop_pos] * y_stop) / self.Q
        # return the radius of the stop
        radius_sp = torch.abs(y_stop) + torch.abs(ybar_stop)
        
        return dist_sp, radius_sp # int, scalar, scalar

    def propagate(self, y0, u0):
        """
        Use paraxial propagation.
        This propagate() is used for recording u and y of different elements.
        """
        u = u0.unsqueeze(0)
        y = y0.unsqueeze(0)
        
        for i in range(self.num_points):
            u0 = u0 - y0 / self.elem_effl[:, i][:, None]
            y0 = y0 + self.elem_dist[:, i][:, None] * u0

            u = torch.cat((u, u0.unsqueeze(0)), dim=0)
            y = torch.cat((y, y0.unsqueeze(0)), dim=0)
            
        return y, u
    
    def plot_set_up_with_trace(self, sys_id=0, M=7):
        """
        Plot elements in 2D.
        """
        h = int(self.elem_radius[sys_id, :].max().cpu().detach().numpy() * 2 + 1) * 5
        w = int(self.elem_dist[sys_id, :].sum().cpu().detach().numpy() + 1) * 5
        fig, ax = plt.subplots(figsize=(w, h))
        
        # calculate radius of each elements
        r = self.elem_radius[sys_id, :].cpu().detach().numpy() 
        z = [0]
        
        ######################################## step 0: draw the 2D setup of the system ########################################
        w = r.max() / 40 # arrow size
        
        for i in range(self.num_points):
            # draw the element
            ax.plot([z[-1], z[-1]], [-r[i], r[i]], color='k')
            
            if self.elem_effl[sys_id, i] > 0:
                ax.plot([z[-1], z[-1]-w], [-r[i], -r[i]+w], color='k')
                ax.plot([z[-1], z[-1]+w], [-r[i], -r[i]+w], color='k')
                ax.plot([z[-1], z[-1]-w], [r[i], r[i]-w], color='k')
                ax.plot([z[-1], z[-1]+w], [r[i], r[i]-w], color='k')
            else:
                ax.plot([z[-1], z[-1]-w], [-r[i], -r[i]-w], color='k')
                ax.plot([z[-1], z[-1]+w], [-r[i], -r[i]-w], color='k')
                ax.plot([z[-1], z[-1]-w], [r[i], r[i]+w], color='k')
                ax.plot([z[-1], z[-1]+w], [r[i], r[i]+w], color='k')
        
            z.append(z[-1] + self.elem_dist[sys_id, i].cpu().detach().numpy())
            
        # draw image plane
        ax.plot([z[-1], z[-1]], [-r[-1], r[-1]], color='k')
        ax.set_aspect(1)
        
        # draw stop
        dist_sp, radius_sp = self.calc_stop()
        z_s = (self.elem_dist[sys_id, :self.stop_pos].sum() - dist_sp[sys_id]).cpu().detach().numpy()
        r_s = radius_sp[sys_id].cpu().detach().numpy()
        ax.plot([z_s-w, z_s+w], [r_s, r_s], color='r')
        ax.plot([z_s-w, z_s+w], [-r_s, -r_s], color='r')
        ax.plot([z_s, z_s], [r_s, r_s+w], color='r')
        ax.plot([z_s, z_s], [-r_s, -r_s-w], color='r')
        
        ######################################## step 1: calculate enpp ########################################
        ybar, y = 0., self.target_enpd / 2
        enpp = (self.y[:, 0] * ybar - self.ybar[:, 0] * y) / self.Q
        
        ######################################## step 2: draw marginal field ########################################
        _y_m = (torch.linspace(-0.5, 0.5, M) * self.target_enpd)[None, :].repeat(self.sys_num, 1) # [sys, M]
        _u_m = torch.ones_like(_y_m) * torch.tan(torch.deg2rad(torch.tensor(0.))) # [sys, M]
        
        if self.stop_pos == 0:
            ax.plot([-dist_sp[sys_id].cpu().detach().numpy(), 0], [_y_m[sys_id].cpu().detach().numpy(), _y_m[sys_id].cpu().detach().numpy()], color='b')
            
        y_m, _ = self.propagate(_y_m, _u_m)
        ax.plot(z, y_m[:, sys_id].cpu().detach().numpy(), color='b')
        
        ######################################## step 3: draw chief field ########################################
        _y_c = _y_m + ((0 - enpp) * torch.tan(torch.deg2rad(self.target_fov)))[:, None] # [sys, M]
        _u_c = torch.ones_like(_y_c) * torch.tan(torch.deg2rad(self.target_fov)) # [sys, M] 

        if self.stop_pos == 0:
            _y_s = _y_c + ((0 - dist_sp) * torch.tan(torch.deg2rad(self.target_fov)))[:, None] # [sys, M]
            ax.plot([-dist_sp[sys_id].cpu().detach().numpy(), 0], [_y_s[sys_id].cpu().detach().numpy(), _y_c[sys_id].cpu().detach().numpy()], color='c')

        y_c, _ = self.propagate(_y_c, _u_c)
        ax.plot(z, y_c[:, sys_id].cpu().detach().numpy(), color='c')
        plt.tight_layout()
        
    #====================================================================================================#
    #------------------------------------------- Optimization -------------------------------------------#
    #====================================================================================================#
    
    def merit_dist(self):
        """
        Consider minimum distances among optical elements.
        Consider the back focal length of the system.
        """
        loss_dist_min = torch.where(self.elem_dist[:, :-1] < self.dist_min, self.dist_min - self.elem_dist[:, :-1], 0).sum(dim=-1)
        loss_bfl = torch.where(self.elem_dist[:, -1] < self.target_bfl, self.target_bfl - self.elem_dist[:, -1], 0)
        loss = loss_bfl + loss_dist_min
        return loss # [sys]
    
    def merit_stop(self):
        """
        merit the position of the stop surface
        """
        if self.stop_pos == 0:
            loss = torch.where(self.ybar[:, 0] < 0., -self.ybar[:, 0], 0.)
            return loss
        else:
            loss_1 = torch.where(self.ybar[:, self.stop_pos-1] > 0., self.ybar[:, self.stop_pos-1], 0.)
            loss_2 = torch.where(self.ybar[:, self.stop_pos] < 0., -self.ybar[:, self.stop_pos], 0.)
            return loss_1 + loss_2 # [sys]
    
    def merit_totr(self):
        """
        merit the total length of the system
        """
        dist_sp, _ = self.calc_stop()
        if self.stop_pos == 0:
            elem_dist = torch.cat([dist_sp[:, None], self.elem_dist], dim=-1)
        else:
            elem_dist = self.elem_dist
            
        loss = torch.where(elem_dist.sum(dim=-1) > self.target_totr, elem_dist.sum(dim=-1) - self.target_totr, 0)
        return loss # [sys]
    
    def merit_f(self):
        """
        merit the f-number of each element
        """
        loss_fno = limit_var(self.elem_fno ** -1, 0., self.target_fno ** -1).sum(dim=-1)
        return loss_fno # [sys]
    
    def merit_angle(self):
        """
        merit the angle of each element
        """
        M = 3
        ######################################## step 1: calculate enpp ########################################
        ybar, y = 0., self.target_enpd / 2
        enpp = (self.y[:, 0] * ybar - self.ybar[:, 0] * y) / self.Q
        
        ######################################## step 2: calc marginal field ########################################
        _y_m = (torch.linspace(-0.5, 0.5, M) * self.target_enpd)[None, :].repeat(self.sys_num, 1) # [sys, M]
        _u_m = torch.ones_like(_y_m) * torch.tan(torch.deg2rad(torch.tensor(0.))) # [sys, M]    
        _, u_m = self.propagate(_y_m, _u_m) # [sys, num_points, M]
        
        ######################################## step 3: calc chief field ########################################
        _y_c = _y_m + ((0 - enpp) * torch.tan(torch.deg2rad(self.target_fov)))[:, None] # [sys, M]
        _u_c = torch.ones_like(_y_c) * torch.tan(torch.deg2rad(self.target_fov)) # [sys, M] 
        _, u_c = self.propagate(_y_c, _u_c) # [sys, num_points, M]
        
        loss = ((u_m[1:, :, :] - u_m[:-1, :, :]) ** 2).amax(dim=[0, -1]) + ((u_c[1:, :, :] - u_c[:-1, :, :]) ** 2).amax(dim=[0, -1])
        return loss # [sys]
    
    def merit_seidel(self):
        SIV, CI, CII = self.calc_seidel()
        loss = (SIV.sum(dim=-1).abs()+
                CI.sum(dim=-1).abs()+ 
                CII.sum(dim=-1).abs())
        return loss # [sys]
    
    def merit_max_radius(self):
        """
        merit the maximum radius of the system
        """
        loss = self.elem_radius.amax(dim=-1)
        return loss # [sys]
    
    def fitness(self):
        """
        Calculate the fitness of each system
        """
        loss_s = self.merit_seidel()
        # loss_a = self.merit_stop() + self.merit_dist() + self.merit_totr() + self.merit_f() + self.merit_max_radius()
        loss_a = self.merit_stop() + self.merit_dist() + self.merit_totr() + self.merit_f()
        loss = loss_s + loss_a
        return loss # [sys]
    
    def optimize_SA(self):
        T = 100
        T_min = 1
        step = 0.001
        alpha = 0.99
        iters = 100
        k = 1
        
        with torch.no_grad():
            y_min = self._y.data
            ybar_min = self._ybar.data
            loss_min = self.fitness()
            
            pbar = tqdm()
            while T >= T_min:
                for i in range(iters):
                    loss = self.fitness()
                    
                    _y = self._y.data
                    _ybar = self._ybar.data
                    
                    self._y.data = self._y.data * (1 + (torch.rand_like(self._y.data) - 0.5) * 2 * step * T)
                    self._ybar.data = self._ybar.data * (1 + (torch.rand_like(self._ybar.data) - 0.5) * 2 * step * T)
                    self.update_y_ybar()
                    self.generate()
                    loss_new = self.fitness()
                    
                    y_min = torch.where((loss_new < loss_min)[:, None], self._y.data, y_min)
                    ybar_min = torch.where((loss_new < loss_min)[:, None], self._ybar.data, ybar_min)
                    loss_min = torch.where(loss_new < loss_min, loss_new, loss_min)
                    
                    valid = loss_new < loss
                    _y[valid] = self._y.data[valid]
                    _ybar[valid] = self._ybar.data[valid]
                    
                    p = torch.exp(-(loss_new - loss) / (k * T))[~valid]
                    r = torch.rand_like(p)
                    valid_bad = r < p
                    _y[~valid][valid_bad] = self._y.data[~valid][valid_bad]
                    _ybar[~valid][valid_bad] = self._ybar.data[~valid][valid_bad]
                    
                    self._y.data = _y
                    self._ybar.data = _ybar
                    self.update_y_ybar()
                    self.generate()
                
                pbar.set_description_str(f'T: {T}, min loss: {loss_min.min().item()}, max loss: {loss_min.max().item()}')
                T = T * alpha
                self._y.data = y_min
                self._ybar.data = ybar_min
                self.update_y_ybar()
                self.generate()
        
        self._y.data = y_min
        self._ybar.data = ybar_min
        self.update_y_ybar()
        self.generate()
    
    def optimize(self, lr, save_dir):
        """
        Optimize self._y and self._ybar
        """
        p = tqdm()
        # optimizer initialization
        optimizer = torch.optim.Adam([self._y, self._ybar], lr=lr)
        # ======================================================== #
        # Merit y, ybar, C, MU
        # ======================================================== #
        loss_min = torch.ones(self.sys_num) * 1e10
        _y_m, _ybar_m = torch.zeros_like(self._y), torch.zeros_like(self._ybar)
        iters_tol = 1
        while iters_tol < self.tol_iters:
            optimizer.zero_grad()
            self.iters += 1
            
            loss = self.fitness()
            p.set_description(f'min loss: {loss.min().item():.6f}, max loss: {loss.max().item():.6f}')
            
            valid = (loss_min > loss)
            loss_min[valid] = loss[valid]
            _y_m[valid], _ybar_m[valid] = self._y.detach().clone()[valid], self._ybar.detach().clone()[valid]
            
            iters_tol = 1 if loss.min() <= loss_min.min() else iters_tol + 1
            loss.sum().backward()
            optimizer.step()
            self.update_y_ybar()
            self.generate()
            
        # ======================================================== #
        # Judge valid system and choose the best id
        # ======================================================== #
        self._y.data, self._ybar.data = _y_m, _ybar_m
        self.update_y_ybar()
        self.generate()
        
        with torch.no_grad():
            loss = self.fitness()
        
        _, idx = torch.topk(loss, 1, largest=False)
        self.idx = int(idx)
        print(f'idx: {self.idx}')
        self.print_info(self.idx)
        
        if save_dir is not None:
            self.plot_set_up_with_trace(self.idx, 3)
            plt.savefig(f'{save_dir}/opt_sys_iter_{self.iters+1}.svg', bbox_inches='tight')
            plt.close()
            self.plot_y_ybar(self.idx)
            plt.savefig(f'{save_dir}/opt_y_ybar_iter_{self.iters+1}.svg', bbox_inches='tight')
            plt.close()
    
    # #====================================================================================================#
    # #--------------------------------------------- Instance ---------------------------------------------#
    # #====================================================================================================#
    def lens_calc_wo_mat(self, elem_id, sys_num, stype, mat_type, mat_cata):
        """
        Calculate lens parameters
        """
        pbar = tqdm()
        vd_threshold = 50.
        
        effl = self.elem_effl[self.idx, elem_id].detach()
        radius = self.elem_radius[self.idx, elem_id].detach()
        group_type = self.structure[elem_id]

        sub_phi = torch.stack([generate_normalized_numbers(len(group_type), -2., 2.) for _ in range(sys_num)], dim=0).requires_grad_()
        sub_Q = ((torch.rand_like(sub_phi) - 0.5) * 10 - 1).requires_grad_()

        sgl_n = torch.zeros_like(sub_phi) # [sys, elem]
        sgl_c = torch.zeros_like(sub_phi) # [sys, elem]

        dbl_phi = (torch.rand(sys_num, len(group_type)) * 3 - 1).requires_grad_() # [sys, elem]
        dbl_n = torch.zeros(sys_num, len(group_type), 2) # [sys, elem, 2]
        dbl_c = torch.zeros(sys_num, len(group_type), 2) # [sys, elem, 2]
        
        with torch.no_grad():
            for i, lens_type in enumerate(group_type):
                if mat_cata[i] == 'G':
                    catalog = glass_catalog
                    catalog_params = glass_catalog_params
                elif mat_cata[i] == 'P':
                    catalog = plastic_catalog
                    catalog_params = plastic_catalog_params
                else:
                    raise ValueError(f"Unknown material catalog for {mat_cata[i]}")
                
                match lens_type:
                    case 'S':
                        for _i_ in range(sys_num):
                            idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            if mat_type[i] == 'K':
                                while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            elif mat_type[i] == 'F':
                                while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            elif mat_type[i] == 'R':
                                idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            else:
                                raise ValueError(f"Unknown material type for S: {mat_type[i]}")
                            sgl_n[_i_, i] = catalog[list(catalog)[idx]]['nd']
                            sgl_c[_i_, i] = catalog[list(catalog)[idx]]['vd'] ** -1
                        
                    case 'D':
                        randmat = torch.rand(sys_num)
                        for _i_ in range(sys_num):
                            idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            if mat_type[i] == 'M':
                                if randmat[_i_] < 0.5:
                                    while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                else:
                                    while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            elif mat_type[i] == 'R':
                                idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            else:
                                raise ValueError(f"Unknown material type for D: {mat_type[i]}")
                            dbl_n[_i_, i, 0] = catalog[list(catalog)[idx]]['nd']
                            dbl_c[_i_, i, 0] = catalog[list(catalog)[idx]]['vd'] ** -1
                        
                        for _i_ in range(sys_num):
                            idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            if mat_type[i] == 'M':
                                if randmat[_i_] > 0.5:
                                    while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                else:
                                    while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            elif mat_type[i] == 'R':
                                idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            else:
                                raise ValueError(f"Unknown material type for D: {mat_type[i]}")
                            dbl_n[_i_, i, 1] = catalog[list(catalog)[idx]]['nd']
                            dbl_c[_i_, i, 1] = catalog[list(catalog)[idx]]['vd'] ** -1
    
        optimizer = torch.optim.Adam([sub_phi, sub_Q, dbl_phi])
        
        _sub_phi = torch.zeros_like(sub_phi)
        _sub_Q = torch.zeros_like(sub_Q)
        _dbl_phi = torch.zeros_like(dbl_phi)
        
        loss_min = torch.ones(sys_num) * 1e10
        iters_tol = 0
        while iters_tol < self.ins_tol_iters:

            phis = sub_phi / sub_phi.sum(dim=-1, keepdim=True)
            sgl_roc = torch.zeros(sys_num, len(group_type), 2) # [sys, elem, 2]
            dbl_roc = torch.zeros(sys_num, len(group_type), 3) # [sys, elem, 3]

            for i, lens_type in enumerate(group_type):
                match lens_type:
                    case 'S':
                        phi = phis[:, i]
                        sgl_roc[:, i, 0] = effl ** -1 * phi * (sub_Q[:, i] + sgl_n[:, i] / (sgl_n[:, i] - 1))
                        sgl_roc[:, i, 1] = effl ** -1 * phi * (sub_Q[:, i] + 1)
                    case 'D':
                        phi = phis[:, i]
                        dbl_roc[:, i, 1] = effl ** -1 * phi * (sub_Q[:, i] + dbl_phi[:, i])
                        dbl_roc[:, i, 0] = effl ** -1 * phi * (sub_Q[:, i] + dbl_n[:, i, 0] * dbl_phi[:, i] / (dbl_n[:, i, 0] - 1))
                        dbl_roc[:, i, 2] = effl ** -1 * phi * (sub_Q[:, i] + (dbl_n[:, i, 1] * dbl_phi[:, i] - 1) / (dbl_n[:, i, 1] - 1))

            loss_sgl_roc = limit_var(sgl_roc, -0.333 / radius, 0.333 / radius).sum(dim=[-2, -1])
            loss_dbl_roc = limit_var(dbl_roc, -0.333 / radius, 0.333 / radius).sum(dim=[-2, -1])
            
            loss = loss_sgl_roc + loss_dbl_roc
            pbar.set_postfix_str(f'{elem_id+1}/{self.num_points}, min loss: {loss.min().item()}, max loss: {loss.max().item()}')
            
            valid = loss_min > loss
            loss_min[valid] = loss[valid]
            
            _sub_phi[valid] = sub_phi[valid]
            _sub_Q[valid] = sub_Q[valid]
            _dbl_phi[valid] = dbl_phi[valid]
            
            valid_quit = valid.any()
            iters_tol = 1 if valid_quit else iters_tol + 1
            
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
        
        _roc = []
        _mat = []
        _stype = []
        _mat_cata = []

        phis = _sub_phi / _sub_phi.sum(dim=-1, keepdim=True)
        for i, lens_type in enumerate(group_type):
            if mat_cata[i] == 'G':
                catalog = glass_catalog
            elif mat_cata[i] == 'P':
                catalog = plastic_catalog
            else:
                raise ValueError(f"Unknown material catalog for {mat_cata[i]}")
            
            match lens_type:
                case 'S':
                    sgl_g1, sgl_g2 = nv_to_g1_g2(sgl_n[:, i], sgl_c[:, i] ** -1, mat_cata[i])
                    sgl_param = torch.stack([sgl_g1, sgl_g2], dim=-1)
                    sgl_idx = fit_get_mat_id(sgl_param, mat_cata=mat_cata[i])
                    sgl_name = [list(catalog)[j] for j in sgl_idx]
                    n = torch.tensor([catalog[name]['nd'] for name in sgl_name])

                    phi = phis[:, i]
                    sgl_roc_1 = effl ** -1 * phi * (_sub_Q[:, i] + n / (n - 1))
                    sgl_roc_2 = effl ** -1 * phi * (_sub_Q[:, i] + 1)
                    
                    _roc.append(sgl_roc_1)
                    _roc.append(sgl_roc_2)
                    _mat.append(sgl_name)
                    _mat.append(['VACUUM'] * sys_num)
                    _stype.append(stype[i])
                    _stype.append(stype[i])
                    _mat_cata.append(mat_cata[i])
                    _mat_cata.append(None)

                case 'D':
                    dbl_a_g1, dbl_a_g2 = nv_to_g1_g2(dbl_n[:, i, 0], dbl_c[:, i, 0] ** -1, mat_cata[i])
                    dbl_b_g1, dbl_b_g2 = nv_to_g1_g2(dbl_n[:, i, 1], dbl_c[:, i, 1] ** -1, mat_cata[i])
                    dbl_a_param = torch.stack([dbl_a_g1, dbl_a_g2], dim=-1)
                    dbl_b_param = torch.stack([dbl_b_g1, dbl_b_g2], dim=-1)
                    dbl_idx_a = fit_get_mat_id(dbl_a_param, mat_cata=mat_cata[i])
                    dbl_idx_b = fit_get_mat_id(dbl_b_param, mat_cata=mat_cata[i])
                    dbl_name_a = [list(catalog)[j] for j in dbl_idx_a]
                    dbl_name_b = [list(catalog)[j] for j in dbl_idx_b]
                    na = torch.tensor([catalog[name]['nd'] for name in dbl_name_a])
                    nb = torch.tensor([catalog[name]['nd'] for name in dbl_name_b])
                            
                    phi = phis[:, i]
                    dbl_roc_2 = effl ** -1 * phi * (_sub_Q[:, i] + _dbl_phi[:, i])
                    dbl_roc_1 = effl ** -1 * phi * (_sub_Q[:, i] + na * _dbl_phi[:, i] / (na - 1))
                    dbl_roc_3 = effl ** -1 * phi * (_sub_Q[:, i] + (nb * _dbl_phi[:, i] - 1) / (nb - 1))
                    
                    _roc.append(dbl_roc_1)
                    _roc.append(dbl_roc_2)
                    _roc.append(dbl_roc_3)
                    _mat.append(dbl_name_a)
                    _mat.append(dbl_name_b)
                    _mat.append(['VACUUM'] * sys_num)
                    _stype.append(stype[i])
                    _stype.append(stype[i])
                    _stype.append(stype[i])
                    _mat_cata.append(mat_cata[i])
                    _mat_cata.append(mat_cata[i])
                    _mat_cata.append(None)
        
        sub_group = {'num': len(_mat), 'radius': radius, 'roc': _roc, 'material': _mat, 'stype': _stype, 'mat_cata': _mat_cata}
        return sub_group
    
    def lens_group(self, sys_num, stype, mat_type, mat_cata):
        """
        Calculate lens parameters
        """
        lens_group = []
        stype = stype.split('|')
        mat_type = mat_type.split('|')
        mat_cata = mat_cata.split('|')
        
        for i in range(self.num_points):
            sub_elem = self.lens_calc_wo_mat(i, sys_num, stype[i], mat_type[i], mat_cata[i]) # [return list]
            lens_group.append(sub_elem)
        return lens_group
    
    
    def lens_instance(self, sys_num, cfg_num, stype, mat_type, mat_cata):
        """
        Initialize the system according to the delano diagram.
        """
        lens_group = self.lens_group(sys_num, stype, mat_type, mat_cata)
        dist_sp, radius_sp = self.calc_stop()
        dist_sp, radius_sp = dist_sp[self.idx], radius_sp[self.idx]
        
        sys = nn.ModuleList([])
        sys.append(OBJECT(material='VACUUM', distance=[None] * cfg_num))
        
        if self.stop_pos == 0:
            common_params = {
                'radius': (self.scale * radius_sp.repeat(sys_num, cfg_num)).tolist(),
                'material': ["VACUUM"] * sys_num,
                'roc': [None] * sys_num,
                'thick': (self.scale * dist_sp.repeat(sys_num, cfg_num)).tolist(),
                'conic': [0.0] * sys_num,
            }
            sys.append(Sphere(**common_params))
        
        stop_pos = 0
        for i in range(self.num_points):
            surf_num = lens_group[i]['num']
            for j in range(surf_num):
                if j + 1 == surf_num:
                    if (i + 1) == self.stop_pos:
                        thick = self.scale * (self.elem_dist[self.idx, i] - dist_sp).repeat(sys_num, cfg_num)
                    else:
                        thick = self.scale * self.elem_dist[self.idx, i].repeat(sys_num, cfg_num)
                else:
                    thick = torch.zeros(sys_num, cfg_num)
                
                common_params = {
                    'radius': (self.scale * self.elem_radius[self.idx, i].repeat(sys_num, cfg_num)).tolist(),
                    'material': lens_group[i]['material'][j],
                    'roc': (self.scale * lens_group[i]['roc'][j] ** -1).tolist(),
                    'thick': thick.tolist(),
                    'conic': [0.0] * sys_num,
                    'mat_cata': lens_group[i]["mat_cata"][j],
                }
                match lens_group[i]['stype'][j]:
                    case 'S':
                        sys.append(Sphere(**common_params))
                    case 'A':
                        sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * 3))
                    case 'Q':
                        sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * 3, rnorm=(self.scale * self.elem_radius[self.idx, i].repeat(sys_num)).tolist()))
                    case 'q':
                        sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * 3, rnorm=(self.scale * self.elem_radius[self.idx, i].repeat(sys_num)).tolist()))
                
                if (i + 1) <= self.stop_pos:
                    stop_pos += 1
                
            if (i + 1) == self.stop_pos:
                common_params = {
                    'radius': (self.scale * radius_sp.repeat(sys_num, cfg_num)).tolist(),
                    'material': ["VACUUM"] * sys_num,
                    'roc': [None] * sys_num,
                    'thick': (self.scale * dist_sp.repeat(sys_num, cfg_num)).tolist(),
                    'conic': [0.0] * sys_num,
                }
                sys.append(Sphere(**common_params))
        
        sys.append(IMAGE(radius=(self.scale * self.elem_radius[self.idx, -1].repeat(sys_num, cfg_num)).tolist()))
        return sys, stop_pos+1