import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from .surface import Sphere, OBJECT, IMAGE, Asphere, Qcon, Qbfs
from .utils import glass_catalog, glass_catalog_params, plastic_catalog, plastic_catalog_params, nv_to_g1_g2, generate_normalized_numbers, fit_get_mat_id, limit_var

class Zoom(nn.Module):
    
    def __init__(self, group_structure:str, group_type:str, sys_num:int, target_fov:list, target_effl:list, target_fno:list,
                 target_totr, target_bfl, FF_min_dist, FM_min_dist, MM_min_dist, stop_pos, stop_fix:bool=True):
        super(Zoom, self).__init__()
        # group_structure: e.g. FMFMF
        # F - Fixed
        # M - Movable
        
        self.group_structure = group_structure
        self.group_type = group_type.split('|')
        self.group_num = len(group_structure)
        self.sys_num = sys_num
        
        self.target_fov = torch.tensor(target_fov) # [cfg]
        self.target_effl = torch.tensor(target_effl) # [cfg]
        self.scale = torch.min(self.target_effl).item() # scalar
        self.target_effl = self.target_effl / self.scale # [cfg]
        
        self.target_fno = torch.tensor(target_fno) # [cfg]
        self.target_enpd = self.target_effl / self.target_fno # [cfg]
        
        self.target_totr = torch.tensor(target_totr) / self.scale
        self.target_bfl = torch.tensor(target_bfl) / self.scale
        
        # hidden attributes
        self.FF_min_dist = FF_min_dist / self.scale
        self.FM_min_dist = FM_min_dist / self.scale
        self.MM_min_dist = MM_min_dist / self.scale
        
        self.stop_pos = stop_pos
        self.stop_fix = stop_fix
        
        # Other attributes
        self.Q = self.linv() # [cfg]
        self.cfg_num = len(self.Q)
        self.img_h = torch.tan(torch.deg2rad(self.target_fov)) * self.target_effl # [cfg]
        self.y_ybar_init()
        self.generate()
        
        self.fst_tol_iters = 500
        self.ins_tol_iters = 100
        self.iters = 0
        self.loss_SA = []
        self.loss_Ae = []
        self.loss_As = []
        
        self.init_k = False
        
    def linv(self):
        """
        Calculate Lagrange (or optical) invariant of system.
        Q = n y ubar - n ybar u
        """
        y = self.target_enpd / 2
        u = torch.tan(torch.deg2rad(self.target_fov))
        n = 1.0        
        return n*y*u # [cfg]
    
    def y_ybar_init(self):
        """
        Initialize y and ybar
        """
        if hasattr(self, '_y'): delattr(self, '_y')
        if hasattr(self, '_ybar'): delattr(self, '_ybar')
        
        self._y = torch.sort(torch.rand(self.sys_num, self.group_num - 1), dim=-1)[0].flip(-1)[:, None, :] * self.target_enpd[None, :, None] / 2 # [sys, cfg, subs]
        self._y = nn.Parameter(self._y)
        self.y = torch.cat([self.target_enpd[None, :, None].repeat(self.sys_num, 1, 1) / 2, self._y, torch.zeros_like(self.target_enpd)[None, :, None].repeat(self.sys_num, 1, 1)], dim=-1)
        
        if self.stop_pos > 0:
            self._ybar = torch.sort(torch.cat([
                -torch.rand(self.sys_num, self.stop_pos), torch.rand(self.sys_num, self.group_num - self.stop_pos - 1)
            ], dim=-1), dim=-1)[0][:, None, :] * self.img_h[None, :, None] # [sys, cfg, subs]
        elif self.stop_pos == 0:
            self._ybar = torch.sort(torch.rand(self.sys_num, self.group_num - 1), dim=-1)[0][:, None, :] * self.img_h[None, :, None] # [sys, cfg, subs]
        self._ybar = nn.Parameter(self._ybar)
        self.ybar = torch.cat([self._ybar[:, :, 0:self.stop_pos], torch.zeros_like(self.Q)[None, :, None].repeat(self.sys_num, 1, 1), self._ybar[:, :, self.stop_pos:], self.img_h[None, :, None].repeat(self.sys_num, 1, 1)], dim=-1)
        
    def update_y_ybar(self):
        self.y = torch.cat([self.target_enpd[None, :, None].repeat(self.sys_num, 1, 1) / 2, self._y, torch.zeros_like(self.target_enpd)[None, :, None].repeat(self.sys_num, 1, 1)], dim=-1)
        self.ybar = torch.cat([self._ybar[:, :, 0:self.stop_pos], torch.zeros_like(self.Q)[None, :, None].repeat(self.sys_num, 1, 1), self._ybar[:, :, self.stop_pos:], self.img_h[None, :, None].repeat(self.sys_num, 1, 1)], dim=-1)
        
    def generate(self):
        """
        Generate effective focal lengths and distances of initial optical elements.
        """
        if self.stop_fix:
            self.y[:, :, self.stop_pos] = self.y.mean(dim=1, keepdim=True)[:, :, self.stop_pos]
        # Calculate the distances among elements
        sub_dist = (self.y[:, :, :-1] * self.ybar[:, :, 1:] - self.ybar[:, :, :-1] * self.y[:, :, 1:]) / self.Q[None, :, None].repeat(self.sys_num, 1, 1) # [sys, cfg, subs]
        # Calculate the effective focal lengths of each elements
        sub_effl = torch.zeros([self.sys_num, self.cfg_num, self.group_num])
        sub_effl[:, :, 1:] = (self.Q[None, :, None].repeat(self.sys_num, 1, 1) * sub_dist[:, :, :-1] * sub_dist[:, :, 1:]) / (
            (self.y[:, :, 1:-1] - self.y[:, :, :-2]) * (self.ybar[:, :, 2:] - self.ybar[:, :, 1:-1]) - 
            (self.y[:, :, 2:] - self.y[:, :, 1:-1]) * (self.ybar[:, :, 1:-1] - self.ybar[:, :, :-2]))
        sub_effl[:, :, 0] = self.y[:, :, 0] * sub_dist[:, :, 0] / (self.y[:, :, 0] - self.y[:, :, 1])
        # Calculate the radius of each elements
        sub_radius = torch.abs(self.y) + torch.abs(self.ybar)
        
        self.sub_effl = sub_effl # [sys, cfg, subs]
        self.sub_dist = sub_dist # [sys, cfg, subs]
        self.sub_radius = sub_radius # [sys, cfg, subs]
        self.sub_fno = sub_effl.abs() / (sub_radius[:, :, 0:-1] * 2)
    
    def calc_u_ubar(self):
        """
        Calculate u and ubar (represent in tan).
        """
        u = torch.zeros([self.sys_num, self.cfg_num, self.group_num + 1])
        ubar = torch.zeros([self.sys_num, self.cfg_num, self.group_num + 1])
        u[:, :, 0] = torch.tan(torch.deg2rad(torch.zeros_like(self.target_fov)))[None, :].repeat(self.sys_num, 1)
        ubar[:, :, 0] = torch.tan(torch.deg2rad(self.target_fov))[None, :].repeat(self.sys_num, 1)
        
        for i in range(self.group_num):
            u[:, :, i+1] = u[:, :, i] - self.y[:, :, i] / self.sub_effl[:, :, i]
            ubar[:, :, i+1] = ubar[:, :, i] - self.ybar[:, :, i] / self.sub_effl[:, :, i]
        return u, ubar
    
    def print_info(self, sys_id=0):
        """
        Print y, ybar, effl and dist of each elements.
        """
        for cfg in range(self.cfg_num):
            print(f'Delano Diagram - Configuration {cfg}')
            print('Units         :   Millimeters')
            print('Field-of-View :   {:f} degree'.format(self.target_fov[cfg]))
            print('Total Length  :   {:f} mm'.format(self.sub_dist[sys_id, cfg].sum()))
            print('Effective Focal Length  :   {:f} mm'.format(self.effl()[sys_id,cfg]))
            print('Entrance Pupil Diameter :   {:f} mm'.format(self.target_enpd[cfg]))
            
            ####################################### Systems #######################################
            itv = 3 # interval between rows        
            print('Surf' + ' ' * 4 + 'Y-Bar' +
                        ' ' * itv + ' ' * 12 + 'Y' +
                        ' ' * itv + ' ' * 16 + 'EFFL' +
                        ' ' * itv + ' ' * 13 + 'Distance' +
                        ' ' * itv + ' ' * 9 + 'F')
            for idx in range(self.group_num + 1):
                msg = '{}'.format(idx+1)
                msg += ' ' * itv * 2 + '{:>17.10E}'.format(self.ybar[sys_id, cfg, idx].item()) + ' ' * itv + '{:>17.10E}'.format(self.y[sys_id, cfg, idx].item())
                if idx != self.group_num:
                    msg += (' ' * itv + '{:>17.10E}'.format(self.sub_effl[sys_id,cfg, idx].item()) +
                            ' ' * itv + '{:>17.10E}'.format(self.sub_dist[sys_id, cfg, idx].item()) +
                            ' ' * itv + '{:>17.10E}'.format((self.sub_fno[sys_id, cfg, idx]).item()))
                print(msg)
    
    def plot_y_ybar(self, sys_id=0):
        lim = max(self.y[sys_id, :].abs().max(), self.ybar[sys_id, :].abs().max()).cpu().detach().numpy() * 1.25
        h, w = int(lim + 1) * 4, int(lim + 1) * 4
        fig, ax = plt.subplots(figsize=(h, w))
        color_list = ['blue', 'brown', 'cyan', 'green', 'olive', 'orange', 'pink', 'purple', 'red']
        for cfg in range(self.cfg_num):
            ax.plot(self.ybar[sys_id, cfg].cpu().detach().numpy(), self.y[sys_id, cfg].cpu().detach().numpy(), color=color_list[cfg])
            ax.scatter(self.ybar[sys_id, cfg].cpu().detach().numpy(), self.y[sys_id, cfg].cpu().detach().numpy(), marker='*', color=color_list[cfg])
        ax.axis('square')
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_xlabel('ybar/mm', fontsize=14)
        ax.set_ylabel('y/mm', fontsize=14)
        ax.grid()
        plt.tight_layout()
        
    def effl(self):
        """
        Calculate the effective focal length of the system.
        """
        OF_ybar = self.ybar[:, :, -1]
        OF_y = self.y[:, :, -1]
        
        k_PM = (self.y[:, :, -2] - self.y[:, :, -1]) / (self.ybar[:, :, -2] - self.ybar[:, :, -1])
        OP_ybar = self.y[:, :, 0] / k_PM + self.ybar[:, :, -1]
        OP_y = self.y[:, :, 0]
        
        effl = ((OF_ybar * OP_y) - (OF_y * OP_ybar)) / self.Q[None, :]
        return effl
        
    def propagate(self, y0, u0):
        """
        Use paraxial propagation.
        This propagate() is used for recording u and y of different elements.
        """
        u = u0.unsqueeze(0) # [1, sys, cfg, M]
        y = y0.unsqueeze(0)
        
        for i in range(self.group_num):
            u0 = u0 - y0 / self.sub_effl[:, :, i][:, :, None]
            y0 = y0 + self.sub_dist[:, :, i][:, :, None] * u0
            
            u = torch.cat([u, u0.unsqueeze(0)], dim=0) # [x, sys, cfg, M]
            y = torch.cat([y, y0.unsqueeze(0)], dim=0) # [x, sys, cfg, M]

        return y, u
    
    def plot_set_up_with_trace(self, sys_id=0, M=7):
        """
        Plot elements in 2D.
        """
        ######################################## step 1: calculate enpp ########################################
        ybar, y = torch.zeros(self.sys_num, self.cfg_num), self.target_enpd[None, :].repeat(self.sys_num, 1) / 2 # [sys, cfg]
        enpp = (self.y[:, :, 0] * ybar - self.ybar[:, :, 0] * y) / self.Q[None, :] # [sys, cfg]

        ######################################## step 2: calculate marginal field ########################################
        _y_m = torch.linspace(-0.5, 0.5, M)[None, None, :] * self.target_enpd[None, :, None].repeat(self.sys_num, 1, 1) # [sys, cfg, M]
        _u_m = torch.zeros_like(_y_m)
        y_m, _ = self.propagate(_y_m, _u_m)
        
        ######################################## step 3: calculate chief field ########################################
        _y_c = _y_m + (0 - enpp)[:, :, None] * torch.tan(torch.deg2rad(self.target_fov))[None, :, None] # [sys, cfg, M]
        _u_c = torch.ones_like(_y_c) * torch.tan(torch.deg2rad(self.target_fov))[None, :, None] # [sys, cfg, M]
        y_c, _ = self.propagate(_y_c, _u_c)
        
        ######################################## step 4: set fig size ########################################
        xlim_1 = torch.max(self.sub_dist[sys_id].sum(dim=-1)).cpu().detach().numpy() * 0.05
        xlim_2 = torch.max(self.sub_dist[sys_id].sum(dim=-1)).cpu().detach().numpy() * 1.05
        ylim = torch.max(self.sub_radius[sys_id]).cpu().detach().numpy() * 1.05
        w = self.sub_radius[sys_id].max().cpu().detach().numpy() / 20 # arrow size
        
        hh = int(math.ceil(ylim * 2))
        ww = int(math.ceil(xlim_1 + xlim_2))
        fig, ax = plt.subplots(figsize=(ww, hh * self.cfg_num))
        plt.axis('off')
        x = 0
        for cfg in range(self.cfg_num):
            x += 1
            ax = plt.subplot(self.cfg_num, 1, x)
            ax.set_xlim([-xlim_1, xlim_2])
            ax.set_ylim([-ylim, ylim])
            
            # calculate radius of each elements
            r = self.sub_radius[sys_id, cfg, :].cpu().detach().numpy() # [subs]
            z = [0]
            
            ######################################## step 5: draw the 2D setup of the system ########################################
            for i in range(self.group_num):
                # draw the element
                ax.plot([z[-1], z[-1]], [-r[i], r[i]], color='k')
                
                if self.sub_effl[sys_id, cfg, i] > 0:
                    ax.plot([z[-1], z[-1]-w], [-r[i], -r[i]+w], color='k')
                    ax.plot([z[-1], z[-1]+w], [-r[i], -r[i]+w], color='k')
                    ax.plot([z[-1], z[-1]-w], [r[i], r[i]-w], color='k')
                    ax.plot([z[-1], z[-1]+w], [r[i], r[i]-w], color='k')
                else:
                    ax.plot([z[-1], z[-1]-w], [-r[i], -r[i]-w], color='k')
                    ax.plot([z[-1], z[-1]+w], [-r[i], -r[i]-w], color='k')
                    ax.plot([z[-1], z[-1]-w], [r[i], r[i]+w], color='k')
                    ax.plot([z[-1], z[-1]+w], [r[i], r[i]+w], color='k')
            
                z.append(z[-1] + self.sub_dist[sys_id, cfg, i].cpu().detach().numpy())
                
            # draw image plane
            ax.plot([z[-1], z[-1]], [-r[-1], r[-1]], color='k')
            ax.set_aspect(1)
            
            # draw stop
            z_s = (self.sub_dist[sys_id, cfg, :self.stop_pos].sum()).cpu().detach().numpy()
            r_s = self.sub_radius[sys_id, cfg, self.stop_pos].cpu().detach().numpy()
            ax.plot([z_s-w, z_s+w], [r_s, r_s], color='r')
            ax.plot([z_s-w, z_s+w], [-r_s, -r_s], color='r')
            ax.plot([z_s, z_s], [r_s, r_s+w], color='r')
            ax.plot([z_s, z_s], [-r_s, -r_s-w], color='r')
            
            ######################################## step 6: draw marginal field ########################################
            ax.plot(z, y_m[:, sys_id, cfg].cpu().detach().numpy(), color='b')
            
            ######################################## step 7: draw chief field ########################################
            ax.plot(z, y_c[:, sys_id, cfg].cpu().detach().numpy(), color='c')
        plt.tight_layout()
    
    #====================================================================================================#
    #------------------------------------------- Optimization -------------------------------------------#
    #====================================================================================================#
    
    def merit_dist(self):
        """
        Consider the back focal length of the system.
        Consider the FF, FM, MM distances of the system.
        """
        loss_dist = torch.zeros([self.sys_num, self.cfg_num, len(self.group_structure)])
        structure = self.group_structure + 'I'
        for i in range(len(self.group_structure)):
            sub = structure[i:i+2]
            match sub:
                case 'FF':
                    loss_dist[:, :, i] = torch.where(self.sub_dist[:, :, i] < self.FF_min_dist, self.FF_min_dist - self.sub_dist[:, :, i], 0.)
                case 'FM' | 'MF':
                    loss_dist[:, :, i] = torch.where(self.sub_dist[:, :, i] < self.FM_min_dist, self.FM_min_dist - self.sub_dist[:, :, i], 0.)
                case 'MM':
                    loss_dist[:, :, i] = torch.where(self.sub_dist[:, :, i] < self.MM_min_dist, self.MM_min_dist - self.sub_dist[:, :, i], 0.)
                case 'MI' | 'FI':
                    loss_dist[:, :, i] = torch.where(self.sub_dist[:, :, i] < self.target_bfl, self.target_bfl - self.sub_dist[:, :, i], 0.)
        return loss_dist.sum(dim=[-2, -1]) # [sys]
    
    def merit_dist_diff(self):
        """
        Merit the absolute position of the Fixed subgroup.
        """
        loss_dist = torch.zeros(self.sys_num, len(self.group_structure)) # [sys, sub]
        for i, sub in enumerate(self.group_structure):
            match sub:
                case 'F':
                    loss_dist[:, i] = ((self.sub_dist[:, :, i:].sum(dim=-1) - torch.mean(self.sub_dist[:, :, i:].sum(dim=-1), dim=-1, keepdim=True)) ** 2).sum(dim=-1)
        return loss_dist.sum(dim=-1) # [sys]
    
    def merit_smooth_zoom(self):
        loss_dist = torch.zeros(self.sys_num, len(self.group_structure)) # [sys, sub]
        for i, sub in enumerate(self.group_structure):
            match sub:
                case 'M':
                    dist_a = self.sub_dist[:, 1:, i:].sum(dim=-1) # [sys, cfg-1]
                    dist_b = self.sub_dist[:, :-1, i:].sum(dim=-1) # [sys, cfg-1]

                    loss_des = torch.where((dist_a - dist_b) > 0., dist_a - dist_b, 0.).sum(dim=-1) # [sys]
                    loss_asc = torch.where((dist_b - dist_a) > 0., dist_b - dist_a, 0.).sum(dim=-1) # [sys]
                    loss_dist[:, i] = torch.where(loss_des > loss_asc, loss_asc, loss_des)
        return loss_dist.sum(dim=-1) # [sys]
    
    def merit_radius_diff(self):
        radius_diff = (self.sub_radius - self.sub_radius.mean(dim=[1], keepdim=True)) ** 2 # [sys, cfg]
        return radius_diff.sum(dim=[-2, -1]) # [sys]
    
    def merit_fno(self):
        """
        Consider the fno of each subgroup.
        """
        loss_fno = (self.sub_fno ** -1).amax(dim=[-2, -1])
        return loss_fno # [sys]
    
    def merit_pow_diff(self):
        """
        Merit the power of each configuration.
        """
        loss_pow = (self.sub_effl - torch.mean(self.sub_effl, dim=1, keepdim=True)).abs()
        return loss_pow.sum(dim=[-2, -1]) # [sys]
    
    def merit_angle(self):
        M = 3
        ######################################## step 1: calculate enpp ########################################
        ybar, y = torch.zeros(self.sys_num, self.cfg_num), self.target_enpd[None, :].repeat(self.sys_num, 1) / 2 # [sys, cfg]
        enpp = (self.y[:, :, 0] * ybar - self.ybar[:, :, 0] * y) / self.Q[None, :] # [sys, cfg]

        ######################################## step 2: calculate marginal field ########################################
        _y_m = torch.linspace(-0.5, 0.5, M)[None, None, :] * self.target_enpd[None, :, None].repeat(self.sys_num, 1, 1) # [sys, cfg, M]
        _u_m = torch.zeros_like(_y_m)
        _, u_m = self.propagate(_y_m, _u_m)
        
        ######################################## step 3: calculate chief field ########################################
        _y_c = _y_m + (0 - enpp)[:, :, None] * torch.tan(torch.deg2rad(self.target_fov))[None, :, None] # [sys, cfg, M]
        _u_c = torch.ones_like(_y_c) * torch.tan(torch.deg2rad(self.target_fov))[None, :, None] # [sys, cfg, M]
        _, u_c = self.propagate(_y_c, _u_c)
    
        loss = ((u_m[1:, :, :, :] - u_m[:-1, :, :, :]).abs()).amax(dim=[-1]).sum(dim=[0, 2]) + ((u_c[1:, :, :, :] - u_c[:-1, :, :, :]).abs()).amax(dim=[-1]).sum(dim=[0, 2])
        return loss # [sys]
    
    def merit_totr(self):
        loss = torch.where(self.sub_dist.sum(dim=-1) > self.target_totr, self.sub_dist.sum(dim=-1) - self.target_totr, 0.) # [sys, cfg]
        return loss.sum(dim=-1)
    
    def fitness_A(self):
        loss = self.merit_dist_diff() + self.merit_pow_diff() + self.merit_dist() + self.merit_smooth_zoom() + self.merit_angle() + self.merit_totr() + self.merit_fno() + self.merit_radius_diff()
        return loss
    
    def fitness_B(self):
        loss = (self.merit_dist_diff() + 1) * (self.merit_pow_diff() + 1) * (self.merit_dist() + 1) * (self.merit_smooth_zoom() + 1) * (self.merit_angle() + self.merit_totr() + self.merit_fno() + self.merit_radius_diff())
        return loss
    
    def optimize_SA(self):
        T = 200
        T_min = 1
        step = 0.001
        alpha = 0.99
        iters = 200
        k = 1
        
        with torch.no_grad():
            y_min = self._y.data
            ybar_min = self._ybar.data
            loss_min = self.fitness_A()
            
            pbar = tqdm()
            while T >= T_min:
                for i in range(iters):
                    loss = self.fitness_A()
                    
                    _y = self._y.data
                    _ybar = self._ybar.data
                    
                    self._y.data = self._y.data * (1 + (torch.rand_like(self._y.data) - 0.5) * 2 * step * T)
                    self._ybar.data = self._ybar.data * (1 + (torch.rand_like(self._ybar.data) - 0.5) * 2 * step * T)
                    self.update_y_ybar()
                    self.generate()
                    loss_new = self.fitness_A()
                    
                    y_min = torch.where((loss_new < loss_min)[:, None, None], self._y.data, y_min)
                    ybar_min = torch.where((loss_new < loss_min)[:, None, None], self._ybar.data, ybar_min)
                    loss_min = torch.where(loss_new < loss_min, loss_new, loss_min)
                    self.loss_SA.append(loss_min.min().item())
                    
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
        pbar = tqdm()
        # optimizer initialization
        optimizer = torch.optim.Adam([self._y, self._ybar], lr = lr)
        # ======================================================== #
        # Merit y, ybar
        # ======================================================== #
        # Step1: ------------------------------------------------- #
        loss_min = torch.ones(self.sys_num) * 1e10
        _y_m, _ybar_m = torch.zeros_like(self._y), torch.zeros_like(self._ybar)
        iters_tol = 1
        while iters_tol < self.fst_tol_iters:
            optimizer.zero_grad()
            self.iters += 1
            loss = self.fitness_A()
            
            valid = loss_min > loss
            loss_min[valid] = loss[valid]
            _y_m[valid], _ybar_m[valid] = self._y[valid], self._ybar[valid]
            pbar.set_postfix_str(f'step: 1/2, min loss: {loss_min.min().item()}, max loss: {loss_min.max().item()}')
            self.loss_Ae.append(loss_min.min().item())
        
            iters_tol = 1 if loss.min() <= loss_min.min() else iters_tol + 1
            
            loss.sum().backward()
            optimizer.step()
            self.update_y_ybar()
            self.generate()
            
        # Step2: ------------------------------------------------- #
        # adjust dist and pow diff
        self._y.data, self._ybar.data = _y_m, _ybar_m
        loss_min = torch.ones(self.sys_num) * 1e10
        iters_tol = 1
        while iters_tol < self.fst_tol_iters:
            optimizer.zero_grad()
            self.iters += 1
            loss = self.fitness_B()
            
            valid = loss_min > loss
            loss_min[valid] = loss[valid]
            _y_m[valid], _ybar_m[valid] = self._y[valid], self._ybar[valid]
            pbar.set_postfix_str(f'step: 2/2, min loss: {loss_min.min().item()}, max loss: {loss_min.max().item()}')
            self.loss_As.append(loss_min.min().item())
        
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
            loss = self.fitness_B()
        
        _, idx = torch.topk(torch.nan_to_num(loss, torch.inf), 1, largest=False)

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
            
    def revise_lens_data(self):
        self.sub_effl = self.sub_effl.mean(dim=1, keepdim=True).repeat(1, self.cfg_num, 1) # [sys, cfg, subs]
        self.sub_fno = self.sub_effl.abs() / (2 * self.sub_radius[:, :, 0:-1]) # [sys, cfg, subs]
        
        structure = self.group_structure + 'I'
        for i in range(len(self.group_structure)):
            sub = structure[i:i+2]
            match sub:
                case 'FF' | 'FI':
                    self.sub_dist[:, :, i] = self.sub_dist[:, :, i].mean(dim=-1, keepdim=True)
                case 'FM' | 'MF' | 'MM' | 'MI':
                    self.sub_dist[:, :, i] = self.sub_dist[:, :, i]
        
    #====================================================================================================#
    #--------------------------------------------- Instance ---------------------------------------------#
    #====================================================================================================#
    def lens_calc_wo_mat(self, sub_i, sys_num, stype, mat_type, mat_cata):
        """
        Use random method to generate the lens system.
        """
        pbar = tqdm()
        vd_threshold = 50.
        
        effl = self.sub_effl[self.idx, :, sub_i].mean().detach()
        radius = self.sub_radius[self.idx, :, sub_i].max().detach()
        group_type = self.group_type[sub_i]
        
        sub_phi = torch.stack([generate_normalized_numbers(len(group_type), -2., 2.) for _ in range(sys_num)], dim=0).requires_grad_() # [sys, elem]
        sub_Q = ((torch.rand_like(sub_phi) - 0.5) * 10 - 1).requires_grad_() # [sys, elem]

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
            pbar.set_postfix_str(f'{sub_i+1}/{self.group_num}, min loss: {loss.min().item()}, max loss: {loss.max().item()}')
            
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
        lens_group = []
        stype = stype.split('|')
        mat_type = mat_type.split('|')
        mat_cata = mat_cata.split('|')
        
        for i in range(self.group_num):
            sub_group = self.lens_calc_wo_mat(i, sys_num, stype[i], mat_type[i], mat_cata[i]) # [return list]
            lens_group.append(sub_group)
        return lens_group
    
    def lens_instance(self, sys_num, cfg_num, stype, mat_type, mat_cata):
        """
        Random initialize the system according to the delano diagram.
        """
        self.generate()
        self.revise_lens_data()
        lens_group = self.lens_group(sys_num, stype, mat_type, mat_cata)
        
        zoom_type = []
        structure = self.group_structure + 'I'
        
        sys = nn.ModuleList([])
        sys.append(OBJECT(material="VACUUM", distance=[None] * self.cfg_num))
        
        for i in range(self.group_num):
            if i == self.stop_pos:
                common_params = {
                    'radius': (self.scale * self.sub_radius[self.idx, :, self.stop_pos][None, :].repeat(sys_num, 1)).tolist(),
                    'material': ['VACUUM'] * sys_num,
                    'roc': [None] * sys_num,
                    'thick': [[0.0] * cfg_num] * sys_num,
                    'conic': [0.0] * sys_num,
                }
                sys.append(Sphere(**common_params))
                
                if structure[i] == 'F':
                    zoom_type.append('FF')
                elif structure[i] == 'M':
                    zoom_type.append('MF')
            
            surf_num = lens_group[i]['num']
            for j in range(surf_num):
                if j + 1 == surf_num:
                    thick = self.scale * self.sub_dist[self.idx, None, :, i].repeat(sys_num, 1)
                else:
                    thick = torch.zeros(sys_num, cfg_num)
                common_params = {
                    'radius': (self.scale * lens_group[i]["radius"][..., None, None].repeat(sys_num, cfg_num)).tolist(),
                    'material': lens_group[i]["material"][j],
                    'roc': (self.scale * lens_group[i]["roc"][j] ** -1).tolist(),
                    'thick': thick.tolist(),
                    'conic': [0.0] * sys_num,
                    'mat_cata': lens_group[i]["mat_cata"][j],
                }
                match lens_group[i]['stype'][j]:
                    case 'Q':
                        sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * 3, rnorm=(self.scale * lens_group[i]["radius"][..., None].repeat(sys_num)).tolist()))
                    case 'q':
                        sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * 3, rnorm=(self.scale * lens_group[i]["radius"][..., None].repeat(sys_num)).tolist()))
                    case 'A':
                        sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * 3))
                    case 'S':
                        sys.append(Sphere(**common_params))
            
                if structure[i] == 'F':
                    if j + 1 == surf_num:
                        if structure[i+1] == 'M':
                            zoom_type.append('FM')
                        else:
                            zoom_type.append('FF')
                    else:
                        zoom_type.append('FF')
                elif structure[i] == 'M':
                    if j + 1 == surf_num:
                        zoom_type.append('MM')
                    else:
                        zoom_type.append('MF')
                    
        sys.append(IMAGE(radius=(self.scale * self.img_h[None, :].repeat(sys_num, 1)).tolist()))
        zoom_type.append('FF')
        stop = sum([lens_group[i]['num'] for i in range(self.stop_pos)]) + 1
        return sys, stop, zoom_type