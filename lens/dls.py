import torch

from .optim import Merit
from .utils import eps

# ============================================================
#  Damped Least Squares (DLS)
# ============================================================

class DLS(Merit):
    def __init__(self, args:dict, **kwargs):
        super(DLS, self).__init__(**kwargs)
        self.args = args
        self.mu = None
        self.max_iter = 10
        self.v = 2.0
        self.min_damp = 1e-12
        self.max_damp = 1e12
        self.weight = None

    def _loss(self, *variables):
        loss_list = []
        weight_list = []

        for idx, para in enumerate(self.params):
            para.copy_(variables[idx])

        self.propagate_all_rays()

        for func in self.args:
            match func:
                case 'EFL':
                    loss_list.append(self.efl_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'FNO':
                    loss_list.append(self.fno_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'SPOT':
                    val = self._spot_loss_vector(self.args[func].get('ref', 'rms'))
                    loss_list.append(val)
                    w = self.args[func]['weight']
                    weight_list.extend([w] * val.shape[0])
                case 'WAVEFRONT':
                    val = self._wavefront_loss_vector(self.args[func].get('mode', 'RMS'))
                    loss_list.append(val)
                    w = self.args[func]['weight']
                    weight_list.extend([w] * val.shape[0])
                case 'DISTOR':
                    absolute = self.args[func].get('abs', True)
                    loss_list.append(self.distor_loss(self.args[func]['target'], absolute))
                    weight_list.append(self.args[func]['weight'])
                case 'BFL':
                    loss_list.append(self.bfl_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'TOTR':
                    loss_list.append(self.totr_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'GLA_MIN_THICK':
                    val = 0
                    if self.args[func].get('td_ratio') is not None:
                        val = val + self.gla_min_thick_loss(td_ratio=self.args[func]['td_ratio'])
                    if self.args[func].get('min_thick') is not None:
                        val = val + self.gla_min_thick_loss(min_thick=self.args[func]['min_thick'])
                    loss_list.append(val)
                    weight_list.append(self.args[func]['weight'])
                case 'GLA_MAX_THICK':
                    val = 0
                    if self.args[func].get('td_ratio') is not None:
                        val = val + self.gla_max_thick_loss(td_ratio=self.args[func]['td_ratio'])
                    if self.args[func].get('max_thick') is not None:
                        val = val + self.gla_max_thick_loss(max_thick=self.args[func]['max_thick'])
                    loss_list.append(val)
                    weight_list.append(self.args[func]['weight'])
                case 'GLA_MAX_MIN_RATIO':
                    loss_list.append(self.gla_max_min_ratio_loss(self.args[func]['max_ratio']))
                    weight_list.append(self.args[func]['weight'])
                case 'AIR_THICK':
                    loss_list.append(self.air_thick_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'SURF_K':
                    loss_list.append(self.surf_k_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'ANGLE':
                    loss_list.append(self.angle_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'CRA':
                    loss_list.append(self.cra_loss(self.args[func]['target']))
                    weight_list.append(self.args[func]['weight'])
                case 'ANGLE_STD':
                    loss_list.append(self.angle_std_loss())
                    weight_list.append(self.args[func]['weight'])
                case 'PUPIL':
                    loss_list.append(self.pupil_loss(self.args[func]['ref_point_n']))
                    weight_list.append(self.args[func]['weight'])
                case 'GLA_Z':
                    loss_list.append(self.gla_z_loss(self.args[func]['z_min']))
                    weight_list.append(self.args[func]['weight'])
                case 'SAG_DIA_MAX_RATIO':
                    loss_list.append(self.sag_dia_max_ratio_loss(self.args[func]['max_ratio']))
                    weight_list.append(self.args[func]['weight'])

        self.weight = torch.tensor(weight_list)
        loss = torch.vstack(loss_list)
        return loss
    
    def _spot_loss_vector(self, ref='rms'):
        x = torch.where(self.v_dic, self.o_dic[-1, :, :, :, :, :, :, 0], 0.)
        y = torch.where(self.v_dic, self.o_dic[-1, :, :, :, :, :, :, 1], 0.)
        if ref == 'rms':
            ref_x = x.sum(dim=-1, keepdim=True) / self.v_dic.sum(dim=-1, keepdim=True)
            ref_y = y.sum(dim=-1, keepdim=True) / self.v_dic.sum(dim=-1, keepdim=True)
        else:
            ref_x = x.gather(-1, self.chief_id_dic[..., None])
            ref_y = y.gather(-1, self.chief_id_dic[..., None])
        dx = x - ref_x
        dy = y - ref_y
        spot_ms = torch.where(self.v_dic, dx ** 2 + dy ** 2, 0.).sum(dim=-1) / self.v_dic.sum(dim=-1)
        spot_ms = spot_ms * self.sys.waveweights[:, None, None, None, None] / self.sys.waveweights.sum()
        return spot_ms.permute(1, 2, 0, 3, 4).reshape(self.sys.sys_num, -1).permute(1, 0)

    def _wavefront_loss_vector(self, mode='RMS'):
        chief_o = torch.gather(self.o_dic[-1, :, :, :, :, :, :, :], dim=-2,
                               index=self.chief_id_dic[..., None, None].repeat(1, 1, 1, 1, 1, 1, 3))
        chief_d = torch.gather(self.d_dic[-1, :, :, :, :, :, :, :], dim=-2,
                               index=self.chief_id_dic[..., None, None].repeat(1, 1, 1, 1, 1, 1, 3))
        chief_t = torch.gather(self.t_dic[:, :, :, :, :, :], dim=-1,
                               index=self.chief_id_dic[:, :, :, :, :, None])
        o = torch.where(self.v_dic[..., None], self.o_dic[-1, :, :, :, :, :, :, :], 0.)
        d = torch.where(self.v_dic[..., None], self.d_dic[-1, :, :, :, :, :, :, :], 0.)
        t = torch.where(self.v_dic, self.t_dic, 0.)
        r_chief = -self.sys.EXPP[None, :, :, None, None, None] / chief_d[:, :, :, :, :, :, 2]
        A = 1.
        B = -2 * ((o - chief_o) * d).sum(dim=-1)
        C = ((o - chief_o) ** 2).sum(dim=-1) - r_chief ** 2
        disc = (B ** 2 - 4 * A * C).clip(eps)
        t1 = (-B + torch.sqrt(disc)) / (2 * A)
        t2 = (-B - torch.sqrt(disc)) / (2 * A)
        t_expp = torch.where(t1 > t2, t1, t2)
        opd = ((chief_t - r_chief) - (t - t_expp)) / self.sys.wavelengths[:, None, None, None, None, None]
        opd = torch.where(self.v_dic, opd, 0.)
        if mode == 'RMS':
            opd_mean = opd.sum(dim=-1, keepdim=True) / self.v_dic.sum(dim=-1, keepdim=True)
            opd_diff = opd - opd_mean
            opd_loss = torch.where(self.v_dic, opd_diff ** 2, 0.).sum(dim=-1) / self.v_dic.sum(dim=-1)
        elif mode == 'TV':
            opd_loss = opd.amax(dim=-1) - opd.amin(dim=-1)
        else:
            raise ValueError(f'Invalid mode: {mode}')
        opd_loss = opd_loss * self.sys.waveweights[:, None, None, None, None] / self.sys.waveweights.sum()
        return opd_loss.permute(1, 2, 0, 3, 4).reshape(self.sys.sys_num, -1).permute(1, 0)
    
    @torch.no_grad()
    def update_variables(self):
        self.params = tuple([i for i in self.sys.parameters() if i.requires_grad])
    
    def jacobian(self):
        self.update_variables()
        for para in self.params: para.requires_grad = False
        jacs = torch.autograd.functional.jacobian(self._loss, self.params)
        for para in self.params: para = para.detach_().requires_grad_() # reset requires_grad
        jacs = [torch.diagonal(jac, dim1=1, dim2=2) for jac in jacs]
        jacs = [jac[:, :, None].repeat(1, 1, self.sys.cfg_num) if jac.dim() == 2 else jac.permute(0, 2, 1) for jac in jacs]
        J = torch.stack(jacs).permute(2, 3, 1, 0) # [sys, cfg, M, N]
        return J # [sys, cfg, M, N]
    
    def calc_loss(self):
        for para in self.params: para.requires_grad = False
        loss = self._loss(*self.params)
        for para in self.params: para = para.detach_().requires_grad_() # reset requires_grad
        return loss # [M, sys]
    
    def step(self):
        J = self.jacobian() # [sys, cfg, M, N]
        W = torch.diag(self.weight ** 2)[None, None, :, :].repeat(self.sys.sys_num, self.sys.cfg_num, 1, 1)
        A = torch.matmul(torch.matmul(J.permute(0, 1, 3, 2), W), J) # [sys, cfg, N, N]
        
        if self.mu is None: self.mu = torch.diagonal(A, dim1=-2, dim2=-1).amax(dim=[-2, -1]).clip(eps) # [sys]
        v = self.v
        for i in range(self.max_iter):
            H = A + self.mu[:, None, None, None] * torch.eye(len(self.params))[None, None, :, :] # [sys, cfg, N, N]
            loss = self.calc_loss()
            try:
                g = torch.matmul(torch.matmul(J.permute(0, 1, 3, 2), W), loss[:, None, :, None].permute(2, 3, 0, 1)) # [sys, cfg, N, 1]
                delta = torch.linalg.solve(H, -g) # [sys, cfg, N, 1]
            except Exception:
                self.mu = (self.mu * v).clip(self.min_damp, self.max_damp)
                v *= 2
                continue
            
            delta = torch.zeros_like(delta) if torch.isnan(delta).any() else delta
            
            for idx, para in enumerate(self.params):
                para.data += delta[:, :, idx, 0].mean(dim=1) if para.dim() == 1 else delta[:, :, idx, 0]
            # self.sys.update()
            loss_new = self.calc_loss()
            
            rho = (torch.diagonal(W, dim1=-2, dim2=-1) * (loss ** 2 - loss_new ** 2).permute(1, 0)[:, None, :]).sum(dim=[1, 2]) / (torch.matmul(delta.permute(0, 1, 3, 2), self.mu[:, None, None, None] * delta - g)).sum(dim=[1, 2, 3]) # [sys]
            if rho.max() > 0:
                self.mu = self.mu * torch.where(1 - (2 * rho - 1) ** 3 > 1/3, 1 - (2 * rho - 1) ** 3, 1/3)
                self.mu = self.mu.clip(self.min_damp, self.max_damp)
                break
            else:
                self.mu = (self.mu * v).clip(self.min_damp, self.max_damp)
                v = 2 * v
                for idx, para in enumerate(self.params):
                    para.data -= delta[:, :, idx, 0].mean(dim=1) if para.dim() == 1 else delta[:, :, idx, 0]
                # self.sys.update()