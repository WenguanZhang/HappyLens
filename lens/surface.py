# standard asphere binary2 freeform Qtype
import torch
import torch.nn as nn

from .solver import Solver
from .utils import normalize, Ray, quaternion_raw_multiply, glass_catalog, glass_catalog_params, plastic_catalog, plastic_catalog_params, g1_g2_to_n, eps, factorial, vaccum_nd, vaccum_vd

class Sphere(nn.Module):
    def __init__(self, radius, material, roc, thick, conic, mat_cata=None, aperture='float', min_r=None, max_r=None):
        super(Sphere, self).__init__()
        
        self.radius = torch.tensor(radius) # [sys, cfg]
        
        self.roc = torch.tensor([1. / x if x is not None else 0. for x in roc]) # [sys]
        self.thick = torch.tensor(thick) # [sys, cfg]
        self.conic = torch.tensor(conic) # [sys]

        self.roc = nn.Parameter(self.roc)
        self.thick = nn.Parameter(self.thick)
        self.conic  = nn.Parameter(self.conic)
        
        self.mat_cata = mat_cata
        if self.mat_cata == 'G':
            catalog = glass_catalog
            catalog_params = glass_catalog_params
        elif self.mat_cata == 'P':
            catalog = plastic_catalog
            catalog_params = plastic_catalog_params
        else:
            catalog = None
        self.material = {
            'name': material, # [sys]
            'nd': [catalog[mat]['nd'] if mat != 'VACUUM' and mat != 'MIRROR' else vaccum_nd for mat in material],
            'vd': [catalog[mat]['vd'] if mat != 'VACUUM' and mat != 'MIRROR' else vaccum_vd for mat in material],
        }
        if 'VACUUM' not in self.material['name'] and 'MIRROR' not in self.material['name']:
            self.material_para = torch.zeros(self.radius.shape[0], 7) # [sys, 7], 0: Schott(0)/Sellmeier(1), 1: A0/K1, 2: A1/L1, 3: A2/K2, 4: A3/L2, 5: A4/K3, 6: A5/L3
            for idx, material in enumerate(self.material['name']):
                self.material_para[idx, 0] = 0 if 'A0' in catalog[material] else 1
                self.material_para[idx, 1] = catalog[material]['A0'] if 'A0' in catalog[material] else catalog[material]['K1']
                self.material_para[idx, 2] = catalog[material]['A1'] if 'A1' in catalog[material] else catalog[material]['L1']
                self.material_para[idx, 3] = catalog[material]['A2'] if 'A2' in catalog[material] else catalog[material]['K2']
                self.material_para[idx, 4] = catalog[material]['A3'] if 'A3' in catalog[material] else catalog[material]['L2']
                self.material_para[idx, 5] = catalog[material]['A4'] if 'A4' in catalog[material] else catalog[material]['K3']
                self.material_para[idx, 6] = catalog[material]['A5'] if 'A5' in catalog[material] else catalog[material]['L3']
        
        if 'VACUUM' not in self.material['name'] and 'MIRROR' not in self.material['name']:
            self.g1 = torch.zeros(self.radius.shape[0]) # [sys]
            self.g2 = torch.zeros(self.radius.shape[0]) # [sys]
            for idx, material in enumerate(self.material['name']):
                index = list(catalog.keys()).index(material)
                self.g1[idx] = catalog_params[index, 0].clone().detach().to(self.radius.device) # [sys]
                self.g2[idx] = catalog_params[index, 1].clone().detach().to(self.radius.device)
            
            self.g1 = nn.Parameter(self.g1)
            self.g2 = nn.Parameter(self.g2)
            
            self.g1.requires_grad = False
            self.g2.requires_grad = False
        
        self.fix_radius = False
        self.thick_tol = torch.zeros_like(self.thick)
        self.solver = Solver(20, 1e-11, 1e-9, 'newton', 512)
        
        self.aperture = aperture # float / circ / obsc
        self.min_r = torch.tensor(min_r) if min_r is not None else torch.zeros(self.radius.shape[0]) # [sys]
        self.max_r = torch.tensor(max_r) if max_r is not None else torch.zeros(self.radius.shape[0]) # [sys]
    
    def update_thickness_tol(self, scale):
        self.thick_tol = torch.randn_like(self.radius) * scale # [sys, cfg]
    
    def thickness(self):
        return self.thick + self.thick_tol
        
    def surface(self, x, y):
        # x, y: [wav, sys, cfg, ang, azi, M]
        r2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * r2 * self.roc[None, :, None, None, None, None] ** 2
        sag = r2 * self.roc[None, :, None, None, None, None] / (1 + torch.sqrt((1 - alpha_r2).clip(eps)))

        surf = sag
        return surf
    
    def surface_sag(self, surf_samp):
        r = torch.linspace(0., 1., surf_samp)
        r = (r[None, None, ...] * self.radius[..., None])[None, :, :, None, None, :]
        return self.surface(r, 0)
    
    def surface_d(self, x, y, vec=True):
        r2 = x ** 2 + y ** 2
        
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * r2 * self.roc[None, :, None, None, None, None] ** 2
        #! Clamp here!
        tmp = torch.sqrt((1 - alpha_r2).clip(eps))
        sag_d = self.roc[None, :, None, None, None, None] * (1 + tmp - 0.5 * alpha_r2) / (tmp * (1 + tmp) ** 2)

        surf_d = sag_d
        if vec:
            surf_dx = surf_d * 2 * x
            surf_dy = surf_d * 2 * y
            surf_dz = -torch.ones_like(surf_d)
            return surf_dx, surf_dy, surf_dz
        else:
            return surf_d
    
    def surface_dd(self, x, y, vec=True):
        r2 = x ** 2 + y ** 2
        
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * r2 * self.roc[None, :, None, None, None, None] ** 2
        #! Clamp here!
        tmp = torch.sqrt((1 - alpha_r2).clip(eps))
        
        A = self.roc[None, :, None, None, None, None] ** 3 * (1 + self.conic[None, :, None, None, None, None])
        B = alpha_r2 ** 2 - 8 * alpha_r2 - 4 * alpha_r2 * tmp + 8 * tmp + 8
        C = 4 * (1 + tmp) ** 4 * tmp ** 3
        
        sag_dd = A * B / C

        surf_dd = sag_dd
        if vec:
            surf_ddx = surf_dd * 4 * x * x + self.surface_d(x, y, vec=False) * 2
            surf_ddy = surf_dd * 4 * y * y + self.surface_d(x, y, vec=False) * 2
            surf_ddxy = surf_dd * 4 * x * y
            return surf_ddx, surf_ddy, surf_ddxy
        else:
            return surf_dd
    
    def inter_normal(self, x, y, mode='forward'):
        ds_dxyz = self.surface_d(x, y)
        if mode == 'reverse':
            ds_dxyz = [-ds_dxyz[0], -ds_dxyz[1], ds_dxyz[2]]
        return normalize(torch.stack(ds_dxyz, dim=-1))
    
    def refractive_index(self, wavelength):
        wavelength = wavelength * 1e3
        if 'VACUUM' in self.material['name'] or 'MIRROR' in self.material['name']:
            nd = torch.tensor(self.material['nd'])[None, ...].repeat(len(wavelength), 1) # [wav, sys]
            return nd
        elif self.g1.requires_grad or self.g2.requires_grad:
            return g1_g2_to_n(self.g1, self.g2, wavelength, self.mat_cata) # [wav, sys]
        else:
            n2 = torch.where(self.material_para[:, 0] == 0,
                             # Schott
                             self.material_para[:, 1][None, ...] + 
                             self.material_para[:, 2][None, ...] * wavelength[..., None] ** 2 + 
                             self.material_para[:, 3][None, ...] * wavelength[..., None] ** -2 + 
                             self.material_para[:, 4][None, ...] * wavelength[..., None] ** -4 + 
                             self.material_para[:, 5][None, ...] * wavelength[..., None] ** -6 + 
                             self.material_para[:, 6][None, ...] * wavelength[..., None] ** -8,
                             # Sellmeier
                             1 + (self.material_para[:, 1][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - self.material_para[:, 2][None, ...]) + 
                                  self.material_para[:, 3][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - self.material_para[:, 4][None, ...]) + 
                                  self.material_para[:, 5][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - self.material_para[:, 6][None, ...])))
            return torch.sqrt(n2) # [wav, sys]
    
    def refract(self, ray:Ray, pre_surf, mode='forward'):
        n1 = pre_surf.refractive_index(ray.wavelength)
        n2 = self.refractive_index(ray.wavelength)
        match mode:
            case 'forward':
                mu = (n1 / n2)[:, :, None, None, None, None]
            case 'reverse':
                mu = (n2 / n1)[:, :, None, None, None, None]
        
        l = ray.d
        
        normal = self.inter_normal(ray.o[..., 0], ray.o[..., 1], mode=mode) * torch.sign(ray.d[..., 2])[..., None]
        
        cos_theta1 = - torch.einsum('...k, ...k', normal, l)
        cos_theta2_2 = 1 - mu ** 2 * (1 - cos_theta1 ** 2)
        
        # Judge invalid rays: total reflection.
        ray.valid &= cos_theta2_2 > 0.
        
        #! Avoid cos_theta2_2 < 0
        cos_theta2 = torch.sqrt(cos_theta2_2.clip(eps))
        
        if 'MIRROR' in self.material['name']:
            vr = l + 2 * cos_theta1[..., None] * normal
        else:
            vr = mu.unsqueeze(-1) * l + (mu * cos_theta1 - cos_theta2)[..., None] * normal
        ray.d = normalize(vr)
        return ray
    
    def intersect(self, ray:Ray, mode='forward'):
        """
        mode: 'forward' / 'reverse'
        """
        return self.solver.solve(self, ray, mode)
    
    def judge_valid(self, o):
        match self.aperture:
            case 'circ':
                valid = torch.ones_like(o[..., 0]).bool()
                r2 = o[..., 1:, 0] ** 2 + o[..., 1:, 1] ** 2
                valid[..., 1:] = (r2 > self.min_r[None, :, None, None, None, None] ** 2) & (r2 < self.max_r[None, :, None, None, None, None] ** 2)
            case 'obsc':
                valid = torch.ones_like(o[..., 0]).bool()
                r2 = o[..., 1:, 0] ** 2 + o[..., 1:, 1] ** 2
                valid[..., 1:] = (r2 < self.min_r[None, :, None, None, None, None] ** 2) | ((r2 > self.max_r[None, :, None, None, None, None] ** 2) & (r2 <= self.radius[None, :, :, None, None, None] ** 2))
            case 'float':
                valid = (o[..., 0] ** 2 + o[..., 1] ** 2) <= self.radius[None, :, :, None, None, None] ** 2
        return valid
    
    def propagate(self, ray:Ray, pre_surf, radius_flag=True):
        
        # Record initial coordinates
        _o = ray.o
        _d = ray.d
        
        ray = self.intersect(ray)
        ray = self.refract(ray, pre_surf)
        
        o = ray.o
        d = ray.d
        # Calculate ray path from the plane to the surface
        t_p2s = (ray.o - _o)[..., 2] / _d[..., 2]
        
        if radius_flag:
        # Judge invalid rays: out of range.
            ray.valid &= self.judge_valid(o)
        
        # Calculate the ray path from the surface to the plane
        t_s2p = (self.thickness()[None, :, :, None, None, None] - ray.o[..., 2]) / ray.d[..., 2]
        
        # Total length from the plane to the plane
        ray.t = (ray.t + 
                 t_p2s * pre_surf.refractive_index(ray.wavelength)[:, :, None, None, None, None] + 
                 t_s2p * self.refractive_index(ray.wavelength)[:, :, None, None, None, None])
        
        # local coordinates
        ray.o = ray.o + t_s2p[..., None] * ray.d
        ray.o[..., 2] = ray.o[..., 2] - self.thickness()[None, :, :, None, None, None]
        
        # local coordinates o, d
        return o, d, ray
    
    def reverse_propagate(self, ray:Ray, pre_surf, radius_flag=True):
        """
        # do not record ray.t
        """
        _o = ray.o
        _d = ray.d
        
        ray = self.intersect(ray, mode='reverse')
        ray = self.refract(ray, pre_surf, mode='reverse')
        
        o = ray.o
        d = ray.d
        
        if radius_flag:
        # Judge invalid rays: out of range.
            ray.valid &= self.judge_valid(o)
        
        # Calculate the ray path from the surface to the plane
        t_s2p = (pre_surf.thickness()[None, :, :, None, None, None] - ray.o[..., 2]) / ray.d[..., 2]
        
        # local coordinates
        ray.o = ray.o + t_s2p[..., None] * ray.d
        ray.o[..., 2] = ray.o[..., 2] - pre_surf.thickness()[None, :, :, None, None, None]
        
        # local coordinates o, d
        return o, d, ray

    def abcd(self, pre_surf, wavelength):
        """
        2x2 block matrix, M = [[A, B], [C, D]]
        Refraction at a curved interface [[1, 0 ], [c*(n1/n2-1), n1/n2]]
        Propagation in free space or in a medium of constant refractive index [[1, d], [0, 1]]
        abcd = P @ R
        """
        if 'MIRROR' in self.material['name']:
            D = -torch.ones_like(self.thick)
            C = -2 * self.roc[:, None] * torch.ones_like(self.thick) # [sys, cfg]
            A = 1 + C * self.thick
            B = D * self.thick
        else:
            D = (pre_surf.refractive_index(wavelength) / self.refractive_index(wavelength))[0, :, None] * torch.ones_like(self.thick) # [sys, cfg]
            C = self.roc[:, None] * (D - 1)
            A = 1 + C * self.thick
            B = D * self.thick
        
        ac = torch.stack([A, C], dim=-1)
        bd = torch.stack([B, D], dim=-1)
        abcd = torch.stack([ac, bd], dim=-1)
        return abcd
    

class Asphere(Sphere):
    def __init__(self, ai_list, **kwargs):
        super(Asphere, self).__init__(**kwargs)
        self.ai_num = len(ai_list)
        for i in range(self.ai_num):
            exec('self.ai{} = torch.tensor(ai_list[{}])'.format(2 * i + 4, i))
            exec('self.ai{} = nn.Parameter(self.ai{})'.format(2 * i + 4, 2 * i + 4))
    
    def surface(self, x, y):
        r2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * r2 * self.roc[None, :, None, None, None, None] ** 2

        #! Clamp here!
        sag = r2 * self.roc[None, :, None, None, None, None] / (1 + torch.sqrt((1 - alpha_r2).clip(eps)))
        
        higher_surf = 0
        for i in range(self.ai_num-1, -1, -1):
            higher_surf = eval('r2 * higher_surf + self.ai{}[None, :, None, None, None, None]'.format(2 * i + 4))
        higher_surf = higher_surf * r2 ** 2
        
        surf = sag + higher_surf
        return surf
    
    def surface_d(self, x, y, vec=True):
        r2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * r2 * self.roc[None, :, None, None, None, None] ** 2
        
        #! Clamp here!
        tmp = torch.sqrt((1 - alpha_r2).clip(eps))
        sag_d = self.roc[None, :, None, None, None, None] * (1 + tmp - 0.5 * alpha_r2) / (tmp * (1 + tmp) ** 2)
        
        higher_surf_d = 0
        for i in range(self.ai_num-1, -1, -1):
            higher_surf_d = eval('r2 * higher_surf_d + (i + 2) * self.ai{}[None, :, None, None, None, None]'.format(2 * i + 4))
        higher_surf_d = higher_surf_d * r2
        
        surf_d = sag_d + higher_surf_d
        if vec:
            surf_dx = surf_d * 2 * x
            surf_dy = surf_d * 2 * y
            surf_dz = -torch.ones_like(surf_d)
            return surf_dx, surf_dy, surf_dz
        else:
            return surf_d
    
    def surface_dd(self, x, y, vec=True):
        r2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * r2 * self.roc[None, :, None, None, None, None] ** 2
        
        #! Clamp here!
        tmp = torch.sqrt((1 - alpha_r2).clip(eps))
        A = self.roc[None, :, None, None, None, None] ** 3 * (1 + self.conic[None, :, None, None, None, None])
        B = alpha_r2 ** 2 - 8 * alpha_r2 - 4 * alpha_r2 * tmp + 8 * tmp + 8
        C = 4 * (1 + tmp) ** 4 * tmp ** 3
        sag_dd = A * B / C
        
        higher_surf_dd = 0
        for i in range(self.ai_num-1, -1, -1):
            higher_surf_dd = eval('r2 * higher_surf_dd + (i + 2) * (i + 1) * self.ai{}[None, :, None, None, None, None]'.format(2 * i + 4))
        
        surf_dd = sag_dd + higher_surf_dd
        if vec:
            surf_ddx = surf_dd * 4 * x * x + self.surface_d(x, y, vec=False) * 2
            surf_ddy = surf_dd * 4 * y * y + self.surface_d(x, y, vec=False) * 2
            surf_ddxy = surf_dd * 4 * x * y
            return surf_ddx, surf_ddy, surf_ddxy
        else:
            return surf_dd
        
class Qcon(Sphere):
    def __init__(self, qi_list, rnorm, **kwargs):
        super(Qcon, self).__init__(**kwargs)
        self.qi_num = len(qi_list)
        self.rnorm = torch.tensor(rnorm) # [sys]
        for i in range(self.qi_num):
            exec('self.qi{} = torch.tensor(qi_list[{}])'.format(i, i))
            exec('self.qi{} = nn.Parameter(self.qi{})'.format(i, i))
            
    def surface(self, x, y):
        h2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * h2 * self.roc[None, :, None, None, None, None] ** 2

        #! Clamp here!
        sag = h2 * self.roc[None, :, None, None, None, None] / (1 + torch.sqrt((1 - alpha_r2).clip(eps)))        
        u2 = h2 / (self.rnorm[None, :, None, None, None, None] ** 2)
        
        higher_surf = 0
        for i in range(self.qi_num):
            Q_con = 0
            for j in range(i+1):
                c = (-1) ** j * factorial(2 * i + 4 - j) / factorial(j) / factorial(i + 4 - j) / factorial(i - j) 
                Q_con += c * u2 ** (i - j + 2)
            higher_surf += eval('self.qi{}[None, :, None, None, None, None] * Q_con'.format(i))
        
        surf = sag + higher_surf
        return surf
    
    def surface_d(self, x, y, vec=True):
        h2 = x ** 2 + y ** 2
        u2 = h2 / (self.rnorm[None, :, None, None, None, None] ** 2)
        
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * h2 * self.roc[None, :, None, None, None, None] ** 2
        
        #! Clamp here!
        tmp = torch.sqrt((1 - alpha_r2).clip(eps))
        sag_d = self.roc[None, :, None, None, None, None] * (1 + tmp - 0.5 * alpha_r2) / (tmp * (1 + tmp) ** 2)
        
        higher_surf_d = 0
        for i in range(self.qi_num):
            Q_con = 0
            for j in range(i+1):
                Q_con += (-1) ** j * factorial(2 * i + 4 - j) / factorial(j) / factorial(i + 4 - j) / factorial(i - j) * u2 ** (i - j + 1) * (i - j + 2)
            higher_surf_d += eval('self.qi{}[None, :, None, None, None, None] * Q_con'.format(i))
            
        surf_d = sag_d + higher_surf_d / (self.rnorm[None, :, None, None, None, None] ** 2)
        if vec:
            surf_dx = surf_d * 2 * x
            surf_dy = surf_d * 2 * y
            surf_dz = -torch.ones_like(surf_d)
            return surf_dx, surf_dy, surf_dz
        else:
            raise ValueError('Qcon surface_d only support vec=True')
        
class Qbfs(Sphere):
    def __init__(self, qi_list, rnorm, **kwargs):
        super(Qbfs, self).__init__(**kwargs)
        self.qi_num = len(qi_list)
        self.rnorm = torch.tensor(rnorm) # [sys]
        for i in range(self.qi_num):
            exec('self.qi{} = torch.tensor(qi_list[{}])'.format(i, i))
            exec('self.qi{} = nn.Parameter(self.qi{})'.format(i, i))
        self.l, self.g, self.k = self.calculate_coeff()
    
    def calculate_coeff(self):
        M = self.qi_num - 1
        l = [None] * (M + 1)
        g = [None] * (M)
        k = [None] * (M - 1)
        
        l[0] = 2.
        l[1] = 19.0 ** 0.5 * 0.5
        g[0] = -0.5
        
        for m in range(2, M+1):
            k[m-2] = (m * (1.0 - m)) / (2.0 * l[m-2])
            g[m-1] = -(1.0 + g[m-2] * k[m-2]) / l[m-1]
            l[m] = (m * (m + 1.0) + 3.0 - g[m-1] ** 2 - k[m-2] ** 2) ** 0.5
        return l, g, k
    
    def calculate_S_dS(self, u2):
        Q0 = torch.ones_like(u2) # u2.shape
        dQ0 = torch.zeros_like(u2) # u2.shape
        S = self.qi0[None, :, None, None, None, None] * Q0
        dS = self.qi0[None, :, None, None, None, None] * dQ0
        
        Q1 = (13.0 - 16.0 * u2) * (19.0 ** -0.5) # u2.shape
        dQ1 = -16.0 / (19.0 ** 0.5) * torch.ones_like(u2) # u2.shape
        S = S + self.qi1[None, :, None, None, None, None] * Q1
        dS = dS + self.qi1[None, :, None, None, None, None] * dQ1
        
        Pm = 6.0 - 8.0 * u2
        Pm_ = 2.0 * torch.ones_like(u2)
        
        dPm = -8.0 * torch.ones_like(u2)
        dPm_ = torch.zeros_like(u2)
        
        Qm_, Qm = Q0, Q1
        dQm_, dQm = dQ0, dQ1

        for m in range(1, self.qi_num-1):
            P  = (2.0 - 4.0 * u2) * Pm - Pm_
            dP = -4.0 * Pm + (2.0 - 4.0 * u2) * dPm - dPm_
            
            g = self.g[m]
            k = self.k[m-1]
            l = self.l[m+1]
            
            Q = (P - g * Qm - k * Qm_) / l
            dQ = (dP - g * dQm - k * dQm_) / l
            
            S = S + eval('self.qi{}[None, :, None, None, None, None]'.format(m+1)) * Q
            dS = dS + eval('self.qi{}[None, :, None, None, None, None]'.format(m+1)) * dQ
            
            Pm_, Pm = Pm, P
            dPm_, dPm = dPm, dP
            Qm_, Qm = Qm, Q
            dQm_, dQm = dQm, dQ
        return S, dS
    
    def surface(self, x, y):
        h2 = x ** 2 + y ** 2
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * h2 * self.roc[None, :, None, None, None, None] ** 2
        
        #! Clamp here!
        sag = h2 * self.roc[None, :, None, None, None, None] / (1 + torch.sqrt((1 - alpha_r2).clip(eps)))
        u2 = h2 / (self.rnorm[None, :, None, None, None, None] ** 2) # [wav, sys, cfg, ang, azi, M]
        
        tmp = torch.sqrt((1.0 - self.conic[None, :, None, None, None, None] * self.roc[None, :, None, None, None, None] ** 2 * h2).clip(eps))
        S, _ = self.calculate_S_dS(u2)
        surf = sag + S * u2 * (1 - u2) * tmp / torch.sqrt((1 - alpha_r2).clip(eps))
        return surf
        
    def surface_d(self, x, y, vec=True):
        h2 = x ** 2 + y ** 2 
        alpha_r2 = (1 + self.conic[None, :, None, None, None, None]) * h2 * self.roc[None, :, None, None, None, None] ** 2
        
        #! Clamp here!
        tmp = torch.sqrt((1 - alpha_r2).clip(eps))
        sag_d = self.roc[None, :, None, None, None, None] * (1 + tmp - 0.5 * alpha_r2) / (tmp * (1 + tmp) ** 2)
        
        u2 = h2 / (self.rnorm[None, :, None, None, None, None] ** 2)
        ctmp = torch.sqrt((1.0 - self.conic[None, :, None, None, None, None] * self.roc[None, :, None, None, None, None] ** 2 * h2).clip(eps))
        S, dS = self.calculate_S_dS(u2)
        f = u2 * (1 - u2) * ctmp / tmp
        
        dtmp = (self.roc[None, :, None, None, None, None] + eps) ** -2 * (h2 + eps) ** -1 - (1 + self.conic[None, :, None, None, None, None])
        
        dS_dr2 = dS / (self.rnorm[None, :, None, None, None, None] ** 2)
        df_dr2 = (1 - 2 * u2) * ctmp / tmp / (self.rnorm[None, :, None, None, None, None] ** 2) + \
                 u2 * (1 - u2).clip(eps) * 0.5 * (1 + dtmp ** -1).clip(eps) ** -0.5 * (dtmp * self.roc[None, :, None, None, None, None] * h2 + eps) ** -2
        
        surf_d = sag_d + df_dr2 * S + f * dS_dr2
        
        if vec:
            surf_dx = surf_d * 2 * x
            surf_dy = surf_d * 2 * y
            surf_dz = -torch.ones_like(surf_d)
            return surf_dx, surf_dy, surf_dz
        else:
            raise ValueError('Qbfs surface_d only support vec=True')
    
    def abcd(self, pre_surf, wavelength):
        """
        2x2 block matrix, M = [[A, B], [C, D]]
        Refraction at a curved interface [[1, 0 ], [c*(n1/n2-1), n1/n2]]
        Propagation in free space or in a medium of constant refractive index [[1, d], [0, 1]]
        abcd = P @ R
        """
        u2 = torch.zeros([1, len(self.roc), 1, 1, 1, 1])
        S, _ = self.calculate_S_dS(u2)
        roc = self.roc + 2. / self.rnorm ** 2 * S[0, :, 0, 0, 0, 0] # [sys]
        if 'MIRROR' in self.material['name']:
            D = -torch.ones_like(self.thick)
            C = -2 * roc[:, None] * torch.ones_like(self.thick) # [sys, cfg]
            A = 1 + C * self.thick
            B = D * self.thick
        else:
            D = (pre_surf.refractive_index(wavelength) / self.refractive_index(wavelength))[0, :, None] * torch.ones_like(self.thick) # [sys, cfg]
            C = roc[:, None] * (D - 1)
            A = 1 + C * self.thick
            B = D * self.thick
        
        ac = torch.stack([A, C], dim=-1)
        bd = torch.stack([B, D], dim=-1)
        abcd = torch.stack([ac, bd], dim=-1)
        return abcd
    
class Binary2(Asphere):
    def __init__(self, diff_order, pi_list, rnorm, **kwargs):
        super(Binary2, self).__init__(**kwargs)
        
        self.diff_order = diff_order
        self.rnorm = torch.tensor(rnorm) # [sys]
        
        self.pi_num = len(pi_list)
        for i in range(self.pi_num):
            exec('self.pi{} = torch.tensor(pi_list[{}])'.format(2 * i + 2, i))
            exec('self.pi{} = nn.Parameter(self.pi{})'.format(2 * i + 2, 2 * i + 2))
    
    def phase(self, x, y):
        return self._phi(x ** 2 + y ** 2)
    
    def _phi(self, r2):
        """
        return the additional phase of this position
        """
        nr2 = r2 / (self.rnorm[None, :, None, None, None, None] ** 2)
        add_phase = 0
        for i in range(self.pi_num-1, -1, -1):
            add_phase = eval('nr2 * add_phase + self.pi{}[None, :, None, None, None, None]'.format(2 * i + 2))
        add_phase = self.diff_order * add_phase * nr2
        return add_phase
    
    def _dphid(self, r2):
        nr2 = r2 / (self.rnorm[None, :, None, None, None, None] ** 2)
        phi_derivative = 0
        for i in range(self.pi_num-1, -1, -1):
            phi_derivative = eval('nr2 * phi_derivative + (i + 1) * self.pi{}[None, :, None, None, None, None]'.format(2 * i + 2))
        phi_derivative = self.diff_order * phi_derivative / (self.rnorm[None, :, None, None, None, None] ** 2)
        return phi_derivative
        
    def refract(self, ray:Ray, pre_surf, mode='forward'):
        n1 = pre_surf.refractive_index(ray.wavelength)
        n2 = self.refractive_index(ray.wavelength)
        match mode:
            case 'forward':
                mu = (n1 / n2)[:, :, None, None, None, None]
            case 'reverse':
                mu = (n2 / n1)[:, :, None, None, None, None]
        
        l = ray.d
        
        x, y =  ray.o[..., 0], ray.o[..., 1]
        normal = self.inter_normal(x, y, mode=mode) * torch.sign(ray.d[..., 2])[..., None]
        beta = ray.wavelength[:, None] / (2 * torch.pi * n2)
        r2 = x ** 2 + y ** 2
        nabla_phi = torch.stack((self._dphid(r2) * 2 * x, self._dphid(r2) * 2 * y, torch.zeros_like(x)), dim = -1)
        nabla_phi = (nabla_phi - torch.einsum('...k,...k->...', nabla_phi, normal)[..., None] * normal)
        
        tmp = mu[..., None] * (l - torch.einsum('...k, ...k', normal, l)[..., None] * normal) + beta[:, :, None, None, None, None, None] * nabla_phi
        
        # Judge invalid rays: total reflection.
        ray.valid &= (1 - torch.sum(tmp ** 2, dim=-1)) > 0.
        
        if 'MIRROR' in self.material['name']:
            raise NotImplementedError('Binary2 mirror surface')
        else:
            vr = tmp - torch.sqrt(1 - torch.sum(tmp ** 2, dim=-1))[..., None] * normal
        ray.d = normalize(vr)
        ray.t = ray.t + self._phi(r2) * ray.wavelength[:, None, None, None, None, None] / (2 * torch.pi)
        return ray
    
    def abcd(self, pre_surf, wavelength):
        """
        2x2 block matrix, M = [[A, B], [C, D]]
        Propagation in free space or in a medium of constant refractive index [[1, d], [0, 1]]
        abcd = P @ R
        """  
        if 'MIRROR' in self.material['name']:
            raise NotImplementedError('Binary2 mirror surface')
        else:
            D = (pre_surf.refractive_index(wavelength) / self.refractive_index(wavelength))[0, :, None] * torch.ones_like(self.thick) # [sys, cfg]
            
            C_ref = self.roc[:, None] * (D - 1)
            C_phi = eval('wavelength / (2 * torch.pi * self.refractive_index(wavelength))[0, :, None] * (2 * self.diff_order * self.pi2 / self.rnorm ** 2)[:, None]')
            C = C_ref + C_phi
            
            A = 1 + C * self.thick
            B = D * self.thick
        
        ac = torch.stack([A, C], dim=-1)
        bd = torch.stack([B, D], dim=-1)
        abcd = torch.stack([ac, bd], dim=-1)
        return abcd
        
class OBJECT(nn.Module):
    def __init__(self, material, distance):
        super(OBJECT, self).__init__()
        self.material = {
            'name': material, # [sys]
            'nd': vaccum_nd if material == 'VACUUM' else None,
            'vd': vaccum_vd if material == 'VACUUM' else None,
        }
        self.distance = distance
    
    def refractive_index(self, wavelength):
        wavelength = wavelength * 1e3
        if 'VACUUM' in self.material['name']:
            nd = torch.tensor(self.material['nd'])[None, ...].repeat(len(wavelength), 1) # [wav, sys]
            return nd
        else:
            raise NotImplementedError
    
    def propagate(self, ray:Ray):
        if None in self.distance:
            ray.t = torch.sum(ray.o * ray.d, dim=-1)
        else:
            ray.t = ray.t + (torch.tensor(self.distance)[None, None, :, None, None, None] / ray.d[..., 2]) * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]
            ray.o = ray.o + ray.d * (torch.tensor(self.distance)[None, None, :, None, None, None] / ray.d[..., 2])[..., None]
        return ray


class IMAGE(nn.Module):
    def __init__(self, radius):
        super(IMAGE, self).__init__()
        self.radius = torch.tensor(radius)
        
    def surface(self, x, y):
        return torch.zeros_like(x)
        
class Dummy(nn.Module):
    def __init__(self, pre_surf, thick):
        super(Dummy, self).__init__()
        self.thick = thick if torch.is_tensor(thick) else torch.tensor(thick)
        self.pre_surf = pre_surf
        
        self.thick = nn.Parameter(self.thick)

    def thickness(self):
        return self.thick    
    
    def abcd(self, pre_surf, wavelength):
        """
        Propagation in free space or in a medium of constant refractive index [[1, d], [0, 1]]
        """
        D = torch.ones_like(self.thick)
        C = torch.zeros_like(self.thick)
        A = torch.ones_like(self.thick)
        B = self.thick
        
        ac = torch.stack([A, C], dim=-1)
        bd = torch.stack([B, D], dim=-1)
        abcd = torch.stack([ac, bd], dim=-1)
        return abcd
    
    def refractive_index(self, wavelength):
        def get_pre_surf(pre_surf):
            if isinstance(pre_surf, Dummy):
                pre_surf = get_pre_surf(pre_surf.pre_surf)
            return pre_surf
        pre_surf = get_pre_surf(self.pre_surf)
        
        material = pre_surf.material
        wavelength = wavelength * 1e3
        if 'VACUUM' in material['name']:
            nd = torch.tensor(material['nd'])[None, ...].repeat(len(wavelength), 1) # [wav, sys]
            return nd
        elif pre_surf.g1.requires_grad or pre_surf.g2.requires_grad:
            return g1_g2_to_n(pre_surf.g1, pre_surf.g2, wavelength, pre_surf.mat_cata) # [wav, sys]
        else:
            material_para = pre_surf.material_para
            n2 = torch.where(material_para[:, 0] == 0,
                             # Schott
                             material_para[:, 1][None, ...] + 
                             material_para[:, 2][None, ...] * wavelength[..., None] ** 2 + 
                             material_para[:, 3][None, ...] * wavelength[..., None] ** -2 + 
                             material_para[:, 4][None, ...] * wavelength[..., None] ** -4 + 
                             material_para[:, 5][None, ...] * wavelength[..., None] ** -6 + 
                             material_para[:, 6][None, ...] * wavelength[..., None] ** -8,
                             # Sellmeier
                             1 + (material_para[:, 1][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - material_para[:, 2][None, ...]) + 
                                  material_para[:, 3][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - material_para[:, 4][None, ...]) + 
                                  material_para[:, 5][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - material_para[:, 6][None, ...])))
            return torch.sqrt(n2) # [wav, sys]
    
    def propagate(self, ray:Ray, pre_surf=None, radius_flag=None):
        o = ray.o
        # from plane to plane
        t = self.thickness()[None, :, :, None, None, None] / ray.d[..., 2]
        ray.t = ray.t + t * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]

        ray.o = o + t[..., None] * ray.d
        ray.o[..., 2] = ray.o[..., 2] - self.thickness()[None, :, :, None, None, None]
        return o, ray.d, ray
    
    
class Coordinate(Dummy):
    def __init__(self, decenter, tilt, flag, **kwargs):
        super(Coordinate, self).__init__(**kwargs)
        # decenter : [decenter_x, decenter_y]
        # tilt : [tilt_x, tilt_y, tilt_z]
        
        self.decenter = -decenter if torch.is_tensor(decenter) else -torch.tensor(decenter) # [sys, cfg, 2]
        self.tilt = -torch.deg2rad(tilt) if torch.is_tensor(tilt) else -torch.deg2rad(torch.tensor(tilt)) # [sys, cfg, 3]
        self.flag = flag
        
        self.decenter = nn.Parameter(self.decenter)
        self.tilt = nn.Parameter(self.tilt)
    
    def propagate(self, ray:Ray, pre_surf=None, radius_flag=None):
        zero = torch.zeros_like(self.thick)
        q_x = torch.stack([torch.cos(self.tilt[:, :, 0] / 2), torch.sin(self.tilt[:, :, 0] / 2), zero, zero], dim=-1)[None, :, :, None, None, None, :]
        q_y = torch.stack([torch.cos(self.tilt[:, :, 1] / 2), zero, torch.sin(self.tilt[:, :, 1] / 2), zero], dim=-1)[None, :, :, None, None, None, :]
        q_z = torch.stack([torch.cos(self.tilt[:, :, 2] / 2), zero, zero, torch.sin(self.tilt[:, :, 2] / 2)], dim=-1)[None, :, :, None, None, None, :]
        
        real_parts = ray.o.new_zeros(ray.o.shape[:-1] + (1,))
        o = torch.cat((real_parts, ray.o), -1) # [wav, sys, cfg, ang, azi, M, 4]
        
        real_parts = ray.d.new_zeros(ray.d.shape[:-1] + (1,))
        d = torch.cat((real_parts, ray.d), -1) # [wav, sys, cfg, ang, azi, M, 4]
        
        if self.flag == 0:
            # decenter -> tilt x -> tilt y -> tilt z
            o[..., 1:3] = o[..., 1:3] + self.decenter[None, :, :, None, None, None, :]
            q_xyz = quaternion_raw_multiply(q_z, quaternion_raw_multiply(q_y, q_x))
            o = quaternion_raw_multiply(q_xyz, quaternion_raw_multiply(o, q_xyz * torch.tensor([1., -1., -1., -1.])))[..., 1:]
            d = quaternion_raw_multiply(q_xyz, quaternion_raw_multiply(d, q_xyz * torch.tensor([1., -1., -1., -1.])))[..., 1:]
        else:
            # tilt z -> tilt y -> tilt x -> decenter
            q_zyx = quaternion_raw_multiply(quaternion_raw_multiply(q_x, q_y), q_z)
            o = quaternion_raw_multiply(q_zyx, quaternion_raw_multiply(o, q_zyx * torch.tensor([1., -1., -1., -1.])))[..., 1:]
            d = quaternion_raw_multiply(q_zyx, quaternion_raw_multiply(d, q_zyx * torch.tensor([1., -1., -1., -1.])))[..., 1:]
            o[..., 0:2] = o[..., 0:2] + self.decenter[None, :, :, None, None, None, :]
        
        # from coordinate break to plane
        t_s2p = (0. - o[..., 2]) / d[..., 2]
        o = o + t_s2p[..., None] * d

        # from plane to plane
        t_p2p = self.thickness()[None, :, :, None, None, None] / d[..., 2]
        ray.t = ray.t + (t_s2p + t_p2p) * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]

        ray.o = o + t_p2p[..., None] * d
        ray.o[..., 2] = ray.o[..., 2] - self.thickness()[None, :, :, None, None, None]
        ray.d = d
        return o, d, ray
    
    
class PACKAGE(nn.Module):
    # used for tolerance analysis and optimization
    def __init__(self, decenter, tilt, pack):
        super(PACKAGE, self).__init__()
        ##* decenter then tilt *##
        self.decenter = -decenter if torch.is_tensor(decenter) else -torch.tensor(decenter) # [sys, cfg, 2]
        self.tilt = -torch.deg2rad(tilt) if torch.is_tensor(tilt) else -torch.deg2rad(torch.tensor(tilt)) # [sys, cfg, 3]
        
        self.decenter = nn.Parameter(self.decenter)
        self.tilt = nn.Parameter(self.tilt)
        
        self.pack = pack
        
    def refractive_index(self, wavelength):
        inner_elements = []
        def recursive_extract(elements):
            for elem in elements:
                if isinstance(elem, PACKAGE):
                    recursive_extract(elem.pack)
                else:
                    inner_elements.append(elem)
        recursive_extract(self.pack)
        material = inner_elements[-1].material
        
        wavelength = wavelength * 1e3
        if 'VACUUM' in material['name'] or 'MIRROR' in material['name']:
            nd = torch.tensor(material['nd'])[None, ...].repeat(len(wavelength), 1) # [wav, sys]
            return nd
        elif inner_elements[-1].g1.requires_grad or inner_elements[-1].g2.requires_grad:
            return g1_g2_to_n(inner_elements[-1].g1, inner_elements[-1].g2, wavelength, inner_elements[-1].mat_cata) # [wav, sys]
        else:
            material_para = inner_elements[-1].material_para
            n2 = torch.where(material_para[:, 0] == 0,
                             # Schott
                             material_para[:, 1][None, ...] + 
                             material_para[:, 2][None, ...] * wavelength[..., None] ** 2 + 
                             material_para[:, 3][None, ...] * wavelength[..., None] ** -2 + 
                             material_para[:, 4][None, ...] * wavelength[..., None] ** -4 + 
                             material_para[:, 5][None, ...] * wavelength[..., None] ** -6 + 
                             material_para[:, 6][None, ...] * wavelength[..., None] ** -8,
                             # Sellmeier
                             1 + (material_para[:, 1][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - material_para[:, 2][None, ...]) + 
                                  material_para[:, 3][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - material_para[:, 4][None, ...]) + 
                                  material_para[:, 5][None, ...] * wavelength[..., None] ** 2 / (wavelength[..., None] ** 2 - material_para[:, 6][None, ...])))
            return torch.sqrt(n2) # [wav, sys]
    
    def transform(self, ray:Ray, pre_surf, thick, flag):
        """
        Transform rays before and after the propagation of the pack.
        flag == 0: decenter -> tilt x -> tilt y -> tilt z
        flag != 0: tilt z -> tilt y -> tilt x -> decenter
        Note that the signs of the tilt and the decenter are different before and after the propagation of the pack.
        """
        zero = torch.zeros_like(thick)

        real_parts = ray.o.new_zeros(ray.o.shape[:-1] + (1,))
        o = torch.cat((real_parts, ray.o), -1) # [wav, sys, cfg, ang, azi, M, 4]
        
        real_parts = ray.d.new_zeros(ray.d.shape[:-1] + (1,))
        d = torch.cat((real_parts, ray.d), -1) # [wav, sys, cfg, ang, azi, M, 4]
        
        if flag == 0:
            # transform before the propagation of the pack
            q_x = torch.stack([torch.cos(self.tilt[:, :, 0] / 2), torch.sin(self.tilt[:, :, 0] / 2), zero, zero], dim=-1)[None, :, :, None, None, None, :]
            q_y = torch.stack([torch.cos(self.tilt[:, :, 1] / 2), zero, torch.sin(self.tilt[:, :, 1] / 2), zero], dim=-1)[None, :, :, None, None, None, :]
            q_z = torch.stack([torch.cos(self.tilt[:, :, 2] / 2), zero, zero, torch.sin(self.tilt[:, :, 2] / 2)], dim=-1)[None, :, :, None, None, None, :]
        
            # decenter -> tilt x -> tilt y -> tilt z
            o[..., 1:3] = o[..., 1:3] + self.decenter[None, :, :, None, None, None, :]
            q_xyz = quaternion_raw_multiply(q_z, quaternion_raw_multiply(q_y, q_x))
            o = quaternion_raw_multiply(q_xyz, quaternion_raw_multiply(o, q_xyz * torch.tensor([1., -1., -1., -1.])))[..., 1:]
            d = quaternion_raw_multiply(q_xyz, quaternion_raw_multiply(d, q_xyz * torch.tensor([1., -1., -1., -1.])))[..., 1:]
        else:
            # transform after the propagation of the pack
            q_x = torch.stack([torch.cos(-self.tilt[:, :, 0] / 2), torch.sin(-self.tilt[:, :, 0] / 2), zero, zero], dim=-1)[None, :, :, None, None, None, :]
            q_y = torch.stack([torch.cos(-self.tilt[:, :, 1] / 2), zero, torch.sin(-self.tilt[:, :, 1] / 2), zero], dim=-1)[None, :, :, None, None, None, :]
            q_z = torch.stack([torch.cos(-self.tilt[:, :, 2] / 2), zero, zero, torch.sin(-self.tilt[:, :, 2] / 2)], dim=-1)[None, :, :, None, None, None, :]
        
            # tilt z -> tilt y -> tilt x -> decenter
            q_zyx = quaternion_raw_multiply(quaternion_raw_multiply(q_x, q_y), q_z)
            o = quaternion_raw_multiply(q_zyx, quaternion_raw_multiply(o, q_zyx * torch.tensor([1., -1., -1., -1.])))[..., 1:]
            d = quaternion_raw_multiply(q_zyx, quaternion_raw_multiply(d, q_zyx * torch.tensor([1., -1., -1., -1.])))[..., 1:]
            o[..., 0:2] = o[..., 0:2] + -self.decenter[None, :, :, None, None, None, :]
        
        # from coordinate break to plane
        t_s2p = (0. - o[..., 2]) / d[..., 2]
        o = o + t_s2p[..., None] * d

        # from plane to plane
        t_p2p = thick[None, :, :, None, None, None] / d[..., 2]
        
        if flag == 0:
            # refractive index of the element before the pack
            ray.t = ray.t + (t_s2p + t_p2p) * pre_surf.refractive_index(ray.wavelength)[:, :, None, None, None, None]
        else:
            # refractive index of the last element of the pack
            ray.t = ray.t + (t_s2p + t_p2p) * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]

        ray.o = o + t_p2p[..., None] * d
        ray.o[..., 2] = ray.o[..., 2] - thick[None, :, :, None, None, None]
        ray.d = d
        return ray
    
    def propagate(self, ray:Ray, pre_surf, radius_flag=True):
        o_list = []
        d_list = []
        
        # judge whether there is a PACKAGE in the pack
        if any(isinstance(elem, PACKAGE) for elem in self.pack):
            ########## 1. transform ##########
            zero = torch.zeros(ray.valid.shape[1], ray.valid.shape[2]) # [sys, cfg]
            ray = self.transform(ray, pre_surf, zero, 0)
            
            ########## 2. propagate elem in pack ##########
            for i, elem in enumerate(self.pack):
                if i == 0:
                    if isinstance(elem, PACKAGE):
                        _o_list, _d_list, ray = elem.propagate(ray, pre_surf, radius_flag)
                        o_list.extend(_o_list)
                        d_list.extend(_d_list)
                    else:
                        _o, _d, ray = elem.propagate(ray, pre_surf, radius_flag)
                        o_list.append(_o)
                        d_list.append(_d)
                else:
                    if isinstance(elem, PACKAGE):
                        _o_list, _d_list, ray = elem.propagate(ray, self.pack[i-1], radius_flag)
                        o_list.extend(_o_list)
                        d_list.extend(_d_list)
                    else:
                        _o, _d, ray = elem.propagate(ray, self.pack[i-1], radius_flag)
                        o_list.append(_o)
                        d_list.append(_d)
                                                
            ########## 3. propagate back ##########
            def get_thick(package):
                # return the total length of the pack and the length of the last element in the pack
                total_thick = 0.
                for elem in package.pack:
                    if isinstance(elem, PACKAGE):
                        thick, _ = get_thick(elem)
                        total_thick = total_thick + thick
                    else:
                        thick = elem.thickness()
                        total_thick = total_thick + thick
                return total_thick, thick

            thick_sum, thick_last = get_thick(self)
            thick_sum = thick_sum - thick_last
            
            t_back = -(thick_sum + thick_last)[None, :, :, None, None, None] / ray.d[..., 2]
            ray.t = ray.t + t_back * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]
            ray.o = ray.o + t_back[..., None] * ray.d
            ray.o[..., 2] = ray.o[..., 2] - -(thick_sum + thick_last)[None, :, :, None, None, None]
            
            ########## 4. transform ##########
            ray = self.transform(ray, pre_surf, thick_sum, 1)
            
            ########## 5. propagate dummy plane ##########
            # from plane to plane
            t = thick_last[None, :, :, None, None, None] / ray.d[..., 2]
            ray.t = ray.t + t * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]

            ray.o = ray.o + t[..., None] * ray.d
            ray.o[..., 2] = ray.o[..., 2] - thick_last[None, :, :, None, None, None]
            
        else:
            # no PACKAGE in the self.pack
            _o, _d, ray = self.propagate_elem(ray, pre_surf, radius_flag)
            o_list.append(_o)
            d_list.append(_d)
        # return the o, d, ray
        return o_list, d_list, ray
    
    def propagate_elem(self, ray:Ray, pre_surf, radius_flag=True):
        ########## 1. transform ##########
        zero = torch.zeros(ray.valid.shape[1], ray.valid.shape[2]) # [sys, cfg]
        ray = self.transform(ray, pre_surf, zero, 0)
        
        ########## 2. propagate elem in pack ##########
        for i, elem in enumerate(self.pack):
            if i == 0:
                _o, _d, ray = elem.propagate(ray, pre_surf, radius_flag)
            else:
                _o, _d, ray = elem.propagate(ray, self.pack[i-1], radius_flag)
        
        ########## 3. propagate back ##########
        thick_sum = sum(elem.thickness() for elem in self.pack) - self.pack[-1].thickness()
        thick_last = self.pack[-1].thickness()
        
        t_back = -(thick_sum + thick_last)[None, :, :, None, None, None] / ray.d[..., 2]
        ray.t = ray.t + t_back * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]
        ray.o = ray.o + t_back[..., None] * ray.d
        ray.o[..., 2] = ray.o[..., 2] - -(thick_sum + thick_last)[None, :, :, None, None, None]
        
        ########## 4. transform ##########
        ray = self.transform(ray, pre_surf, thick_sum, 1)
        
        ########## 5. propagate dummy plane ##########
        # from plane to plane
        t = thick_last[None, :, :, None, None, None] / ray.d[..., 2]
        ray.t = ray.t + t * self.refractive_index(ray.wavelength)[:, :, None, None, None, None]

        ray.o = ray.o + t[..., None] * ray.d
        ray.o[..., 2] = ray.o[..., 2] - thick_last[None, :, :, None, None, None]
        return _o, _d, ray