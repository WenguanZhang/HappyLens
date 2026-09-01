import torch
from .utils import Ray, eps

class Solver(object):
    def __init__(self, MAXITER, TOLERANCE_TIGHT, TOLERANCE_LOOSE, METHOD, surf_samp):
        super(Solver, self).__init__()
        # There are the parameters controlling the accuracy of ray tracing.
        self.MAXITER = MAXITER
        # self.MAXSTEP = MAXSTEP
        self.TOLERANCE_TIGHT = TOLERANCE_TIGHT  # in [mm], i.e. 0.1 [nm] here (up to <10 [nm])
        self.TOLERANCE_LOOSE = TOLERANCE_LOOSE  # in [mm], i.e. 1 [nm] here (up to <10 [nm])
        self.METHOD = METHOD
        
        self.surf_samp = surf_samp
        
    def solve(self, surf, ray, mode='forward'):
        """
        mode: 'forward' or 'reverse'
        """
        if self.METHOD == 'contraction':
            ray = self.contraction(surf, ray, mode)
        elif self.METHOD == 'newton':
            ray = self.newton(surf, ray, mode)
        elif self.METHOD == 'halley':
            ray = self.halley(surf, ray, mode)
        else:
            raise Warning('METHOD does not exist!')
        
        return ray
    
    def contraction(self, surf, ray:Ray, mode):
        """
        Contraction method to find the root of the ray-surface intersection point.
        Only for small fov.
        """
        ox, oy, oz = (ray.o[..., i] for i in range(3))
        dx, dy, dz = (ray.d[..., i] for i in range(3))
        
        with torch.no_grad():
            ox = torch.where(ray.valid, ox, 0.)
            oy = torch.where(ray.valid, oy, 0.)
            
            it = 0
            res = 1e8 * torch.ones_like(oz)
            match mode:
                case 'forward':
                    t = torch.ones_like(oz) * surf.surface_sag(self.surf_samp).min(dim=-1, keepdim=True)[0] / dz
                case 'reverse':
                    t = torch.ones_like(oz) * (-surf.surface_sag(self.surf_samp)).min(dim=-1, keepdim=True)[0] / dz
            ox_ = ox + t * dx
            oy_ = oy + t * dy
            out_radius = (ox_ ** 2 + oy_ ** 2) > surf.radius[None, :, :, None, None, None] ** 2
            A = dx ** 2 + dy ** 2
            B = 2 * (ox_ * dx + oy_ * dy)
            C = ox_ ** 2 + oy_ ** 2 - surf.radius[None, :, :, None, None, None] ** 2
            DELTA = B ** 2 - 4 * A * C
            #! Avoid DELTA < 0 and A = 0
            t1 = (-B + torch.sqrt(DELTA.clip(eps))) / (2 * A.clip(eps))
            t2 = (-B - torch.sqrt(DELTA.clip(eps))) / (2 * A.clip(eps))
            if out_radius.any():
                tx = torch.min(torch.stack([torch.abs(t1[out_radius]), torch.abs(t2[out_radius])], dim=-1), dim=-1)[0]
            else:
                tx = torch.zeros_like(t[out_radius])
            t[out_radius] = t[out_radius] + tx
            
            decay_iter = 5
            while (torch.abs(res[ray.valid]) > self.TOLERANCE_TIGHT).any() and (it < self.MAXITER):
                it += 1
                alpha = 1. if it < decay_iter else 1. - (it - decay_iter) / it
                match mode:
                    case 'forward':
                        t = torch.where(ray.valid, alpha * (surf.surface(ox + t * dx, oy + t * dy) / dz) + (1 - alpha) * t, 0.)
                        res = surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
                    case 'reverse':
                        t = torch.where(ray.valid, alpha * (-surf.surface(ox + t * dx, oy + t * dy) / dz) + (1 - alpha) * t, 0.)
                        res = -surf.surface(ox + t * dx, oy + t * dy) - (t * dz)                
            t = torch.where(torch.abs(res) < self.TOLERANCE_LOOSE, t, 0.)
        
        match mode:
            case 'forward':
                t = surf.surface(ox + t * dx, oy + t * dy) / dz
            case 'reverse':
                t = -surf.surface(ox + t * dx, oy + t * dy) / dz
        # print(it)
        ray.o = ray.o + t.unsqueeze(-1) * ray.d
        # Judge invalid rays: no intersection or does not converge.
        match mode:
            case 'forward':
                ray.valid &= torch.abs(surf.surface(ox + t * dx, oy + t * dy) - (t * dz)) < self.TOLERANCE_LOOSE
            case 'reverse':
                ray.valid &= torch.abs(-surf.surface(ox + t * dx, oy + t * dy) - (t * dz)) < self.TOLERANCE_LOOSE
        # local coordinate of the previous surface
        return ray
    
    def newton(self, surf, ray:Ray, mode):
        """
        Newton method to find the root of the ray-surface intersection point.
        """
        ox, oy, oz = (ray.o[..., i] for i in range(3))
        dx, dy, dz = (ray.d[..., i] for i in range(3))
        
        with torch.no_grad():
            ox = torch.where(ray.valid, ox, 0.)
            oy = torch.where(ray.valid, oy, 0.)
            
            it = 0
            res = 1e8 * torch.ones_like(oz)
            match mode:
                case 'forward':
                    t = torch.ones_like(oz) * surf.surface_sag(self.surf_samp).min(dim=-1, keepdim=True)[0] / dz
                case 'reverse':
                    t = torch.ones_like(oz) * (-surf.surface_sag(self.surf_samp)).min(dim=-1, keepdim=True)[0] / dz
            ox_ = ox + t * dx
            oy_ = oy + t * dy
            out_radius = (ox_ ** 2 + oy_ ** 2) > surf.radius[None, :, :, None, None, None] ** 2
            A = dx ** 2 + dy ** 2
            B = 2 * (ox_ * dx + oy_ * dy)
            C = ox_ ** 2 + oy_ ** 2 - surf.radius[None, :, :, None, None, None] ** 2
            DELTA = B ** 2 - 4 * A * C
            #! Avoid DELTA < 0 and A = 0
            t1 = (-B + torch.sqrt(DELTA.clip(eps))) / (2 * A.clip(eps))
            t2 = (-B - torch.sqrt(DELTA.clip(eps))) / (2 * A.clip(eps))
            if out_radius.any():
                tx = torch.min(torch.stack([torch.abs(t1[out_radius]), torch.abs(t2[out_radius])], dim=-1), dim=-1)[0]
            else:
                tx = torch.zeros_like(t[out_radius])
            t[out_radius] = t[out_radius] + tx
            
            while (torch.abs(res[ray.valid]) > self.TOLERANCE_TIGHT).any() and (it < self.MAXITER):
                it += 1
                s_dx, s_dy, s_dz = surf.surface_d(ox + t * dx, oy + t * dy)
                if mode == 'reverse':
                    s_dx = -s_dx
                    s_dy = -s_dy
                df = s_dx * dx + s_dy * dy + s_dz * dz
                match mode:
                    case 'forward':
                        res = surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
                    case 'reverse':
                        res = -surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
                t = torch.where(ray.valid, t - (res / df), 0.)
            t = torch.where(torch.abs(res) < self.TOLERANCE_LOOSE, t, 0.)
            
        s_dx, s_dy, s_dz = surf.surface_d(ox + t * dx, oy + t * dy)
        if mode == 'reverse':
            s_dx = -s_dx
            s_dy = -s_dy
        df = s_dx * dx + s_dy * dy + s_dz * dz
        match mode:
            case 'forward':
                res = surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
            case 'reverse':
                res = -surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
        t = t - (res / df)
        # print(it)
        ray.o = ray.o + t.unsqueeze(-1) * ray.d
        # Judge invalid rays: no intersection or does not converge.
        ray.valid &= torch.abs(res) < self.TOLERANCE_LOOSE
        # local coordinate of the previous surface
        return ray
        
    def halley(self, surf, ray:Ray, mode):
        """
        Halley method to find the root of the ray-surface intersection point.
        """
        ox, oy, oz = (ray.o[..., i] for i in range(3))
        dx, dy, dz = (ray.d[..., i] for i in range(3))
        
        with torch.no_grad():
            ox = torch.where(ray.valid, ox, 0.)
            oy = torch.where(ray.valid, oy, 0.)
            
            it = 0
            res = 1e8 * torch.ones_like(oz)
            match mode:
                case 'forward':
                    t = torch.ones_like(oz) * surf.surface_sag(self.surf_samp).min(dim=-1, keepdim=True)[0] / dz
                case 'reverse':
                    t = torch.ones_like(oz) * (-surf.surface_sag(self.surf_samp)).max(dim=-1, keepdim=True)[0] / dz
            ox_ = ox + t * dx
            oy_ = oy + t * dy
            out_radius = (ox_ ** 2 + oy_ ** 2) > surf.radius[None, :, :, None, None, None] ** 2
            A = dx ** 2 + dy ** 2
            B = 2 * (ox_ * dx + oy_ * dy)
            C = ox_ ** 2 + oy_ ** 2 - surf.radius[None, :, :, None, None, None] ** 2
            DELTA = B ** 2 - 4 * A * C
            #! Avoid DELTA < 0 and A = 0
            t1 = (-B + torch.sqrt(DELTA.clip(eps))) / (2 * A.clip(eps))
            t2 = (-B - torch.sqrt(DELTA.clip(eps))) / (2 * A.clip(eps))
            if out_radius.any():
                tx = torch.min(torch.stack([torch.abs(t1[out_radius]), torch.abs(t2[out_radius])], dim=-1), dim=-1)[0]
            else:
                tx = torch.zeros_like(t[out_radius])
            t[out_radius] = t[out_radius] + tx
            
            while (torch.abs(res[ray.valid]) > self.TOLERANCE_TIGHT).any() and (it < self.MAXITER):
                it += 1
                match mode:
                    case 'forward':
                        res = surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
                    case 'reverse':
                        res = -surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
                
                s_dx, s_dy, s_dz = surf.surface_d(ox + t * dx, oy + t * dy)
                if mode == 'reverse':
                    s_dx = -s_dx
                    s_dy = -s_dy
                df = s_dx * dx + s_dy * dy + s_dz * dz
                
                s_ddx, s_ddy, s_ddxy = surf.surface_dd(ox + t * dx, oy + t * dy)
                if mode == 'reverse':
                    s_ddx = -s_ddx
                    s_ddy = -s_ddy
                    s_ddxy = -s_ddxy
                ddf = s_ddx * dx ** 2 + s_ddxy * 2 * dx * dy + s_ddy * dy * dy
                
                t = torch.where(ray.valid, t - (2 * res * df / (2 * df * df - res * ddf)), 0.)
            t = torch.where(torch.abs(res) < self.TOLERANCE_LOOSE, t, 0.)
            
        match mode:
            case 'forward':
                res = surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
            case 'reverse':
                res = -surf.surface(ox + t * dx, oy + t * dy) - (t * dz)
        s_dx, s_dy, s_dz = surf.surface_d(ox + t * dx, oy + t * dy)
        if mode == 'reverse':
            s_dx = -s_dx
            s_dy = -s_dy
        df = s_dx * dx + s_dy * dy + s_dz * dz
        s_ddx, s_ddy, s_ddxy = surf.surface_dd(ox + t * dx, oy + t * dy)
        if mode == 'reverse':
            s_ddx = -s_ddx
            s_ddy = -s_ddy
            s_ddxy = -s_ddxy
        ddf = s_ddx * dx ** 2 + s_ddxy * 2 * dx * dy + s_ddy * dy * dy
        t = t - (2 * res * df / (2 * df * df - res * ddf))
        # print(it)
        ray.o = ray.o + t.unsqueeze(-1) * ray.d
        # Judge invalid rays: no intersection or does not converge.
        ray.valid &= torch.abs(res) < self.TOLERANCE_LOOSE
        # local coordinate of the previous surface
        return ray