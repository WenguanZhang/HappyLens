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
        if surf.is_sphere():
            return self.analytic(surf, ray, mode)
        elif self.METHOD == 'newton':
            return self.newton(surf, ray, mode)
        else:
            raise Warning('METHOD does not exist!')

    def analytic(self, surf, ray:Ray, mode):
        """Intersect the vertex branch of a conic, including spheres and planes.

        At large fields the vertex plane can lie outside the clear aperture;
        an analytic intersection avoids an unreliable Newton starting point.
        Negative travel is allowed because rays use local vertex planes.
        """
        curvature = surf.roc[None, :, None, None, None, None]
        conic = surf.conic[None, :, None, None, None, None]
        if mode == 'reverse':
            curvature = -curvature
        origin, direction = ray.o, ray.d
        # c(x² + y²) + (1 + k)cz² - 2z = 0 along o + t d.
        a = curvature * (direction.square().sum(-1) + conic * direction[..., 2].square())
        b = 2 * (curvature * ((origin * direction).sum(-1) + conic * origin[..., 2] * direction[..., 2]) - direction[..., 2])
        c = curvature * (origin.square().sum(-1) + conic * origin[..., 2].square()) - 2 * origin[..., 2]
        discriminant = b.square() - 4 * a * c
        root = discriminant.clamp_min(1e-24).sqrt()
        q = -.5 * (b + torch.where(b >= 0, root, -root))
        # Planes and some parabolic/hyperbolic ray directions are linear.
        linear = a == 0
        safe_a = torch.where(linear, torch.ones_like(a), a)
        safe_q = torch.where(q.abs() > 1e-24, q, torch.ones_like(q))
        times = torch.stack((q / safe_a, c / safe_q), dim=-1)
        zs = origin[..., 2, None] + times * direction[..., 2, None]
        branch = 1 - (1 + conic[..., None]) * curvature[..., None] * zs >= -1e-10
        index = torch.where(branch, times.abs(), torch.inf).argmin(-1, keepdim=True)
        travel = times.gather(-1, index).squeeze(-1)
        travel = torch.where(linear, c / safe_q, travel)
        vertex = 1 - (1 + conic) * curvature * (origin[..., 2] + travel * direction[..., 2]) >= -1e-10
        valid = (discriminant > 0) & vertex & (~linear | (b.abs() > 1e-12))
        ray.valid &= valid & torch.isfinite(travel)
        ray.o = origin + travel[..., None] * direction
        return ray
    
    def newton(self, surf, ray:Ray, mode):
        """
        Newton method to find the root of the ray-surface intersection point.
        """
        ox, oy, oz = (ray.o[..., i] for i in range(3))
        dx, dy, dz = (ray.d[..., i] for i in range(3))
        
        # Keep position gradients and copy the mask before in-place updates.
        valid = ray.valid.clone()
        ox = torch.where(valid, ox, 0.)
        oy = torch.where(valid, oy, 0.)

        with torch.no_grad():
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
