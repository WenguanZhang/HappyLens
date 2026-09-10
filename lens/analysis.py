import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from functools import reduce

from .system import System
from .surface import Coordinate, Dummy
from .utils import Ray, normalize, RayleighSommerfeldPsfOp, CoherentPsfOp, eps

class Analysis(nn.Module):
    def __init__(self, system:System):
        super(Analysis, self).__init__()
        self.sys = system
        self.color_list = ['blue', 'green', 'red', 'brown', 'cyan', 'olive', 'orange', 'pink', 'purple', 'gray', 'gold', 'peru', 'navy', 'teal', 'coral', 'lavender', 'salmon', 'lime', 'indigo', 'aqua', 'tan']
        
    @torch.no_grad()
    def single_ray_trace(self, sys_id, cfg_id, ray:Ray):
        """
        Tracing a single ray with
        the normalized pupil coordinates [Px, Py],
        the [wavelength],
        the [view] (in degree),
        and use the [global coordinates] or [local coordinates]
        """
        view = torch.rad2deg(torch.arccos(ray.d[0, sys_id, cfg_id, 0, 0, 0, -1]))
        # propagate the ray and get its intersection on each surface
        ray, oss, dss = self.sys.propagate(ray, radius_flag=True, record=True)
        ###############################
        # print the trace of single ray
        ###############################
        # the head
        print('Ray Trace Data \n')
        print('Units         :   Millimeters')
        print('Wavelength    :   {:0.6f}  nm'.format(ray.wavelength[0] * 1e6))
        print('Field-of-View :   {:f} degree (represent in angle) '.format(view))

        # rays data
        print('Real Ray Trace Data: \n')
        itv = 4  # interval between rows
        print('Surf' + ' ' * itv + ' ' * 4 + 'X-coordinate' +
                       ' ' * itv + ' ' * 5 + 'Y-coordinate' +
                       ' ' * itv + ' ' * 5 + 'Z-coordinate' +
                       ' ' * itv + ' ' * 5 + 'X-cosine' +
                       ' ' * itv + ' ' * 5 + 'Y-cosine' +
                       ' ' * itv + ' ' * 5 + 'Z-cosine')
        surf_no = 0
        
        for idx in range(len(self.sys.extract_surfs())): # for oss and dss the 1 dimension is the surface index
            os, ds = oss[idx, 0, sys_id, cfg_id, 0, 0, 0, :], dss[idx, 0, sys_id, cfg_id, 0, 0, 0, :]
            msg = '{:>3d}'.format(surf_no)
   
            msg += ' ' * itv + '{:>17.10E}'.format(os[0].item()) + ' ' * itv + \
                    '{:>17.10E}'.format(os[1].item()) + \
                    ' ' * itv + '{:>17.10E}'.format(os[2].item())  # xyz coordinates
            msg += ' ' * itv + '{:>13.10f}'.format(ds[0].item()) + ' ' * itv + \
                    '{:>13.10f}'.format(ds[1].item()) + \
                    ' ' * itv + '{:>13.10f}'.format(ds[2].item())  # direction cosines

            print(msg)
            surf_no += 1
            
    @torch.no_grad()
    def sample_ray_1d(self, sampling, wavelength):
        """
        Just for drawing the system.
        For no tolerance system.
        Ray sampling (Consider y).
        The angle and azimuth are in degree.
        """
        if None in self.sys.system[0].distance:
            # Step1: Generate a bunch of rays to determine the sampling range.
            dz = torch.cos(torch.deg2rad(self.sys.max_view[..., None] * self.sys.norm_views[None, ...])) # [cfg, ang]
            dx = torch.zeros_like(dz)
            dy = torch.sin(torch.deg2rad(self.sys.max_view[..., None] * self.sys.norm_views[None, ...])) # [cfg, ang]
            d = normalize(torch.stack([dx, dy, dz], dim=-1))[None, :, :, None, None, :].repeat(self.sys.sys_num, 1, 1, 1, self.sys.surf_samp+1, 1)

            sample_radius = self.sys.system[1].radius[:, :, None] + self.sys.system[1].surface(self.sys.system[1].radius[None, :, :, None, None, None], 0)[0, :, :, :, 0, 0] * (dy / dz).abs()[None, :, :]
            oy = torch.cat((torch.tensor([0.]), torch.linspace(-1., 1., self.sys.surf_samp)))[None, None, None, :] * sample_radius[:, :, :, None] # [sys, cfg, ang, M]
            o = torch.stack([torch.zeros_like(oy), oy, torch.zeros_like(oy)], dim=-1)[:, :, :, None, :, :]
            
            ray = Ray(o=o, d=d, wavelength=wavelength)
            
            # Step2: Calculate the upper and lower sampling boundaries.
            ray = self.sys.propagate(ray)
            oy_min = torch.min(torch.where(ray.valid[0], oy[:, :, :, None, :], torch.inf), dim=-1)[0] # [sys, cfg, ang, 1]
            oy_max = torch.max(torch.where(ray.valid[0], oy[:, :, :, None, :], -torch.inf), dim=-1)[0] # [sys, cfg, ang, 1]
            
            # Step3: Generate valid rays
            oy = torch.linspace(0., 1., sampling)[None, None, None, None, :] * (oy_max - oy_min)[:, :, :, :, None] + oy_min[:, :, :, :, None] # [sys, cfg, ang, 1, M]
            o = torch.stack([torch.zeros_like(oy), oy, torch.zeros_like(oy)], dim=-1) # [sys, cfg, ang, 1, M, 3]
            d = normalize(torch.stack([dx, dy, dz], dim=-1))[None, :, :, None, None, :].repeat(self.sys.sys_num, 1, 1, 1, sampling, 1)
            ray = Ray(o=o, d=d, wavelength=wavelength)
        else:
            # Step1: Generate a bunch of rays to determine the sampling range.
            sample_radius = self.sys.system[1].radius # [sys, cfg]
            oy_surf = (torch.cat((torch.tensor([0.]), torch.linspace(-1., 1., self.sys.surf_samp)))[None, None, :] * sample_radius[:, :, None])[None, :, :, None, None, :] # [1, sys, cfg, 1, 1, M]
            ox_surf = torch.zeros_like(oy_surf)
            oz_surf = self.sys.system[1].surface(ox_surf, oy_surf)
            o_surf = torch.stack([ox_surf, oy_surf, oz_surf], dim=-1).squeeze(0) # [sys, cfg, 1, 1, M, 3]

            obj_dist = torch.tensor(self.sys.system[0].distance)[None, :] + self.sys.ENPP # [sys, cfg]
            oy = -obj_dist[:, :, None] * torch.tan(torch.deg2rad(self.sys.max_view[..., None] * self.sys.norm_views[None, ...]))[None, :, :]
            ox = torch.zeros_like(oy) # [sys, cfg, ang]
            oz = -torch.tensor(self.sys.system[0].distance)[None, :, None] * torch.ones_like(ox)
            o = torch.stack([ox, oy, oz], dim=-1)[:, :, :, None, None, :].repeat(1, 1, 1, 1, self.sys.surf_samp+1, 1) # [sys, cfg, ang, 1, M, 3]
            d = normalize(o_surf - o) # [sys, cfg, ang, 1, M, 3]
            
            ray = Ray(o=o, d=d, wavelength=wavelength)
            
            # Step2: Calculate the upper and lower sampling boundaries.
            ray = self.sys.propagate(ray)
            oy_surf_min = torch.min(torch.where(ray.valid[0], oy_surf[0].repeat(1, 1, len(self.sys.norm_views), 1, 1), torch.inf), dim=-1)[0][:, :, :, 0]
            oy_surf_max = torch.max(torch.where(ray.valid[0], oy_surf[0].repeat(1, 1, len(self.sys.norm_views), 1, 1), -torch.inf), dim=-1)[0][:, :, :, 0]
            
            # Step3: Generate valid rays
            oy_surf = (torch.linspace(0., 1., sampling)[None, None, None, :] * (oy_surf_max - oy_surf_min)[:, :, :, None] + oy_surf_min[:, :, :, None])[None, :, :, :, None, :]
            ox_surf = torch.zeros_like(oy_surf) # [wav, sys, cfg, ang, 1, M]
            oz_surf = self.sys.system[1].surface(ox_surf, oy_surf)
            o_surf = torch.stack([ox_surf, oy_surf, oz_surf], dim=-1).squeeze(0) # [sys, cfg, ang, 1, M, 3]
            o = torch.stack([ox, oy, oz], dim=-1)[:, :, :, None, None, :].repeat(1, 1, 1, 1, sampling, 1)
            d = normalize(o_surf - o)
            ray = Ray(o=o, d=d, wavelength=wavelength)
            
        return ray
            
    @torch.no_grad()
    def plot_setup_with_trace(self, sys_id=0, cfg_id=0, M=3):
        """Plot a non-tolerance system and rays in the global YZ view.

        Local YZ surface sections and rays are transformed before plotting;
        this is a sectional layout, not a full 3D silhouette renderer.
        """
        dtype = self.sys.wavelengths.dtype
        device = self.sys.wavelengths.device
        margin = 1 + self.sys.clear_margin

        def apply_coordinate(origin, basis, coordinate):
            angles = coordinate.tilt[sys_id, cfg_id]
            zero = angles.new_tensor(0.)
            one = angles.new_tensor(1.)
            cx, cy, cz = torch.cos(angles)
            sx, sy, sz = torch.sin(angles)
            rx = torch.stack((one, zero, zero,
                              zero, cx, -sx,
                              zero, sx, cx)).reshape(3, 3)
            ry = torch.stack((cy, zero, sy,
                              zero, one, zero,
                              -sy, zero, cy)).reshape(3, 3)
            rz = torch.stack((cz, -sz, zero,
                              sz, cz, zero,
                              zero, zero, one)).reshape(3, 3)
            shift = torch.cat((coordinate.decenter[sys_id, cfg_id], angles.new_zeros(1)))

            # Invert the transform order used by Coordinate.propagate.
            if coordinate.flag == 0:
                old_to_new = rz @ ry @ rx
                offset = old_to_new @ shift
            else:
                old_to_new = rx @ ry @ rz
                offset = shift

            new_to_old = old_to_new.transpose(0, 1)
            return (origin - basis @ new_to_old @ offset, basis @ new_to_old)

        def to_global(points, frame):
            origin, basis = frame
            return points @ basis.transpose(0, 1) + origin

        # Frames match System.propagate records, including auxiliary surfaces.
        origin = torch.zeros(3, dtype=dtype, device=device)
        basis = torch.eye(3, dtype=dtype, device=device)
        record_frames = [(origin, basis)]
        surface_frames = {}

        for surface_id, surface in enumerate(self.sys.system[1:-1], start=1):
            if isinstance(surface, Coordinate):
                origin, basis = apply_coordinate(origin, basis, surface)
            record_frames.append((origin, basis))
            if not isinstance(surface, Coordinate):
                surface_frames[surface_id] = record_frames[-1]
            if hasattr(surface, 'thickness'):
                origin = origin + basis[:, 2] * surface.thickness()[sys_id, cfg_id]

        record_frames.append((origin, basis))

        fig, ax = plt.subplots(figsize=(8, 6))

        def draw(points, color='k'):
            points = points.detach().cpu()
            ax.plot(points[..., 2], points[..., 1], color=color)

        def material_name(surface):
            material = getattr(surface, 'material', None)
            if material is None:
                return None
            names = material['name']
            return names if isinstance(names, str) else names[sys_id]

        def outer_radius(surface):
            if surface.aperture in ('circ', 'square'):
                return surface.max_r[sys_id]
            return surface.radius[sys_id, cfg_id]

        # Share barrel radii within coaxial lenses, not across folds or prisms.
        rim_radii = {}
        lens_group = []

        def finish_lens_group():
            if len(lens_group) > 1:
                radius = torch.stack([
                    torch.maximum(outer_radius(self.sys.system[i]), self.sys.system[i].radius[sys_id, cfg_id])
                    for i in lens_group
                ]).max()
                for i in lens_group:
                    rim_radii[i] = radius * margin
            lens_group.clear()

        for surface_id, frame in surface_frames.items():
            surface = self.sys.system[surface_id]
            name = material_name(surface)
            if (not hasattr(surface, 'radius') or name in (None, 'MIRROR') or surface.aperture == 'square'):
                finish_lens_group()
                continue
            if lens_group:
                first_origin, first_basis = surface_frames[lens_group[0]]
                offset = (frame[0] - first_origin) @ first_basis
                if (not torch.allclose(frame[1], first_basis, rtol=0., atol=1e-9)
                        or not torch.allclose(offset[:2], torch.zeros_like(offset[:2]), rtol=0., atol=1e-9)):
                    finish_lens_group()
            lens_group.append(surface_id)
            if name == 'VACUUM':
                finish_lens_group()
        finish_lens_group()

        def surface_segments(surface, rim_radius=None):
            aperture = getattr(surface, 'aperture', 'float')
            if aperture == 'circ':
                limits = ((-surface.max_r[sys_id], -surface.min_r[sys_id]),
                          (surface.min_r[sys_id], surface.max_r[sys_id]))
            else:
                radius = outer_radius(surface)
                limits = ((-radius, radius),)

            segments = []
            for lower, upper in limits:
                radial = torch.linspace(lower, upper, self.sys.surf_samp, dtype=dtype, device=device) * margin
                x, y = torch.zeros_like(radial), radial
                shaped_x = x[None, None, None, None, None, :].expand(1, 1, self.sys.cfg_num, 1, 1, -1)
                shaped_y = y[None, None, None, None, None, :].expand_as(shaped_x)
                sag = surface.surface(shaped_x, shaped_y)[0, sys_id, cfg_id, 0, 0]
                segments.append(torch.stack((x, y, sag), dim=-1))
            if rim_radius is not None:
                lower, upper = segments[0][0].clone(), segments[-1][-1].clone()
                lower[1], upper[1] = -rim_radius, rim_radius
                segments[0] = torch.cat((lower[None, :], segments[0]))
                segments[-1] = torch.cat((segments[-1], upper[None, :]))
            return segments

        # Connect outer edges through glass; mirrors do not change the medium.
        previous_edge = None
        medium = material_name(self.sys.system[0])
        for surface_id, surface in enumerate(self.sys.system[1:-1], start=1):
            if isinstance(surface, Dummy):
                continue
            if not hasattr(surface, 'surface') or not hasattr(surface, 'radius'):
                continue

            frame = surface_frames[surface_id]
            segments = [to_global(segment, frame) for segment in surface_segments(surface, rim_radii.get(surface_id))]
            name = material_name(surface)
            is_air_stop = (surface_id == self.sys.stop_id and name == 'VACUUM' and medium == 'VACUUM')
            if is_air_stop:
                stop_radius = outer_radius(surface)
                width = stop_radius / 10
                stop_lines = (
                    ((-width, stop_radius), (width, stop_radius)),
                    ((-width, -stop_radius), (width, -stop_radius)),
                    ((0., stop_radius), (0., stop_radius + width)),
                    ((0., -stop_radius), (0., -stop_radius - width)),
                )
                for line in stop_lines:
                    local = torch.zeros((2, 3), dtype=dtype, device=device)
                    local[0, 2], local[1, 2] = line[0][0], line[1][0]
                    local[0, 1], local[1, 1] = line[0][1], line[1][1]
                    draw(to_global(local, frame), 'r')
            else:
                for segment in segments:
                    draw(segment)

            edge = torch.stack((segments[0][0], segments[-1][-1]))
            if previous_edge is not None and medium not in (None, 'VACUUM', 'MIRROR'):
                direct = torch.linalg.vector_norm(previous_edge - edge, dim=-1).sum()
                crossed = torch.linalg.vector_norm(previous_edge - edge.flip(0), dim=-1).sum()
                if crossed < direct:
                    edge = edge.flip(0)
                draw(torch.stack((previous_edge, edge)))

            previous_edge = edge
            if name != 'MIRROR':
                medium = name

        image_radius = self.sys.system[-1].radius[sys_id, cfg_id]
        radial = torch.linspace(-image_radius, image_radius, self.sys.surf_samp, dtype=dtype, device=device)
        local_image = torch.zeros((self.sys.surf_samp, 3), dtype=dtype, device=device)
        local_image[:, 1] = radial
        draw(to_global(local_image, record_frames[-1]))
        ax.set_aspect(1)
        
        ray = self.sample_ray_1d(sampling=M, wavelength=self.sys.wavelengths[self.sys.p_wvl])
        ray, o_dic, _ = self.sys.propagate(ray, radius_flag=True, record=True)

        if len(record_frames) != o_dic.shape[0]:
            raise RuntimeError('layout frames do not match the recorded ray surfaces')

        # Hide auxiliary intersections on backward extensions of folded rays.
        visible_records = [0] + [i for i, s in enumerate(self.sys.system[1:-1], 1) if not isinstance(s, Dummy)] + [len(record_frames) - 1]

        for j in range(ray.o.shape[3]):
            valid = ray.valid[0, sys_id, cfg_id, j, 0, :]
            local_points = o_dic[:, 0, sys_id, cfg_id, j, 0, valid, :]
            if local_points.shape[1] == 0:
                continue
            global_points = torch.stack([
                to_global(local_points[i], record_frames[i])
                for i in visible_records
            ])
            draw(global_points, self.color_list[j % len(self.color_list)])

        ax.set_xlabel('Global Z (mm)')
        ax.set_ylabel('Global Y (mm)')
        ax.set_title(f'System {sys_id}, Configuration {cfg_id}\n'
                     f'{self.sys.EFFL[sys_id, cfg_id]:.2f} mm, {self.sys.max_view[cfg_id]:.2f}°, F{self.sys.FNO[sys_id, cfg_id]:.2f}',
                     fontsize=10, pad=5)
        ax.margins(x=0.02, y=0.02)
        fig.tight_layout(pad=0.4)

        # Trim equal-aspect whitespace without clipping labels or scaling axes.
        fig.canvas.draw()
        bounds = ax.get_tightbbox(fig.canvas.get_renderer()).padded(6)
        axes_bounds = ax.get_window_extent().frozen()
        fig.set_size_inches(bounds.width / fig.dpi, bounds.height / fig.dpi)
        ax.set_position([(axes_bounds.x0 - bounds.x0) / bounds.width,
                         (axes_bounds.y0 - bounds.y0) / bounds.height,
                         axes_bounds.width / bounds.width,
                         axes_bounds.height / bounds.height])
    
    @torch.no_grad()
    def psf(self, sys_id, cfg_id, pupil_samp, image_samp, image_delta, norm_view:float, azimuth:float, wavelength:float=None, vig:dict=None, split_channel=False, show=True):
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square', vig=vig)
        ray = self.sys.propagate(ray, radius_flag=True, record=False)
        # back to the exit pupil plane
        t_img_ep = self.sys.EXPP[None, :, :, None, None, None] / ray.d[..., 2]
        
        if wavelength == None:
            rel_o = ray.o[self.sys.p_wvl, sys_id, cfg_id, 0, 0][ray.chief_id[self.sys.p_wvl, sys_id, cfg_id, 0, 0]]
            waveweights = (self.sys.waveweights / self.sys.waveweights.sum()).tolist()
        else:
            rel_o = ray.o[0, sys_id, cfg_id, 0, 0][ray.chief_id[0, sys_id, cfg_id, 0, 0]]
            waveweights = [1.]
        
        image_delta = image_delta * 1e-3
        line_sample = torch.linspace(-int((image_samp - 1) / 2), int((image_samp - 1) / 2), image_samp) * image_delta
        y, x = torch.meshgrid(-line_sample, line_sample, indexing='ij')
        grid = rel_o + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
        
        if split_channel:
            psf_mul = torch.zeros((len(ray.wavelength), image_samp, image_samp))
        else:
            psf_mul = torch.zeros((image_samp, image_samp))
        for w, wave in enumerate(ray.wavelength):
            k = 2 * torch.pi / wave
            if 1: # 1: RS, 0: Coherent
                o = (ray.o[w, sys_id, cfg_id, 0, 0] + t_img_ep[w, sys_id, cfg_id, 0, 0][..., None] * ray.d[w, sys_id, cfg_id, 0, 0])[ray.valid[w, sys_id, cfg_id, 0, 0]]
                t = (ray.t[w, sys_id, cfg_id, 0, 0] + t_img_ep[w, sys_id, cfg_id, 0, 0])[ray.valid[w, sys_id, cfg_id, 0, 0]]
                s = -torch.sign(t_img_ep[w, sys_id, cfg_id, 0, 0][ray.valid[w, sys_id, cfg_id, 0, 0]])
                psf = RayleighSommerfeldPsfOp.apply(o, s * t, grid, s * k, -self.sys.EXPP[sys_id, cfg_id])
            else:
                o = ray.o[w, sys_id, cfg_id, 0, 0][ray.valid[w, sys_id, cfg_id, 0, 0]]
                d = ray.d[w, sys_id, cfg_id, 0, 0][ray.valid[w, sys_id, cfg_id, 0, 0]]
                t = ray.t[w, sys_id, cfg_id, 0, 0][ray.valid[w, sys_id, cfg_id, 0, 0]]
                psf = CoherentPsfOp.apply(o, d, grid, t, k)
                
            if split_channel:
                psf_mul[w, ...] = psf / psf.sum()
            else:
                psf_mul += psf / psf.sum() * waveweights[w]
        
        if show:
            if split_channel:
                for w in range(len(ray.wavelength)):
                    fig, ax = plt.subplots(figsize=(4, 3))
                    im = ax.imshow(psf_mul[w, ...].cpu(), cmap='jet')
                    plt.colorbar(im)
            else:
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(psf_mul.cpu(), cmap='jet')
                plt.colorbar(im)
        return psf_mul
    
    
    @torch.no_grad()
    def wavefront(self, sys_id, cfg_id, pupil_samp, norm_view:float, azimuth:float, wavelength:float=None, vig:dict=None, use_exit_pupil_shape=True, show=True):
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square', vig=vig)
        ray = self.sys.propagate(ray, radius_flag=True, record=False)        
        
        p_wvl = self.sys.p_wvl if wavelength == None else 0
        wave = ray.wavelength[p_wvl].item()
        
        rel_o = ray.o[p_wvl, sys_id, cfg_id, 0, 0][ray.chief_id[p_wvl, sys_id, cfg_id, 0, 0]] # [3]
        rel_d = ray.d[p_wvl, sys_id, cfg_id, 0, 0][ray.chief_id[p_wvl, sys_id, cfg_id, 0, 0]] # [3]
        t_chief = ray.t[p_wvl, sys_id, cfg_id, 0, 0][ray.chief_id[p_wvl, sys_id, cfg_id, 0, 0]] # [1]
        
        o = ray.o[p_wvl, sys_id, cfg_id, 0, 0] # [M, 3]
        d = ray.d[p_wvl, sys_id, cfg_id, 0, 0] # [M, 3]
        t = ray.t[p_wvl, sys_id, cfg_id, 0, 0] # [M]
        
        r_chief = -self.sys.EXPP[sys_id, cfg_id] / rel_d[2] # [1]
        
        A = 1.
        B = -2 * ((o - rel_o[None, :]) * d).sum(dim=-1) # [M]
        C = ((o - rel_o[None, :]) ** 2).sum(dim=-1) - r_chief ** 2 # [M]
        t1 = (-B + torch.sqrt(B ** 2 - 4 * A * C)) / (2 * A) # [M]
        t2 = (-B - torch.sqrt(B ** 2 - 4 * A * C)) / (2 * A) # [M]
        t_expp = torch.where(r_chief >= 0, t1, t2) # [M] select the exit-sphere side using the signed chief distance
        opd = (t_chief - r_chief) - (t - t_expp) # [M]
        
        if use_exit_pupil_shape:
            o_expp = o - t_expp[:, None] * d # [M, 3]
            chief_o_expp = rel_o - r_chief * rel_d # [3]
            
            o_expp = o_expp[ray.valid[p_wvl, sys_id, cfg_id, 0, 0]] - chief_o_expp # [M, 3]
            ox_expp, oy_expp, oz_expp = o_expp[:, 0], o_expp[:, 1], o_expp[:, 2]
            
            # rotate the plane (x axis)
            ori_d = torch.tensor([0., 0., 1.])
            cos_theta = (rel_d * ori_d).sum()
            sin_theta = torch.sqrt(1. - cos_theta ** 2)
            y = (ox_expp).cpu().detach().numpy()[1:] # [M]
            x = (oy_expp * cos_theta - oz_expp * sin_theta).cpu().detach().numpy()[1:] # [M]
            opd = opd[ray.valid[p_wvl, sys_id, cfg_id, 0, 0]].cpu().detach().numpy()[1:] # [M]
        
            from scipy.interpolate import griddata
            from scipy.spatial import Delaunay
            import numpy as np
            
            points = np.asarray([x, y]).T
            tri = Delaunay(points)
            
            # calculate the circumcenter radius of each triangle
            A, B, C = points[tri.simplices[:, 0]], points[tri.simplices[:, 1]], points[tri.simplices[:, 2]]
            c, a, b = np.linalg.norm(A - B, axis=1), np.linalg.norm(B - C, axis=1), np.linalg.norm(C - A, axis=1)
            s = (a + b + c) * 0.5
            area_sq = s * (s - a) * (s - b) * (s - c)
            radii = a * b * c / (4.0 * np.sqrt(area_sq))

            edge_lengths = np.concatenate([a, b, c]) # [X]
            alpha = np.median(edge_lengths) * 2.0 # threshold for circumcenter radius
            valid_triangles = (radii < alpha)
            
            # generate grid points for interpolation
            scale = (x ** 2 + y ** 2).max() ** 0.5
            xs, ys = np.mgrid[-1.:1.:1j * pupil_samp, -1.:1.:1j * pupil_samp]
            grid_points = np.column_stack([xs.ravel(), ys.ravel()]) * scale # [pupil_samp**2, 2]
            
            # find grid points that are inside the valid triangles
            tri_indices = tri.find_simplex(grid_points) # [pupil_samp**2]
            valid_mask = np.array([tri_idx >= 0 and valid_triangles[tri_idx] for tri_idx in tri_indices])
            
            # interpolate the wavefront data
            wf = np.full(len(grid_points), np.nan)
            wf[valid_mask] = griddata(points, opd, grid_points[valid_mask], method='linear') / wave
            wf = wf.reshape(pupil_samp, pupil_samp)
            
        else:
            opd = torch.where(ray.valid[p_wvl, sys_id, cfg_id, 0, 0], opd, torch.nan)
            wf = opd[1:].reshape(pupil_samp, pupil_samp) / wave
            wf = wf.T.cpu()
            opd = opd[ray.valid[p_wvl, sys_id, cfg_id, 0, 0]].cpu().detach().numpy()[1:] # [M]
        
        if show:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(wf, cmap='jet', extent=[-1., 1., -1., 1.], origin='lower', aspect='equal')
            ax.set_xlim(-1., 1.)
            ax.set_ylim(-1., 1.)
            ax.set_xlabel('x (normalized)')
            ax.set_ylabel('y (normalized)')
            ax.set_title(f'Wavefront (normalized pupil coordinates): {norm_view * self.sys.max_view[cfg_id]:.2f}°')
            plt.colorbar(im)
        
        return opd
    
    
    @torch.no_grad()
    def mtf(self, sys_id, cfg_id, pupil_samp, image_samp, image_delta, norm_view=None, azimuth=None, wavelength:float=None, vig:dict=None, freq_max=None, freq_delta=None, show=True):        
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square', vig=vig)
        ray = self.sys.propagate(ray, radius_flag=True, record=False)
        
        angle = self.sys.norm_views * self.sys.max_view[cfg_id] if norm_view is None else torch.tensor(norm_view) * self.sys.max_view[cfg_id]
        angle = angle.unsqueeze(0) if angle.numel() == 1 else angle
        azimuth = self.sys.azimuths if azimuth is None else azimuth
        azimuth = torch.tensor([azimuth]) if isinstance(azimuth, float) else azimuth
        
        # back to the exit pupil plane 
        t_img_ep = self.sys.EXPP[None, :, :, None, None, None] / ray.d[..., 2]
        
        image_delta = image_delta * 1e-3
        line_sample = torch.linspace(-int((image_samp - 1) / 2), int((image_samp - 1) / 2), image_samp) * image_delta
        y, x = torch.meshgrid(-line_sample, line_sample, indexing='ij')
        
        if freq_max is None:   
            freq_max = 1 / image_delta / 2
        if freq_delta is None:        
            freq_delta = 1
        freq_cut = 1 / image_delta / 2
        if freq_cut < freq_max:
            raise Exception('freq_max can not be larger than freq_cut: {} lp/mm'.format(freq_cut))
        
        num_points = int(freq_max / freq_delta) + 1
        T_all = torch.zeros([len(angle), len(azimuth), num_points])
        S_all = torch.zeros([len(angle), len(azimuth), num_points])
        freq = torch.linspace(0, freq_max, num_points)
        pad_points = int(freq_cut / freq_delta - image_samp / 2) + 1
        index = int(pad_points + image_samp / 2)
        
        for i, ang in enumerate(angle):
            for j, azi in enumerate(azimuth):
                if wavelength == None:
                    rel_o = ray.o[self.sys.p_wvl, sys_id, cfg_id, i, j][ray.chief_id[self.sys.p_wvl, sys_id, cfg_id, i, j]]
                    waveweights = (self.sys.waveweights / self.sys.waveweights.sum()).tolist()
                else:
                    rel_o = ray.o[0, sys_id, cfg_id, i, j][ray.chief_id[0, sys_id, cfg_id, i, j]]
                    waveweights = [1.]
            
                grid = rel_o + torch.stack([x, y, torch.zeros_like(x)], dim=-1)
        
                psf_mul = torch.zeros((image_samp, image_samp))
                for w, wave in enumerate(ray.wavelength):
                    k = 2 * torch.pi / wave
                    if 1: # 1: RS, 0: Coherent
                        o = (ray.o[w, sys_id, cfg_id, i, j] + t_img_ep[w, sys_id, cfg_id, i, j][..., None] * ray.d[w, sys_id, cfg_id, i, j])[ray.valid[w, sys_id, cfg_id, i, j]]
                        t = (ray.t[w, sys_id, cfg_id, i, j] + t_img_ep[w, sys_id, cfg_id, i, j])[ray.valid[w, sys_id, cfg_id, i, j]]
                        s = -torch.sign(t_img_ep[w, sys_id, cfg_id, i, j][ray.valid[w, sys_id, cfg_id, i, j]])
                        psf = RayleighSommerfeldPsfOp.apply(o, s * t, grid, s * k, -self.sys.EXPP[sys_id, cfg_id])
                    else:
                        o = ray.o[w, sys_id, cfg_id, i, j][ray.valid[w, sys_id, cfg_id, i, j]]
                        d = ray.d[w, sys_id, cfg_id, i, j][ray.valid[w, sys_id, cfg_id, i, j]]
                        t = ray.t[w, sys_id, cfg_id, i, j][ray.valid[w, sys_id, cfg_id, i, j]]
                        psf = CoherentPsfOp.apply(o, d, grid, t, k)
                    
                    psf_mul += psf / psf.sum() * waveweights[w]

                psf_pad = torch.nn.functional.pad(psf_mul, [pad_points, pad_points, pad_points, pad_points])
                T = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf_pad)))[:, index][index: index + num_points]
                S = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf_pad)))[index, :][index: index + num_points]
                
                T_all[i, j, :] = T
                S_all[i, j, :] = S
        
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            c = -1
            for i, ang in enumerate(angle):
                for j, azi in enumerate(azimuth):
                    c += 1
                    ax.plot(freq.cpu(), T_all[i, j, :].cpu(), '-', label=f'ang: {ang:.2f} - azi: {azi:.2f}: T', color=self.color_list[c])
                    ax.plot(freq.cpu(), S_all[i, j, :].cpu(), '-.', label=f'ang: {ang:.2f} - azi: {azi:.2f}: S', color=self.color_list[c])
            plt.xlim(0)
            plt.ylim(0)
            plt.grid()
            
            ax.set_ylabel('MTF')
            ax.set_xlabel('Frequency (lp/mm)')
            ax.xaxis.set_major_locator(plt.MaxNLocator(11))
            ax.yaxis.set_major_locator(plt.MaxNLocator(11))
            ax.tick_params(axis='both')
            plt.legend()
            
        return freq, T_all, S_all
    
    
    @torch.no_grad()
    def spot_diagram(self, sys_id=0, cfg_id=0, sampling=7, samp_method='ring'):
        rays = self.sys.sample_ray_2d(sampling, samp_method=samp_method, vig=self.sys.vig)
        rays = self.sys.propagate(rays, radius_flag=True, record=False)
        
        rays_mean = {} 
        spot_rms = {}
        lim = 0
        for i, view in enumerate(self.sys.norm_views * self.sys.max_view[cfg_id]):
            for j, azimuth in enumerate(self.sys.azimuths):
                rays_wxy = torch.tensor([])
                for k, wavelength in enumerate(self.sys.wavelengths):
                    rays_xy = rays.o[k, sys_id, cfg_id, i, j, :, 0:2][rays.valid[k, sys_id, cfg_id, i, j, :]]
                    rays_wxy = torch.cat([rays_wxy, rays_xy], dim=0)

                rays_mean[i, j] = rays_wxy.mean(dim=0) # [M, 2]
                spot_rms[i, j] = torch.sqrt(torch.mean(torch.sum((rays_wxy - rays_mean[i, j]) ** 2, dim=-1)))
                lim = torch.abs(rays_wxy - rays_mean[i, j]).max().cpu() if lim < torch.abs(rays_wxy - rays_mean[i, j]).max().cpu() else lim
        
        airy = 1.22 * self.sys.wavelengths[self.sys.p_wvl] * self.sys.FNO[sys_id, cfg_id]
        lim_max = (max([lim, airy]) * 2).cpu()
        
        fig, ax = plt.subplots(figsize=(4*len(self.sys.norm_views), 4*len(self.sys.azimuths)))
        plt.axis('off')
        x = 0
        for j, azimuth in enumerate(self.sys.azimuths):
            for i, view in enumerate(self.sys.norm_views * self.sys.max_view[cfg_id]):
                x += 1
                ax = plt.subplot(len(self.sys.azimuths), len(self.sys.norm_views), x)
                import matplotlib.patches as patches
                # create a circle
                circle = patches.Circle((0, 0), airy.cpu(), edgecolor='k', facecolor='none', linewidth=1)
                # add the circle to the plot
                ax.add_patch(circle)
                
                for k, wavelength in enumerate(self.sys.wavelengths):
                    rays_xy = rays.o[k, sys_id, cfg_id, i, j, :, 0:2][rays.valid[k, sys_id, cfg_id, i, j, :]] - rays_mean[i, j]
                    ax.scatter(rays_xy[:, 0].cpu(), rays_xy[:, 1].cpu(), s=0.1, color=self.color_list[k], label=f'{wavelength * 1e6:.2f} nm')
                
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlim([-lim_max, lim_max])
                ax.set_ylim([-lim_max, lim_max])

                units_str = '[mm]'
                ax.set_xlabel('x ' + units_str)
                ax.set_ylabel('y ' + units_str)
                ax.set_title(f'view: {view:.2f} degree, \n azimuth: {azimuth:.2f} degree \n RMS: {float(1000 * spot_rms[i, j]):.4f} um')

        plt.legend()
        plt.tight_layout()
    
    
    @torch.no_grad()
    def relative_illumination(self, sys_id, cfg_id, pupil_samp, field_samp, wavelength=None, show=True):
        #! not very accuracy
        def area(sample_d):
            # sample_d: [M, 2]
            s = 2
            delta = torch.sqrt(torch.sum((sample_d[None, :, :] - sample_d[:, None, :]) ** 2, axis=-1)) / s
            pixels = (2 / torch.sort(delta, dim=-1)[0][:, 1].mean()).int() + 1
            
            x_pixels = ((sample_d[..., 0] + 1) * pixels / 2).int()
            y_pixels = ((sample_d[..., 1] + 1) * pixels / 2).int()
        
            mask = torch.zeros(pixels, pixels)
            mask[y_pixels, x_pixels] = 1.

            kernel = torch.ones((1, 1, 2 * s + 1, 2 * s + 1))
            mask = torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0), kernel, padding=s)
            mask = (mask > 0.).to(kernel.dtype)
            
            mask = torch.nn.functional.conv2d(mask, kernel, padding=s)
            mask = (mask.squeeze().flip(0) == (2 * s + 1) ** 2).to(kernel.dtype)
            
            # plt.figure()
            # plt.imshow(mask.cpu(), cmap='gray')
            # plt.axis('off')
            
            return mask.sum() * (2. / pixels) ** 2
        
        angle = torch.linspace(0, 1, field_samp).tolist()
        rays = self.sys.sample_ray_2d(pupil_samp, angle, 0., wavelength)
        rays = self.sys.propagate(rays, radius_flag=True, record=False)
        
        rl = torch.zeros(rays.valid.shape[-3])
        for i in range(rays.valid.shape[-3]):
            d = rays.d[0, sys_id, cfg_id, i, 0, :, 0:2][rays.valid[0, sys_id, cfg_id, i, 0]]
            off_axis = area(d)
            rl[i] = off_axis
        rl = rl / rl[0]
        
        if show:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.plot([ang * self.sys.max_view[cfg_id].item() for ang in angle], rl.cpu())
            ax.set_ylim([0., 1.])
            ax.set_xlim([0., self.sys.max_view[cfg_id].item()])
            ax.set_xticks([ang * self.sys.max_view[cfg_id].item() for ang in angle])
        return rl
    
    
    @torch.no_grad()
    def distortion(self, sys_id, cfg_id, pupil_samp, field_samp, wavelength=None, show=True):
        # f-tan(theta) distortion
        surfs = self.sys.extract_surfs()
        abcds = [elem.abcd(surfs[i], self.sys.wavelengths[self.sys.p_wvl][..., None]) for i, elem in enumerate(surfs[1:-1])]
        abcd = reduce((lambda x, y: torch.matmul(y, x)), abcds)
        theta = torch.deg2rad(torch.tensor(1e-4) * self.sys.max_view)
        ini = torch.stack([-self.sys.ENPP * theta[None, :], torch.ones_like(self.sys.ENPP) * theta[None, :]], dim=-1)
        l = torch.matmul(abcd, ini.unsqueeze(-1))
        difl = l[:, :, 0, 0] / torch.tan(theta)
        
        angle = torch.linspace(0, 1, field_samp)
        rays = self.sys.sample_ray_2d(pupil_samp, angle.tolist(), 0., wavelength)
        rays = self.sys.propagate(rays, radius_flag=True, record=False)
        
        target_y = difl[:, :, None, None] * torch.tan(torch.deg2rad(angle[None, :] * self.sys.max_view[:, None]))[None, :, :, None] * torch.cos(torch.deg2rad(torch.tensor([0.])))[None, None, None, :]
        chief_y = torch.gather(rays.o[:, :, :, :, :, :, 1], dim=-1, index=rays.chief_id.unsqueeze(-1)).squeeze(-1)
        distortion_ms = (chief_y - target_y.unsqueeze(0)) / (target_y.unsqueeze(0) + eps) # [wav, sys, cfg, ang]
        
        if show:
            fig, ax = plt.subplots(figsize=(6, 6))
            for i in range(len(rays.wavelength)):
                im = ax.plot(distortion_ms[i, sys_id, cfg_id, :].cpu(), angle.cpu(), label=f'{rays.wavelength[i] * 1e6:.2f} nm')
                ax.set_ylim([0., 1.])
                plt.legend()
        return distortion_ms # [sys]
    
    
    @torch.no_grad()
    def save_analysis_results(self, path, sys_id=0, loss=0., samp_rays=3, samp_method='ring'):
        self.sys.save_json(sys_id, f'{path}/sys_{sys_id}_{loss:.4f}.json')
        for i in range(self.sys.cfg_num):
            self.plot_setup_with_trace(sys_id, i, 3)
            plt.savefig(f'{path}/setup_sys_{sys_id}_cfg_{i}_{loss:.4f}.svg', bbox_inches='tight')
            plt.close()
        
            self.spot_diagram(sys_id, i, samp_rays * 2 + 1, samp_method=samp_method)
            plt.savefig(f'{path}/spot_sys_{sys_id}_cfg_{i}_{loss:.4f}.svg', bbox_inches='tight')
            plt.close()
            
            
    @torch.no_grad()
    def psf_spot(self, sys_id, cfg_id, pupil_samp, image_samp, image_delta, norm_view:float, azimuth:float, wavelength:float=None, vig:dict=None, split_channel=False, show=True):
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square', vig=vig)
        ray = self.sys.propagate(ray, radius_flag=True, record=False)
        
        if wavelength == None:
            rel_o = -ray.o[self.sys.p_wvl, sys_id, cfg_id, 0, 0][ray.chief_id[self.sys.p_wvl, sys_id, cfg_id, 0, 0]]
            waveweights = (self.sys.waveweights / self.sys.waveweights.sum()).tolist()
        else:
            rel_o = -ray.o[0, sys_id, cfg_id, 0, 0][ray.chief_id[0, sys_id, cfg_id, 0, 0]]
            waveweights = [1.]
        
        image_delta = image_delta * 1e-3 # um -> mm
        psf_range = [-int((image_samp - 1) / 2) * image_delta, int((image_samp - 1) / 2) * image_delta]
        x_min, x_max = psf_range
        y_min, y_max = psf_range
        
        if split_channel:
            psf_mul = torch.zeros((len(ray.wavelength), image_samp, image_samp))
        else:
            psf_mul = torch.zeros((image_samp, image_samp))
        for w, wave in enumerate(ray.wavelength):
            points = -ray.o[w, sys_id, cfg_id, 0, 0, :, 0:2]
            
            point_shift = points - rel_o[None, 0:2]
            ra = ray.valid[w, sys_id, cfg_id, 0, 0] * (point_shift[..., 0].abs() < psf_range[1] - 0.1 * image_delta) * (point_shift[..., 1].abs() < psf_range[1] - 0.1 * image_delta)
            point_shift = point_shift * ra.unsqueeze(-1)
            
            points = point_shift

            # ==> Normalize points to the range [0, 1]
            points_normalized = torch.zeros_like(points)
            points_normalized[:, 0] = (points[:, 1] - y_max) / (y_min - y_max)
            points_normalized[:, 1] = (points[:, 0] - x_min) / (x_max - x_min)
            
            ks = image_samp
            # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
            pixel_indices_float = points_normalized * (ks - 1)
            w_b = pixel_indices_float[..., 0] - pixel_indices_float[..., 0].floor()
            w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()

            # ==> Pixel indices
            pixel_indices_tl = pixel_indices_float.floor().long()
            pixel_indices_tr = torch.stack((pixel_indices_float[:, 0], pixel_indices_float[:, 1]+1), dim=-1).floor().long()
            pixel_indices_bl = torch.stack((pixel_indices_float[:, 0]+1, pixel_indices_float[:, 1]), dim=-1).floor().long()
            pixel_indices_br = pixel_indices_tl + 1
            
            obliq = ray.d[w, sys_id, cfg_id, 0, 0, :, 2] ** 2
            grid = torch.zeros(ks, ks)
            grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*obliq, accumulate=True)
            grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*obliq, accumulate=True)
            grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*obliq, accumulate=True)
            grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*obliq, accumulate=True)

            psf = grid / grid.max()
            if split_channel:
                psf_mul[w, ...] = psf.flip(0).flip(1) / psf.sum()
            else:
                psf_mul += psf.flip(0).flip(1) / psf.sum() * waveweights[w]
        
        if show:
            if split_channel:
                for w in range(len(ray.wavelength)):
                    fig, ax = plt.subplots(figsize=(4, 3))
                    im = ax.imshow(psf_mul[w, ...].cpu(), cmap='jet')
                    plt.colorbar(im)
            else:
                fig, ax = plt.subplots(figsize=(4, 3))
                im = ax.imshow(psf_mul.cpu(), cmap='jet')
                plt.colorbar(im)
        return psf_mul
