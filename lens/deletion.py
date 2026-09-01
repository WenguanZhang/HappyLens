import torch
from functools import reduce

from .optim import Merit
from .utils import plot_loss_pie

class Deletion(Merit):
    def __init__(self, **kwargs):
        super(Deletion, self).__init__(**kwargs)
        
    def params_lr(self, lr):
        param_list = []
        # for deletion
        for name, params in self.sys.system.named_parameters():
            print(name)
            if name.endswith('roc'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif name.endswith('thick'):
                param_list.append({'params': params, 'lr': lr})
            elif name.endswith('conic'):
                param_list.append({'params': params, 'lr': lr})
        return param_list
    
    def forward_loss(self, del_id:list, args:dict, **kwargs):
        """
        writer: tensorboard writer
        count: iteration count
        path: save loss pie path
        """
        writer = kwargs.get('writer', None)
        count = kwargs.get('count', None)
        path = kwargs.get('path', None)

        self.propagate_all_rays()
        loss, label = [], []
        for func in args:
            label.append(func)
            match func:
                case 'EFL':
                    efl_loss = self.efl_loss(args[func]['target'])
                    if writer: writer.add_scalar('optics/A/efl_loss', efl_loss[self.sys.valid].mean(), count)
                    loss.append(efl_loss * args[func]['weight'])
                case 'FNO':
                    fno_loss = self.fno_loss(args[func]['target'])
                    if writer: writer.add_scalar('optics/A/fno_loss', fno_loss[self.sys.valid].mean(), count)
                    loss.append(fno_loss * args[func]['weight'])
                case 'SPOT':
                    k = args[func]['k'] if 'k' in args[func] else 1.
                    spot_loss = self.spot_loss(args[func]['ref'], k=k, efl=args['EFL']['target'])
                    if writer: writer.add_scalar('optics/A/spot_loss', spot_loss[self.sys.valid].mean(), count)
                    loss.append(spot_loss * args[func]['weight'])
                case 'DISTOR':
                    absolute = args[func]['abs'] if 'abs' in args[func] else True
                    distor_loss = self.distor_loss(args[func]['target'], absolute)
                    if writer: writer.add_scalar('optics/A/distor_loss', distor_loss[self.sys.valid].mean(), count)
                    loss.append(distor_loss * args[func]['weight'])
                case 'BFL':
                    bfl_loss = self.bfl_loss(args[func]['target'], del_id)
                    if writer: writer.add_scalar('optics/B/bfl_loss', bfl_loss[self.sys.valid].mean(), count)
                    loss.append(bfl_loss * args[func]['weight'])
                case 'TOTR':
                    totr_loss = self.totr_loss(args[func]['target'])
                    if writer: writer.add_scalar('optics/B/totr_loss', totr_loss[self.sys.valid].mean(), count)
                    loss.append(totr_loss * args[func]['weight'])
                case 'GLA_MIN_THICK':
                    gla_min_thick_loss = self.gla_min_thick_loss(args[func]['td_ratio'], del_id)
                    if writer: writer.add_scalar('optics/B/gla_min_thick_loss', gla_min_thick_loss[self.sys.valid].mean(), count)
                    loss.append(gla_min_thick_loss * args[func]['weight'])
                case 'GLA_MAX_THICK':
                    gla_max_thick_loss = self.gla_max_thick_loss(args[func]['td_ratio'], del_id)
                    if writer: writer.add_scalar('optics/B/gla_max_thick_loss', gla_max_thick_loss[self.sys.valid].mean(), count)
                    loss.append(gla_max_thick_loss * args[func]['weight'])
                case 'AIR_THICK':
                    air_thick_loss = self.air_thick_loss(args[func]['target'], del_id)
                    if writer: writer.add_scalar('optics/B/air_thick_loss', air_thick_loss[self.sys.valid].mean(), count)
                    loss.append(air_thick_loss * args[func]['weight'])
                case 'SURF_K':
                    surf_k_loss = self.surf_k_loss(args[func]['target'])
                    if writer: writer.add_scalar('optics/B/surf_k_loss', surf_k_loss[self.sys.valid].mean(), count)
                    loss.append(surf_k_loss * args[func]['weight'])
        
        if path: plot_loss_pie(loss, label, self.sys.valid, path)
        loss = reduce(lambda x, y: x + y, loss)
        return loss # [sys]

    def bfl_loss(self, target, del_id:list=[]):
        surfs = self.sys.extract_surfs()
        
        if del_id and del_id[1] == len(surfs) - 2: # if the del surf is the last surf
            elem_last = surfs[del_id[0] - 1]
            acc_thick = sum(elem.thick for elem in surfs[del_id[0]-1:del_id[1]+1]) # the accumulated thick from (the del first surf) to (the surf just after the del surf)
        else:
            elem_last = surfs[-2]
            acc_thick = elem_last.thick
        
        r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
        r = (r[None, None, :] * elem_last.radius[:, :, None])[None, :, :, None, None, :]
        sag = elem_last.surface(r, 0).max(dim=-1)[0].squeeze(dim=[0, 3, 4])
        sag_res = (acc_thick - sag) - target
        sag_loss = torch.where(sag_res < 0., -sag_res, 0.)
        return (sag_loss ** 2).sum(dim=-1)
    
    def gla_min_thick_loss(self, td_ratio, del_id:list=[]):
        edges_res = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, elem in enumerate(surfs[1:-1]):
            if 'VACUUM' not in elem.material['name']:
                if not del_id or ((i + 1) < del_id[0] or (i + 1) > del_id[1]):
                    elem_aft = surfs[i + 2]
                    r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                    r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :]
                    sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4])
                    sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                    edge_res = (elem.thick[:, :, None] - sag + sag_aft).min(dim=-1)[0] - td_ratio * torch.max(elem.radius, elem_aft.radius) * 2
                    edges_res = torch.cat([edges_res, torch.where(edge_res < 0, -edge_res, 0).unsqueeze(0)])
        return (edges_res ** 2).sum(dim=[0, -1])
    
    def gla_max_thick_loss(self, td_ratio, del_id:list=[]):
        edges_res = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, elem in enumerate(surfs[1:-1]):
            if 'VACUUM' not in elem.material['name']:
                if not del_id or ((i + 1) < del_id[0] or (i + 1) > del_id[1]):
                    elem_aft = surfs[i + 2]
                    r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                    r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :]
                    sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4])
                    sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                    edge_res = (elem.thick[:, :, None] - sag + sag_aft).max(dim=-1)[0] - td_ratio * torch.min(elem.radius, elem_aft.radius) * 2
                    edges_res = torch.cat([edges_res, torch.where(edge_res > 0, edge_res, 0).unsqueeze(0)])
        return (edges_res ** 2).sum(dim=[0, -1])
    
    def air_thick_loss(self, target, del_id:list=[]):
        airs_res = torch.tensor([])
        surfs = self.sys.extract_surfs()

        for i, elem in enumerate(surfs[1:-2]):  # skip the last air
            if 'VACUUM' in elem.material['name']: # start count from the air surf
                if i == 0 and self.sys.stop_id == 1: # if the stop is the first surface, skip it
                    r = elem.radius[None, :, :, None, None, None] # [wav, sys, cfg, ang, azi, M]
                    air_res = elem.thick + surfs[1].surface(r, 0).squeeze(dim=[0, 3, 4, 5]) # [sys, cfg]
                    airs_res = torch.cat([airs_res, torch.where(air_res < 0, -air_res, 0).unsqueeze(0)])
                    continue
                
                if del_id:
                    if (i + 2) == del_id[0]: # if the surf id is just before the del surf
                        if del_id[1] == len(surfs) - 2: # if the del surf is the last surf
                            continue
                        elem_aft = surfs[del_id[1] + 1] # the surf just after the del surf
                        acc_thick = sum(dsurf.thick for dsurf in surfs[del_id[0]-1:del_id[1]+1]) # the accumulated thick from (the del first surf) to (the surf just after the del surf)
                    elif (i + 1) == del_id[1]: # if the surf id is the del last surf
                        continue
                    else:
                        elem_aft = surfs[i + 2] # other common surf 
                        acc_thick = elem.thick
                else:
                    elem_aft = surfs[i + 2]
                    acc_thick = elem.thick

                r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :] # [wav, sys, cfg, ang, azi, M]
                sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4]) # [sys, cfg, M]
                sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                air_res = (acc_thick[:, :, None] - sag + sag_aft).min(dim=-1)[0] - target # [sys, cfg]
                airs_res = torch.cat([airs_res, torch.where(air_res < 0, -air_res, 0).unsqueeze(0)])
        return (airs_res ** 2).sum(dim=[0, -1])
    
    @torch.no_grad()
    def find_del_surfs(self):
        """
        Return the start and end surface index of the surfaces to be deleted.
        """
        surfs = self.sys.extract_surfs()
        ray = self.sys.sample_ray_2d(self.samp_rays, azimuth=0.)
        ray, o, d = self.sys.propagate(ray, radius_flag=True, record=True)
        
        s_efl = torch.zeros(len(surfs)-2)
        s_abr = torch.zeros(len(surfs)-2)
        
        # Calculate the contribution of each surface.
        for i, surf in enumerate(surfs[1:-1]):
            ox = torch.where(ray.valid, o[i + 1][..., 0], 0.)
            oy = torch.where(ray.valid, o[i + 1][..., 1], 0.)
            oz = torch.where(ray.valid, o[i + 1][..., 2], 0.)
            
            normal = -surf.inter_normal(ox, oy) # [wav, sys, cfg, ang, azi, M, 3]
            d_pre = d[i] # [wav, sys, cfg, ang, azi, M, 3]
            d_aft = d[i + 1] # [wav, sys, cfg, ang, azi, M, 3]
            
            n_pre = surfs[i].refractive_index(self.sys.wavelengths)
            n_aft = surf.refractive_index(self.sys.wavelengths)
            
            #! power contribution
            t_pre = torch.nan_to_num(torch.where(ray.valid, oy / d_pre[..., 1], 0.), 0.) # [wav, sys, cfg, ang, azi, M]
            t_aft = torch.nan_to_num(torch.where(ray.valid, oy / d_aft[..., 1], 0.), 0.) # [wav, sys, cfg, ang, azi, M]
            l_pre = torch.sqrt((ox - t_pre * d_pre[..., 0]) ** 2 + (oz - t_pre * d_pre[..., 2]) ** 2) # [wav, sys, cfg, ang, azi, M]
            l_aft = torch.sqrt((ox - t_aft * d_aft[..., 0]) ** 2 + (oz - t_aft * d_aft[..., 2]) ** 2) # [wav, sys, cfg, ang, azi, M]
            s_efl[i] = torch.nan_to_num((n_aft[..., None, None, None, None, None] / l_aft - n_pre[..., None, None, None, None, None] / l_pre) * normal[..., 1].abs(), 0.).sum()
        
            #! aberration contribution
            Ax = n_pre[..., None, None, None, None, None] * torch.sqrt(1 - torch.einsum('...k, ...k', normal, d_pre).clip(-1., 1.) ** 2) # [wav, sys, cfg, ang, azi, M]
            s_abr[i] = torch.where(ray.valid, Ax * (torch.arccos(d_aft[..., 2]) / n_aft[..., None, None, None, None, None] - torch.arccos(d_pre[..., 2]) / n_pre[..., None, None, None, None, None]), 0.).sum()
        
        # Extract elements surfs.
        elems = []
        start = 1
        while start < len(surfs)-2:
            if 'VACUUM' not in surfs[start].material['name']:
                for end in range(start, len(surfs)-2):
                    if 'VACUUM' in surfs[end+1].material['name']:
                        elems.append([start, end+1])
                        start = end + 1
                        break
                else:
                    break
            else:
                start += 1
        
        # Calculate power and aberration sum.
        e_efl = torch.zeros(len(elems))
        e_abr = torch.zeros(len(elems))
        for i, idx in enumerate(elems):
            e_efl[i] = s_efl[idx[0] - 1:idx[1]].sum()
            e_abr[i] = s_abr[idx[0] - 1:idx[1]].sum()
        
        # info
        # print(elems)
        # print(s_efl)
        # print(s_abr)
        # print(e_efl)
        # print(e_abr)
        
        # Return the del surf id.
        idx = torch.argmin(e_efl.abs() * e_abr.abs())
        if self.sys.stop_id in elems[idx]:
            _, indices = torch.sort(e_efl.abs() * e_abr.abs())
            idx = indices[1]
        return elems[idx] # list: [a, b]
    
    def del_surf_loss(self, del_id):
        surfs = self.sys.extract_surfs()
        reses = torch.tensor([])
        flats = torch.tensor([])
        for i in range(del_id[0], del_id[1]):
            elem = surfs[i]
            elem_aft = surfs[i + 1]
            r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
            r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :]
            sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4])
            sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
            reses = torch.cat([reses, (elem.thick[:, :, None] - sag + sag_aft).abs().max(dim=-1)[0].unsqueeze(0)]) # [surf_id, sys, cfg]
            flats = torch.cat([flats, sag.abs().max(dim=-1)[0].unsqueeze(0)]) # [surf_id, sys, cfg]
        flats = torch.cat([flats, sag_aft.abs().max(dim=-1)[0].unsqueeze(0)])
        return reses.amax(dim=[0, -1]), flats.amax(dim=[0, -1])