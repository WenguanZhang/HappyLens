import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm

import torch
import torch.nn as nn

from .system import System
from .surface import IMAGE
from .utils import length, RayleighSommerfeldPsfOp, CoherentPsfOp, eps, limit_var, plot_loss_pie, glass_catalog_params, plastic_catalog_params, find_key, zernike_wavefront

class Merit(nn.Module):
    def __init__(self, system:System, samp_rays=6):
        super(Merit, self).__init__()
        self.sys = system
        self.samp_rays = samp_rays * 2 + 1
        
    def params_lr(self, lr, scale=10.):
        param_list = []
        # for refinement
        for name, params in self.sys.system.named_parameters():
            print(name)
            if name.endswith('roc'):
                param_list.append({'params': params, 'lr': lr / scale})
            elif name.endswith('thick'):
                param_list.append({'params': params, 'lr': lr * scale})
            elif name.endswith('conic'):
                param_list.append({'params': params, 'lr': lr})
            elif name.endswith('g1'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif name.endswith('g2'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif 'ai' in name:
                param_list.append({'params': params, 'lr': lr * scale ** -(int(name.split('.ai')[-1]) - 1)})
            elif 'qi' in name:
                param_list.append({'params': params, 'lr': lr})
            elif name.endswith('decenter'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
            elif name.endswith('tilt'):
                param_list.append({'params': params, 'lr': lr * 1e-1})
        return param_list
    
    def forward_loss(self, args:dict, **kwargs):
        """
        writer: tensorboard writer.
        count: iteration count.
        path: save loss pie path.
        """
        writer = kwargs.get('writer', None)
        count = kwargs.get('count', None)
        path = kwargs.get('path', None)

        self.propagate_all_rays()
        valids = torch.ones_like(self.sys.valid).bool()
        valids &= torch.gather(self.v_dic, -1, self.chief_id_dic.unsqueeze(-1)).squeeze(-1).prod(0).prod(-1).prod(-1).prod(-1).bool() # [sys]
        valids &= (self.v_dic.sum(dim=-1) > 2).prod(0).prod(-1).prod(-1).prod(-1).bool()
        loss, label = [], []
        for func in args:
            label.append(func)
            match func:
                case 'EFL':
                    efl_loss = self.efl_loss(args[func]['target'])
                    valids &= ~torch.isnan(efl_loss)
                    if writer: writer.add_scalar('optics/A/efl_loss', efl_loss[valids].mean(), count)
                    loss.append(efl_loss * args[func]['weight'])
                case 'FNO':
                    fno_loss = self.fno_loss(args[func]['target'])
                    valids &= ~torch.isnan(fno_loss)
                    if writer: writer.add_scalar('optics/A/fno_loss', fno_loss[valids].mean(), count)
                    loss.append(fno_loss * args[func]['weight'])
                case 'SPOT':
                    k = args[func]['k'] if 'k' in args[func] else 1.
                    spot_loss = self.spot_loss(args[func]['ref'], k=k, efl=args['EFL']['target'])
                    valids &= ~torch.isnan(spot_loss)
                    if writer: writer.add_scalar('optics/A/spot_loss', spot_loss[valids].mean(), count)
                    loss.append(spot_loss * args[func]['weight'])
                case 'WAVEFRONT':
                    wavefront_loss = self.wavefront_loss(args[func]['mode'])
                    valids &= ~torch.isnan(wavefront_loss)
                    if writer: writer.add_scalar('optics/A/wavefront_loss', wavefront_loss[valids].mean(), count)
                    loss.append(wavefront_loss * args[func]['weight'])
                case 'DISTOR':
                    absolute = args[func]['abs'] if 'abs' in args[func] else True
                    distor_loss = self.distor_loss(args[func]['target'], absolute)
                    valids &= ~torch.isnan(distor_loss)
                    if writer: writer.add_scalar('optics/A/distor_loss', distor_loss[valids].mean(), count)
                    loss.append(distor_loss * args[func]['weight'])
                case 'LATERAL':
                    lateral_loss = self.lateral_loss(args[func]['ref'])
                    valids &= ~torch.isnan(lateral_loss)
                    if writer: writer.add_scalar('optics/A/lateral_loss', lateral_loss[valids].mean(), count)
                    loss.append(lateral_loss * args[func]['weight'])
                case 'BFL':
                    bfl_loss = self.bfl_loss(args[func]['target'])
                    valids &= ~torch.isnan(bfl_loss)
                    if writer: writer.add_scalar('optics/B/bfl_loss', bfl_loss[valids].mean(), count)
                    loss.append(bfl_loss * args[func]['weight'])
                case 'TOTR':
                    totr_loss = self.totr_loss(args[func]['target'])
                    valids &= ~torch.isnan(totr_loss)
                    if writer: writer.add_scalar('optics/B/totr_loss', totr_loss[valids].mean(), count)
                    loss.append(totr_loss * args[func]['weight'])
                case 'GLA_MIN_THICK':
                    gla_min_thick_loss = 0
                    if args[func].get('td_ratio') is not None:
                        gla_min_thick_loss = gla_min_thick_loss + self.gla_min_thick_loss(td_ratio=args[func]['td_ratio'])
                    if args[func].get('min_thick') is not None:
                        gla_min_thick_loss = gla_min_thick_loss + self.gla_min_thick_loss(min_thick=args[func]['min_thick'])
                    valids &= ~torch.isnan(gla_min_thick_loss)
                    if writer: writer.add_scalar('optics/B/gla_min_thick_loss', gla_min_thick_loss[valids].mean(), count)
                    loss.append(gla_min_thick_loss * args[func]['weight'])
                case 'GLA_MAX_THICK':
                    gla_max_thick_loss = 0
                    if args[func].get('td_ratio') is not None:
                        gla_max_thick_loss = gla_max_thick_loss + self.gla_max_thick_loss(td_ratio=args[func]['td_ratio'])
                    if args[func].get('max_thick') is not None:
                        gla_max_thick_loss = gla_max_thick_loss + self.gla_max_thick_loss(max_thick=args[func]['max_thick'])
                    valids &= ~torch.isnan(gla_max_thick_loss)
                    if writer: writer.add_scalar('optics/B/gla_max_thick_loss', gla_max_thick_loss[valids].mean(), count)
                    loss.append(gla_max_thick_loss * args[func]['weight'])
                case 'GLA_MAX_MIN_RATIO':
                    gla_max_min_ratio_loss = self.gla_max_min_ratio_loss(args[func]['max_ratio'])
                    valids &= ~torch.isnan(gla_max_min_ratio_loss)
                    if writer: writer.add_scalar('optics/B/gla_max_min_ratio_loss', gla_max_min_ratio_loss[valids].mean(), count)
                    loss.append(gla_max_min_ratio_loss * args[func]['weight'])
                case 'SAG_DIA_MAX_RATIO':
                    sag_dia_max_ratio_loss = self.sag_dia_max_ratio_loss(args[func]['max_ratio'])
                    valids &= ~torch.isnan(sag_dia_max_ratio_loss)
                    if writer: writer.add_scalar('optics/B/sag_dia_max_ratio_loss', sag_dia_max_ratio_loss[valids].mean(), count)
                    loss.append(sag_dia_max_ratio_loss * args[func]['weight'])
                case 'AIR_THICK':
                    air_thick_loss = self.air_thick_loss(args[func]['target'])
                    valids &= ~torch.isnan(air_thick_loss)
                    if writer: writer.add_scalar('optics/B/air_thick_loss', air_thick_loss[valids].mean(), count)
                    loss.append(air_thick_loss * args[func]['weight'])
                case 'SURF_K':
                    surf_k_loss = self.surf_k_loss(args[func]['target'])
                    valids &= ~torch.isnan(surf_k_loss)
                    if writer: writer.add_scalar('optics/B/surf_k_loss', surf_k_loss[valids].mean(), count)
                    loss.append(surf_k_loss * args[func]['weight'])
                case 'ANGLE':
                    angle_loss = self.angle_loss(args[func]['target'])
                    valids &= ~torch.isnan(angle_loss)
                    if writer: writer.add_scalar('optics/C/angle_loss', angle_loss[valids].mean(), count)
                    loss.append(angle_loss * args[func]['weight'])
                case 'CRA':
                    cra_loss = self.cra_loss(args[func]['target'])
                    valids &= ~torch.isnan(cra_loss)
                    if writer: writer.add_scalar('optics/C/cra_loss', cra_loss[valids].mean(), count)
                    loss.append(cra_loss * args[func]['weight'])
                case 'ANGLE_STD':
                    angle_std_loss = self.angle_std_loss()
                    valids &= ~torch.isnan(angle_std_loss)
                    if writer: writer.add_scalar('optics/C/angle_std_loss', angle_std_loss[valids].mean(), count)
                    loss.append(angle_std_loss * args[func]['weight'])
                case 'PUPIL':
                    pupil_loss = self.pupil_loss(args[func]['ref_point_n'])
                    valids &= ~torch.isnan(pupil_loss)
                    if writer: writer.add_scalar('optics/C/pupil_loss', pupil_loss[valids].mean(), count)
                    loss.append(pupil_loss * args[func]['weight'])
                case 'GLA_Z':
                    gla_z_loss = self.gla_z_loss(args[func]['z_min'])
                    valids &= ~torch.isnan(gla_z_loss)
                    if writer: writer.add_scalar('optics/C/gla_z_loss', gla_z_loss[valids].mean(), count)
                    loss.append(gla_z_loss * args[func]['weight'])
                case 'SURF_GAP':
                    surf_gap_loss = []
                    for i in args[func]:                    
                        surf_gap_loss.append(self.surf_gap_loss(args[func][i]['s_pre'], args[func][i]['s_aft'], args[func][i]['target'], args[func][i]['mode']) * args[func][i]['weight']) 
                    surf_gap_loss = reduce(lambda x, y: x + y, surf_gap_loss)
                    valids &= ~torch.isnan(surf_gap_loss)
                    if writer: writer.add_scalar('optics/C/surf_gap_loss', surf_gap_loss[valids].mean(), count)
                    loss.append(surf_gap_loss)
                case 'ROC':
                    roc_loss = []
                    for i in args[func]:
                        roc_loss.append(self.roc_loss(args[func][i]['s_id'], args[func][i]['sign']) * args[func][i]['weight'])
                    roc_loss = reduce(lambda x, y: x + y, roc_loss)
                    valids &= ~torch.isnan(roc_loss)
                    if writer: writer.add_scalar('optics/C/roc_loss', roc_loss[valids].mean(), count)
                    loss.append(roc_loss)
        
        self.sys.valid = valids
        if path: plot_loss_pie(loss, label, self.sys.valid, path)
        loss = reduce(lambda x, y: x + y, loss)
        return loss # [sys]
    
    @torch.no_grad()
    def update_system(self, rmax=None, avg_cfg=False, fit_material=True, update_radius=True, quick_focus=True):
        """
        Update the material / paraxial data / surface radius / image position.
        """
        surfs = self.sys.extract_surfs()
        if fit_material:
            for idx in range(1, len(surfs)-1): self.sys.material_fit(idx)
        self.sys.update()
        if update_radius: self.update_radius(rmax)
        self.sys.update()
        if quick_focus: self.quickfocus(avg_cfg)
        self.sys.update()
    
    @torch.no_grad()
    def update_stop_radius(self, target:list|float, stop_fix=True):
        """
        Update the stop radius of the system.
        target: [cfg]
        Remember to use the self.update_system() after this function.
        """
        target = torch.tensor(target)[None, ...].repeat(self.sys.sys_num, 1) # [sys, cfg]
        if stop_fix:
            scale = (self.sys.FNO / target).amax(dim=-1, keepdim=True) # [sys]
        else:
            scale = (self.sys.FNO / target)
        self.sys.system[self.sys.stop_id].radius = self.sys.system[self.sys.stop_id].radius * scale
        self.sys.update()
        
    @torch.no_grad()
    def reborn_bad_system(self, args, optimizer=None):
        """
        Generate new systems to replace the invalid ones.
        args: dict for loss function.
        radius: [sys, cfg]
        roc: [sys]
        thick: [sys, cfg]
        conic: [sys]
        g1: [sys]
        g2: [sys]
        ai/qi: [sys]
        others: to be continue
        """
        loss = self.forward_loss(args) # get valid and invalid systems
        idx = torch.nan_to_num(loss, torch.inf).argsort() # small -> large
        
        def revise_param(param, idx):
            valid_data = param.data[self.sys.valid] # [M, cfg] or [M]
            valid_num = valid_data.shape[0]
            invalid_num = param.data.shape[0] - valid_num # [N]
            
            if invalid_num < valid_num:
                param.data[~self.sys.valid] = param[idx[:invalid_num]]
            else:
                if param.dim() == 1:
                    revise_data = valid_data.repeat(invalid_num // valid_num + 1)
                else:
                    revise_data = valid_data.repeat(invalid_num // valid_num + 1, 1) # [X, cfg]
                param.data[~self.sys.valid] = revise_data[:invalid_num]
        
        # correct invalid system
        surfs = self.sys.extract_surfs()
        for i, surf in enumerate(surfs[1:-1]):
            revise_param(surf.radius, idx)
            revise_param(surf.thick, idx)
            revise_param(surf.roc, idx)
            revise_param(surf.conic, idx)
            revise_param(surf.min_r, idx)
            revise_param(surf.max_r, idx)
            
            if hasattr(surf, 'ai_num'):
                for j in range(surf.ai_num):
                    revise_param(eval(f'surf.ai{2 * j + 4}'), idx)
            
            if hasattr(surf, 'qi_num'):
                revise_param(surf.rnorm, idx)
                for j in range(surf.qi_num):
                    revise_param(eval(f'surf.qi{j}'), idx)
            
            if hasattr(surf, 'g1') and hasattr(surf, 'g2'):
                revise_param(surf.g1, idx)
                revise_param(surf.g2, idx)
        self.sys.update()
        
        if optimizer:
            for i in optimizer.state.keys():
                for j in optimizer.state[i].keys():
                    if optimizer.state[i][j].shape:
                        revise_param(optimizer.state[i][j], idx)
                
    @torch.no_grad()
    def random_flip_elements(self, args, mask=None, p_fix=0.2, top_pick=False):
        if mask is None: mask = torch.zeros(self.sys.sys_num).bool()
        ori_data = self.sys.extract_opt_data()

        surfs = self.sys.extract_surfs()
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
        num_elems = len(elems)
        
        if top_pick:
            loss = self.forward_loss(args) # [sys]
            _, top_id = torch.topk(loss, int(self.sys.sys_num * p_fix), largest=False)
        else:
            top_id = torch.randperm(self.sys.sys_num)[0:int(self.sys.sys_num * p_fix)]
        
        new_data = self.sys.extract_opt_data()
        for i in range(self.sys.sys_num):
            if (i in top_id) or mask[i]:
                continue
            else:
                elem_id = torch.randint(0, num_elems, (1,)).item()
                s_start, s_end = elems[elem_id]
                if s_start + 1 == s_end:
                    x = new_data[s_start-1]['roc'][i].detach().clone()
                    new_data[s_start-1]['roc'][i] = -new_data[s_end-1]['roc'][i]
                    new_data[s_end-1]['roc'][i] = -x
                    
                if s_start + 2 == s_end:
                    x = new_data[s_start-1]['roc'][i].detach().clone()
                    new_data[s_start-1]['roc'][i] = -new_data[s_end-1]['roc'][i]
                    new_data[s_end-1]['roc'][i] = -x
                    new_data[s_start]['roc'][i] = -new_data[s_start]['roc'][i]
                    
                    thi = new_data[s_start-1]['thick'][i].detach().clone()
                    new_data[s_start-1]['thick'][i] = new_data[s_start]['thick'][i]
                    new_data[s_start]['thick'][i] = thi
                    
                    if new_data[s_start-1].get('g1', None) is not None and new_data[s_start].get('g1', None) is not None:
                        g1 = new_data[s_start-1]['g1'][i].detach().clone()
                        new_data[s_start-1]['g1'][i] = new_data[s_start]['g1'][i]
                        new_data[s_start]['g1'][i] = g1
                        
                        g2 = new_data[s_start-1]['g2'][i].detach().clone()
                        new_data[s_start-1]['g2'][i] = new_data[s_start]['g2'][i]
                        new_data[s_start]['g2'][i] = g2
                        
                    else:
                        g1 = self.sys.system[s_start].g1[i].detach().clone()
                        g2 = self.sys.system[s_start].g2[i].detach().clone()
                        self.sys.system[s_start].g1.data[i] = self.sys.system[s_end-1].g1[i].detach().clone()
                        self.sys.system[s_start].g2.data[i] = self.sys.system[s_end-1].g2[i].detach().clone()
                        self.sys.system[s_end-1].g1.data[i] = g1
                        self.sys.system[s_end-1].g2.data[i] = g2
                        
                r = self.sys.system[s_start].radius[i].detach().clone()
                self.sys.system[s_start].radius.data[i] = self.sys.system[s_end].radius[i].detach().clone()
                self.sys.system[s_end].radius.data[i] = r
                    
        self.sys.fit_opt_data(new_data)
        self.sys.update()
        
        loss = self.forward_loss(args)
        for s in ori_data:
            for key in ori_data[s]:
                new_data[s][key][~self.sys.valid] = ori_data[s][key][~self.sys.valid]
        self.sys.fit_opt_data(new_data)
        for idx in range(1, len(surfs)-1): self.sys.material_fit(idx)
        self.sys.update()
        
    @torch.no_grad()
    def random_change_materials(self, args, mask=None, p_fix=0.2, top_pick=False):
        if mask is None: mask = torch.zeros(self.sys.sys_num).bool()
        ori_data = self.sys.extract_opt_data()

        surfs = self.sys.extract_surfs()
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
        num_elems = len(elems)
        
        if top_pick:
            loss = self.forward_loss(args) # [sys]
            _, top_id = torch.topk(loss, int(self.sys.sys_num * p_fix), largest=False)
        else:
            top_id = torch.randperm(self.sys.sys_num)[0:int(self.sys.sys_num * p_fix)]
        
        new_data = self.sys.extract_opt_data()
        for i in range(self.sys.sys_num):
            if (i in top_id) or mask[i]:
                continue
            else:
                elem_id = torch.randint(0, num_elems, (1,)).item()
                s_start, _ = elems[elem_id]
                s_start = s_start - 1
                if 'g1' in new_data[s_start].keys() and 'g2' in new_data[s_start].keys():
                    if surfs[s_start+1].mat_cata == 'G':
                        catalog_params = glass_catalog_params
                    elif surfs[s_start+1].mat_cata == 'P':
                        catalog_params = plastic_catalog_params
                    else:
                        raise ValueError(f'Unknown material type {surfs[s_start+1].mat_cata}')
                    index = torch.randint(0, catalog_params.shape[0], (1,)).item()
                    new_data[s_start]['g1'][i] = catalog_params[index, 0].clone().detach() # [sys]
                    new_data[s_start]['g2'][i] = catalog_params[index, 1].clone().detach()
                    
        self.sys.fit_opt_data(new_data)
        self.sys.update()
        
        loss = self.forward_loss(args)
        for s in ori_data:
            for key in ori_data[s]:
                new_data[s][key][~self.sys.valid] = ori_data[s][key][~self.sys.valid]
        self.sys.fit_opt_data(new_data)
        for idx in range(1, len(surfs)-1): self.sys.material_fit(idx)
        self.sys.update()
        
    @torch.no_grad()
    def random_perturb_roc_thick(self, args, scale, mask=None, p_fix=0.2, top_pick=False):
        if mask is None: mask = torch.zeros(self.sys.sys_num).bool()
        ori_data = self.sys.extract_opt_data()
        surfs = self.sys.extract_surfs()
        surfs_id = []
        for idx, surf in enumerate(surfs):
            if surf.__class__.__name__ in ['Sphere', 'Asphere', 'Qbfs', 'Qcon']:
                if surf.roc.requires_grad and surf.thick.requires_grad:
                    surfs_id.append(idx)

        if top_pick:
            loss = self.forward_loss(args) # [sys]
            _, top_id = torch.topk(loss, int(self.sys.sys_num * p_fix), largest=False)
        else:
            top_id = torch.randperm(self.sys.sys_num)[0:int(self.sys.sys_num * p_fix)]
        
        new_data = self.sys.extract_opt_data()
        for i in range(self.sys.sys_num):
            if (i in top_id) or mask[i]:
                continue
            else:
                x = torch.randint(0, len(surfs_id), (1,)).item()
                surf_id = surfs_id[x]
                new_data[surf_id-1]['roc'][i] = new_data[surf_id-1]['roc'][i] * scale * (torch.rand(1) * 2 - 1)
                
                x = torch.randint(0, len(surfs_id), (1,)).item()
                surf_id = surfs_id[x]
                new_data[surf_id-1]['thick'][i] = new_data[surf_id-1]['thick'][i] * scale * torch.rand(1)
        
        self.sys.fit_opt_data(new_data)
        self.sys.update()
        
        loss = self.forward_loss(args)
        for s in ori_data:
            for key in ori_data[s]:
                new_data[s][key][~self.sys.valid] = ori_data[s][key][~self.sys.valid]
        self.sys.fit_opt_data(new_data)
        self.sys.update()
        
    @torch.no_grad()
    def random_perturb_Qtype(self, args, scale, mask=None, p_fix=0.2, top_pick=False):
        if mask is None: mask = torch.zeros(self.sys.sys_num).bool()
        ori_data = self.sys.extract_opt_data()
        surfs = self.sys.extract_surfs()
        q_surf = []
        for idx, surf in enumerate(surfs):
            if surf.__class__.__name__ == 'Qbfs' or surf.__class__.__name__ == 'Qcon':
                q_surf.append(idx)
        if q_surf == []:
            return
        
        if top_pick:
            loss = self.forward_loss(args) # [sys]
            _, top_id = torch.topk(loss, int(self.sys.sys_num * p_fix), largest=False)
        else:
            top_id = torch.randperm(self.sys.sys_num)[0:int(self.sys.sys_num * p_fix)]
        
        new_data = self.sys.extract_opt_data()
        for i in range(self.sys.sys_num):
            if (i in top_id) or mask[i]:
                continue
            else:
                x = torch.randint(0, len(q_surf), (1,)).item()
                surf_id = q_surf[x]
                
                configurations = [
                    (9, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                    (8, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
                    (7, [0, 1, 2, 3, 4, 5, 6, 7]),
                    (6, [0, 1, 2, 3, 4, 5, 6]),
                    (5, [0, 1, 2, 3, 4, 5]),
                    (4, [0, 1, 2, 3, 4]),
                    (3, [0, 1, 2, 3]),
                    (2, [0, 1, 2]),
                    (1, [0, 1]),
                ]
                
                for max_exp, all_exps in configurations:
                    key = f'qi{max_exp}'
                    if key in new_data[surf_id-1]:
                        q = torch.randint(0, len(all_exps), (1,)).item()
                        new_data[surf_id-1][f'qi{q}'][i] = new_data[surf_id-1][f'qi{q}'][i] * (1. + scale * (torch.rand(1) * 2 - 1))
                        break
        
        self.sys.fit_opt_data(new_data)
        self.sys.update()
        
        loss = self.forward_loss(args)
        for s in ori_data:
            for key in ori_data[s]:
                new_data[s][key][~self.sys.valid] = ori_data[s][key][~self.sys.valid]
        self.sys.fit_opt_data(new_data)
        self.sys.update()
    
    @torch.no_grad()
    def random_perturb_asphere(self, args, mask=None, p_fix=0.2, epsilon=1.e-2, sampling=101, top_pick=False):
        if mask is None: mask = torch.zeros(self.sys.sys_num).bool()
        ori_data = self.sys.extract_opt_data()
        surfs = self.sys.extract_surfs()
        asp = []
        for idx, surf in enumerate(surfs):
            if surf.__class__.__name__ == 'Asphere':
                asp.append(idx)
        if asp == []:
            return
        
        if top_pick:
            loss = self.forward_loss(args) # [sys]
            _, top_id = torch.topk(loss, int(self.sys.sys_num * p_fix), largest=False)
        else:
            top_id = torch.randperm(self.sys.sys_num)[0:int(self.sys.sys_num * p_fix)]
        
        new_data = self.sys.extract_opt_data()
        for i in range(self.sys.sys_num):
            if (i in top_id) or mask[i]:
                continue
            else:
                x = torch.randint(0, len(asp), (1,)).item()
                surf_id = asp[x]
                radius = surfs[surf_id].radius[i].max()

                configurations = [
                    (16, [4, 6, 8, 10, 12, 14, 16]),
                    (14, [4, 6, 8, 10, 12, 14]),
                    (12, [4, 6, 8, 10, 12]),
                    (10, [4, 6, 8, 10]),
                    (8, [4, 6, 8]),
                    (6, [4, 6]),
                ]
                
                ai_list = torch.tensor([])
                for max_exp, all_exps in configurations:
                    key = f'ai{max_exp}'
                    if key in new_data[surf_id-1]:
                        r = torch.linspace(0, radius, sampling)
                        M = torch.stack([r ** exp for exp in all_exps]).T
                        for exp in all_exps:
                            if exp <= max_exp:
                                ai_list = torch.cat([ai_list, new_data[surf_id-1][f'ai{exp}'][i][None, ...]])
                        
                        U, s, Vt = torch.linalg.svd(M, full_matrices=False)
                        
                        k = 2
                        V_small = Vt.T[:, -k:]
                        weights = torch.randn(k)
                        weights /= torch.linalg.norm(weights)
                        v_rand = V_small @ weights
                        v_rand /= torch.norm(v_rand)

                        Mv = M @ v_rand
                        scale = epsilon / torch.max(torch.abs(Mv))
                        delta = scale * v_rand
                        
                        for exp_id, exp in enumerate(all_exps):
                            exp_key = f'ai{exp}'
                            new_data[surf_id-1][exp_key][i] += delta[exp_id]
                        break
        
        self.sys.fit_opt_data(new_data)
        self.sys.update()
        
        loss = self.forward_loss(args)
        for s in ori_data:
            for key in ori_data[s]:
                new_data[s][key][~self.sys.valid] = ori_data[s][key][~self.sys.valid]
        self.sys.fit_opt_data(new_data)
        self.sys.update()
    
    def propagate_all_rays(self):
        ray = self.sys.sample_ray_2d(self.samp_rays, vig=self.sys.vig)
        ray, o, d = self.sys.propagate(ray, radius_flag=True, record=True)
        #! dataset
        self.o_dic = o # [surf, wav, sys, cfg, ang, azi, M, 3]
        self.d_dic = d # [surf, wav, sys, cfg, ang, azi, M, 3]
        self.t_dic = ray.t # [wav, sys, cfg, ang, azi, M]
        self.v_dic = ray.valid # [wav, sys, cfg, ang, azi, M]
        self.chief_id_dic = ray.chief_id # [wav, sys, cfg, ang, azi]
    
    @torch.no_grad()
    def update_radius(self, rmax=None, only_img=False):
        """
        Remove tolerances from the system to update the radius.
        """
        ray = self.sys.sample_ray_2d(self.samp_rays * 20 + 1, vig=self.sys.vig, samp_method='line')
        # propagate to the first surface
        ray = self.sys.system[0].propagate(ray)
        o = ray.o.unsqueeze(0)
        
        system = self.sys.extract_surfs()
        for i, elem in enumerate(system[1:self.sys.stop_id]):
            o_s, _, ray = elem.propagate(ray, system[i], radius_flag=elem.fix_radius)
            o = torch.cat([o, o_s.unsqueeze(0)], dim=0)
            
        o_s, _, ray = system[self.sys.stop_id].propagate(ray, system[self.sys.stop_id-1], radius_flag=True)
        o = torch.cat([o, o_s.unsqueeze(0)], dim=0)
        
        for i, elem in enumerate(system[self.sys.stop_id+1:-1]):
            o_s, _, ray = elem.propagate(ray, system[i+self.sys.stop_id], radius_flag=elem.fix_radius)
            o = torch.cat([o, o_s.unsqueeze(0)], dim=0) # [surf, wav, sys, cfg, ang, azi, M, 3]
        
        o = torch.cat([o, ray.o.unsqueeze(0)], dim=0)
        if only_img:
            self.sys.system[-1].radius = torch.where(ray.valid, length(o[-1, :, :, :, :, :, :, 0:2]), -torch.inf).amax(dim=[0, 3, 4, 5]) # [sys, cfg]
            return
        
        for s, elem in enumerate(system[1:]):
            if (s+1 != self.sys.stop_id):
                radius = torch.where(ray.valid, length(o[s+1, :, :, :, :, :, :, 0:2]), -torch.inf).amax(dim=[0, 3, 4, 5]) # [sys, cfg]
                if isinstance(elem, IMAGE):
                    elem.radius = radius.amax(dim=-1, keepdim=True).repeat(1, self.sys.cfg_num)
                else:
                    if elem.fix_radius == False:
                        elem.radius = radius.amax(dim=-1, keepdim=True).repeat(1, self.sys.cfg_num).clamp(min=0., max=rmax)
                
    @torch.no_grad()
    def quickfocus(self, avg_cfg=False):
        surfs = self.sys.extract_surfs()
        ray = self.sys.sample_ray_2d(self.samp_rays, vig=self.sys.vig)
        ray = self.sys.propagate(ray, radius_flag=True, record=False)
        
        ox, oy, vx, vy = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        ox = torch.where(ray.valid, ray.o[..., 0], torch.nan)
        oy = torch.where(ray.valid, ray.o[..., 1], torch.nan)
        ox = ox - torch.nanmean(ox, dim=[0, -1], keepdim=True)
        oy = oy - torch.nanmean(oy, dim=[0, -1], keepdim=True)
        
        dx = torch.where(ray.valid, ray.d[..., 0], torch.nan)
        dy = torch.where(ray.valid, ray.d[..., 1], torch.nan)
        dz = torch.where(ray.valid, ray.d[..., 2], torch.nan)
        vx = dx / dz - torch.nanmean(dx, dim=[0, -1], keepdim=True) / torch.nanmean(dz, dim=[0, -1], keepdim=True)
        vy = dy / dz - torch.nanmean(dy, dim=[0, -1], keepdim=True) / torch.nanmean(dz, dim=[0, -1], keepdim=True)
        
        if avg_cfg:
            distance = - (ox * vx + oy * vy).nansum(dim=[0, 2, 3, 4, 5])[:, None] / (vx ** 2 + vy ** 2).nansum(dim=[0, 2, 3, 4, 5])[:, None]
        else:
            distance = - (ox * vx + oy * vy).nansum(dim=[0, 3, 4, 5]) / (vx ** 2 + vy ** 2).nansum(dim=[0, 3, 4, 5])
        surfs[-2].thick.data = surfs[-2].thick + distance
    
    
    @torch.no_grad()
    def genetic_system(self, args, iters, elitism_rate=0.1, mutation_rate=0.1, mutation_strength=0.1, writer=None, count=None):
        """
        iters: number of iterations
        elitism_rate: percentage of elite systems to carry over to the next generation
        mutation_rate: probability of mutation for each parameter
        mutation_strength: standard deviation of the Gaussian noise added during mutation
        """
        def crossover(data):
            batch_size = data.shape[0]
            alpha = torch.rand(batch_size // 2) if data.dim() == 1 else torch.rand(batch_size // 2, 1)
            child1 = alpha * data[:-1:2] + (1 - alpha) * data[1::2]
            child2 = alpha * data[1::2] + (1 - alpha) * data[:-1:2]
            child = torch.cat([child1, child2], dim=0)
            return child
        
        def mutate(data, rate, step):
            batch_size = data.shape[0]
            mask = torch.rand(batch_size) < rate
            noise = mask * step * (torch.rand(batch_size) - 0.5) * 2
            if data.dim() == 1:
                data_new = data + noise
            else:
                data_new = data + noise[:, None]
            return data_new
    
        elitism_num = max(int(elitism_rate * self.sys.sys_num), 1)

        if hasattr(self.sys, 'zoom_type'):
            if None in self.sys.zoom_type:
                avg_cfg = False
            else:
                avg_cfg = (self.sys.zoom_type[-2][-1] == 'F')
        else:
            avg_cfg = False
        self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=True, update_radius=False, quick_focus=True)
            
        opt_data_min = self.sys.extract_opt_data()
        fit_material_flag = False if find_key(opt_data_min, 'g1') == [] and find_key(opt_data_min, 'g2') == [] else True
        loss_min = self.forward_loss(args)
        valid_min = self.sys.valid
        loss_min = torch.nan_to_num(loss_min, torch.inf)
        
        pbar = tqdm(total=iters)
        for i in range(iters):
            opt_data = self.sys.extract_opt_data()
            loss = self.forward_loss(args) # [sys]
            pbar.set_description_str(f'iters: {i}, loss_min: {loss_min.mean().item():.4f}, loss: {loss.mean().item():.4f}')
            pbar.update(1)
            
            # record the best system
            valid = (loss < loss_min) & self.sys.valid
            for s in opt_data:
                for key in opt_data[s]:
                    opt_data_min[s][key][valid] = opt_data[s][key][valid]
            loss_min[valid] = loss[valid]
            if writer is not None and count is not None:
                count += 1
                writer.add_scalar('total/loss', loss_min[valid_min | valid].min(), count)
                writer.add_scalar('total/loss_mean', loss_min[valid_min | valid].mean(), count)
                writer.add_scalar('total/valid', (valid_min | valid).sum(), count)
            
            # select parents
            max_val = torch.max(loss[self.sys.valid])
            inverted_fitness = (max_val - loss + eps)[self.sys.valid]
            probabilities = inverted_fitness / torch.sum(inverted_fitness)
            indices = torch.multinomial(probabilities, len(probabilities), replacement=True)
            parents = {s: {key: opt_data[s][key][self.sys.valid][indices].detach().clone() for key in opt_data[s]} for s in opt_data}
            
            # create new generation
            new_data = {s: {key: torch.zeros_like(opt_data[s][key]) for key in opt_data[s]} for s in opt_data}
            elite_indices = torch.argsort(loss[self.sys.valid])[:elitism_num]
            for s in opt_data:
                for key in opt_data[s]:
                    new_data[s][key][:elitism_num] = opt_data[s][key][self.sys.valid][elite_indices]
            
            indices = torch.randperm(len(probabilities))
            for s in parents:
                for key in parents[s]:
                    match key:
                        case 'thick':
                            strength = mutation_strength * 1e1
                        case 'conic':
                            strength = mutation_strength * 1e0
                        case 'roc':
                            strength = mutation_strength * 1e-1
                        case 'g1':
                            strength = mutation_strength * 1e-2
                        case 'g2':
                            strength = mutation_strength * 1e-2
                        case _:
                            if key.startswith('ai'):
                                x = int(key.split('ai')[1]) // 2
                                strength = mutation_strength * 1e-1 ** x
                            elif key.startswith('qi'):
                                strength = mutation_strength * 1e-3
                        
                    child = crossover(parents[s][key][indices])
                    child = mutate(child, mutation_rate, strength)
                    
                    if child.shape[0] >= self.sys.sys_num - elitism_num:
                        new_data[s][key][elitism_num:] = child[:self.sys.sys_num - elitism_num]
                    else:
                        remain = self.sys.sys_num - elitism_num - child.shape[0]
                        indices = torch.randint(0, len(probabilities), [remain])
                        remain_data = mutate(parents[s][key][indices], mutation_rate, strength)
                        new_data[s][key][elitism_num:] = torch.cat([child, remain_data], dim=0)
                
            self.sys.fit_opt_data(new_data)
            self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)
        
        self.sys.fit_opt_data(opt_data_min)
        self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)
        
        if writer is not None and count is not None:
            return count
    
    @torch.no_grad()
    def differential_evolution_system(self, args, iters, F=0.5, CR=0.5, writer=None, count=None):
        """
        iters: number of iterations
        F: differential weight
        CR: crossover probability
        """
        if hasattr(self.sys, 'zoom_type'):
            if None in self.sys.zoom_type:
                avg_cfg = False
            else:
                avg_cfg = (self.sys.zoom_type[-2][-1] == 'F')
        else:
            avg_cfg = False
        self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=True, update_radius=False, quick_focus=True)
        
        opt_data = self.sys.extract_opt_data()
        fit_material_flag = False if find_key(opt_data, 'g1') == [] and find_key(opt_data, 'g2') == [] else True
        
        pbar = tqdm(total=iters)
        for i in range(iters):
            opt_data = self.sys.extract_opt_data()
            if writer is not None and count is not None:
                count += 1
                loss = self.forward_loss(args, writer=writer, count=count) # [sys]
                writer.add_scalar('total/loss', loss[self.sys.valid].min(), count)
                writer.add_scalar('total/loss_mean', loss[self.sys.valid].mean(), count)
                writer.add_scalar('total/valid', self.sys.valid.sum(), count)
            else:
                loss = self.forward_loss(args) # [sys]
            pbar.set_description_str(f'iters: {i}, F: {F}, CR: {CR}, loss_min: {loss.min().item():.4f}, loss: {loss.mean().item():.4f}')
            pbar.update(1)
            
            sol1 = torch.randperm(self.sys.sys_num)
            sol2 = torch.randperm(self.sys.sys_num)
            sol3 = torch.randperm(self.sys.sys_num)
            
            # create new generation
            new_data = {s: {key: torch.zeros_like(opt_data[s][key]) for key in opt_data[s]} for s in opt_data}
            for s in opt_data:
                for key in opt_data[s]:
                    if opt_data[s][key].dim() == 1:
                        new_data[s][key] = opt_data[s][key][sol1] + F * (opt_data[s][key][sol2] - opt_data[s][key][sol3])
                    else:
                        new_data[s][key] = opt_data[s][key][sol1] + F * (opt_data[s][key][sol2] - opt_data[s][key][sol3])
                    K = torch.rand(self.sys.sys_num) < CR
                    new_data[s][key][~K] = opt_data[s][key][~K]

            self.sys.fit_opt_data(new_data)
            self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)
            loss_new = self.forward_loss(args) # [sys]
            
            valid = (loss_new < loss) & self.sys.valid
            for s in opt_data:
                for key in opt_data[s]:
                    opt_data[s][key][valid] = new_data[s][key][valid]
            
            self.sys.fit_opt_data(opt_data)
            self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)
        
        if writer is not None and count is not None:
            return count
    
    @torch.no_grad()
    def differential_evolution_system_shade(self, args, iters, F=0.5, CR=0.5, c=10, p=0.2, writer=None, count=None):
        """
        iters: number of iterations
        F: differential weight
        CR: crossover probability
        c: channel of F and CR
        p: percentage of top solutions
        """
        F = F * torch.ones(c)
        CR = CR * torch.ones(c)
        
        if hasattr(self.sys, 'zoom_type'):
            if None in self.sys.zoom_type:
                avg_cfg = False
            else:
                avg_cfg = (self.sys.zoom_type[-2][-1] == 'F')
        else:
            avg_cfg = False
        self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=True, update_radius=False, quick_focus=True)
        
        opt_data = self.sys.extract_opt_data()
        fit_material_flag = False if find_key(opt_data, 'g1') == [] and find_key(opt_data, 'g2') == [] else True
        archive_data = {s: {key: opt_data[s][key].detach().clone() for key in opt_data[s]} for s in opt_data}
            
        pbar = tqdm(total=iters)
        for i in range(iters):
            opt_data = self.sys.extract_opt_data()
            if writer is not None and count is not None:
                count += 1
                loss = self.forward_loss(args, writer=writer, count=count) # [sys]
                writer.add_scalar('total/loss', loss[self.sys.valid].min(), count)
                writer.add_scalar('total/loss_mean', loss[self.sys.valid].mean(), count)
                writer.add_scalar('total/valid', self.sys.valid.sum(), count)
            else:
                loss = self.forward_loss(args) # [sys]
            pbar.set_description_str(f'iters: {i}, F: {F[i % c]}, CR: {CR[i % c]}, loss_min: {loss.min().item():.4f}, loss: {loss.mean().item():.4f}')
            pbar.update(1)
            
            sol1 = torch.randperm(self.sys.sys_num)
            sol2 = torch.randperm(self.sys.sys_num)
            _, top_id = torch.topk(loss, min(int(self.sys.sys_num * p) + 1, self.sys.sys_num), largest=False)
            nice_id = top_id[torch.randint(0, len(top_id), [self.sys.sys_num])]
            
            # create new generation
            r = torch.randint(0, c, [self.sys.sys_num])
            
            samp_F = torch.hstack([torch.Tensor.cauchy_(torch.ones(1), F[x], 0.1).clip(1e-16, 1.) for x in r])
            samp_CR = (CR[r] + torch.randn(self.sys.sys_num) * 0.1).clip(0., 1.)
            
            new_data = {s: {key: torch.zeros_like(opt_data[s][key]) for key in opt_data[s]} for s in opt_data}
            for s in opt_data:
                for key in opt_data[s]:
                    data = archive_data[s][key][sol2] if torch.rand(1) < 0.5 else opt_data[s][key][sol2]
                    if opt_data[s][key].dim() == 1:
                        new_data[s][key] = opt_data[s][key] + samp_F * (opt_data[s][key][sol1] - data) + samp_F * (opt_data[s][key][nice_id] - opt_data[s][key])
                    else:
                        new_data[s][key] = opt_data[s][key] + samp_F[:, None] * (opt_data[s][key][sol1] - data) + samp_F[:, None] * (opt_data[s][key][nice_id] - opt_data[s][key])
                    K = torch.rand(self.sys.sys_num) < samp_CR
                    new_data[s][key][~K] = opt_data[s][key][~K]

            self.sys.fit_opt_data(new_data)
            self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)
            loss_new = self.forward_loss(args) # [sys]
            
            valid = (loss_new < loss) & self.sys.valid
            for s in opt_data:
                for key in opt_data[s]:
                    archive_data[s][key][valid] = opt_data[s][key][valid]
                    opt_data[s][key][valid] = new_data[s][key][valid]
                    
            wk = (loss - loss_new)[valid]
            wk = wk / (wk.sum() + eps)
                    
            # update F and CR
            if valid.any():
                CR[i % c] = (samp_CR[valid] * wk).sum()
                F[i % c] = (samp_F[valid] ** 2 * wk).sum() / (samp_F[valid] * wk).sum()
            
            self.sys.fit_opt_data(opt_data)
            self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)
    
        if writer is not None and count is not None:
            return count
    
    @torch.no_grad()
    def simulated_annealing_system(self, args, T=10, T_min=1, step=0.001, alpha=0.9, iter=20, ptresh=0.5, writer=None, count=None):
        """
        T: initial temperature
        T_min: minimum temperature
        step: step size
        alpha: temperature decay rate
        iter: number of iterations
        ptresh: probability threshold for opt data perturbation
        """
        @torch.no_grad()
        def perturb_opt_data(opt_data, step, p):
            """
            Perturb the opt data.
            p: float (0~1)
            step: float
            """
            opt_data_new = {}
            for i in opt_data:
                opt_data_new[i] = {}
                for key in opt_data[i]:
                    match key:
                        case 'thick':
                            one_step = step * 1e1
                        case 'conic':
                            one_step = step * 1e0
                        case 'roc':
                            one_step = step * 1e-1
                        case 'g1':
                            one_step = step * 1e-2
                        case 'g2':
                            one_step = step * 1e-2
                        case _:
                            if key.startswith('ai'):
                                x = int(key.split('ai')[1]) // 2
                                one_step = step * 1e-1 ** x
                            elif key.startswith('qi'):
                                one_step = step * 1e-3
                            
                    data = opt_data[i][key]
                    mask = torch.rand(self.sys.sys_num) < p
                    noise = mask * one_step * (torch.rand(self.sys.sys_num) - 0.5) * 2
                    if data.dim() == 1:
                        opt_data_new[i][key] = data + noise
                    else:
                        opt_data_new[i][key] = data + noise[:, None]
            return opt_data_new
  
        k = 1
        if hasattr(self.sys, 'zoom_type'):
            if None in self.sys.zoom_type:
                avg_cfg = False
            else:
                avg_cfg = (self.sys.zoom_type[-2][-1] == 'F')
        else:
            avg_cfg = False
        self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=True, update_radius=False, quick_focus=True)
        
        opt_data_min = self.sys.extract_opt_data()
        loss_min = self.forward_loss(args)
        valid_min = self.sys.valid
        loss_min = torch.nan_to_num(loss_min, torch.inf)
        fit_material_flag = False if find_key(opt_data_min, 'g1') == [] and find_key(opt_data_min, 'g2') == [] else True
        
        pbar = tqdm()
        while T >= T_min:
            for i in range(iter):
                opt_data = self.sys.extract_opt_data()
                loss = self.forward_loss(args) # [sys]
                
                opt_data_new = perturb_opt_data(opt_data, step * T, ptresh)
                self.sys.fit_opt_data(opt_data_new)
                self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=True, update_radius=False, quick_focus=True)
        
                opt_data_new = self.sys.extract_opt_data()
                loss_new = self.forward_loss(args) # [sys]
                
                valid = (loss_new < loss_min) & self.sys.valid
                for s in opt_data:
                    for key in opt_data[s]:
                        opt_data_min[s][key][valid] = opt_data_new[s][key][valid]
                loss_min[valid] = loss_new[valid]
                if writer is not None and count is not None:
                    count += 1
                    writer.add_scalar('total/loss', loss_min[valid_min | valid].min(), count)
                    writer.add_scalar('total/loss_mean', loss_min[valid_min | valid].mean(), count)
                    writer.add_scalar('total/valid', (valid_min | valid).sum(), count)
    
                valid = (loss_new < loss) & self.sys.valid
                for s in opt_data:
                    for key in opt_data[s]:
                        opt_data[s][key][valid] = opt_data_new[s][key][valid]
                
                p = torch.exp(-(loss_new - loss) / (k * T))[~valid]
                r = torch.rand_like(p)
                valid_bad = r < p
                for s in opt_data:
                    for key in opt_data[s]:
                        opt_data[s][key][~valid][valid_bad] = opt_data_new[s][key][~valid][valid_bad]
                
                self.sys.fit_opt_data(opt_data)
                self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)

                pbar.set_description_str(f'T: {T:.4f}, loss_min: {loss_min.mean().item():.4f}, loss: {loss.mean().item():.4f}, loss_new: {loss_new.mean().item():.4f}')
            T = T * alpha

        self.sys.fit_opt_data(opt_data_min)
        self.update_system(rmax=None, avg_cfg=avg_cfg, fit_material=fit_material_flag, update_radius=False, quick_focus=True)
        
        if writer is not None and count is not None:
            return count

    #===================================================================================================#
    #------------------------------------------ Loss Function ------------------------------------------#
    #===================================================================================================#
    def lateral_loss(self, ref='rms'):
        """
        ref: chief mean
        """
        x = torch.where(self.v_dic, self.o_dic[-1, :, :, :, :, :, :, 0], 0.) # [wav, sys, cfg, ang, azi, M]
        y = torch.where(self.v_dic, self.o_dic[-1, :, :, :, :, :, :, 1], 0.) # [wav, sys, cfg, ang, azi, M]
        
        match ref:
            case 'rms':
                ref_x = x.sum(dim=-1, keepdim=True) / self.v_dic.sum(dim=-1, keepdim=True) # [wav, sys, cfg, ang, azi, 1]
                ref_y = y.sum(dim=-1, keepdim=True) / self.v_dic.sum(dim=-1, keepdim=True) # [wav, sys, cfg, ang, azi, 1]
            case 'chief':
                ref_x = x.gather(-1, self.chief_id_dic[..., None]) # [wav, sys, cfg, ang, azi, 1]
                ref_y = y.gather(-1, self.chief_id_dic[..., None]) # [wav, sys, cfg, ang, azi, 1]
                
        res = (ref_x - ref_x.mean(dim=0, keepdim=True)).abs() + (ref_y - ref_y.mean(dim=0, keepdim=True)).abs() # [wav, sys, cfg, ang, azi, 1]
        scale = 1.22 * self.sys.FNO * self.sys.wavelengths[self.sys.p_wvl] # [sys, cfg] / mm
        res = res[..., 0] / scale[None, :, :, None, None]
        return res.sum(dim=[0, 2, 3, 4])
    
    def spot_loss(self, ref='rms', k=1., efl=None):
        """
        ref: rms/ideal/chief
        """
        ignore_lat = False
        x = torch.where(self.v_dic, self.o_dic[-1, :, :, :, :, :, :, 0], 0.) # [wav, sys, cfg, ang, azi, M]
        y = torch.where(self.v_dic, self.o_dic[-1, :, :, :, :, :, :, 1], 0.) # [wav, sys, cfg, ang, azi, M]
        
        match ref:
            case 'rms':
                if ignore_lat:
                    ref_x = x.sum(dim=-1, keepdim=True) / self.v_dic.sum(dim=-1, keepdim=True) # [wav, sys, cfg, ang, azi, 1]
                    ref_y = y.sum(dim=-1, keepdim=True) / self.v_dic.sum(dim=-1, keepdim=True) # [wav, sys, cfg, ang, azi, 1]
                else:
                    ref_x = x.sum(dim=[0, -1], keepdim=True) / self.v_dic.sum(dim=[0, -1], keepdim=True) # [1, sys, cfg, ang, azi, 1]
                    ref_y = y.sum(dim=[0, -1], keepdim=True) / self.v_dic.sum(dim=[0, -1], keepdim=True) # [1, sys, cfg, ang, azi, 1]
            case 'chief':
                if ignore_lat:
                    ref_x = x.gather(-1, self.chief_id_dic[..., None]) # [wav, sys, cfg, ang, azi, 1]
                    ref_y = y.gather(-1, self.chief_id_dic[..., None]) # [wav, sys, cfg, ang, azi, 1]
                else:
                    ref_x = x.gather(-1, self.chief_id_dic[..., None])[self.sys.p_wvl][None, ...] # [1, sys, cfg, ang, azi, 1]
                    ref_y = y.gather(-1, self.chief_id_dic[..., None])[self.sys.p_wvl][None, ...] # [1, sys, cfg, ang, azi, 1]
            case 'ideal':
                efl = torch.tensor(efl) if isinstance(efl, list) else torch.tensor([efl])
                target_x = efl[:, None, None] * torch.tan(torch.deg2rad(self.sys.norm_views[None, :] * self.sys.max_view[:, None]))[:, :, None] * torch.sin(torch.deg2rad(self.sys.azimuths))[None, None, :] # [cfg, ang, azi]
                target_y = efl[:, None, None] * torch.tan(torch.deg2rad(self.sys.norm_views[None, :] * self.sys.max_view[:, None]))[:, :, None] * torch.cos(torch.deg2rad(self.sys.azimuths))[None, None, :] # [cfg, ang, azi]
                
                ref_x = target_x[None, None, ..., None] # [1, 1, cfg, ang, azi, 1]
                ref_y = target_y[None, None, ..., None] # [1, 1, cfg, ang, azi, 1]
                
        dx = x - ref_x
        dy = y - ref_y
        spot_ms = torch.where(self.v_dic, dx ** 2 + dy ** 2, 0.) # [wav, sys, cfg, ang, azi, M]
        weight = torch.where(self.v_dic, torch.exp((k * (dx ** 2 + dy ** 2)).clip(0., 1.)), 0.) # [wav, sys, cfg, ang, azi, M]
        spot_loss = (spot_ms * weight).sum(dim=-1) * self.sys.waveweights[:, None, None, None, None] / self.sys.waveweights.sum() # [wav, sys, cfg, ang, azi]
        return spot_loss.sum(dim=[0, 2, 3, 4])
        
    def wavefront_loss(self, mode='rms'):
        """
        Calculate the wavefront loss.
        """
        chief_o = torch.gather(self.o_dic[-1, :, :, :, :, :, :, :], dim=-2, index=self.chief_id_dic[..., None, None].repeat(1, 1, 1, 1, 1, 1, 3)) # [wav, sys, cfg, ang, azi, 1, 3]
        chief_d = torch.gather(self.d_dic[-1, :, :, :, :, :, :, :], dim=-2, index=self.chief_id_dic[..., None, None].repeat(1, 1, 1, 1, 1, 1, 3)) # [wav, sys, cfg, ang, azi, 1, 3]
        chief_t = torch.gather(self.t_dic[:, :, :, :, :, :], dim=-1, index=self.chief_id_dic[:, :, :, :, :, None]) # [wav, sys, cfg, ang, azi, 1]
        
        o = torch.where(self.v_dic[..., None], self.o_dic[-1, :, :, :, :, :, :, :], 0.) # [wav, sys, cfg, ang, azi, M, 3]
        d = torch.where(self.v_dic[..., None], self.d_dic[-1, :, :, :, :, :, :, :], 0.) # [wav, sys, cfg, ang, azi, M, 3]
        t = torch.where(self.v_dic, self.t_dic, 0.) # [wav, sys, cfg, ang, azi]
        
        r_chief = -self.sys.EXPP[None, :, :, None, None, None] / chief_d[:, :, :, :, :, :, 2]
        
        A = 1.
        B = -2 * ((o - chief_o) * d).sum(dim=-1) # [wav, sys, cfg, ang, azi, M]
        C = ((o - chief_o) ** 2).sum(dim=-1) - r_chief ** 2 # [wav, sys, cfg, ang, azi, M, 3]
        t1 = (-B + torch.sqrt((B ** 2 - 4 * A * C).clip(eps))) / (2 * A) # [wav, sys, cfg, ang, azi, M]
        t2 = (-B - torch.sqrt((B ** 2 - 4 * A * C).clip(eps))) / (2 * A) # [wav, sys, cfg, ang, azi, M]
        t_expp = torch.where(t1 > t2, t1, t2) # [wav, sys, cfg, ang, azi, M]

        opd = (chief_t - r_chief) - (t - t_expp) # [wav, sys, cfg, ang, azi, M]
        opd = torch.where(self.v_dic, opd, 0.) # [wav, sys, cfg, ang, azi, M]
        opd = opd / self.sys.wavelengths[:, None, None, None, None, None] # [wav, sys, cfg, ang, azi, M]

        if mode == 'rms':
            opd_loss = torch.where(self.v_dic, opd ** 2, 0.).sum(dim=-1) / self.v_dic.sum(dim=-1) # [wav, sys, cfg, ang, azi]
        elif mode == 'tv':
            opd_loss = opd.amax(dim=-1) - opd.amin(dim=-1) # [wav, sys, cfg, ang, azi]
        else:
            raise ValueError('Invalid mode.')
        opd_loss = opd_loss * self.sys.waveweights[:, None, None, None, None] / self.sys.waveweights.sum() # [wav, sys, cfg, ang, azi]

        return opd_loss.sum(dim=[0, 2, 3, 4])
    
    def efl_loss(self, target):
        # paraxial ray tracing
        target = torch.tensor(target)
        surfs = self.sys.extract_surfs()
        abcds = [elem.abcd(surfs[i], self.sys.wavelengths[self.sys.p_wvl][..., None]) for i, elem in enumerate(surfs[1:-1])]
        abcd = reduce((lambda x, y: torch.matmul(y, x)), abcds)
        effl = -1 / abcd[:, :, 1, 0]
        return ((effl - target[None, ...]) ** 2).sum(dim=-1)
    
    def fno_loss(self, target):
        # paraxial ray tracing
        target = torch.tensor(target)
        surfs = self.sys.extract_surfs()
        abcds = [elem.abcd(surfs[i], self.sys.wavelengths[self.sys.p_wvl][..., None]) for i, elem in enumerate(surfs[1:-1])]
        
        abcd_pre_s = reduce((lambda x, y: torch.matmul(y, x)), abcds[:self.sys.stop_id - 1]) if self.sys.stop_id > 1 else torch.tensor([[1., 0], [0., 1.]])[None, None, :, :].repeat(self.sys.sys_num, self.sys.cfg_num, 1, 1)
        abcd = reduce((lambda x, y: torch.matmul(y, x)), abcds)
        
        enpd = surfs[self.sys.stop_id].radius / abcd_pre_s[:, :, 0, 0] * 2
        effl = -1 / abcd[:, :, 1, 0]
        fno = effl / enpd
        fno_loss = torch.where(fno > target[None, ...], fno - target[None, ...], 0.) ** 2 # [sys, cfg]
        return fno_loss.sum(dim=-1)
    
    def totr_loss(self, target):
        surfs = self.sys.extract_surfs()
        if self.sys.stop_id == 1:
            totr = torch.max(sum(elem.thick for elem in surfs[1:-1]), sum(elem.thick for elem in surfs[2:-1]))
        else:
            totr = sum(elem.thick for elem in surfs[1:-1])
        totr_res = totr - target
        totr_loss = torch.where(totr_res > 0., totr_res, 0.)
        return (totr_loss ** 2).sum(dim=-1)
    
    def bfl_loss(self, target):
        surfs = self.sys.extract_surfs()
        
        elem_last = surfs[-2]
        acc_thick = elem_last.thick
        
        r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
        r = (r[None, None, :] * elem_last.radius[:, :, None])[None, :, :, None, None, :]
        sag = elem_last.surface(r, 0).max(dim=-1)[0].squeeze(dim=[0, 3, 4])
        sag_res = (acc_thick - sag) - target
        sag_loss = torch.where(sag_res < 0., -sag_res, 0.)
        return (sag_loss ** 2).sum(dim=-1)
    
    def gla_min_thick_loss(self, td_ratio=None, min_thick=None, ircf=False):
        x = 2 if ircf else 0
        edges_res = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, elem in enumerate(surfs[1:-1-x]):
            if 'VACUUM' not in elem.material['name']:
                elem_aft = surfs[i + 2]
                r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :]
                sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4])
                sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                if td_ratio is not None:
                    ref_thick = td_ratio * torch.max(elem.radius, elem_aft.radius) * 2
                elif min_thick is not None:
                    ref_thick = min_thick
                edge_res = (elem.thick[:, :, None] - sag + sag_aft).amin(dim=-1) - ref_thick
                edges_res = torch.cat([edges_res, torch.where(edge_res < 0, -edge_res, 0).unsqueeze(0)])
        return (edges_res ** 2).sum(dim=[0, -1])
    
    def gla_max_thick_loss(self, td_ratio=None, max_thick=None, ircf=False):
        x = 2 if ircf else 0
        edges_res = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, elem in enumerate(surfs[1:-1-x]):
            if 'VACUUM' not in elem.material['name']:
                elem_aft = surfs[i + 2]
                r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :]
                sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4])
                sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                if td_ratio is not None:
                    ref_thick = td_ratio * torch.min(elem.radius, elem_aft.radius) * 2
                elif max_thick is not None:
                    ref_thick = max_thick
                edge_res = (elem.thick[:, :, None] - sag + sag_aft).max(dim=-1)[0] - ref_thick
                edges_res = torch.cat([edges_res, torch.where(edge_res > 0, edge_res, 0).unsqueeze(0)])
        return (edges_res ** 2).sum(dim=[0, -1])
    
    def gla_max_min_ratio_loss(self, max_ratio, ircf=False):
        x = 2 if ircf else 0
        ratio_list = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, elem in enumerate(surfs[1:-1-x]):
            if 'VACUUM' not in elem.material['name']:
                elem_aft = surfs[i + 2]
                r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :]
                sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4])
                sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                edge_res_min = (elem.thick[:, :, None] - sag + sag_aft).amin(dim=-1)
                edge_res_max = (elem.thick[:, :, None] - sag + sag_aft).amax(dim=-1)
                ratio = edge_res_min / edge_res_max
                ratio_list = torch.cat([ratio_list, torch.where(ratio < max_ratio ** -1, max_ratio ** -1 - ratio, 0).unsqueeze(0)])
        return (ratio_list ** 2).sum(dim=[0, -1])
    
    def sag_dia_max_ratio_loss(self, max_ratio, ircf=False):
        x = 2 if ircf else 0
        ratio_list = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, elem in enumerate(surfs[1:-1-x]):
            dia = elem.radius * 2 # [sys, cfg]
            r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
            r = (r[None, None, :] * elem.radius[:, :, None])[None, :, :, None, None, :]
            sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4]) # [sys, cfg, M]
            ratio = (sag / dia[..., None]).abs().amax(dim=-1)
            ratio_list = torch.cat([ratio_list, torch.where(ratio > max_ratio, ratio - max_ratio, 0).unsqueeze(0)])
        return (ratio_list ** 2).sum(dim=[0, -1])
    
    def air_thick_loss(self, target):
        airs_res = torch.tensor([])
        surfs = self.sys.extract_surfs()

        for i, elem in enumerate(surfs[1:-2]):  # skip the last air
            if 'VACUUM' in elem.material['name']: # start count from the air surf
                if i == 0 and self.sys.stop_id == 1: # if the stop is the first surface, skip it
                    r = elem.radius[None, :, :, None, None, None] # [wav, sys, cfg, ang, azi, M]
                    air_res = elem.thick + surfs[2].surface(r, 0).squeeze(dim=[0, 3, 4, 5]) # [sys, cfg]
                    airs_res = torch.cat([airs_res, torch.where(air_res < 0, -air_res, 0).unsqueeze(0)])
                    continue
                
                elem_aft = surfs[i + 2]
                acc_thick = elem.thick

                r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :] # [wav, sys, cfg, ang, azi, M]
                sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4]) # [sys, cfg, M]
                sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                air_res = (acc_thick[:, :, None] - sag + sag_aft).amin(dim=-1) - target # [sys, cfg]
                airs_res = torch.cat([airs_res, torch.where(air_res < 0, -air_res, 0).unsqueeze(0)])
        return (airs_res ** 2).sum(dim=[0, -1])
    
    def surf_gap_loss(self, s_pre:int, s_aft:int, target:float, mode:str):
        """
        mode: lt/gt/eq
        """
        surfs = self.sys.extract_surfs()
        r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
        r = (r[None, None, :] * torch.min(surfs[s_pre].radius, surfs[s_aft].radius)[:, :, None])[None, :, :, None, None, :]
        sag = surfs[s_pre].surface(r, 0).squeeze(dim=[0, 3, 4])
        sag_aft = surfs[s_aft].surface(r, 0).squeeze(dim=[0, 3, 4])
        thick = sum(elem.thick for elem in surfs[s_pre:s_aft])
        edge_res = (thick[:, :, None] - sag + sag_aft).amin(dim=-1)
        match mode:
            case 'lt':
                edge_loss = torch.where(edge_res > target, edge_res - target, 0.)
            case 'gt':
                edge_loss = torch.where(edge_res < target, target - edge_res, 0.)
            case 'eq':
                edge_loss = (edge_res - target).abs()
        return (edge_loss ** 2).sum(dim=[-1])
    
    def gla_z_loss(self, z_min, ircf=False):
        x = 2 if ircf else 0
        z = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, elem in enumerate(surfs[1:-1-x]):
            if 'VACUUM' not in elem.material['name']:
                elem_aft = surfs[i + 2]
                A = elem.radius * 2 * elem.roc[:, None] # [sys, cfg]
                B = elem_aft.radius * 2 * elem_aft.roc[:, None] # [sys, cfg]
                Z = (A - B).abs().amin(dim=-1) / 4
                z = torch.cat([z, Z.unsqueeze(0)]) # [surf, sys]
        loss_z = torch.where(z < z_min, z_min - z, 0.)
        return (loss_z ** 2).sum(dim=[0])
    
    def angle_loss(self, target=None):
        surfs = self.sys.extract_surfs()[1:-1]
        k = torch.cos(torch.deg2rad(torch.tensor(target))) if target != None else 1.
        loss = torch.zeros(len(surfs), self.sys.sys_num, self.sys.cfg_num, self.v_dic.shape[-1]) # incident angle
        
        for i, surf in enumerate(surfs):
            ox = torch.where(self.v_dic, self.o_dic[i + 1][..., 0], 0.)
            oy = torch.where(self.v_dic, self.o_dic[i + 1][..., 1], 0.)
            
            normal = -surf.inter_normal(ox, oy) # [wav, sys, cfg, ang, azi, M, 3]
            d_pre = self.d_dic[i] # [wav, sys, cfg, ang, azi, M, 3]
            d_aft = self.d_dic[i + 1] # [wav, sys, cfg, ang, azi, M, 3]

            theta_1 = torch.where(self.v_dic, torch.einsum('...k, ...k -> ...', normal, d_pre), 1.) # [wav, sys, cfg, ang, azi, M]
            theta_2 = torch.where(self.v_dic, torch.einsum('...k, ...k -> ...', normal, d_aft), 1.) # [wav, sys, cfg, ang, azi, M]
            theta_min = torch.where(theta_1 < theta_2, theta_1, theta_2) # [wav, sys, cfg, ang, azi, M]
            loss[i] = torch.where(theta_min < k, k - theta_min, 0.).amax(dim=[0, 3, 4]) # [sys, cfg, M]

        angle_loss = loss.sum(dim=-1) # [surfs, sys, cfg]
        return angle_loss.sum(dim=[0, -1])
    
    def cra_loss(self, target=None):
        chief_d_img = torch.gather(self.d_dic[-1, :, :, :, :, :, :, :], dim=-2, index=self.chief_id_dic[..., None, None].repeat(1, 1, 1, 1, 1, 1, 3)) # [wav, sys, cfg, ang, azi, 1, 3]
        normal = torch.zeros_like(chief_d_img)
        normal[..., -1] = 1.
        theta = torch.einsum('...k, ...k -> ...', normal, chief_d_img) # [wav, sys, cfg, ang, azi, 1]
        k = torch.cos(torch.deg2rad(torch.tensor(target))) if target != None else 0.
        loss = torch.where(theta < k, k - theta, 0.).sum(dim=[0, 2, 3, 4, 5])
        return loss # [sys]
    
    def angle_std_loss(self):
        d_surf_num = self.d_dic.shape[0]
        dx = torch.where(self.v_dic[None, ...].repeat(d_surf_num, 1, 1, 1, 1, 1, 1), self.d_dic[:, :, :, :, :, :, :, 0], 0.) # [surfs, wav, sys, cfg, ang, azi, M]
        dy = torch.where(self.v_dic[None, ...].repeat(d_surf_num, 1, 1, 1, 1, 1, 1), self.d_dic[:, :, :, :, :, :, :, 1], 0.) # [surfs, wav, sys, cfg, ang, azi, M]
        dz = torch.where(self.v_dic[None, ...].repeat(d_surf_num, 1, 1, 1, 1, 1, 1), self.d_dic[:, :, :, :, :, :, :, 2], 0.) # [surfs, wav, sys, cfg, ang, azi, M]
        
        d_dx = dx - dx.sum(dim=[0, -1], keepdim=True) / self.v_dic.sum(dim=-1)[None, :, :, :, :, :, None] / d_surf_num # [surfs, wav, sys, cfg, ang, azi, M]
        d_dy = dy - dy.sum(dim=[0, -1], keepdim=True) / self.v_dic.sum(dim=-1)[None, :, :, :, :, :, None] / d_surf_num # [surfs, wav, sys, cfg, ang, azi, M]
        d_dz = dz - dz.sum(dim=[0, -1], keepdim=True) / self.v_dic.sum(dim=-1)[None, :, :, :, :, :, None] / d_surf_num # [surfs, wav, sys, cfg, ang, azi, M]
        
        d_diff = torch.where(self.v_dic[None, ...].repeat(d_surf_num, 1, 1, 1, 1, 1, 1), d_dx ** 2 + d_dy ** 2 + d_dz ** 2, 0.) # [surfs, wav, sys, cfg, ang, azi, M]
        loss_ms = d_diff.sum(dim=[0, -1]) / self.v_dic.sum(dim=-1) / d_surf_num # [wav, sys, cfg, ang, azi]
        return loss_ms.amax(dim=[3]).sum(dim=[0, 2, 3]) # [sys]
    
    def surf_k_loss(self, target=None):
        surfs = self.sys.extract_surfs()[1:-1]
        k = torch.cos(torch.deg2rad(torch.tensor(target))) if target != None else 1.        
        loss = torch.zeros(len(surfs), self.sys.sys_num, self.sys.cfg_num, self.v_dic.shape[-1]) # normal angle
        
        for i, surf in enumerate(surfs):
            ox = torch.where(self.v_dic, self.o_dic[i + 1][..., 0], 0.)
            oy = torch.where(self.v_dic, self.o_dic[i + 1][..., 1], 0.)
            
            normal = -surf.inter_normal(ox, oy) # [wav, sys, cfg, ang, azi, M, 3]
            axis = torch.zeros_like(normal) # [wav, sys, cfg, ang, azi, M, 3]
            axis[..., 2] = 1.

            norm = torch.where(self.v_dic, torch.einsum('...k, ...k -> ...', normal, axis), 1.) # [wav, sys, cfg, ang, azi, M]
            loss[i] = torch.where(norm < k, k - norm, 0.).amax(dim=[0, 3, 4]) # [sys, cfg, M]
            
        k_loss = loss.sum(dim=-1) # [surfs, sys, cfg]
        return k_loss.sum(dim=[0, -1])
    
    def distor_loss(self, target=None, absolute=True):
        """ f-tan(theta) distortion
        Args:
            target (float, optional): Target distortion. Defaults to None.
            absolute (bool, optional): Defaults to True.
                If absolute is True, minimize the absolute distortion within the target range.
                If absolute is False, minimize the distortion within 0. and the target value.
        """
        with torch.no_grad():
            surfs = self.sys.extract_surfs()
            abcds = [elem.abcd(surfs[i], self.sys.wavelengths[self.sys.p_wvl][..., None]) for i, elem in enumerate(surfs[1:-1])]
            abcd = reduce((lambda x, y: torch.matmul(y, x)), abcds)
            theta = torch.deg2rad(torch.tensor(1e-4) * self.sys.max_view)
            ini = torch.stack([-self.sys.ENPP * theta[None, :], torch.ones_like(self.sys.ENPP) * theta[None, :]], dim=-1)
            l = torch.matmul(abcd, ini.unsqueeze(-1))
            difl = l[:, :, 0, 0] / torch.tan(theta)
            target_x = difl[:, :, None, None] * torch.tan(torch.deg2rad(self.sys.norm_views[None, :] * self.sys.max_view[:, None]))[None, :, :, None] * torch.sin(torch.deg2rad(self.sys.azimuths))[None, None, None, :]
            target_y = difl[:, :, None, None] * torch.tan(torch.deg2rad(self.sys.norm_views[None, :] * self.sys.max_view[:, None]))[None, :, :, None] * torch.cos(torch.deg2rad(self.sys.azimuths))[None, None, None, :]
        
        chief_x = torch.gather(self.o_dic[-1, :, :, :, :, :, :, 0], dim=-1, index=self.chief_id_dic.unsqueeze(-1)).squeeze(-1)[self.sys.p_wvl] # [sys, cfg, ang, azi]
        chief_y = torch.gather(self.o_dic[-1, :, :, :, :, :, :, 1], dim=-1, index=self.chief_id_dic.unsqueeze(-1)).squeeze(-1)[self.sys.p_wvl] # [sys, cfg, ang, azi]
        
        target_length = torch.sqrt((target_x ** 2 + target_y ** 2).clip(min=eps))
        chief_length = torch.sqrt((chief_x ** 2 + chief_y ** 2).clip(min=eps))
        dist = chief_length - target_length
        
        if absolute:
            distortion_loss = limit_var(dist, -torch.abs(target * target_length), torch.abs(target * target_length))
        else:
            if target > 0.:
                distortion_loss = limit_var(dist, 0., target * target_length)
            elif target < 0.:
                distortion_loss = limit_var(dist, target * target_length, 0.)
        return distortion_loss.amax(dim=[1, 2, 3]) # [sys]

    def pupil_loss(self, ref_point_n=8):
        """
        Optimize the pupil distortion on the exit pupil plane.
        ref_point_n: number of reference points on the exit pupil plane.
        """
        t_img_ep = self.sys.EXPP[None, :, :, None, None, None] / self.d_dic[-1][..., 2] # [wav, sys, cfg, ang, azi, M]
        o_exp = (self.o_dic[-1] + t_img_ep[..., None] * self.d_dic[-1])
        
        ref_x = o_exp[..., 0].gather(-1, self.chief_id_dic[..., None]) # [wav, sys, cfg, ang, azi, 1]
        ref_y = o_exp[..., 1].gather(-1, self.chief_id_dic[..., None]) # [wav, sys, cfg, ang, azi, 1]
        o_exp_ref = torch.cat([ref_x[..., None], ref_y[..., None], (torch.ones_like(ref_x) * self.sys.EXPP[None, :, :, None, None, None])[..., None]], dim=-1) # [wav, sys, cfg, ang, azi, 1, 3]
        o_exp = o_exp - o_exp_ref # [wav, sys, cfg, ang, azi, M, 3]
        
        theta_interval = 2 * torch.pi / ref_point_n
        thetas = torch.arange(0, 2 * torch.pi, theta_interval) # [ref_point_n]
        x = self.sys.EXPD[:, :, None] * torch.cos(thetas)[None, None, :] / 2 # [sys, cfg, ref_point_n]
        y = self.sys.EXPD[:, :, None] * torch.sin(thetas)[None, None, :] / 2 # [sys, cfg, ref_point_n]
        z = torch.zeros_like(x) # [sys, cfg, ref_point_n]
        target = torch.stack([x, y, z], dim=-1) # [sys, cfg, ref_point_n, 3]
        
        res = length(o_exp[:, :, :, :, :, :, None, :] - target[None, :, :, None, None, None, :, :]) # [wav, sys, cfg, ang, azi, M, ref_point_n]
        res = torch.where(self.v_dic[..., None], res, torch.inf).amin(dim=-2).amax(dim=-1) # [wav, sys, cfg, ang, azi]
        res = torch.where(res == torch.inf, torch.nan, res) / (self.sys.EXPD[None, :, :, None, None] / 2) # [wav, sys, cfg, ang, azi]
        return res.sum(dim=[0, 2, 3, 4])
    
    def roc_loss(self, surf_id, sign:str):
        """
        sign: p: positive, n: negative
        """
        roc = self.sys.extract_surfs()[surf_id].roc # [sys]
        match sign:
            case 'p':
                roc_loss = torch.where(roc < 0., -roc, 0.)
            case 'n':
                roc_loss = torch.where(roc > 0., roc, 0.)
        return roc_loss
    
    def seidel_loss(self):
        """
        Seidel loss function.
        Only for Sphere systems.
        """
        surfs = self.sys.extract_surfs()
        abcds = [elem.abcd(surfs[i], self.sys.wavelengths[self.sys.p_wvl][..., None]) for i, elem in enumerate(surfs[1:-1])]
        
        ubar0 = torch.tan(self.sys.max_view.deg2rad())[None, :].repeat(self.sys.sys_num, 1) # [sys, cfg]
        ybar0 = ubar0 * (0 - self.sys.ENPP)
        u0 = torch.zeros(self.sys.sys_num, self.sys.cfg_num)
        y0 = self.sys.ENPD / 2
        
        ubar = ubar0[None, ...]
        ybar = ybar0[None, ...]
        u = u0[None, ...]
        y = y0[None, ...]
        n = torch.ones([1, self.sys.sys_num])
        dn = torch.zeros([1, self.sys.sys_num])
        c = torch.zeros([1, self.sys.sys_num])

        for i in range(1, len(self.sys.system)-1):
            abcd = abcds[i-1]
            
            ybar_ubar = torch.stack([ybar0, ubar0], dim=-1)
            y_u = torch.stack([y0, u0], dim=-1)
            
            ybar_ubar = torch.matmul(abcd, ybar_ubar.unsqueeze(-1)).squeeze(-1)
            y_u = torch.matmul(abcd, y_u.unsqueeze(-1)).squeeze(-1)
            
            ybar0, ubar0 = ybar_ubar[..., 0], ybar_ubar[..., 1]
            y0, u0 = y_u[..., 0], y_u[..., 1]
            
            ybar = torch.cat([ybar, ybar0[None, ...]], dim=0)
            ubar = torch.cat([ubar, ubar0[None, ...]], dim=0)
            y = torch.cat([y, y0[None, ...]], dim=0)
            u = torch.cat([u, u0[None, ...]], dim=0)
            n = torch.cat([n, self.sys.system[i].refractive_index(self.sys.wavelengths)[self.sys.p_wvl][None, :]], dim=0)
            _, wave_id = torch.sort(self.sys.wavelengths)
            n1 = self.sys.system[i].refractive_index(self.sys.wavelengths)[wave_id[0]]
            n2 = self.sys.system[i].refractive_index(self.sys.wavelengths)[wave_id[-1]]
            dn = torch.cat([dn, (n1 - n2)[None, :]], dim=0)
            c = torch.cat([c, self.sys.system[i].roc[None, :]], dim=0)
        
        ybar = torch.cat([ybar, ybar0[None, ...]], dim=0)
        ubar = torch.cat([ubar, ubar0[None, ...]], dim=0)
        y = torch.cat([y, y0[None, ...]], dim=0)
        u = torch.cat([u, u0[None, ...]], dim=0)
        n = torch.cat([n, torch.ones([1, self.sys.sys_num])], dim=0)
        dn = torch.cat([dn, torch.zeros([1, self.sys.sys_num])], dim=0)
        c = torch.cat([c, torch.zeros([1, self.sys.sys_num])], dim=0)
        
        A = n[:-1, :, None] * u[:-1] + n[:-1, :, None] * y[:-1] * c[1:, :, None]
        Abar = n[:-1, :, None] * ubar[:-1] + n[:-1, :, None] * ybar[:-1] * c[1:, :, None]
        H = Abar * y[:-1] - A * ybar[:-1] # Q = n0 * y0 * ubar0
        
        SI = -A ** 2 * y[:-1] * (u[1:] / n[1:, :, None] - u[:-1] / n[:-1, :, None]) # [surf, sys, cfg]
        SII = -Abar * A * y[:-1] * (u[1:] / n[1:, :, None] - u[:-1] / n[:-1, :, None]) # [surf, sys, cfg]
        SIII = -Abar ** 2 * y[:-1] * (u[1:] / n[1:, :, None] - u[:-1] / n[:-1, :, None]) # [surf, sys, cfg]
        SIV = H ** 2 * ((n[1:] - n[:-1]) * c[1:] / (n[1:] * n[:-1]))[:, :, None] # [surf, sys, cfg]
        SV = Abar / (A + eps) * (SIII + SIV) # [surf, sys, cfg]
        CI = -A * y[:-1] * ((dn / n)[1:, :, None] - (dn / n)[:-1, :, None]) # [surf, sys, cfg]
        CII = -Abar * y[:-1] * ((dn / n)[1:, :, None] - (dn / n)[:-1, :, None]) # [surf, sys, cfg]
        return SI.sum(dim=[0, -1]), SII.sum(dim=[0, -1]), SIII.sum(dim=[0, -1]), SIV.sum(dim=[0, -1]), SV.sum(dim=[0, -1]), CI.sum(dim=[0, -1]), CII.sum(dim=[0, -1])
    
    #=====================================================================================================#
    #------------------------------------------ PSF Calculation ------------------------------------------#
    #=====================================================================================================#
    @torch.no_grad()
    def relative_illumination(self, pupil_samp, norm_view, azimuth, wavelength=None):
        if wavelength == None:
            wavelength = self.sys.wavelengths.tolist()[self.sys.p_wvl]
        elif ~isinstance(wavelength, float):
            raise ValueError("wavelength must be a float number.")
            
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
            
            return mask.sum() * (2. / pixels) ** 2
        
        rays = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength)
        rays = self.sys.propagate(rays, radius_flag=True, record=False)
        
        ray_ref = self.sys.sample_ray_2d(pupil_samp, 0., 0., wavelength)
        ray_ref = self.sys.propagate(ray_ref, radius_flag=True, record=False)
        
        rl = torch.zeros(rays.valid.shape[1], rays.valid.shape[2], rays.valid.shape[3], rays.valid.shape[4]) # [sys, cfg, ang, azi]
        for _sys in range(rays.valid.shape[1]):
            for _cfg in range(rays.valid.shape[2]):
                d_ref = ray_ref.d[0, _sys, _cfg, 0, 0, :, 0:2][ray_ref.valid[0, _sys, _cfg, 0, 0]]
                rl_ref = area(d_ref)

                for _ang in range(rays.valid.shape[3]):
                    for _azi in range(rays.valid.shape[4]):
                        d = rays.d[0, _sys, _cfg, _ang, _azi, :, 0:2][rays.valid[0, _sys, _cfg, _ang, _azi]]
                        off_rl = area(d)
                        rl[_sys, _cfg, _ang, _azi] = off_rl / rl_ref
        return rl # [sys, cfg, ang, azi]
    
    
    @torch.no_grad()
    def distortion(self, pupil_samp, norm_view, azimuth, wavelength=None):
        if wavelength == None:
            wavelength = self.sys.wavelengths.tolist()[self.sys.p_wvl]
        elif ~isinstance(wavelength, float):
            raise ValueError("wavelength must be a float number.")
        
        surfs = self.sys.extract_surfs()
        abcds = [elem.abcd(surfs[i], self.sys.wavelengths[self.sys.p_wvl][..., None]) for i, elem in enumerate(surfs[1:-1])]
        abcd = reduce((lambda x, y: torch.matmul(y, x)), abcds)
        theta = torch.deg2rad(torch.tensor(1e-4) * self.sys.max_view)
        ini = torch.stack([-self.sys.ENPP * theta[None, :], torch.ones_like(self.sys.ENPP) * theta[None, :]], dim=-1)
        l = torch.matmul(abcd, ini.unsqueeze(-1))
        difl = l[:, :, 0, 0] / torch.tan(theta) # [sys, cfg]
        
        rays = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength)
        rays = self.sys.propagate(rays, radius_flag=True, record=False)
        
        norm_view = torch.tensor([norm_view]) if isinstance(norm_view, float) else torch.tensor(norm_view)
        azimuth = torch.tensor([azimuth]) if isinstance(azimuth, float) else torch.tensor(azimuth)

        target_y = difl[:, :, None, None] * torch.tan(torch.deg2rad(norm_view[None, :] * self.sys.max_view[:, None]))[None, :, :, None] * torch.cos(torch.deg2rad(azimuth))[None, None, None, :] # [sys, cfg, ang, azi]
        target_x = difl[:, :, None, None] * torch.tan(torch.deg2rad(norm_view[None, :] * self.sys.max_view[:, None]))[None, :, :, None] * torch.sin(torch.deg2rad(azimuth))[None, None, None, :] # [sys, cfg, ang, azi]
        
        chief_x = torch.gather(rays.o[:, :, :, :, :, :, 0], dim=-1, index=rays.chief_id.unsqueeze(-1)).squeeze(-1)[0] # [sys, cfg, ang, azi]
        chief_y = torch.gather(rays.o[:, :, :, :, :, :, 1], dim=-1, index=rays.chief_id.unsqueeze(-1)).squeeze(-1)[0] # [sys, cfg, ang, azi]
        
        vec_x = chief_x - target_x
        vec_y = chief_y - target_y
        vec_xy = torch.stack([vec_x, vec_y], dim=-1)
        return vec_xy # [sys, cfg, ang, azi, 2]
        
    
    def psf_rs(self, pupil_samp, image_samp, image_delta, norm_view, azimuth, wavelength:float=None, auto=True, chief_o=False):
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square')
        ray = self.sys.propagate(ray, radius_flag=True, record=False)
        # back to the exit pupil plane 
        t_img_ep = self.sys.EXPP[None, :, :, None, None, None] / ray.d[..., 2]

        image_delta = image_delta * 1e-3
        line_sample = torch.linspace(-int((image_samp - 1) / 2), int((image_samp - 1) / 2), image_samp) * image_delta
        y, x = torch.meshgrid(-line_sample, line_sample, indexing='ij')
        
        psfs = torch.zeros(ray.valid.shape[0], ray.valid.shape[1], ray.valid.shape[2], ray.valid.shape[3], ray.valid.shape[4], image_samp, image_samp)
        if chief_o: grids = torch.zeros(ray.valid.shape[1], ray.valid.shape[2], ray.valid.shape[3], ray.valid.shape[4], 2)
        for w, wave in enumerate(ray.wavelength):
            for _sys in range(ray.valid.shape[1]):
                for _cfg in range(ray.valid.shape[2]):
                    for _ang in range(ray.valid.shape[3]):
                        for _azi in range(ray.valid.shape[4]):
                            if wavelength == None:
                                rel_o = ray.o[self.sys.p_wvl, _sys, _cfg, _ang, _azi][ray.chief_id[self.sys.p_wvl, _sys, _cfg, _ang, _azi]]
                            else:
                                rel_o = ray.o[0, _sys, _cfg, _ang, _azi][ray.chief_id[0, _sys, _cfg, _ang, _azi]]

                            k = 2 * torch.pi / wave
                            grid = rel_o + torch.stack([x, y, torch.zeros_like(x)], dim=-1) # [M, M, 3]
                            o = (ray.o[w, _sys, _cfg, _ang, _azi] + t_img_ep[w, _sys, _cfg, _ang, _azi][..., None] * ray.d[w, _sys, _cfg, _ang, _azi])[ray.valid[w, _sys, _cfg, _ang, _azi]]
                            t = (ray.t[w, _sys, _cfg, _ang, _azi] + t_img_ep[w, _sys, _cfg, _ang, _azi])[ray.valid[w, _sys, _cfg, _ang, _azi]]
                
                            if auto:
                                r = length(grid[:, :, None, :] - o[None, None, ...]).clip(min=eps)
                                amp = torch.einsum('ijk->ij', (-self.sys.EXPP[_sys, _cfg] / r) * (1j * k - 1 / r) * torch.exp(1j * k * (t + r)) / r)
                                psf = torch.abs(amp) ** 2
                            else:
                                psf = RayleighSommerfeldPsfOp.apply(o, t, grid, k, -self.sys.EXPP[_sys, _cfg])
                
                            psfs[w, _sys, _cfg, _ang, _azi] = psf / psf.sum()
                            if chief_o: grids[_sys, _cfg, _ang, _azi] = rel_o[:2]
        
        if chief_o:
            return psfs, grids.permute(4, 0, 1, 2, 3) # [wav, sys, cfg, ang, azi, M, M], [2, sys, cfg, ang, azi]
        else:
            return psfs # [wav, sys, cfg, ang, azi, M, M]
        
        
    def psf_rs_err(self, pupil_samp, image_samp, image_delta, norm_view, azimuth, zernike_err:dict, wavelength:float=None, auto=True, chief_o=False):
        # zernike_err - unit: wavelength
        
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square')
        ray = self.sys.propagate(ray, radius_flag=True, record=False)
        # back to the exit pupil plane 
        t_img_ep = self.sys.EXPP[None, :, :, None, None, None] / ray.d[..., 2]
        
        image_delta = image_delta * 1e-3
        line_sample = torch.linspace(-int((image_samp - 1) / 2), int((image_samp - 1) / 2), image_samp) * image_delta
        y, x = torch.meshgrid(-line_sample, line_sample, indexing='ij')
        
        def make_pupil_grid(sampling):
            xs = torch.linspace(-1.0, 1.0, sampling)
            ys = torch.linspace(-1.0, 1.0, sampling)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')

            rho = torch.sqrt(xx**2 + yy**2)
            theta = torch.atan2(yy, xx)
            mask = rho <= 1.0

            return xx, yy, rho, theta, mask
        
        with torch.no_grad():
            wave = self.sys.wavelengths.tolist()[self.sys.p_wvl] if wavelength == None else wavelength
            zernike_err = {j: c * wave for j, c in zernike_err.items()}
            xx, yy, rho, theta, mask = make_pupil_grid(pupil_samp)
            wf_error = zernike_wavefront(rho, theta, zernike_err)
            wf_error[~mask] = 0.
            wf_error = wf_error.flatten()
        
        psfs = torch.zeros(ray.valid.shape[0], ray.valid.shape[1], ray.valid.shape[2], ray.valid.shape[3], ray.valid.shape[4], image_samp, image_samp)
        if chief_o: grids = torch.zeros(ray.valid.shape[1], ray.valid.shape[2], ray.valid.shape[3], ray.valid.shape[4], 2)
        for w, wave in enumerate(ray.wavelength):
            for _sys in range(ray.valid.shape[1]):
                for _cfg in range(ray.valid.shape[2]):
                    for _ang in range(ray.valid.shape[3]):
                        for _azi in range(ray.valid.shape[4]):
                            if wavelength == None:
                                rel_o = ray.o[self.sys.p_wvl, _sys, _cfg, _ang, _azi][ray.chief_id[self.sys.p_wvl, _sys, _cfg, _ang, _azi]]
                            else:
                                rel_o = ray.o[0, _sys, _cfg, _ang, _azi][ray.chief_id[0, _sys, _cfg, _ang, _azi]]
                                
                            k = 2 * torch.pi / wave
                            grid = rel_o + torch.stack([x, y, torch.zeros_like(x)], dim=-1) # [M, M, 3]
                            _o = (ray.o[w, _sys, _cfg, _ang, _azi] + t_img_ep[w, _sys, _cfg, _ang, _azi][..., None] * ray.d[w, _sys, _cfg, _ang, _azi])[ray.valid[w, _sys, _cfg, _ang, _azi]]
                            _t = (ray.t[w, _sys, _cfg, _ang, _azi] + t_img_ep[w, _sys, _cfg, _ang, _azi])[ray.valid[w, _sys, _cfg, _ang, _azi]]
                            
                            # Add zernike wavefront error to the wavefront
                            t = _t[1:] + wf_error[ray.valid[w, _sys, _cfg, _ang, _azi, 1:]]
                            o = _o[1:]
                            
                            if auto:
                                r = length(grid[:, :, None, :] - o[None, None, ...]).clip(min=eps)
                                amp = torch.einsum('ijk->ij', (-self.sys.EXPP[_sys, _cfg] / r) * (1j * k - 1 / r) * torch.exp(1j * k * (t + r)) / r)
                                psf = torch.abs(amp) ** 2
                            else:
                                psf = RayleighSommerfeldPsfOp.apply(o, t, grid, k, -self.sys.EXPP[_sys, _cfg])
                                
                            psfs[w, _sys, _cfg, _ang, _azi] = psf / psf.sum()
                            if chief_o: grids[_sys, _cfg, _ang, _azi] = rel_o[:2]
        
        if chief_o:
            return psfs, grids.permute(4, 0, 1, 2, 3) # [wav, sys, cfg, ang, azi, M, M], [2, sys, cfg, ang, azi]
        else:
            return psfs # [wav, sys, cfg, ang, azi, M, M]
        
        
    def psf_co(self, pupil_samp, image_samp, image_delta, norm_view, azimuth, wavelength:float=None, auto=True):
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square')
        ray = self.sys.propagate(ray, radius_flag=True, record=False)

        image_delta = image_delta * 1e-3
        line_sample = torch.linspace(-int((image_samp - 1) / 2), int((image_samp - 1) / 2), image_samp) * image_delta
        y, x = torch.meshgrid(-line_sample, line_sample, indexing='ij')
        
        psfs = torch.zeros(ray.valid.shape[0], ray.valid.shape[1], ray.valid.shape[2], ray.valid.shape[3], ray.valid.shape[4], image_samp, image_samp)
        for w, wave in enumerate(ray.wavelength):
            for _sys in range(ray.valid.shape[1]):
                for _cfg in range(ray.valid.shape[2]):
                    for _ang in range(ray.valid.shape[3]):
                        for _azi in range(ray.valid.shape[4]):
                            if wavelength == None:
                                rel_o = ray.o[self.sys.p_wvl, _sys, _cfg, _ang, _azi][ray.chief_id[self.sys.p_wvl, _sys, _cfg, _ang, _azi]]
                            else:
                                rel_o = ray.o[0, _sys, _cfg, _ang, _azi][ray.chief_id[0, _sys, _cfg, _ang, _azi]]

                            k = 2 * torch.pi / wave
                            grid = rel_o + torch.stack([x, y, torch.zeros_like(x)], dim=-1)          
                            o = ray.o[w, _sys, _cfg, _ang, _azi][ray.valid[w, _sys, _cfg, _ang, _azi]]
                            d = ray.d[w, _sys, _cfg, _ang, _azi][ray.valid[w, _sys, _cfg, _ang, _azi]]
                            t = ray.t[w, _sys, _cfg, _ang, _azi][ray.valid[w, _sys, _cfg, _ang, _azi]]

                            if auto:
                                r = grid[:, :, None, :] - o[None, None, :, :]
                                dr = torch.einsum('...k,...k->...', d[None, None, :, :], r)
                                amp = torch.einsum('...k->...', torch.exp(1j * k * (t + dr)) * d[None, None, :, 2])
                                psf = torch.abs(amp) ** 2
                            else:
                                psf = CoherentPsfOp.apply(o, d, grid, t, k)
                    
                            psfs[w, _sys, _cfg, _ang, _azi] = psf / psf.sum()
        
        return psfs # [wav, sys, cfg, ang, azi, M, M]
    
    
    def psf_spot(self, pupil_samp, image_samp, image_delta, norm_view, azimuth, wavelength:float=None, auto=True):
        ray = self.sys.sample_ray_2d(pupil_samp, norm_view, azimuth, wavelength, samp_method='square')
        ray = self.sys.propagate(ray, radius_flag=True, record=False)

        image_delta = image_delta * 1e-3
        psf_range = [-int((image_samp - 1) / 2) * image_delta, int((image_samp - 1) / 2) * image_delta]
        x_min, x_max = psf_range
        y_min, y_max = psf_range
        
        psfs = torch.zeros(ray.valid.shape[0], ray.valid.shape[1], ray.valid.shape[2], ray.valid.shape[3], ray.valid.shape[4], image_samp, image_samp)
        for w, wave in enumerate(ray.wavelength):
            for _sys in range(ray.valid.shape[1]):
                for _cfg in range(ray.valid.shape[2]):
                    for _ang in range(ray.valid.shape[3]):
                        for _azi in range(ray.valid.shape[4]):
                            if wavelength == None:
                                rel_o = ray.o[self.sys.p_wvl, _sys, _cfg, _ang, _azi][ray.chief_id[self.sys.p_wvl, _sys, _cfg, _ang, _azi]]
                            else:
                                rel_o = ray.o[0, _sys, _cfg, _ang, _azi][ray.chief_id[0, _sys, _cfg, _ang, _azi]]
    
                            o = ray.o[w, _sys, _cfg, _ang, _azi]
                            
                            point_shift = o[:, 0:2] - rel_o[None, 0:2]
                            ra = ray.valid[w, _sys, _cfg, 0, 0] * (point_shift[..., 0].abs() < psf_range[1] - 0.1 * image_delta) * (point_shift[..., 1].abs() < psf_range[1] - 0.1 * image_delta)
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
                            
                            obliq = ray.d[w, _sys, _cfg, _ang, _azi, :, 2] ** 2
                            grid = torch.zeros(ks, ks)
                            grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*obliq, accumulate=True)
                            grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*obliq, accumulate=True)
                            grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*obliq, accumulate=True)
                            grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*obliq, accumulate=True)

                            psf = grid / grid.max()
                            
                            psfs[w, _sys, _cfg, _ang, _azi] = psf / psf.sum()
        return psfs # [wav, sys, cfg, ang, azi, M, M]
    
    
    def psf_to_rgb(self, psfs, psfs_weight, show=True):
        # psfs_weight: [3, wav]
        # psfs: [wav, sys, cfg, ang, azi, M, M]
        psfs_weight = psfs_weight / psfs_weight.sum(dim=-1, keepdim=True)
        psf_rgb = torch.einsum('ij, j...->i...', psfs_weight, psfs)
        
        if show:
            for _sys in range(psf_rgb.shape[1]):
                for _cfg in range(psf_rgb.shape[2]):
                    fig, ax = plt.subplots(figsize=(4 * psf_rgb.shape[3], 3 * psf_rgb.shape[4]))
                    plt.axis('off')
                    x = 0
                    for _azi in range(psf_rgb.shape[4]):
                        for _ang in range(psf_rgb.shape[3]):
                            x += 1
                            ax = plt.subplot(psf_rgb.shape[4], psf_rgb.shape[3], x)
                            im = ax.imshow((psf_rgb[:, _sys, _cfg, _ang, _azi] / psf_rgb[:, _sys, _cfg, _ang, _azi].max()).permute(1, 2, 0).detach().cpu().numpy())
                            plt.colorbar(im)
                    plt.tight_layout()
        
        return psf_rgb # [3, sys, cfg, ang, azi, M, M]
    
###################################################################################################################################################################
class MeritZ(Merit):
    def __init__(self, **kwargs):
        super(MeritZ, self).__init__(**kwargs)
    
    def forward_loss(self, args:dict, **kwargs):
        """
        writer: tensorboard writer.
        count: iteration count.
        path: save loss pie path.
        adjust_weight: whether to adjust weight.
        """
        writer = kwargs.get('writer', None)
        count = kwargs.get('count', None)
        path = kwargs.get('path', None)

        self.propagate_all_rays()
        valids = torch.ones_like(self.sys.valid).bool()
        valids &= torch.gather(self.v_dic, -1, self.chief_id_dic.unsqueeze(-1)).squeeze(-1).prod(0).prod(-1).prod(-1).prod(-1).bool() # [sys]
        valids &= (self.v_dic.sum(dim=-1) > 2).prod(0).prod(-1).prod(-1).prod(-1).bool()
        loss, label = [], []
        for func in args:
            label.append(func)
            match func:
                case 'EFL':
                    efl_loss = self.efl_loss(args[func]['target'])
                    valids &= ~torch.isnan(efl_loss)
                    if writer: writer.add_scalar('optics/A/efl_loss', efl_loss[valids].mean(), count)
                    loss.append(efl_loss * args[func]['weight'])
                case 'FNO':
                    fno_loss = self.fno_loss(args[func]['target'])
                    valids &= ~torch.isnan(fno_loss)
                    if writer: writer.add_scalar('optics/A/fno_loss', fno_loss[valids].mean(), count)
                    loss.append(fno_loss * args[func]['weight'])
                case 'SPOT':
                    k = args[func]['k'] if 'k' in args[func] else 1.
                    spot_loss = self.spot_loss(args[func]['ref'], k=k, efl=args['EFL']['target'])
                    valids &= ~torch.isnan(spot_loss)
                    if writer: writer.add_scalar('optics/A/spot_loss', spot_loss[valids].mean(), count)
                    loss.append(spot_loss * args[func]['weight'])
                case 'WAVEFRONT':
                    wavefront_loss = self.wavefront_loss(args[func]['mode'])
                    valids &= ~torch.isnan(wavefront_loss)
                    if writer: writer.add_scalar('optics/A/wavefront_loss', wavefront_loss[valids].mean(), count)
                    loss.append(wavefront_loss * args[func]['weight'])
                case 'DISTOR':
                    absolute = args[func]['abs'] if 'abs' in args[func] else True
                    distor_loss = self.distor_loss(args[func]['target'], absolute)
                    valids &= ~torch.isnan(distor_loss)
                    if writer: writer.add_scalar('optics/A/distor_loss', distor_loss[valids].mean(), count)
                    loss.append(distor_loss * args[func]['weight'])
                case 'LATERAL':
                    lateral_loss = self.lateral_loss(args[func]['ref'])
                    valids &= ~torch.isnan(lateral_loss)
                    if writer: writer.add_scalar('optics/A/lateral_loss', lateral_loss[valids].mean(), count)
                    loss.append(lateral_loss * args[func]['weight'])
                case 'BFL':
                    bfl_loss = self.bfl_loss(args[func]['target'])
                    valids &= ~torch.isnan(bfl_loss)
                    if writer: writer.add_scalar('optics/B/bfl_loss', bfl_loss[valids].mean(), count)
                    loss.append(bfl_loss * args[func]['weight'])
                case 'TOTR':
                    totr_loss = self.totr_loss(args[func]['target'])
                    valids &= ~torch.isnan(totr_loss)
                    if writer: writer.add_scalar('optics/B/totr_loss', totr_loss[valids].mean(), count)
                    loss.append(totr_loss * args[func]['weight'])
                case 'GLA_MIN_THICK':
                    if args[func].get('td_ratio') is not None:
                        gla_min_thick_loss = self.gla_min_thick_loss(td_ratio=args[func]['td_ratio'])
                    elif args[func].get('min_thick') is not None:
                        gla_min_thick_loss = self.gla_min_thick_loss(min_thick=args[func]['min_thick'])
                    valids &= ~torch.isnan(gla_min_thick_loss)
                    if writer: writer.add_scalar('optics/B/gla_min_thick_loss', gla_min_thick_loss[valids].mean(), count)
                    loss.append(gla_min_thick_loss * args[func]['weight'])
                case 'GLA_MAX_THICK':
                    if args[func].get('td_ratio') is not None:
                        gla_max_thick_loss = self.gla_max_thick_loss(td_ratio=args[func]['td_ratio'])
                    elif args[func].get('max_thick') is not None:
                        gla_max_thick_loss = self.gla_max_thick_loss(max_thick=args[func]['max_thick'])
                    valids &= ~torch.isnan(gla_max_thick_loss)
                    if writer: writer.add_scalar('optics/B/gla_max_thick_loss', gla_max_thick_loss[valids].mean(), count)
                    loss.append(gla_max_thick_loss * args[func]['weight'])
                case 'GLA_MAX_MIN_RATIO':
                    gla_max_min_ratio_loss = self.gla_max_min_ratio_loss(args[func]['max_ratio'])
                    valids &= ~torch.isnan(gla_max_min_ratio_loss)
                    if writer: writer.add_scalar('optics/B/gla_max_min_ratio_loss', gla_max_min_ratio_loss[valids].mean(), count)
                    loss.append(gla_max_min_ratio_loss * args[func]['weight'])
                case 'SAG_DIA_MAX_RATIO':
                    sag_dia_max_ratio_loss = self.sag_dia_max_ratio_loss(args[func]['max_ratio'])
                    valids &= ~torch.isnan(sag_dia_max_ratio_loss)
                    if writer: writer.add_scalar('optics/B/sag_dia_max_ratio_loss', sag_dia_max_ratio_loss[valids].mean(), count)
                    loss.append(sag_dia_max_ratio_loss * args[func]['weight'])
                case 'AIR_THICK':
                    air_thick_loss = self.air_thick_loss(FF_target=args[func]['FF_target'], FM_target=args[func]['FM_target'], MM_target=args[func]['MM_target'])
                    valids &= ~torch.isnan(air_thick_loss)
                    if writer: writer.add_scalar('optics/B/air_thick_loss', air_thick_loss[valids].mean(), count)
                    loss.append(air_thick_loss * args[func]['weight'])
                case 'FIX_LENS':
                    fix_lens_loss = self.fix_lens_loss()
                    valids &= ~torch.isnan(fix_lens_loss)
                    if writer: writer.add_scalar('optics/B/fix_lens_loss', fix_lens_loss[valids].mean(), count)
                    loss.append(fix_lens_loss * args[func]['weight'])
                case 'SMOOTH_ZOOM':
                    smooth_zoom_loss = self.smooth_zoom_loss()
                    valids &= ~torch.isnan(smooth_zoom_loss)
                    if writer: writer.add_scalar('optics/B/smooth_zoom_loss', smooth_zoom_loss[valids].mean(), count)
                    loss.append(smooth_zoom_loss * args[func]['weight'])
                case 'SURF_K':
                    surf_k_loss = self.surf_k_loss(args[func]['target'])
                    valids &= ~torch.isnan(surf_k_loss)
                    if writer: writer.add_scalar('optics/B/surf_k_loss', surf_k_loss[valids].mean(), count)
                    loss.append(surf_k_loss * args[func]['weight'])
                case 'ANGLE':
                    angle_loss = self.angle_loss(args[func]['target'])
                    valids &= ~torch.isnan(angle_loss)
                    if writer: writer.add_scalar('optics/C/angle_loss', angle_loss[valids].mean(), count)
                    loss.append(angle_loss * args[func]['weight'])
                case 'CRA':
                    cra_loss = self.cra_loss(args[func]['target'])
                    valids &= ~torch.isnan(cra_loss)
                    if writer: writer.add_scalar('optics/C/cra_loss', cra_loss[valids].mean(), count)
                    loss.append(cra_loss * args[func]['weight'])
                case 'ANGLE_STD':
                    angle_std_loss = self.angle_std_loss()
                    valids &= ~torch.isnan(angle_std_loss)
                    if writer: writer.add_scalar('optics/C/angle_std_loss', angle_std_loss[valids].mean(), count)
                    loss.append(angle_std_loss * args[func]['weight'])
                case 'PUPIL':
                    pupil_loss = self.pupil_loss(args[func]['ref_point_n'])
                    valids &= ~torch.isnan(pupil_loss)
                    if writer: writer.add_scalar('optics/C/pupil_loss', pupil_loss[valids].mean(), count)
                    loss.append(pupil_loss * args[func]['weight'])
                case 'GLA_Z':
                    gla_z_loss = self.gla_z_loss(args[func]['z_min'])
                    valids &= ~torch.isnan(gla_z_loss)
                    if writer: writer.add_scalar('optics/C/gla_z_loss', gla_z_loss[valids].mean(), count)
                    loss.append(gla_z_loss * args[func]['weight'])
                case 'ROC':
                    roc_loss = []
                    for i in args[func]:
                        roc_loss.append(self.roc_loss(args[func][i]['s_id'], args[func][i]['sign']) * args[func][i]['weight'])
                    roc_loss = reduce(lambda x, y: x + y, roc_loss)
                    valids &= ~torch.isnan(roc_loss)
                    if writer: writer.add_scalar('optics/C/roc_loss', roc_loss[valids].mean(), count)
                    loss.append(roc_loss)
        
        self.sys.valid = valids
        if path: plot_loss_pie(loss, label, self.sys.valid, path)
        loss = reduce(lambda x, y: x + y, loss)
        return loss # [sys]
    
    def air_thick_loss(self, FF_target, FM_target, MM_target):
        airs_res = torch.tensor([])
        surfs = self.sys.extract_surfs()

        for i, elem in enumerate(surfs[1:-2]):  # skip the last air
            if 'VACUUM' in elem.material['name']: # start count from the air surf
                if i == 0 and self.sys.stop_id == 1: # if the stop is the first surface, skip it
                    r = elem.radius[None, :, :, None, None, None] # [wav, sys, cfg, ang, azi, M]
                    air_res = elem.thick + surfs[1].surface(r, 0).squeeze(dim=[0, 3, 4, 5]) # [sys, cfg]
                    airs_res = torch.cat([airs_res, torch.where(air_res < 0, -air_res, 0).unsqueeze(0)])
                    continue
                
                elem_aft = surfs[i + 2]
                acc_thick = elem.thick

                r = torch.linspace(0., 1., self.sys.surf_samp) * (1 + self.sys.clear_margin)
                r = (r[None, None, :] * torch.min(elem.radius, elem_aft.radius)[:, :, None])[None, :, :, None, None, :] # [wav, sys, cfg, ang, azi, M]
                sag = elem.surface(r, 0).squeeze(dim=[0, 3, 4]) # [sys, cfg, M]
                sag_aft = elem_aft.surface(r, 0).squeeze(dim=[0, 3, 4])
                if self.sys.zoom_type[i][-1] == 'F': 
                    air_res = (acc_thick[:, :, None] - sag + sag_aft).amin(dim=-1) - FF_target # [sys, cfg]
                else:
                    if self.sys.zoom_type[i][0] == 'F':
                        air_res = (acc_thick[:, :, None] - sag + sag_aft).amin(dim=-1) - FM_target # [sys, cfg]
                    else:
                        if self.sys.zoom_type[i+1][0] == 'F':
                            air_res = (acc_thick[:, :, None] - sag + sag_aft).amin(dim=-1) - FM_target # [sys, cfg]
                        else:
                            air_res = (acc_thick[:, :, None] - sag + sag_aft).amin(dim=-1) - MM_target # [sys, cfg]
                            
                airs_res = torch.cat([airs_res, torch.where(air_res < 0, -air_res, 0).unsqueeze(0)])
        return (airs_res ** 2).sum(dim=[0, -1])
    
    def fix_lens_loss(self):
        fix_lens = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, _ in enumerate(surfs[1:-1]):
            if self.sys.zoom_type[i] == 'FF':
                subs = sum(elem.thick for elem in surfs[1:-1][i:])
                fix_lens = torch.cat([fix_lens, subs.unsqueeze(0)]) # [surfs, sys, cfg]
        if fix_lens.numel() == 0:
            return torch.zeros(self.sys.sys_num)
        else:
            loss = fix_lens - fix_lens.mean(dim=-1, keepdim=True)
            return (loss ** 2).sum(dim=[0, -1])
    
    def smooth_zoom_loss(self):
        sort_loss = torch.tensor([])
        surfs = self.sys.extract_surfs()
        for i, _ in enumerate(surfs[1:-1]):
            if self.sys.zoom_type[i] == 'MM':
                thick = sum(elem.thick for elem in surfs[1:-1][i:])
                loss_des = torch.where((thick[:, 1:] - thick[:, :-1]) > 0., thick[:, 1:] - thick[:, :-1], 0.).sum(dim=-1) # [sys]
                loss_asc = torch.where((thick[:, :-1] - thick[:, 1:]) > 0., thick[:, :-1] - thick[:, 1:], 0.).sum(dim=-1) # [sys]
                loss = torch.where(loss_des > loss_asc, loss_asc, loss_des) # [sys]
                sort_loss = torch.cat([sort_loss, loss.unsqueeze(0)]) # [surfs, sys]
        return sort_loss.sum(dim=0) # [sys]
    