import json
import torch
import torch.nn as nn
from functools import reduce
from tqdm import tqdm
from einops import rearrange

from .surface import OBJECT, IMAGE, Sphere, Asphere, Qcon, Qbfs, Binary2, Coordinate, Dummy, PACKAGE
from .delano import Delano
from.zoom import Zoom
from .utils import fit_get_mat_id, glass_catalog, glass_catalog_params, plastic_catalog, plastic_catalog_params, list_convert, Ray, pupil_distribution, normalize, eps, length

class System(nn.Module):
    def __init__(self, wavelengths, p_wvl, max_view, waveweights=None, sys_num=1, cfg_num=1, pre_samp:int=None, stop_max_samp_ang:float=45., samp_method='square', fix_radius_surf:list=[], norm_views=None, azimuths=None, vig=None, **kwargs):
        super(System, self).__init__()
        self.sys_num = sys_num
        self.cfg_num = cfg_num
        self.valid = torch.ones(self.sys_num).bool()
        
        self.max_view = torch.tensor(max_view) if isinstance(max_view, list) else torch.tensor(max_view).repeat(self.cfg_num)
        self.norm_views = torch.tensor([0., 0.3, 0.5, 0.707, 0.85, 1.0]) if norm_views is None else torch.tensor(norm_views)
        self.azimuths = torch.tensor([0., 90., 180., 270.]) if azimuths is None else torch.tensor(azimuths)
        self.wavelengths = torch.tensor(wavelengths)
        self.waveweights = torch.tensor(waveweights) if waveweights is not None else torch.ones(len(wavelengths))
        self.p_wvl = p_wvl
        
        self.clear_margin = 0.005
        self.samp_method = samp_method
        self.surf_samp = 512
        self.vig_chief = False
        self.vig = vig
        
        # for large pupil aberration systems
        self.pre_samp = pre_samp
        self.stop_max_samp_ang = torch.tensor(stop_max_samp_ang).deg2rad()
        
        if 'file' in kwargs:
            self.system, self.stop_id, self.zoom_type = self.read_sys(kwargs['file'])
            self.update()
            for i in fix_radius_surf: self.system[i].fix_radius = True
            self.samp_margin = 0.
        elif 'delano' in kwargs:
            self.paraxial_opt = True
            self.paraxial_scale = 20 # 100/20/4/1
            self.system, self.stop_id = self.delano_sys(kwargs['delano'][0], kwargs['delano'][1], kwargs['delano'][2], kwargs['delano'][3], kwargs['delano'][4])
            self.update()
            for i in fix_radius_surf: self.system[i].fix_radius = True
            self.samp_margin = 0.
        elif 'zoom' in kwargs:
            self.paraxial_opt = True
            self.paraxial_scale = 20 # 100/20/4/1
            self.system, self.stop_id, self.zoom_type = self.zoom_sys(kwargs['zoom'][0], kwargs['zoom'][1], kwargs['zoom'][2], kwargs['zoom'][3], kwargs['zoom'][4])
            self.update()
            for i in fix_radius_surf: self.system[i].fix_radius = True
            self.samp_margin = 0.
        elif 'random' in kwargs:
            self.paraxial_scale = 20 # 100/20/4/1
            self.system, self.stop_id = self.rand_sys(kwargs['random'][0], kwargs['random'][1], kwargs['random'][2], kwargs['random'][3], kwargs['random'][4], kwargs['random'][5])
            self.update()
            for i in fix_radius_surf: self.system[i].fix_radius = True
            self.samp_margin = 0.
        elif 'random_zoom' in kwargs:
            self.paraxial_scale = 20 # 100/20/4/1
            self.system, self.stop_id, self.zoom_type = self.rand_zoom_sys(kwargs['random_zoom'][0], kwargs['random_zoom'][1], kwargs['random_zoom'][2], kwargs['random_zoom'][3], kwargs['random_zoom'][4], kwargs['random_zoom'][5],  kwargs['random_zoom'][6])
            self.update()
            for i in fix_radius_surf: self.system[i].fix_radius = True
            self.samp_margin = 0.

    def read_sys(self, file):
        with open(file) as file:
            lens_dict = json.load(file)
        file.close()
        zoom_type = []
        
        sys = nn.ModuleList([])
        sys.append(OBJECT(material=lens_dict['OBJECT']['material'], distance=lens_dict['OBJECT']['distance'] if isinstance(lens_dict['OBJECT']['distance'], list) else [lens_dict['OBJECT']['distance']] * self.cfg_num))
        for i, item in enumerate(list(lens_dict)[1:-1]):
            if lens_dict[item]['material'] in glass_catalog:
                mat_cata = 'G'
            elif lens_dict[item]['material'] in plastic_catalog:
                mat_cata = 'P'
            else:
                mat_cata = None
            common_params = {
                'radius': [lens_dict[item]['radius'] if isinstance(lens_dict[item]['radius'], list) else [lens_dict[item]['radius']] * self.cfg_num] * self.sys_num,
                'material': [lens_dict[item]['material']] * self.sys_num,
                'roc': [lens_dict[item]['roc']] * self.sys_num,
                'thick': [lens_dict[item]['thick'] if isinstance(lens_dict[item]['thick'], list) else [lens_dict[item]['thick']] * self.cfg_num] * self.sys_num,
                'conic': [lens_dict[item]['conic']] * self.sys_num,
                'mat_cata': mat_cata,
                'aperture': lens_dict[item].get('aperture', 'float'),
                'min_r': [lens_dict[item].get('min_r', 0.)] * self.sys_num,
                'max_r': [lens_dict[item].get('max_r', 0.)] * self.sys_num,
            }
            if lens_dict[item]['type'] == 'Standard':
                sys.append(Sphere(**common_params))
            elif lens_dict[item]['type'] == 'Asphere':
                sys.append(Asphere(**common_params,
                    ai_list=[[x] * self.sys_num for x in lens_dict[item]['ai_list']],
                ))
            elif lens_dict[item]['type'] == 'Qcon':
                sys.append(Qcon(**common_params,
                    qi_list=[[x] * self.sys_num for x in lens_dict[item]['qi_list']],
                    rnorm=[lens_dict[item]['rnorm']] * self.sys_num,
                ))
            elif lens_dict[item]['type'] == 'Qbfs':
                sys.append(Qbfs(**common_params,
                    qi_list=[[x] * self.sys_num for x in lens_dict[item]['qi_list']],
                    rnorm=[lens_dict[item]['rnorm']] * self.sys_num,
                ))
            elif lens_dict[item]['type'] == 'Binary2':
                sys.append(Binary2(**common_params,
                    ai_list=[[x] * self.sys_num for x in lens_dict[item]['ai_list']],
                    diff_order=lens_dict[item]['diff_order'],
                    pi_list=[[x] * self.sys_num for x in lens_dict[item]['pi_list']],
                    rnorm=[lens_dict[item]['rnorm']] * self.sys_num,
               ))
                
            if 'zoom_type' in lens_dict[item]:
                zoom_type.append(lens_dict[item]['zoom_type'])
            else:
                zoom_type.append(None)
            
            if lens_dict[item]['stop']:
                stop_id = i + 1
        sys.append(IMAGE(radius=[lens_dict['IMAGE']['radius'] if isinstance(lens_dict['IMAGE']['radius'], list) else [lens_dict['IMAGE']['radius']] * self.cfg_num] * self.sys_num))
        zoom_type.append('FF')
        return sys, stop_id, zoom_type

        
    def delano_sys(self, delano:Delano, stype:str, mat_type:str, mat_cata:str, merit:dict):
        if self.paraxial_opt == False:
            sys, stop_id = delano.lens_instance(self.sys_num, self.cfg_num, stype, mat_type, mat_cata)
            return sys, stop_id
        
        sys_num = self.sys_num * self.paraxial_scale
        asp_terms = 7
        qcon_terms = 10
        
        sys, stop_id = delano.lens_instance(sys_num, self.cfg_num, stype, mat_type, mat_cata)
        sys[stop_id].roc.requires_grad_(False)
        
        #! -> Optimize using paraxial ray tracing
        scale = merit['EFL']['target']
        sys = self.simulated_annealing(sys, stop_id, sys_num, merit, scale)
        
        #! -> Update the radius of the system
        with torch.no_grad():
            loss, R = self.paraxial_loss(sys, stop_id, sys_num, merit)
        _, idx = torch.topk(torch.nan_to_num(loss, torch.inf), self.sys_num, largest=False)
        new_sys = nn.ModuleList([])
        new_sys.append(OBJECT(material='VACUUM', distance=[None] * self.cfg_num))
        for i, elem in enumerate(sys[1:-1]):
            common_params = {
                'radius': (R[i, idx]).tolist(),
                'material': [elem.material['name'][i] for i in idx],
                'roc': (elem.roc[idx] ** -1).tolist() if (elem.roc[idx] != 0.).any() else [None] * self.sys_num,
                'thick': elem.thick[idx].tolist(),
                'conic': [0.0] * self.sys_num,
                'mat_cata': elem.mat_cata,
            }
            match elem.__class__.__name__:
                case 'Sphere':
                    new_sys.append(Sphere(**common_params))
                case 'Asphere':
                    new_sys.append(Asphere(**common_params, ai_list=[[0.0] * self.sys_num] * asp_terms))
                case 'Qcon':
                    new_sys.append(Qcon(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
                case 'Qbfs':
                    new_sys.append(Qbfs(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
        new_sys.append(IMAGE(radius=((self.max_view.deg2rad().tan() * merit['EFL']['target']).repeat(self.sys_num, self.cfg_num)).tolist()))
        return new_sys, stop_id
    
    
    def zoom_sys(self, zoom:Zoom, stype:str, mat_type:str, mat_cata:str, merit:dict):
        if self.paraxial_opt == False:
            sys, stop_id, zoom_type = zoom.lens_instance(self.sys_num, self.cfg_num, stype, mat_type, mat_cata)
            return sys, stop_id, zoom_type
        
        sys_num = self.sys_num * self.paraxial_scale
        asp_terms = 7
        qcon_terms = 10
        
        sys, stop_id, zoom_type = zoom.lens_instance(sys_num, self.cfg_num, stype, mat_type, mat_cata)
        sys[stop_id].roc.requires_grad_(False)
        
        scale = min(merit['EFL']['target'])
        sys = self.simulated_annealing(sys, stop_id, sys_num, merit, scale)
        
        #! -> Update the radius of the system
        with torch.no_grad():
            loss, R = self.paraxial_loss(sys, stop_id, sys_num, merit)
        _, idx = torch.topk(torch.nan_to_num(loss, torch.inf), self.sys_num, largest=False)
        new_sys = nn.ModuleList([])
        new_sys.append(OBJECT(material='VACUUM', distance=[None] * self.cfg_num))
        for i, elem in enumerate(sys[1:-1]):
            common_params = {
                'radius': (R[i, idx].amax(dim=-1)[:, None].repeat(1, self.cfg_num)).tolist(),
                'material': [elem.material['name'][i] for i in idx],
                'roc': (elem.roc[idx] ** -1).tolist() if (elem.roc[idx] != 0.).any() else [None] * self.sys_num,
                'thick': elem.thick[idx].tolist(),
                'conic': [0.0] * self.sys_num,
                'mat_cata': elem.mat_cata,
            }
            match elem.__class__.__name__:
                case 'Sphere':
                    new_sys.append(Sphere(**common_params))
                case 'Asphere':
                    new_sys.append(Asphere(**common_params, ai_list=[[0.0] * self.sys_num] * asp_terms))
                case 'Qcon':
                    new_sys.append(Qcon(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
                case 'Qbfs':
                    new_sys.append(Qbfs(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
        new_sys.append(IMAGE(radius=(self.max_view.deg2rad().tan() * torch.tensor(merit['EFL']['target']))[None, :].repeat(self.sys_num, 1).tolist()))    
        return new_sys, stop_id, zoom_type


    def rand_zoom_sys(self, structure:str, elem_type:str, surf_type:str, mat_type:str, mat_cata:str, stop_pos:int, merit:dict):
        """
        Only for prime lens.
        """
        sys_num = self.sys_num * self.paraxial_scale
        asp_terms = 7
        qcon_terms = 10
        
        elem_type = elem_type.split('|')
        surf_type = surf_type.split('|')
        mat_type = mat_type.split('|')
        mat_cata = mat_cata.split('|')
        
        surf_nums = sum([len(surfs) for surfs in surf_type]) * 2 + 1
        thick_scale = (merit['TOTR']['target'] - merit['BFL']['target']) / (surf_nums - 1)
        roc_scale = 1.e2 * max(merit['EFL']['target'])
        vd_threshold = 50.
        
        sys = nn.ModuleList([])
        sys.append(OBJECT(material='VACUUM', distance=[None] * self.cfg_num))
        
        ss = 1
        zoom_type = []
        #! -> initialize the system
        for g, g_type in enumerate(structure):
            if g == stop_pos:
                common_params = {
                    'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                    'material': ["VACUUM"] * sys_num,
                    'roc': [None] * sys_num,
                    'thick': (thick_scale * torch.zeros(sys_num, self.cfg_num)).tolist(),
                    'conic': [0.0] * sys_num,
                }
                sys.append(Sphere(**common_params))
                if g_type == 'F':
                    zoom_type.append('FF')
                elif g_type == 'M':
                    zoom_type.append('MF')
                else:
                    raise ValueError(f"Unknown group type: {g_type}")
                stop_id = ss
            
            for s, stype in enumerate(surf_type[g]):
                if mat_cata[g][s] == 'G':
                    catalog = glass_catalog
                    catalog_params = glass_catalog_params
                elif mat_cata[g][s] == 'P':
                    catalog = plastic_catalog
                    catalog_params = plastic_catalog_params
                else:
                    raise ValueError(f"Unknown material catalog for {mat_cata[g][s]}")
                
                if elem_type[g][s] == 'S':
                    # surf pre
                    material = []
                    for _ in range(sys_num):
                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                        if mat_type[g][s] == 'K':
                            while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            material.append(list(catalog)[idx])
                        elif mat_type[g][s] == 'F':
                            while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            material.append(list(catalog)[idx])
                        elif mat_type[g][s] == 'R':
                            material.append(list(catalog)[idx])
                        else:
                            raise ValueError(f"Unknown material type for S: {mat_type[g][s]}")
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                        'mat_cata': mat_cata[g][s],
                    }
                    match stype:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    if g_type == 'F':
                        zoom_type.append('FF')
                    elif g_type == 'M':
                        zoom_type.append('MF')
                    else:
                        raise ValueError(f"Unknown group type: {g_type}")
                    ss += 1

                    # surf after
                    material = ['VACUUM'] * sys_num
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                    }
                    match stype:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    if g_type == 'F':
                        if s != len(surf_type[g]) - 1:
                            zoom_type.append('FF')
                        else:
                            if g != len(structure) - 1:
                                if structure[g + 1] == 'F':
                                    zoom_type.append('FF')
                                else:
                                    zoom_type.append('FM')
                            else:
                                zoom_type.append('FF')
                    elif g_type == 'M':
                        if s != len(surf_type[g]) - 1:
                            zoom_type.append('MF')
                        else:
                            zoom_type.append('MM')
                    else:
                        raise ValueError(f"Unknown group type: {g_type}")
                    ss += 1
                elif elem_type[g][s] == 'D':
                    randmat = torch.rand(sys_num)
                    
                    # surf pre
                    material = []
                    for _i_ in range(sys_num):
                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                        if mat_type[g][s] == 'M':
                            if randmat[_i_] < 0.5:
                                while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                            else:
                                while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                        elif mat_type[g][s] == 'R':
                            material.append(list(catalog)[idx])
                        else:
                            raise ValueError(f"Unknown material type for D: {mat_type[g][s]}")
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                        'mat_cata': mat_cata[g][s],
                    }
                    match stype:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    if g_type == 'F':
                        zoom_type.append('FF')
                    elif g_type == 'M':
                        zoom_type.append('MF')
                    else:
                        raise ValueError(f"Unknown group type: {g_type}")
                    ss += 1
                    
                    # surf middle
                    material = []
                    for _i_ in range(sys_num):
                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                        if mat_type[g][s] == 'M':
                            if randmat[_i_] > 0.5:
                                while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                            else:
                                while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                        elif mat_type[g][s] == 'R':
                            material.append(list(catalog)[idx])
                        else:
                            raise ValueError(f"Unknown material type for D: {mat_type[g][s]}")
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                        'mat_cata': mat_cata[g][s],
                    }
                    match stype:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    if g_type == 'F':
                        zoom_type.append('FF')
                    elif g_type == 'M':
                        zoom_type.append('MF')
                    else:
                        raise ValueError(f"Unknown group type: {g_type}")
                    ss += 1
                    
                    # surf after
                    material = ['VACUUM'] * sys_num
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                    }
                    match stype:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    if g_type == 'F':
                        if s != len(surf_type[g]) - 1:
                            zoom_type.append('FF')
                        else:
                            if g != len(structure) - 1:
                                if structure[g + 1] == 'F':
                                    zoom_type.append('FF')
                                else:
                                    zoom_type.append('FM')
                            else:
                                zoom_type.append('FF')
                    elif g_type == 'M':
                        if s != len(surf_type[g]) - 1:
                            zoom_type.append('MF')
                        else:
                            zoom_type.append('MM')
                    else:
                        raise ValueError(f"Unknown group type: {g_type}")
                    ss += 1
                else:
                    raise ValueError(f"Unknown element type: {elem_type[g][s]}")
                
        sys[-1].thick.data = torch.ones_like(sys[-1].thick) * merit['BFL']['target']                    
        height = self.max_view.deg2rad().tan() * torch.tensor(merit['EFL']['target'])
        sys.append(IMAGE(radius=(height.max().repeat(sys_num, self.cfg_num)).tolist()))
        zoom_type.append('FF')
        
        # get sys, stop_id, zoom_type        
        sys[stop_id].roc.requires_grad_(False)
        
        scale = min(merit['EFL']['target'])
        sys = self.simulated_annealing(sys, stop_id, sys_num, merit, scale)
        
        #! -> Update the radius of the system
        with torch.no_grad():
            loss, R = self.paraxial_loss(sys, stop_id, sys_num, merit)
        _, idx = torch.topk(torch.nan_to_num(loss, torch.inf), self.sys_num, largest=False)
        new_sys = nn.ModuleList([])
        new_sys.append(OBJECT(material='VACUUM', distance=[None] * self.cfg_num))
        for i, elem in enumerate(sys[1:-1]):
            common_params = {
                'radius': (R[i, idx].amax(dim=-1)[:, None].repeat(1, self.cfg_num)).tolist(),
                'material': [elem.material['name'][i] for i in idx],
                'roc': (elem.roc[idx] ** -1).tolist() if (elem.roc[idx] != 0.).any() else [None] * self.sys_num,
                'thick': elem.thick[idx].tolist(),
                'conic': [0.0] * self.sys_num,
                'mat_cata': elem.mat_cata,
            }
            match elem.__class__.__name__:
                case 'Sphere':
                    new_sys.append(Sphere(**common_params))
                case 'Asphere':
                    new_sys.append(Asphere(**common_params, ai_list=[[0.0] * self.sys_num] * asp_terms))
                case 'Qcon':
                    new_sys.append(Qcon(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
                case 'Qbfs':
                    new_sys.append(Qbfs(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
        new_sys.append(IMAGE(radius=(self.max_view.deg2rad().tan() * torch.tensor(merit['EFL']['target']))[None, :].repeat(self.sys_num, 1).tolist()))    
        return new_sys, stop_id, zoom_type

    
    def rand_sys(self, structure:str, surf_type:str, mat_type:str, mat_cata:str, stop_pos:int, merit:dict):
        """
        Only for prime lens.
        """
        sys_num = self.sys_num * self.paraxial_scale
        asp_terms = 7
        qcon_terms = 10
        
        structure = structure.split('|')
        surf_type = surf_type.split('|')
        mat_type = mat_type.split('|')
        mat_cata = mat_cata.split('|')
        surf_nums = len(structure) + structure.count('D') + 1 # stop surface

        thick_scale = (merit['TOTR']['target'] - merit['BFL']['target']) / (surf_nums - 1)
        roc_scale = 1.e2 * merit['EFL']['target']
        vd_threshold = 50.
        
        sys = nn.ModuleList([])
        sys.append(OBJECT(material='VACUUM', distance=[None] * self.cfg_num))
        
        if stop_pos == 0:
            common_params = {
                'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                'material': ["VACUUM"] * sys_num,
                'roc': [None] * sys_num,
                'thick': (thick_scale * torch.zeros(sys_num, self.cfg_num)).tolist(),
                'conic': [0.0] * sys_num,
            }
            sys.append(Sphere(**common_params))
            stop_id = 1

        #! -> initialize the system
        ss = 0
        for j, s in enumerate(structure):
            if mat_cata[j] == 'G':
                catalog = glass_catalog
                catalog_params = glass_catalog_params
            elif mat_cata[j] == 'P':
                catalog = plastic_catalog
                catalog_params = plastic_catalog_params
            else:
                raise ValueError(f"Unknown material catalog for {mat_cata[j]}")
            
            match s:
                case 'S':
                    ss += 2
                    material = []
                    for _ in range(sys_num):
                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                        if mat_type[j] == 'K':
                            while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            material.append(list(catalog)[idx])
                        elif mat_type[j] == 'F':
                            while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                            material.append(list(catalog)[idx])
                        elif mat_type[j] == 'R':
                            material.append(list(catalog)[idx])
                        else:
                            raise ValueError(f"Unknown material type for S: {mat_type[j]}")
                        
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                        'mat_cata': mat_cata[j],
                    }
                    match surf_type[j]:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    
                    
                    material = ['VACUUM'] * sys_num
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                    }
                    match surf_type[j]:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    
                case 'D':
                    ss += 3
                    randmat = torch.rand(sys_num)
                    material = []
                    for _i_ in range(sys_num):
                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                        if mat_type[j] == 'M':
                            if randmat[_i_] < 0.5:
                                while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                            else:
                                while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                        elif mat_type[j] == 'R':
                            material.append(list(catalog)[idx])
                        else:
                            raise ValueError(f"Unknown material type for D: {mat_type[j]}")
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                        'mat_cata': mat_cata[j],
                    }
                    match surf_type[j]:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    
                    
                    material = []
                    for _i_ in range(sys_num):
                        idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                        if mat_type[j] == 'M':
                            if randmat[_i_] > 0.5:
                                while catalog[list(catalog)[idx]]['vd'] < vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                            else:
                                while catalog[list(catalog)[idx]]['vd'] > vd_threshold:
                                    idx = torch.randint(0, catalog_params.shape[0], (1,)).tolist()[0]
                                material.append(list(catalog)[idx])
                        elif mat_type[j] == 'R':
                            material.append(list(catalog)[idx])
                        else:
                            raise ValueError(f"Unknown material type for D: {mat_type[j]}")
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                        'mat_cata': mat_cata[j],
                    }
                    match surf_type[j]:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                    
                
                    material = ['VACUUM'] * sys_num
                    common_params = {
                        'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                        'material': material,
                        'roc': torch.randn(sys_num) * roc_scale,
                        'thick': (torch.rand(sys_num) * thick_scale)[:, None].repeat(1, self.cfg_num).tolist(),
                        'conic': [0.0] * sys_num,
                    }
                    match surf_type[j]:
                        case 'S':
                            sys.append(Sphere(**common_params))
                        case 'A':
                            sys.append(Asphere(**common_params, ai_list=[[0.0] * sys_num] * asp_terms))
                        case 'Q':
                            sys.append(Qcon(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
                        case 'q':
                            sys.append(Qbfs(**common_params, qi_list=[[0.0] * sys_num] * qcon_terms, rnorm=torch.ones(sys_num).tolist()))
            if j + 1 == stop_pos:
                common_params = {
                    'radius': (torch.ones(sys_num, self.cfg_num)).tolist(),
                    'material': ["VACUUM"] * sys_num,
                    'roc': [None] * sys_num,
                    'thick': (thick_scale * torch.zeros(sys_num, self.cfg_num)).tolist(),
                    'conic': [0.0] * sys_num,
                }
                sys.append(Sphere(**common_params))
                stop_id = ss + 1
                    
        sys[-1].thick.data = torch.ones_like(sys[-1].thick) * merit['BFL']['target']                    
        sys.append(IMAGE(radius=((self.max_view.deg2rad().tan() * merit['EFL']['target']).repeat(sys_num, self.cfg_num)).tolist()))
        
        # get sys, stop_id
        sys[stop_id].roc.requires_grad_(False)
        
        #! -> Optimize using paraxial ray tracing
        scale = merit['EFL']['target']
        sys = self.simulated_annealing(sys, stop_id, sys_num, merit, scale)
        
        #! -> Update the radius of the system
        with torch.no_grad():
            loss, R = self.paraxial_loss(sys, stop_id, sys_num, merit)
        _, idx = torch.topk(torch.nan_to_num(loss, torch.inf), self.sys_num, largest=False)
        new_sys = nn.ModuleList([])
        new_sys.append(OBJECT(material='VACUUM', distance=[None] * self.cfg_num))
        for i, elem in enumerate(sys[1:-1]):
            common_params = {
                'radius': (R[i, idx]).tolist(),
                'material': [elem.material['name'][i] for i in idx],
                'roc': (elem.roc[idx] ** -1).tolist() if (elem.roc[idx] != 0.).any() else [None] * self.sys_num,
                'thick': elem.thick[idx].tolist(),
                'conic': [0.0] * self.sys_num,
                'mat_cata': elem.mat_cata,
            }
            match elem.__class__.__name__:
                case 'Sphere':
                    new_sys.append(Sphere(**common_params))
                case 'Asphere':
                    new_sys.append(Asphere(**common_params, ai_list=[[0.0] * self.sys_num] * asp_terms))
                case 'Qcon':
                    new_sys.append(Qcon(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
                case 'Qbfs':
                    new_sys.append(Qbfs(**common_params, qi_list=[[0.0] * self.sys_num] * qcon_terms, rnorm=(R[i, idx].amax(dim=-1)).tolist()))
        new_sys.append(IMAGE(radius=((self.max_view.deg2rad().tan() * merit['EFL']['target']).repeat(self.sys_num, self.cfg_num)).tolist()))
        return new_sys, stop_id
    
    
    @torch.no_grad()
    def simulated_annealing(self, sys, stop_id, sys_num, merit, scale, T=100., T_min=1., step=0.001, alpha=0.95, iter=100, ptresh=0.5):
        k = 1
        def extract_opt_data(sys):
            """
            Extract the opt data from the system.
            """
            opt_data = {}
            for i, surf in enumerate(sys[1:-1]):
                opt_data[i] = {}
                if surf.thick.requires_grad:
                    opt_data[i]['thick'] = surf.thick.detach().clone()
                if surf.roc.requires_grad:
                    opt_data[i]['roc'] = surf.roc.detach().clone()
            return opt_data
        
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
                            one_step = step * scale
                        case 'roc':
                            one_step = step / scale
                    data = opt_data[i][key]
                    mask = torch.rand(sys_num) < p
                    noise = mask * one_step * (torch.rand(sys_num) - 0.5) * 2
                    if data.dim() == 1:
                        opt_data_new[i][key] = data + noise
                    else:
                        opt_data_new[i][key] = data + noise[:, None]
            return opt_data_new
        
        def fit_opt_data(sys, opt_data):
            """
            Fit the opt data to the system.
            """
            for i, surf in enumerate(sys[1:-1]):
                for key in opt_data[i]:
                    data = opt_data[i][key]
                    exec(f'surf.{key}.data = data')
        
        opt_data_min = extract_opt_data(sys)
        loss_min, _ = self.paraxial_loss(sys, stop_id, sys_num, merit)
        
        pbar = tqdm()
        while T >= T_min:
            for i in range(iter):
                opt_data = extract_opt_data(sys)
                loss, _ = self.paraxial_loss(sys, stop_id, sys_num, merit)
                
                opt_data_new = perturb_opt_data(opt_data, step * T, ptresh)
                fit_opt_data(sys, opt_data_new)
                loss_new, _ = self.paraxial_loss(sys, stop_id, sys_num, merit)
                
                valid = loss_new < loss_min
                for s in opt_data_min:
                    for key in opt_data_min[s]:
                        opt_data_min[s][key][valid] = opt_data_new[s][key][valid]
                loss_min[valid] = loss_new[valid]
                
                valid = loss_new < loss
                for s in opt_data:
                    for key in opt_data[s]:
                        opt_data[s][key][valid] = opt_data_new[s][key][valid]
                
                p = torch.exp(-(loss_new - loss) / (k * T))[~valid]
                r = torch.rand_like(p)
                valid_bad = r < p
                for s in opt_data:
                    for key in opt_data[s]:
                        opt_data[s][key][~valid][valid_bad] = opt_data_new[s][key][~valid][valid_bad]

                fit_opt_data(sys, opt_data)
                pbar.set_description_str(f'T: {T:.4f}, loss_min: {loss_min.min().item():.4f}, loss_mean: {loss_min.mean().item():.4f}, new_loss_mean: {loss_new.mean().item():.4f}')
            T = T * alpha
            fit_opt_data(sys, opt_data_min)
        
        fit_opt_data(sys, opt_data_min)
        return sys
        
    
    def paraxial_loss(self, sys, stop_id, sys_num, merit):
        """
        Calculate the paraxial loss of the system.
        """
        abcds = [elem.abcd(sys[i], self.wavelengths[self.p_wvl][..., None]) for i, elem in enumerate(sys[1:-1])]
        abcd_pre_s = reduce((lambda x, y: torch.matmul(y, x)), abcds[:stop_id - 1]) if stop_id > 1 else torch.tensor([[1., 0], [0., 1.]])[None, None, :, :].repeat(sys_num, self.cfg_num, 1, 1)
        abcd = reduce((lambda x, y: torch.matmul(y, x)), abcds)
        
        EFFL = -1 / abcd[:, :, 1, 0] # [sys, cfg]

        ENPP = abcd_pre_s[:, :, 0, 1] / abcd_pre_s[:, :, 0, 0]
        ubar0 = torch.tan(self.max_view.deg2rad())[None, :].repeat(sys_num, 1) # [sys, cfg]
        ybar0 = ubar0 * (0 - ENPP)
        u0 = torch.zeros(sys_num, self.cfg_num)
        ENPD = (torch.tensor(merit['EFL']['target']) / torch.tensor(merit['FNO']['target']))[None, ...].repeat(sys_num, 1)
        y0 = ENPD / 2
        
        ubar = ubar0[None, ...]
        ybar = ybar0[None, ...]
        u = u0[None, ...]
        y = y0[None, ...]
        n = torch.ones([1, sys_num])
        dn = torch.zeros([1, sys_num])
        c = torch.zeros([1, sys_num])
        
        for i in range(1, len(sys)-1):
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
            n = torch.cat([n, sys[i].refractive_index(self.wavelengths)[self.p_wvl][None, :]], dim=0)
            _, wave_id = torch.sort(self.wavelengths)
            n1 = sys[i].refractive_index(self.wavelengths)[wave_id[0]]
            n2 = sys[i].refractive_index(self.wavelengths)[wave_id[-1]]
            dn = torch.cat([dn, (n1 - n2)[None, :]], dim=0)
            c = torch.cat([c, sys[i].roc[None, :]], dim=0)
            
        delta_d = ((0 - y0) / (u0 + eps)).mean(dim=-1) # [sys]
        sys[-2].thick.data = sys[-2].thick.data + delta_d[:, None]
        
        ybar = torch.cat([ybar, ybar0[None, ...]], dim=0)
        ubar = torch.cat([ubar, ubar0[None, ...]], dim=0)
        y = torch.cat([y, y0[None, ...]], dim=0)
        u = torch.cat([u, u0[None, ...]], dim=0)
        n = torch.cat([n, torch.ones([1, sys_num])], dim=0)
        dn = torch.cat([dn, torch.zeros([1, sys_num])], dim=0)
        c = torch.cat([c, torch.zeros([1, sys_num])], dim=0)

        A = n[:-1, :, None] * u[:-1] + n[:-1, :, None] * y[:-1] * c[1:, :, None]
        Abar = n[:-1, :, None] * ubar[:-1] + n[:-1, :, None] * ybar[:-1] * c[1:, :, None]
        H = Abar * y[:-1] - A * ybar[:-1] # Q = n0 * y0 * ubar0
        
        SI = -A ** 2 * y[:-1] * (u[1:] / n[1:, :, None] - u[:-1] / n[:-1, :, None]) # [surf, sys, cfg]
        loss_SI = SI.sum(dim=[0, -1]).abs()
        SII = -Abar * A * y[:-1] * (u[1:] / n[1:, :, None] - u[:-1] / n[:-1, :, None]) # [surf, sys, cfg]
        loss_SII = SII.sum(dim=[0, -1]).abs()
        SIII = -Abar ** 2 * y[:-1] * (u[1:] / n[1:, :, None] - u[:-1] / n[:-1, :, None]) # [surf, sys, cfg]
        loss_SIII = SIII.sum(dim=[0, -1]).abs()
        SIV = H ** 2 * ((n[1:] - n[:-1]) * c[1:] / (n[1:] * n[:-1]))[:, :, None] # [surf, sys, cfg]
        loss_SIV = SIV.sum(dim=[0, -1]).abs()
        SV = Abar / (A + eps) * (SIII + SIV) # [surf, sys, cfg]
        loss_SV = SV.sum(dim=[0, -1]).abs()
        CI = -A * y[:-1] * ((dn / n)[1:, :, None] - (dn / n)[:-1, :, None]) # [surf, sys, cfg]
        loss_CI = CI.sum(dim=[0, -1]).abs()
        CII = -Abar * y[:-1] * ((dn / n)[1:, :, None] - (dn / n)[:-1, :, None]) # [surf, sys, cfg]
        loss_CII = CII.sum(dim=[0, -1]).abs()
        
        air_thick = torch.tensor([]) # [N, sys, cfg]
        gla_thick = torch.tensor([]) # [M, sys, cfg]
        R = y.abs() + ybar.abs() # [X, sys, cfg]
        ttl = 0
        for i, elem in enumerate(sys[1:-2]):
            ttl += elem.thick
            r = torch.max(R[i], R[i+1])
            if 'VACUUM' in elem.material['name']:
                center_thick = elem.thick
                sag_1 = r ** 2 * elem.roc[:, None] / (1 + torch.sqrt((1 - r ** 2 * elem.roc[:, None] ** 2).clip(eps)))
                sag_2 = r ** 2 * sys[i+2].roc[:, None] / (1 + torch.sqrt((1 - r ** 2 * sys[i+2].roc[:, None] ** 2).clip(eps)))
                edge_thick = elem.thick + (sag_2 - sag_1)
                min_thick = torch.min(center_thick, edge_thick)
                air_thick = torch.cat([air_thick, min_thick[None, ...]], dim=0)
            else:
                center_thick = elem.thick
                sag_1 = r ** 2 * elem.roc[:, None] / (1 + torch.sqrt((1 - r ** 2 * elem.roc[:, None] ** 2).clip(eps)))
                sag_2 = r ** 2 * sys[i+2].roc[:, None] / (1 + torch.sqrt((1 - r ** 2 * sys[i+2].roc[:, None] ** 2).clip(eps)))
                edge_thick = elem.thick + (sag_2 - sag_1)
                min_thick = torch.min(center_thick, edge_thick)
                gla_thick = torch.cat([gla_thick, min_thick[None, ...]], dim=0)
        BFL = sys[-2].thick
        
        target_efl = torch.tensor(merit['EFL']['target'])[None, ...].repeat(sys_num, 1) # [sys, cfg]
        efl_loss = (EFFL - target_efl).abs().mean(dim=-1) # [sys]
        if 'target' in merit['AIR_THICK']:
            air_loss = torch.where(air_thick < merit['AIR_THICK']['target'], merit['AIR_THICK']['target'] - air_thick, torch.zeros_like(air_thick)).amax(dim=[0, -1])
        elif 'FF_target' in merit['AIR_THICK']:
            air_loss = torch.where(air_thick < merit['AIR_THICK']['FF_target'], merit['AIR_THICK']['FF_target'] - air_thick, torch.zeros_like(air_thick)).amax(dim=[0, -1])
            
        gla_min_loss = torch.where(gla_thick < merit['GLA_MIN_THICK']['min_thick'], merit['GLA_MIN_THICK']['min_thick'] - gla_thick, torch.zeros_like(gla_thick)).amax(dim=[0, -1])
        gla_max_loss = torch.where(gla_thick > merit['GLA_MAX_THICK']['max_thick'], gla_thick - merit['GLA_MAX_THICK']['max_thick'], torch.zeros_like(gla_thick)).amax(dim=[0, -1])
        bfl_loss = torch.where(BFL < merit['BFL']['target'], merit['BFL']['target'] - BFL, torch.zeros_like(BFL)).amax(dim=-1)
        totr_loss = torch.where(ttl > merit['TOTR']['target'] * 1.333, ttl - merit['TOTR']['target'] * 1.333, torch.zeros_like(ttl)).amax(dim=-1)
        
        loss_roc = torch.where(c[1:-1, :, None].abs() > (R[0:-2] ** -1) * 0.333, c[1:-1, :, None].abs() - 0.333 * (R[0:-2] ** -1), torch.zeros_like(c[1:-1, :, None])).amax(dim=[0, -1])
        loss_fst = efl_loss + bfl_loss + totr_loss
        loss_gap =  gla_min_loss + gla_max_loss + air_loss
        loss_sei = loss_SI + loss_SII + loss_SIII + loss_SIV + loss_SV + loss_CI + loss_CII
        
        loss = (1 + loss_fst) * (1 + loss_sei) * (1 + loss_roc) * (1 + loss_gap)
        return loss, R
    
    
    @torch.no_grad()
    def update(self):
        surfs = self.extract_surfs()
        abcds = [elem.abcd(surfs[i], self.wavelengths[self.p_wvl][..., None]) for i, elem in enumerate(surfs[1:-1])]
        
        abcd_pre_s = reduce((lambda x, y: torch.matmul(y, x)), abcds[:self.stop_id - 1]) if self.stop_id > 1 else torch.tensor([[1., 0], [0., 1.]])[None, None, :, :].repeat(self.sys_num, self.cfg_num, 1, 1)
        abcd_aft_s = reduce((lambda x, y: torch.matmul(y, x)), abcds[self.stop_id - 1:])
        abcd = reduce((lambda x, y: torch.matmul(y, x)), abcds)
        
        self.ENPP = abcd_pre_s[:, :, 0, 1] / abcd_pre_s[:, :, 0, 0]
        self.ENPD = surfs[self.stop_id].radius / abcd_pre_s[:, :, 0, 0] * 2
        self.EXPP = -abcd_aft_s[:, :, 0, 1] / abcd_aft_s[:, :, 1, 1]
        self.EXPD = surfs[self.stop_id].radius * (abcd_aft_s[:, :, 0, 0] - abcd_aft_s[:, :, 0, 1] * abcd_aft_s[:, :, 1, 0] / abcd_aft_s[:, :, 1, 1]) * 2
        
        self.EFFL = -1 / abcd[:, :, 1, 0]
        self.FNO = self.EFFL / self.ENPD
        self.TOTR = reduce((lambda x, y: x + y), [elem.thick for i, elem in enumerate(surfs[1:-1])])
        
        
    ######################################### Ray Tracing #########################################
    def propagate(self, ray:Ray, radius_flag=True, record=False):
        # for chief ray calculation
        o_pupil = ray.o + ray.d * (self.ENPP[None, :, :, None, None, None] / ray.d[..., 2])[..., None]
        
        # propagate to the first surface
        ray = self.system[0].propagate(ray)
        
        # record initialization
        o = ray.o.unsqueeze(0)
        d = ray.d.unsqueeze(0)

        # propagate and record
        for i, elem in enumerate(self.system[1:-1]):
            o_s, d_s, ray = elem.propagate(ray, self.system[i], radius_flag)
            
            if isinstance(o_s, list):
                o_s = torch.stack([item for item in o_s])
                o = torch.cat([o, o_s], dim=0)
            else:
                o = torch.cat([o, o_s.unsqueeze(0)], dim=0)
            
            if isinstance(d_s, list):
                d_s = torch.stack([item for item in d_s])
                d = torch.cat([d, d_s], dim=0)
            else:
                d = torch.cat([d, d_s.unsqueeze(0)], dim=0)
        
        o = torch.cat([o, ray.o.unsqueeze(0)], dim=0)
        d = torch.cat([d, ray.d.unsqueeze(0)], dim=0)
        
        if self.vig_chief:
            with torch.no_grad():
                ox = torch.where(ray.valid, o_pupil[..., 0], torch.nan)
                oy = torch.where(ray.valid, o_pupil[..., 1], torch.nan)
                ox_mean_diff = ox - torch.nanmean(ox, dim=-1, keepdim=True)
                oy_mean_diff = oy - torch.nanmean(oy, dim=-1, keepdim=True)
                diffs = torch.sqrt(ox_mean_diff**2 + oy_mean_diff**2)
                # if all rays are invalid, set chief_id to 0
                ray.chief_id = torch.argmin(torch.nan_to_num(diffs, nan=torch.inf), dim=-1)
                
        if record:
            return ray, o, d
        else:
            return ray
        
        
    def reverse_propagate(self, ray:Ray):
        """
        not for tolerance system
        """
        surfs = self.extract_surfs()
        t_s2p = (surfs[-2].thickness()[None, :, :, None, None, None] - ray.o[..., 2]) / ray.d[..., 2]
        ray.o = ray.o + t_s2p[..., None] * ray.d
        ray.o[..., 2] = ray.o[..., 2] - surfs[-2].thickness()[None, :, :, None, None, None]
        for i in range(len(surfs)-2, 1, -1):
            _, _, ray = surfs[i].reverse_propagate(ray, surfs[i-1], radius_flag=False)
        ray = surfs[1].intersect(ray, mode='reverse')
        ray = surfs[1].refract(ray, surfs[0], mode='reverse')
        return ray


    @torch.no_grad()
    def pre_samp_ray(self, views, wavelengths, pre_samp):
        """
        sample_range: [sys, cfg, ang, 4]
        views: [cfg, ang]
        """
        surfs = self.extract_surfs()
        
        sampling = pre_samp * 2 + 1
        samp_range = self.ENPD[:, :, None].repeat(1, 1, views.shape[1]) # [sys, cfg, ang]
        delta = samp_range * 2 / (sampling - 1) # [sys, cfg, ang]
        _xy = torch.linspace(-1., 1., sampling) # [M]
        
        oz = torch.zeros(sampling * 2 + 1)[None, None, None, None, :].repeat(self.sys_num, self.cfg_num, 1, 1, 1) # [sys, cfg, M]
        o = torch.stack([torch.zeros_like(oz), torch.zeros_like(oz), oz], dim=-1) # [sys, cfg, 1, 1, M, 3]
        dy = torch.linspace(0., 1., sampling * 2 + 1) * torch.sin(self.stop_max_samp_ang) # [M]
        dx = torch.zeros_like(dy) # [M]
        dz = (1. - dy ** 2 - dx ** 2) ** 0.5 # [M]
        d = torch.stack([dx, dy, dz], dim=-1)[None, None, None, None, :, :].repeat(self.sys_num, self.cfg_num, 1, 1, 1, 1) # [sys, cfg, 1, 1, M, 3]
        ray = Ray(o, d, wavelengths)
        for i in range(self.stop_id, 1, -1):
            _, _, ray = surfs[i].reverse_propagate(ray, surfs[i-1], radius_flag=False)
        ray = surfs[1].intersect(ray, mode='reverse')
        ray = surfs[1].refract(ray, surfs[0], mode='reverse')
        
        dy_ = torch.where(ray.valid, ray.d[..., 1], torch.nan) # [wav, sys, cfg, 1, 1, M]
        dy_ = rearrange(dy_, 'wav sys cfg 1 1 M -> sys cfg (wav 1 1 M)')[:, :, None, :].repeat(1, 1, views.shape[1], 1) # [sys, cfg, ang, N]
        
        oy_ = torch.where(ray.valid, ray.o[..., 1], torch.nan) # [wav, sys, cfg, 1, 1, M]
        oy_ = -rearrange(oy_, 'wav sys cfg 1 1 M -> sys cfg (wav 1 1 M)')[:, :, None, :].repeat(1, 1, views.shape[1], 1) # [sys, cfg, ang, N]
        
        if None in surfs[0].distance:
            target_dy = torch.sin(views)[None, :, :, None].repeat(self.sys_num, 1, 1, 1) # [sys, cfg, ang, 1]
            fst_oy = torch.gather(oy_, dim=-1, index=torch.argmin(torch.abs(dy_ - target_dy).nan_to_num(nan=torch.inf), dim=-1, keepdim=True))[..., 0] # [sys, cfg, ang]
            
            dz = torch.cos(views) # [cfg, ang]
            dx = torch.zeros_like(dz)
            dy = torch.sin(views) # [cfg, ang]
            d = normalize(torch.stack([dx, dy, dz], dim=-1))[None, :, :, None, None, :].repeat(self.sys_num, 1, 1, 1, sampling, 1) # [sys, cfg, ang, 1, M, 3]
            
            _o1y = samp_range[:, :, :, None, None] * _xy[None, None, None, None, :] + fst_oy[:, :, :, None, None] # [sys, cfg, ang, 1, M]
            _o1z = surfs[1].surface(0., _o1y[None, ...])[0] # [sys, cfg, ang, 1, M]
            o1y = _o1y - _o1z / d[..., 2] * d[..., 1] # [sys, cfg, ang, 1, M]
            o1 = torch.stack([torch.zeros_like(o1y), o1y, torch.zeros_like(o1y)], dim=-1) # [sys, cfg, ang, 1, M, 3]
            
            _o2x = samp_range[:, :, :, None, None] * _xy[None, None, None, None, :] # [sys, cfg, ang, 1, M]
            _o2y = fst_oy[:, :, :, None, None].repeat(1, 1, 1, 1, sampling) # [sys, cfg, ang, 1, M]
            _o2z = surfs[1].surface(_o2x[None, ...], _o2y[None, ...])[0] # [sys, cfg, ang, 1, M]
            o2y = _o2y - _o2z / d[..., 2] * d[..., 1] # [sys, cfg, ang, 1, M]
            o2 = torch.stack([_o2x, o2y, torch.zeros_like(_o2x)], dim=-1) # [sys, cfg, ang, 1, M, 3]
            
            o = torch.cat([o1, o2], dim=3) # [sys, cfg, ang, 2, M, 3]
            ray = Ray(o, d.repeat(1, 1, 1, 2, 1, 1), wavelengths)
            ray = surfs[0].propagate(ray)
            for i, elem in enumerate(surfs[1:self.stop_id]):
                _, _, ray = elem.propagate(ray, surfs[i], radius_flag=False)
            _, _, ray = surfs[self.stop_id].propagate(ray, surfs[self.stop_id-1], radius_flag=True)
            _oy = torch.where(ray.valid.sum(dim=0, dtype=bool), _o1y, torch.nan)[:, :, :, 0, :] # [sys, cfg, ang, M]
            _ox = torch.where(ray.valid.sum(dim=0, dtype=bool), _o2x, torch.nan)[:, :, :, 1, :] # [sys, cfg, ang, M]
            
            oy_min = torch.nan_to_num(_oy, nan=torch.inf).amin(dim=-1) - delta # [sys, cfg, ang]
            oy_max = torch.nan_to_num(_oy, nan=-torch.inf).amax(dim=-1) + delta # [sys, cfg, ang]
            
            oz_min = surfs[1].surface(0., oy_min[None, ..., None, None])[0, :, :, :, 0, 0] # [sys, cfg, ang]
            oz_max = surfs[1].surface(0., oy_max[None, ..., None, None])[0, :, :, :, 0, 0] # [sys, cfg, ang]
            
            oy_min = oy_min - oz_min / dz[None, ...] * dy[None, ...] # [sys, cfg, ang]
            oy_max = oy_max - oz_max / dz[None, ...] * dy[None, ...] # [sys, cfg, ang]
            
            ox_min = torch.nan_to_num(_ox, nan=torch.inf).amin(dim=-1) - delta # [sys, cfg, ang]
            ox_max = torch.nan_to_num(_ox, nan=-torch.inf).amax(dim=-1) + delta # [sys, cfg, ang]
            
            sample_range = torch.stack([ox_min, ox_max, oy_min, oy_max], dim=-1) # [sys, cfg, ang, 4]
        else:
            obj_d = torch.tensor(surfs[0].distance)[None, :] + self.ENPP # [sys, cfg]
            obj_y = -obj_d[:, :, None] * torch.tan(views)[None, :, :] # [sys, cfg, ang]
            obj_x = torch.zeros_like(obj_y)
            obj_z = -torch.tensor(surfs[0].distance)[None, :, None] * torch.ones_like(obj_y) # [sys, cfg, ang]
            obj_o = torch.stack([obj_x, obj_y, obj_z], dim=-1)[:, :, :, None, None, :].repeat(1, 1, 1, 1, sampling, 1) # [sys, cfg, ang, 1, M, 3]
            
            ox_ = torch.zeros_like(oy_) # [sys, cfg, ang, N]
            oz_ = surfs[1].surface(ox_[None, :, :, :, None, :], oy_[None, :, :, :, None, :])[0, :, :, :, 0, :] # [sys, cfg, ang, N]
            fst_o = torch.stack([ox_, oy_, oz_], dim=-1)[:, :, :, None, :, :] # [sys, cfg, ang, 1, N, 3]
            
            dy = normalize(fst_o - obj_o.repeat(1, 1, 1, 1, len(wavelengths), 1))[:, :, :, 0, :, 1] # [sys, cfg, ang, N]
            fst_oy = torch.gather(oy_, dim=-1, index=torch.argmin(torch.abs(dy - dy_).nan_to_num(nan=torch.inf), dim=-1, keepdim=True))[..., 0] # [sys, cfg, ang]
            
            _o1y = samp_range[:, :, :, None, None] * _xy[None, None, None, None, :] + fst_oy[:, :, :, None, None] # [sys, cfg, ang, 1, M]
            _o1z = surfs[1].surface(0., _o1y[None, ...])[0] # [sys, cfg, ang, 1, M]
            o1 = torch.stack([torch.zeros_like(_o1y), _o1y, _o1z], dim=-1) # [sys, cfg, ang, 1, M, 3]
            
            _o2x = samp_range[:, :, :, None, None] * _xy[None, None, None, None, :] # [sys, cfg, ang, 1, M]
            _o2y = fst_oy[:, :, :, None, None].repeat(1, 1, 1, 1, sampling) # [sys, cfg, ang, 1, M]
            _o2z = surfs[1].surface(_o2x[None, ...], _o2y[None, ...])[0] # [sys, cfg, ang, 1, M]
            o2 = torch.stack([_o2x, _o2y, _o2z], dim=-1) # [sys, cfg, ang, 1, M, 3]
            
            o = torch.cat([o1, o2], dim=3) # [sys, cfg, ang, 2, M, 3]
            d = normalize(o - obj_o.repeat(1, 1, 1, 2, 1, 1)) # [sys, cfg, ang, 2, M, 3]
            ray = Ray(obj_o.repeat(1, 1, 1, 2, 1, 1), d, wavelengths)
            
            ray = surfs[0].propagate(ray)
            for i, elem in enumerate(surfs[1:self.stop_id]):
                _, _, ray = elem.propagate(ray, surfs[i], radius_flag=False)
            _, _, ray = surfs[self.stop_id].propagate(ray, surfs[self.stop_id-1], radius_flag=True)
            _oy = torch.where(ray.valid.sum(dim=0, dtype=bool), _o1y, torch.nan)[:, :, :, 0, :] # [sys, cfg, ang, M]
            _ox = torch.where(ray.valid.sum(dim=0, dtype=bool), _o2x, torch.nan)[:, :, :, 1, :] # [sys, cfg, ang, M]
            
            obj_o = torch.stack([obj_x, obj_y, obj_z], dim=-1) # [sys, cfg, ang, 3]
            
            oy_min = torch.nan_to_num(_oy, nan=torch.inf).amin(dim=-1) - delta # [sys, cfg, ang]
            _oy_min_oz = surfs[1].surface(0., oy_min[None, ..., None, None])[0, :, :, :, 0, 0] # [sys, cfg, ang]
            _oy_min_o = torch.stack([torch.zeros_like(_oy_min_oz), oy_min, _oy_min_oz], dim=-1) # [sys, cfg, ang, 3]
            _oy_min_d = normalize(_oy_min_o - obj_o) # [sys, cfg, ang, 3]
            oy_min = _oy_min_o[..., 1] - _oy_min_o[..., 2] / _oy_min_d[..., 2] * _oy_min_d[..., 1] # [sys, cfg, ang]
            
            oy_max = torch.nan_to_num(_oy, nan=-torch.inf).amax(dim=-1) + delta # [sys, cfg, ang]
            _oy_max_oz = surfs[1].surface(0., oy_max[None, ..., None, None])[0, :, :, :, 0, 0] # [sys, cfg, ang]
            _oy_max_o = torch.stack([torch.zeros_like(_oy_max_oz), oy_max, _oy_max_oz], dim=-1) # [sys, cfg, ang, 3]
            _oy_max_d = normalize(_oy_max_o - obj_o) # [sys, cfg, ang, 3]
            oy_max = _oy_max_o[..., 1] - _oy_max_o[..., 2] / _oy_max_d[..., 2] * _oy_max_d[..., 1] # [sys, cfg, ang]
            
            ox_min = torch.nan_to_num(_ox, nan=torch.inf).amin(dim=-1) - delta # [sys, cfg, ang]
            _ox_min_oz = surfs[1].surface(ox_min[None, ..., None, None], fst_oy[None, ..., None, None])[0, :, :, :, 0, 0] # [sys, cfg, ang]
            _ox_min_o = torch.stack([ox_min, fst_oy, _ox_min_oz], dim=-1) # [sys, cfg, ang, 3]
            _ox_min_d = normalize(_ox_min_o - obj_o) # [sys, cfg, ang, 3]
            ox_min = _ox_min_o[..., 0] - _ox_min_o[..., 2] / _ox_min_d[..., 2] * _ox_min_d[..., 0] # [sys, cfg, ang]
            
            ox_max = torch.nan_to_num(_ox, nan=-torch.inf).amax(dim=-1) + delta # [sys, cfg, ang]
            _ox_max_oz = surfs[1].surface(ox_max[None, ..., None, None], fst_oy[None, ..., None, None])[0, :, :, :, 0, 0] # [sys, cfg, ang]
            _ox_max_o = torch.stack([ox_max, fst_oy, _ox_max_oz], dim=-1) # [sys, cfg, ang, 3]
            _ox_max_d = normalize(_ox_max_o - obj_o) # [sys, cfg, ang, 3]
            ox_max = _ox_max_o[..., 0] - _ox_max_o[..., 2] / _ox_max_d[..., 2] * _ox_max_d[..., 0] # [sys, cfg, ang]
            
            sample_range = torch.stack([ox_min, ox_max, oy_min, oy_max], dim=-1) # [sys, cfg, ang, 4]
        
        return sample_range
    
    
    @torch.no_grad()
    def sample_ray_2d(self, sampling, norm_view=None, azimuth=None, wavelength=None, pre_samp=None, samp_method=None, vig=None):
        # parallel waves, sys, cfg, views, azimuths, rays_num
        # rays.o: [waves, sys, cfg, views, azimuths, rays_num, 3]
        if norm_view == None:
            views = torch.deg2rad(self.max_view[..., None] * self.norm_views[None, :]) # [cfg, ang]
        elif isinstance(norm_view, list):
            views = torch.deg2rad(self.max_view[..., None] * torch.tensor(norm_view)[None, :]) # [cfg, ang]
        elif isinstance(norm_view, float):
            views = torch.deg2rad(self.max_view[..., None] * torch.tensor([norm_view])[None, :]) # [cfg, ang]
        else: raise TypeError("Please input list/float data!")
        
        if azimuth == None:
            azimuths = torch.deg2rad(self.azimuths)
        elif isinstance(azimuth, list):
            azimuths = torch.deg2rad(torch.tensor(azimuth))
        elif isinstance(azimuth, float):
            azimuths = torch.deg2rad(torch.tensor([azimuth]))
        else: raise TypeError("Please input list/float data!")
            
        if wavelength == None:
            wavelengths = self.wavelengths
        elif isinstance(wavelength, list):
            wavelengths = torch.tensor(wavelength)
        elif isinstance(wavelength, float):
            wavelengths = torch.tensor([wavelength])
        else: raise TypeError("Please input list/float data!")
        
        if pre_samp == None:
            pre_samp = self.pre_samp
        
        samp_method = self.samp_method if samp_method is None else samp_method
        xy = pupil_distribution(sampling, sampling, samp_method) # [M, 2], [-1, 1]
        if vig != None:
            vig_up = torch.tensor(vig['VUY'])
            vig_up = vig_up[None, :].repeat(self.cfg_num, 1) if vig_up.dim() == 1 else vig_up # [cfg, ang]
            vig_dw = torch.tensor(vig['VLY'])
            vig_dw = vig_dw[None, :].repeat(self.cfg_num, 1) if vig_dw.dim() == 1 else vig_dw # [cfg, ang]
            vig_le = torch.tensor(vig['VUX'])
            vig_le = vig_le[None, :].repeat(self.cfg_num, 1) if vig_le.dim() == 1 else vig_le # [cfg, ang]
            vig_ri = torch.tensor(vig['VLX'])
            vig_ri = vig_ri[None, :].repeat(self.cfg_num, 1) if vig_ri.dim() == 1 else vig_ri # [cfg, ang]
            
            _xy = torch.zeros_like(xy)[None, None, :, :].repeat(self.cfg_num, views.shape[-1], 1, 1) # [cfg, ang, M, 2]
            _xy[:, :, :, 0] = (1 - vig_le + 1 - vig_ri)[:, :, None] / 2 * xy[None, None, :, 0] + (vig_ri - vig_le)[:, :, None] / 2
            _xy[:, :, :, 1] = (1 - vig_up + 1 - vig_dw)[:, :, None] / 2 * xy[None, None, :, 1] + (vig_dw - vig_up)[:, :, None] / 2
        else:
            _xy = xy[None, None, :, :].repeat(self.cfg_num, views.shape[-1], 1, 1) # [cfg, ang, M, 2]
            
        if None in self.system[0].distance:
            dz = torch.cos(views)[:, :, None].repeat(1, 1, len(azimuths)) # [cfg, ang, azi]
            dx = torch.sin(views)[:, :, None] * torch.sin(azimuths)[None, None, :]
            dy = torch.sin(views)[:, :, None] * torch.cos(azimuths)[None, None, :]
            d = normalize(torch.stack([dx, dy, dz], dim=-1))[None, :, :, :, None, :].repeat(self.sys_num, 1, 1, 1, xy.shape[0], 1)
            if pre_samp:
                sample_range = self.pre_samp_ray(views, wavelengths, pre_samp) # [sys, cfg, ang, 4]
                ox = (_xy[None, ..., 0] + 1) / 2 * (sample_range[..., 1][..., None] - sample_range[..., 0][..., None]) + sample_range[..., 0][..., None] # [sys, cfg, ang, M]
                oy = (_xy[None, ..., 1] + 1) / 2 * (sample_range[..., 3][..., None] - sample_range[..., 2][..., None]) + sample_range[..., 2][..., None] # [sys, cfg, ang, M]
                
                _x = ox[:, :, :, None, :] * torch.cos(-azimuths)[None, None, None, :, None] - oy[:, :, :, None, :] * torch.sin(-azimuths)[None, None, None, :, None] # [sys, cfg, ang, azi, M]
                _y = ox[:, :, :, None, :] * torch.sin(-azimuths)[None, None, None, :, None] + oy[:, :, :, None, :] * torch.cos(-azimuths)[None, None, None, :, None] # [sys, cfg, ang, azi, M]
                o = torch.stack([_x, _y, torch.zeros_like(_x)], dim=-1) # [sys, cfg, ang, azi, M, 3]
            else:
                x = _xy[:, :, None, :, 0] * torch.cos(-azimuths)[None, None, :, None] - _xy[:, :, None, :, 1] * torch.sin(-azimuths)[None, None, :, None] # [cfg, ang, azi, M]
                y = _xy[:, :, None, :, 0] * torch.sin(-azimuths)[None, None, :, None] + _xy[:, :, None, :, 1] * torch.cos(-azimuths)[None, None, :, None] # [cfg, ang, azi, M]
                _xy = torch.stack([x, y], dim=-1) # [cfg, ang, azi, M, 2]
                
                sample_radius = self.ENPD * 0.5 * (1. + self.samp_margin) # [sys, cfg]
                o_xy = _xy[None, :, :, :, :, :] * sample_radius[:, :, None, None, None, None] # [sys, cfg, ang, azi, M, 2]
                o = torch.stack([o_xy[..., 0], o_xy[..., 1], torch.ones_like(o_xy[..., 0]) * self.ENPP[:, :, None, None, None]], dim=-1) # [sys, cfg, ang, azi, M, 3]
                
                p_dz = torch.ones_like(views)[:, :, None].repeat(1, 1, len(azimuths)) # [cfg, ang, azi]
                p_dx = views[:, :, None] * torch.sin(azimuths)[None, None, :]
                p_dy = views[:, :, None] * torch.cos(azimuths)[None, None, :]
                p_d = torch.stack([p_dx, p_dy, p_dz], dim=-1)[None, :, :, :, None, :].repeat(self.sys_num, 1, 1, 1, xy.shape[0], 1)
                
                t = o[..., 2] / p_d[..., 2]
                o = o - t[..., None] * p_d            
        else:
            obj_d = torch.tensor(self.system[0].distance)[None, :] + self.ENPP # [sys, cfg]
            ox = -obj_d[:, :, None, None] * torch.tan(views)[None, :, :, None] * torch.sin(azimuths)[None, None, None, :] # [sys, cfg, ang, azi]
            oy = -obj_d[:, :, None, None] * torch.tan(views)[None, :, :, None] * torch.cos(azimuths)[None, None, None, :] # [sys, cfg, ang, azi]
            oz = -torch.tensor(self.system[0].distance)[None, :, None, None] * torch.ones_like(ox) # [sys, cfg, ang, azi]
            o = torch.stack([ox, oy, oz], dim=-1)[:, :, :, :, None, :].repeat(1, 1, 1, 1, xy.shape[0], 1) # [sys, cfg, ang, azi, M, 3]
            if pre_samp:
                sample_range = self.pre_samp_ray(views, wavelengths, pre_samp) # [sys, cfg, ang, 4]
                ox = (_xy[None, ..., 0] + 1) / 2 * (sample_range[..., 1][..., None] - sample_range[..., 0][..., None]) + sample_range[..., 0][..., None] # [sys, cfg, ang, M]
                oy = (_xy[None, ..., 1] + 1) / 2 * (sample_range[..., 3][..., None] - sample_range[..., 2][..., None]) + sample_range[..., 2][..., None] # [sys, cfg, ang, M]
                
                _x = ox[:, :, :, None, :] * torch.cos(-azimuths)[None, None, None, :, None] - oy[:, :, :, None, :] * torch.sin(-azimuths)[None, None, None, :, None] # [sys, cfg, ang, azi, M]
                _y = ox[:, :, :, None, :] * torch.sin(-azimuths)[None, None, None, :, None] + oy[:, :, :, None, :] * torch.cos(-azimuths)[None, None, None, :, None] # [sys, cfg, ang, azi, M]
                o_pla = torch.stack([_x, _y, torch.zeros_like(_x)], dim=-1) # [sys, cfg, ang, azi, M, 3]
                d = normalize(o_pla - o) # [sys, cfg, ang, azi, M, 3]
            else:
                x = _xy[:, :, None, :, 0] * torch.cos(-azimuths)[None, None, :, None] - _xy[:, :, None, :, 1] * torch.sin(-azimuths)[None, None, :, None] # [cfg, ang, azi, M]
                y = _xy[:, :, None, :, 0] * torch.sin(-azimuths)[None, None, :, None] + _xy[:, :, None, :, 1] * torch.cos(-azimuths)[None, None, :, None] # [cfg, ang, azi, M]
                _xy = torch.stack([x, y], dim=-1) # [cfg, ang, azi, M, 2]
                
                sample_radius = self.ENPD * 0.5 * (1. + self.samp_margin) # [sys, cfg]
                o_xy = _xy[None, :, :, :, :, :] * sample_radius[:, :, None, None, None, None] # [sys, cfg, ang, azi, M, 2]
                o_pupil = torch.stack([o_xy[..., 0], o_xy[..., 1], self.ENPP[:, :, None, None, None].repeat(1, 1, views.shape[-1], len(azimuths), xy.shape[0])], dim=-1) # [sys, cfg, ang, azi, M, 3]
                d = normalize(o_pupil - o) # [sys, cfg, ang, azi, M, 3]
        
        return Ray(o, d, wavelengths)


    ######################################### Basic Functions #########################################
    @torch.no_grad()
    def fit_opt_data(self, opt_data):
        """
        Fit the opt data to the system.
        """
        surfs = self.extract_surfs()
        for i, surf in enumerate(surfs[1:-1]):
            for key in opt_data[i]:
                data = opt_data[i][key]
                exec(f'surf.{key}.data = data.detach().clone()')
    
    @torch.no_grad()
    def extract_opt_data(self):
        """
        Extract the opt data from the system.
        """
        opt_data = {}
        surfs = self.extract_surfs()
        for i, surf in enumerate(surfs[1:-1]):
            opt_data[i] = {}
            
            if surf.thick.requires_grad:
                opt_data[i]['thick'] = surf.thick.detach().clone()
                
            if surf.conic.requires_grad:
                opt_data[i]['conic'] = surf.conic.detach().clone()

            if surf.roc.requires_grad:
                opt_data[i]['roc'] = surf.roc.detach().clone()
            
            if hasattr(surf, 'ai_num'):
                for j in range(surf.ai_num):
                    if eval(f'surf.ai{2 * j + 4}.requires_grad'):
                        opt_data[i][f'ai{2 * j + 4}'] = eval(f'surf.ai{2 * j + 4}.detach().clone()')
            
            if hasattr(surf, 'qi_num'):
                for j in range(surf.qi_num):
                    if eval(f'surf.qi{j}.requires_grad'):
                        opt_data[i][f'qi{j}'] = eval(f'surf.qi{j}.detach().clone()')
        
            if hasattr(surf, 'g1') and hasattr(surf, 'g2'):
                if surf.g1.requires_grad:
                    opt_data[i]['g1'] = surf.g1.detach().clone()
                if surf.g2.requires_grad:
                    opt_data[i]['g2'] = surf.g2.detach().clone()
                    
        return opt_data
    
    @torch.no_grad()
    def extract_all_sys_data(self):
        """
        Extract the opt data from the system.
        """
        opt_data = {}
        surfs = self.extract_surfs()
        for i, surf in enumerate(surfs[1:-1]):
            opt_data[i] = {}
            
            opt_data[i]['radius'] = surf.radius.detach().clone()
            opt_data[i]['roc'] = surf.roc.detach().clone()
            opt_data[i]['thick'] = surf.thick.detach().clone()
            opt_data[i]['conic'] = surf.conic.detach().clone()
            opt_data[i]['min_r'] = surf.min_r.detach().clone()
            opt_data[i]['max_r'] = surf.max_r.detach().clone()
            
            if hasattr(surf, 'ai_num'):
                for j in range(surf.ai_num):
                    opt_data[i][f'ai{2 * j + 4}'] = eval(f'surf.ai{2 * j + 4}.detach().clone()')
            
            if hasattr(surf, 'qi_num'):
                opt_data[i]['rnorm'] = surf.rnorm.detach().clone()
                for j in range(surf.qi_num):
                    opt_data[i][f'qi{j}'] = eval(f'surf.qi{j}.detach().clone()')
        
            if hasattr(surf, 'g1') and hasattr(surf, 'g2'):
                opt_data[i]['g1'] = surf.g1.detach().clone()
                opt_data[i]['g2'] = surf.g2.detach().clone()
                
        return opt_data
    
    def extract_surfs(self):
        def extract_inner_elements(package):
            inner_elements = []
            
            def recursive_extract(elements):
                for elem in elements:
                    if isinstance(elem, PACKAGE):
                        recursive_extract(elem.pack)
                    else:
                        inner_elements.append(elem)
            
            recursive_extract(package.pack)
            return inner_elements
        
        surfs = []
        for elem in self.system:
            if isinstance(elem, PACKAGE):
                surfs.extend(extract_inner_elements(elem))
            else:
                surfs.append(elem)
        return surfs
    
    def extract_tols(self):
        def extract_tilt_decenter(package):
            variables = {}
            variables['decenter-tilt'] = [package.decenter, package.tilt]
                
            for i, elem in enumerate(package.pack):
                if isinstance(elem, PACKAGE):
                    sub_vars = extract_tilt_decenter(elem)
                    for key, value in sub_vars.items():
                        variables[f'{i}_{key}'] = value
            return variables

        tols = {}
        for i, elem in enumerate(self.system):
            if isinstance(elem, PACKAGE):
                sub_vars = extract_tilt_decenter(elem)
                for key, value in sub_vars.items():
                    tols[f'{i}_{key}'] = value
        return tols
    
    def print_sys_para(self, sys_id=0, cfg_id=0, logger=None):
        if logger is None:
            print('System Parameters')
        else:
            logger.info('System Parameters')
        
        names = []
        for i, item in enumerate(self.extract_surfs()):
            for j in item.state_dict():
                if j not in names:
                    names.append(j)
        
        itv = 4
        title = 'Surface' + ' ' * itv + ' '
        title += 'radius' + ' ' * itv + ' ' * (17 - len('radius'))
        for i in names:
            if "pre_surf" not in i:
                if "decenter" in i:
                    title += f'{i} X' + ' ' * itv + ' ' * (15 - len(i))
                    title += f'{i} Y' + ' ' * itv + ' ' * (15 - len(i))
                elif "tilt" in i:
                    title += f'{i} X' + ' ' * itv + ' ' * (15 - len(i))
                    title += f'{i} Y' + ' ' * itv + ' ' * (15 - len(i))
                    title += f'{i} Z' + ' ' * itv + ' ' * (15 - len(i))
                else:
                    title += i + ' ' * itv + ' ' * (17 - len(i))
                    
        if logger is None:
            print(title)
        else:
            logger.info(title)
        
        for i, item in enumerate(self.extract_surfs()):
            msg = '{:7}'.format(i)
            if isinstance(item, Sphere):
                msg += ' ' * itv + '{:>17.10E}'.format(item.radius[sys_id, cfg_id])
            else:
                msg += ' ' * itv + '{:>17}'.format('-')
            for j in names:
                if "pre_surf" not in j:
                    if j in item.state_dict():
                        if "decenter" in j:
                            data = item.state_dict()[j][sys_id, cfg_id, 0].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(-data)
                            data = item.state_dict()[j][sys_id, cfg_id, 1].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(-data)
                        elif "tilt" in j:
                            data = torch.rad2deg(item.state_dict()[j][sys_id, cfg_id, 0]).item()
                            msg += ' ' * itv + '{:>17.10E}'.format(-data)
                            data = torch.rad2deg(item.state_dict()[j][sys_id, cfg_id, 1]).item()
                            msg += ' ' * itv + '{:>17.10E}'.format(-data)
                            data = torch.rad2deg(item.state_dict()[j][sys_id, cfg_id, 2]).item()
                            msg += ' ' * itv + '{:>17.10E}'.format(-data)
                        else:
                            data = item.state_dict()[j][sys_id].item() if len(item.state_dict()[j].shape) == 1 else item.state_dict()[j][sys_id, cfg_id].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(data)
                    else:
                        if "decenter" in j:
                            msg += ' ' * itv + '{:>17}'.format('-')
                            msg += ' ' * itv + '{:>17}'.format('-')
                        elif "tilt" in j:
                            msg += ' ' * itv + '{:>17}'.format('-')
                            msg += ' ' * itv + '{:>17}'.format('-')
                            msg += ' ' * itv + '{:>17}'.format('-')
                        else:
                            msg += ' ' * itv + '{:>17}'.format('-')
            if logger is None:
                print(msg)
            else:
                logger.info(msg)

    def print_sys_grad(self, sys_id=0, cfg_id=0, logger=None):
        if logger is None:
            print('System Parameters Gradient')
        else:
            logger.info('System Parameters Gradient')
        
        names = []
        for i, item in enumerate(self.extract_surfs()):
            for j in item.state_dict():
                if j not in names:
                    names.append(j)
        
        itv = 4
        title = 'Surface' + ' ' * itv + ' '
        for i in names:
            if "pre_surf" not in i:
                if "decenter" in i:
                    title += f'{i} X' + ' ' * itv + ' ' * (17 - len(i))
                    title += f'{i} Y' + ' ' * itv + ' ' * (17 - len(i))
                elif "tilt" in i:
                    title += f'{i} X' + ' ' * itv + ' ' * (17 - len(i))
                    title += f'{i} Y' + ' ' * itv + ' ' * (17 - len(i))
                    title += f'{i} Z' + ' ' * itv + ' ' * (17 - len(i))
                else:
                    title += i + ' ' * itv + ' ' * (17 - len(i))
            
        if logger is None:
            print(title)
        else:
            logger.info(title)
        
        for i, item in enumerate(self.extract_surfs()):
            msg = '{:7}'.format(i)
            for j in names:
                if "pre_surf" not in j:
                    if j in item.state_dict():
                        if "decenter" in j:
                            data = 0 if item.get_parameter(j).grad is None else item.get_parameter(j).grad[sys_id, cfg_id, 0].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(data)
                            data = 0 if item.get_parameter(j).grad is None else item.get_parameter(j).grad[sys_id, cfg_id, 1].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(data)
                        elif "tilt" in j:
                            data = 0 if item.get_parameter(j).grad is None else item.get_parameter(j).grad[sys_id, cfg_id, 0].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(data)
                            data = 0 if item.get_parameter(j).grad is None else item.get_parameter(j).grad[sys_id, cfg_id, 1].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(data)
                            data = 0 if item.get_parameter(j).grad is None else item.get_parameter(j).grad[sys_id, cfg_id, 2].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(data)
                        else:
                            if item.get_parameter(j).grad is None:
                                data = 0
                            else:
                                data = item.get_parameter(j).grad[sys_id].item() if len(item.get_parameter(j).grad.shape) == 1 else item.get_parameter(j).grad[sys_id, cfg_id].item()
                            msg += ' ' * itv + '{:>17.10E}'.format(data)
                    else:
                        if "decenter" in j:
                            msg += ' ' * itv + '{:>17}'.format('-')
                            msg += ' ' * itv + '{:>17}'.format('-')
                        elif "tilt" in j:
                            msg += ' ' * itv + '{:>17}'.format('-')
                            msg += ' ' * itv + '{:>17}'.format('-')
                            msg += ' ' * itv + '{:>17}'.format('-')
                        else:
                            msg += ' ' * itv + '{:>17}'.format('-')
        
            if logger is None:
                print(msg)
            else:
                logger.info(msg)
                
    def print_tol_para(self, sys_id=0, cfg_id=0, logger=None):
        if logger is None:
            print('Tolerance Parameters')
        else:
            logger.info('Tolerance Parameters')
        tols = self.extract_tols()
        
        itv = 4
        msg = 'Package' + ' ' * 10 + 'Decenter X' + ' ' * 11 + 'Decenter Y' + ' ' * 11 + 'Tilt X' + ' ' * 15 + 'Tilt Y' + ' ' * 15 + 'Tilt Z'
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
        
        for item in tols:
            msg = (item.split('_decenter-tilt')[0] + ' ' * (17 - len(item.split('decenter-tilt')[0])) + 
                f'{-tols[item][0][sys_id, cfg_id, 0].item():>17.10e}' + ' ' * itv + 
                f'{-tols[item][0][sys_id, cfg_id, 1].item():>17.10e}' + ' ' * itv + 
                f'{-tols[item][1][sys_id, cfg_id, 0].rad2deg().item():>17.10e}' + ' ' * itv + 
                f'{-tols[item][1][sys_id, cfg_id, 1].rad2deg().item():>17.10e}' + ' ' * itv + 
                f'{-tols[item][1][sys_id, cfg_id, 2].rad2deg().item():>17.10e}')
            if logger is None:
                print(msg)
            else:
                logger.info(msg)
    
    def print_tol_grad(self, sys_id=0, cfg_id=0, logger=None):
        if logger is None:
            print('Tolerance Parameters Gradient')
        else:
            logger.info('Tolerance Parameters Gradient')
        tols = self.extract_tols()
        
        itv = 4
        msg = 'Package' + ' ' * 10 + 'Decenter X' + ' ' * 11 + 'Decenter Y' + ' ' * 11 + 'Tilt X' + ' ' * 15 + 'Tilt Y' + ' ' * 15 + 'Tilt Z'
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

        for item in tols:
            msg = (item.split('_decenter-tilt')[0] + ' ' * (17 - len(item.split('decenter-tilt')[0])) + 
                f'{0 if tols[item][0].grad is None else tols[item][0].grad[sys_id, cfg_id, 0].item():>17.10e}' + ' ' * itv + 
                f'{0 if tols[item][0].grad is None else tols[item][0].grad[sys_id, cfg_id, 1].item():>17.10e}' + ' ' * itv + 
                f'{0 if tols[item][1].grad is None else tols[item][1].grad[sys_id, cfg_id, 0].item():>17.10e}' + ' ' * itv + 
                f'{0 if tols[item][1].grad is None else tols[item][1].grad[sys_id, cfg_id, 1].item():>17.10e}' + ' ' * itv + 
                f'{0 if tols[item][1].grad is None else tols[item][1].grad[sys_id, cfg_id, 2].item():>17.10e}')    
            if logger is None:
                print(msg)
            else:
                logger.info(msg)
                
    def avg_sys_para_grad(self, surf_id, param_name:str='thick'):
        """
        params: "thick"
        """
        def average_gradients_hook(grad):
            grad[:] = grad.mean(dim=-1, keepdim=True)
            return grad
        
        if param_name in self.extract_surfs()[surf_id].state_dict().keys():
            dict(self.extract_surfs()[surf_id].named_parameters())[param_name].register_hook(average_gradients_hook)
        else:
            raise Warning(f'Surf {surf_id} does not have {param_name}!')
    
    def freeze_sys_param(self, surf_id, param_name:str):
        """
        params: "all" or "thick"/"roc"/"conic"/"g1"/"g2"...
        """
        if param_name == 'all':
            for name, param in self.extract_surfs()[surf_id].named_parameters():
                param.requires_grad = False
        elif param_name in self.extract_surfs()[surf_id].state_dict().keys():
            dict(self.extract_surfs()[surf_id].named_parameters())[param_name].requires_grad = False
        else:
            return Warning(f'Surf {surf_id} does not have {param_name}!')
        
    def unfreeze_sys_param(self, surf_id, param_name:str):
        """
        params: "all" or "thick"/"roc"/""conic"/"g1"/"g2"...
        """
        if param_name == 'all':
            for name, param in self.extract_surfs()[surf_id].named_parameters():
                param.requires_grad = True
        elif param_name in self.extract_surfs()[surf_id].state_dict().keys():
            dict(self.extract_surfs()[surf_id].named_parameters())[param_name].requires_grad = True
        else:
            return Warning(f'Surf {surf_id} does not have {param_name}!')
    
    def material_fit(self, surf_id, method='M'):
        """
        Transform the virtual material to the real catalog material.
        M: Mahalanoobis distance
        E: Euclidean distance
        """
        surf = self.extract_surfs()[surf_id]
        if 'VACUUM' in surf.material['name']:
            return "VACUUM"
        elif 'MIRROR' in surf.material['name']:
            return "MIRROR"
        else:
            with torch.no_grad():
                params = torch.stack([surf.g1, surf.g2], dim=-1) # [sys, 2]
                if surf.mat_cata == 'G':
                    catalog_params = glass_catalog_params.to(params.device)
                    catalog = glass_catalog
                elif surf.mat_cata == 'P':
                    catalog_params = plastic_catalog_params.to(params.device)
                    catalog = plastic_catalog
                else:
                    raise ValueError(f"Unknown material catalog for {surf.mat_cata}")
                idx = fit_get_mat_id(params, method, surf.mat_cata) # [sys]
            
            names = list(catalog)
            for i in range(self.sys_num):
                name = names[idx[i]]
                data = catalog[name]
                surf.material['name'][i] = name
                surf.material['nd'][i] = data['nd']
                surf.material['vd'][i] = data['vd']
                surf.material_para[i, 0] = 0 if 'A0' in data else 1
                surf.material_para[i, 1] = data['A0'] if 'A0' in data else data['K1']
                surf.material_para[i, 2] = data['A1'] if 'A1' in data else data['L1']
                surf.material_para[i, 3] = data['A2'] if 'A2' in data else data['K2']
                surf.material_para[i, 4] = data['A3'] if 'A3' in data else data['L2']
                surf.material_para[i, 5] = data['A4'] if 'A4' in data else data['K3']
                surf.material_para[i, 6] = data['A5'] if 'A5' in data else data['L3']
                               
                surf.g1.data[i] = catalog_params[idx[i], 0].clone().detach()
                surf.g2.data[i] = catalog_params[idx[i], 1].clone().detach()
            return "GLASS/PLASTIC"
    
    def freeze_tol_param(self, tols_id:str, param_name:str):
        """
        tols_id: x / x_x / x_x_x / ...
        params: "all" or "decenter"/"tilt"
        """
        tols = self.extract_tols()
        if param_name == 'all':
            tols[f'{tols_id}_decenter-tilt'][0].requires_grad = False
            tols[f'{tols_id}_decenter-tilt'][1].requires_grad = False
        elif param_name == 'decenter':
            tols[f'{tols_id}_decenter-tilt'][0].requires_grad = False
        elif param_name == 'tilt':
            tols[f'{tols_id}_decenter-tilt'][1].requires_grad = False
        else:
            raise Warning(f' {tols_id} {param_name} does not exist!')
    
    def unfreeze_tol_param(self, tols_id:str, param_name:str):
        """
        tols_id: x / x_x / x_x_x / ...
        params: "all" or "decenter"/"tilt"
        """
        tols = self.extract_tols()
        if param_name == 'all':
            tols[f'{tols_id}_decenter-tilt'][0].requires_grad = True
            tols[f'{tols_id}_decenter-tilt'][1].requires_grad = True
        elif param_name == 'decenter':
            tols[f'{tols_id}_decenter-tilt'][0].requires_grad = True
        elif param_name == 'tilt':
            tols[f'{tols_id}_decenter-tilt'][1].requires_grad = True
        else:
            raise Warning(f' {tols_id} {param_name} does not exist!')
    
    def set_tol_param(self, tols_id:str, decenter:torch.Tensor=None, tilt:torch.Tensor=None):
        """
        decenter: [sys, cfg, 2]
        tilt: [sys, cfg, 3] (deg)
        tols_id: x / x_x / x_x_x / ...
        """
        tols = self.extract_tols()
        if decenter != None:
            tols[f'{tols_id}_decenter-tilt'][0].data = -decenter[None, None, :].repeat(self.sys_num, self.cfg_num, 1)
        if tilt != None:
            tols[f'{tols_id}_decenter-tilt'][1].data = -torch.deg2rad(tilt)[None, None, :].repeat(self.sys_num, self.cfg_num, 1)
    
    def rand_tol_param(self, decenter_scale, tilt_scale):
        """
        Random reset all the tol parameters.
        tilt: (deg)
        """
        tols = self.extract_tols()
        for item in tols:
            tols[item][0].data = decenter_scale * torch.randn(self.sys_num, self.cfg_num, 2)
            tols[item][1].data = tilt_scale * torch.deg2rad(torch.randn(self.sys_num, self.cfg_num, 3))
    
    def ini_tol_sys(self, tols_dic:dict):
        """
        Initialize tolerance settings.
        """
        def package_elements(start_surf, end_surf, decenter, tilt):
            elements_to_package = self.system[start_surf:end_surf + 1]
            pack = PACKAGE(decenter, tilt, elements_to_package)
            self.system = self.system[:start_surf] + [pack] + self.system[end_surf + 1:]
        
        delta_id = 0
        for item in tols_dic:
            start, end = int(item.split('_')[0]) - delta_id, int(item.split('_')[1]) - delta_id
            delta_id += end - start
            decenter = tols_dic[item]['decenter'] if torch.is_tensor(tols_dic[item]['decenter']) else torch.tensor(tols_dic[item]['decenter'])
            tilt = tols_dic[item]['tilt'] if torch.is_tensor(tols_dic[item]['tilt']) else torch.tensor(tols_dic[item]['tilt'])
            decenter = decenter[None, None, :].repeat(self.sys_num, self.cfg_num, 1)
            tilt = tilt[None, None, :].repeat(self.sys_num, self.cfg_num, 1)
            package_elements(start, end, decenter, tilt)
        print(self.system)
    
    def remove_tol_param(self):
        """
        Remove all tolerance PACKAGE.
        """
        self.rand_decenter_tilt_thick_param(0., 0., 0.)
        sys = nn.ModuleList([])
        sys.extend(self.extract_surfs())
        self.system = sys
        
    def rand_decenter_tilt_thick_param(self, decenter_scale, tilt_scale, thick_scale):
        """
        Random reset all the decenter, tilt, and thick tolerance parameters.
        """
        self.rand_tol_param(decenter_scale, tilt_scale)
        for i, surf in enumerate(self.extract_surfs()[1:-1]):
            surf.update_thickness_tol(thick_scale)
            
    def print_thick_tol_para(self, sys_id=0, cfg_id=0, logger=None):
        """
        Print all the thickness tolerance parameters.
        """
        if logger is None:
            print('Thickness Tolerance Data')
        else:
            logger.info('Thickness Tolerance Data')
            
        for i, surf in enumerate(self.extract_surfs()[1:-1]):
            msg = f'surf {i+1}: ' + '{:>17}'.format(surf.thick_tol[sys_id, cfg_id])
        
            if logger is None:
                print(msg)
            else:
                logger.info(msg)
    
    
    
    ######################################### Others #########################################
    def tilt_decenter_elements(self, start_surf, end_surf, decenter, tilt):
        """
        Can be replaced by the PACKAGE
        """
        decenter = (decenter if torch.is_tensor(decenter) else torch.tensor(decenter))[None, None, :].repeat(self.sys_num, self.cfg_num, 1)
        tilt = (tilt if torch.is_tensor(tilt) else torch.tensor(tilt))[None, None, :].repeat(self.sys_num, self.cfg_num, 1)
        
        if start_surf == end_surf:
            thick_last = self.system[end_surf].thick.clone()
            
            s1 = Coordinate(pre_surf=self.system[start_surf-1], thick=torch.zeros_like(thick_last), decenter=decenter, tilt=tilt, flag=0)
            s2 = Coordinate(pre_surf=self.system[end_surf], thick=thick_last, decenter=-decenter, tilt=-tilt, flag=1)
            
            self.system[start_surf].thick.data = torch.zeros_like(thick_last)
            
            self.system.insert(start_surf, s1)
            self.system.insert(end_surf+2, s2)
            
            self.stop_id += 1 if start_surf <= self.stop_id else 0
            self.stop_id += 1 if end_surf+2 <= self.stop_id else 0
        else:
            thick_sum = sum(self.system[i].thick for i in range(start_surf, end_surf))
            thick_last = self.system[end_surf].thick.clone()

            s1 = Coordinate(pre_surf=self.system[start_surf-1], thick=torch.zeros_like(thick_last), decenter=decenter, tilt=tilt, flag=0)
            s2 = Coordinate(pre_surf=self.system[end_surf], thick=thick_sum, decenter=-decenter, tilt=-tilt, flag=1)
            s3 = Dummy(pre_surf=self.system[end_surf], thick=thick_last)

            self.system[end_surf].thick.data = -thick_sum

            self.system.insert(start_surf, s1)
            self.system.insert(end_surf+2, s2)
            self.system.insert(end_surf+3, s3)
            
            self.stop_id += 1 if start_surf <= self.stop_id else 0
            self.stop_id += 1 if end_surf+2 <= self.stop_id else 0
            self.stop_id += 1 if end_surf+3 <= self.stop_id else 0
            
    def del_surfs(self, del_id):
        """
        Only for systems with no tolerance.
        """
        if self.stop_id >= del_id[0] and self.stop_id <= del_id[1]:
            raise Warning('The stop surface cannot be deleted!')
        
        if del_id[0] != 1:
            self.system[del_id[0]-1].thick.data += sum(self.system[i].thick for i in range(del_id[0], del_id[1]+1))
        sys = self.system[:del_id[0]] + self.system[del_id[1] + 1:]
        self.system = nn.ModuleList(sys)
        
        if del_id[1] < self.stop_id:
            self.stop_id -= (del_id[1] - del_id[0] + 1)
    
    def convert_sph_to_asp(self, surf_id:int, surf_type:str, order:int):
        """
        surf_type: "Asphere"/"Qcon"/"Qbfs"
        """
        common_params = {
            'radius': self.system[surf_id].radius.tolist(),
            'material': self.system[surf_id].material['name'],
            'roc': (self.system[surf_id].roc ** -1).tolist(),
            'thick': (self.system[surf_id].thick).tolist(),
            'conic': self.system[surf_id].conic.tolist(),
            'mat_cata': self.system[surf_id].mat_cata,
            }
        if surf_type == 'Asphere':
            self.system[surf_id] = Asphere(**common_params, ai_list=[[0.0] * self.sys_num] * order)
        elif surf_type == 'Qcon':
            rnorm = self.system[surf_id].radius.amax(dim=-1) * 1.2
            self.system[surf_id] = Qcon(**common_params, qi_list=[[0.0] * self.sys_num] * order, rnorm=rnorm.tolist())
        elif surf_type == 'Qbfs':
            rnorm = self.system[surf_id].radius.amax(dim=-1) * 1.2
            self.system[surf_id] = Qbfs(**common_params, qi_list=[[0.0] * self.sys_num] * order, rnorm=rnorm.tolist())
        else:
            raise Warning(f'Unknown surf type {surf_type}!')
    
    def add_IRCF(self, thick:float, dist:float, mat:str, sampling:int=3):
        """
        thick: IRCF thickness
        dist: distance to the image sensor
        mat: material of the IRCF
        """
        self.system[-2].thick.data = self.system[-2].thick.data - (thick + dist)
        if hasattr(self, 'zoom_type'):
            if self.zoom_type[-2] != None:
                self.zoom_type.append('FF')
                self.zoom_type.append('FF')
            else:
                self.zoom_type.append(None)
                self.zoom_type.append(None)
            
        front_surf = Sphere(radius=[[1.] * self.cfg_num] * self.sys_num,
                            material=[mat] * self.sys_num,
                            roc=[None] * self.sys_num,
                            thick=[[thick] * self.cfg_num] * self.sys_num,
                            conic=[0.] * self.sys_num,
                            mat_cata='G')
        back_surf = Sphere(radius=[[1.] * self.cfg_num] * self.sys_num,
                            material=['VACUUM'] * self.sys_num,
                            roc=[None] * self.sys_num,
                            thick=[[dist] * self.cfg_num] * self.sys_num,
                            conic=[0.] * self.sys_num)
        self.system.insert(len(self.system)-1, front_surf)
        self.system.insert(len(self.system)-1, back_surf)
        
        ray = self.sample_ray_2d(sampling * 2 + 1, azimuth=0., vig=self.vig)
        # propagate to the first surface
        ray = self.system[0].propagate(ray)
        for i, elem in enumerate(self.system[1:-3]):
            _, _, ray = elem.propagate(ray, self.system[i], radius_flag=True)
            
        o_sf, _, ray = self.system[-3].propagate(ray, self.system[-4], radius_flag=False)
        o_sb, _, ray = self.system[-2].propagate(ray, self.system[-3], radius_flag=False)
        
        radius_sf = torch.where(ray.valid, length(o_sf[:, :, :, :, :, :, 0:2]), -torch.inf).amax(dim=[0, 3, 4, 5]) # [sys, cfg]
        radius_sb = torch.where(ray.valid, length(o_sb[:, :, :, :, :, :, 0:2]), -torch.inf).amax(dim=[0, 3, 4, 5]) # [sys, cfg]
        self.system[-3].radius = radius_sf.amax(dim=-1, keepdim=True).repeat(1, self.cfg_num)
        self.system[-2].radius = radius_sb.amax(dim=-1, keepdim=True).repeat(1, self.cfg_num)
        self.update()
            
    def save_json(self, sys_id, save_path):
        json_file = open(save_path, mode='w')
        json_content={}
        json_content["OBJECT"] = {"material": "VACUUM",
                                  "distance": list_convert(self.system[0].distance)}
        
        for idx, surf in enumerate(self.system[1:-1]):
            if isinstance(surf, Asphere):
                ai_list = []
                for i in range(surf.ai_num):
                    exec('ai_list.append(list_convert(surf.ai{}[sys_id].tolist()))'.format(2 * i + 4))
            
                json_content["Surface{}".format(idx+1)] = {
                    "type": "Asphere",
                    "zoom_type": self.zoom_type[idx] if hasattr(self, 'zoom_type') else None,
                    "stop": (idx + 1) == self.stop_id,
                    "radius": list_convert(surf.radius[sys_id].tolist()),
                    "material": surf.material['name'][sys_id],
                    "roc": list_convert([1 / surf.roc[sys_id].item() if surf.roc[sys_id].item() != 0 else None]),
                    "thick": list_convert(surf.thick[sys_id].tolist()),
                    "conic": list_convert(surf.conic[sys_id].tolist()),
                    "ai_list": ai_list
                }
            elif isinstance(surf, Qcon):
                qi_list = []
                for i in range(surf.qi_num):
                    exec('qi_list.append(list_convert(surf.qi{}[sys_id].tolist()))'.format(i))
            
                json_content["Surface{}".format(idx+1)] = {
                    "type": "Qcon",
                    "zoom_type": self.zoom_type[idx] if hasattr(self, 'zoom_type') else None,
                    "stop": (idx + 1) == self.stop_id,
                    "radius": list_convert(surf.radius[sys_id].tolist()),
                    "material": surf.material['name'][sys_id],
                    "roc": list_convert([1 / surf.roc[sys_id].item() if surf.roc[sys_id].item() != 0 else None]),
                    "thick": list_convert(surf.thick[sys_id].tolist()),
                    "conic": list_convert(surf.conic[sys_id].tolist()),
                    "qi_list": qi_list,
                    "rnorm": list_convert(surf.rnorm[sys_id].tolist()),
                }
            elif isinstance(surf, Qbfs):
                qi_list = []
                for i in range(surf.qi_num):
                    exec('qi_list.append(list_convert(surf.qi{}[sys_id].tolist()))'.format(i))
            
                json_content["Surface{}".format(idx+1)] = {
                    "type": "Qbfs",
                    "zoom_type": self.zoom_type[idx] if hasattr(self, 'zoom_type') else None,
                    "stop": (idx + 1) == self.stop_id,
                    "radius": list_convert(surf.radius[sys_id].tolist()),
                    "material": surf.material['name'][sys_id],
                    "roc": list_convert([1 / surf.roc[sys_id].item() if surf.roc[sys_id].item() != 0 else None]),
                    "thick": list_convert(surf.thick[sys_id].tolist()),
                    "conic": list_convert(surf.conic[sys_id].tolist()),
                    "qi_list": qi_list,
                    "rnorm": list_convert(surf.rnorm[sys_id].tolist()),
                }
            elif isinstance(surf, Sphere):
                json_content["Surface{}".format(idx+1)] = {
                    "type": "Standard",
                    "zoom_type": self.zoom_type[idx] if hasattr(self, 'zoom_type') else None,
                    "stop": (idx + 1) == self.stop_id,
                    "radius": list_convert(surf.radius[sys_id].tolist()),
                    "material": surf.material['name'][sys_id],
                    "roc": list_convert([1 / surf.roc[sys_id].item() if surf.roc[sys_id].item() != 0 else None]),
                    "thick": list_convert(surf.thick[sys_id].tolist()),
                    "conic": list_convert(surf.conic[sys_id].tolist()),
                }
            if surf.aperture != 'float':
                json_content["Surface{}".format(idx+1)]["aperture"] = surf.aperture
                json_content["Surface{}".format(idx+1)]["min_r"] = surf.min_r[sys_id].item()
                json_content["Surface{}".format(idx+1)]["max_r"] = surf.max_r[sys_id].item()
    
        json_content["IMAGE"] = {"radius": list_convert(self.system[-1].radius[sys_id].tolist())}
        json.dump(json_content, json_file, indent=4)
        json_file.close()
        