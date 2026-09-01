# %%
import torch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)

import sys as system
system.path.append('..')
import lens
# %%
name = 'zoom_3x'
args = lens.GetYaml(f'../lens_yaml/{name}.yaml')
lens.configure_material_catalog(getattr(args, 'MATERIAL_CATALOG', None))
torch.set_default_device(f'{args.DEVICE}')
lens.set_random_seed(args.SEED)

waveweights_rgb = torch.tensor([args.WAVEWEIGHTS_R, args.WAVEWEIGHTS_G, args.WAVEWEIGHTS_B])

sys = f'../lens_json/{name}.json'
sys = lens.System(wavelengths=args.WAVELENGTHS, waveweights=args.WAVEWEIGHTS, p_wvl=args.P_WAVE, 
                  max_view=args.MAX_VIEW, 
                  sys_num=args.SYS_NUM, cfg_num=args.CFG_NUM, 
                  pre_samp=args.PRE_SAMP, fix_radius_surf=args.FIX_RADIUS_SURF, 
                  norm_views=args.NORM_VIEWS, azimuths=args.AZIMUTHS, vig=args.VIG,
                  file=sys)
print(f'EFFL: {sys.EFFL}')
print(f'FNO: {sys.FNO}')
print(f'ENPD: {sys.ENPD}')
print(f'ENPP: {sys.ENPP}')
print(f'EXPD: {sys.EXPD}')
print(f'EXPP: {sys.EXPP}')
print(f'TOTR: {sys.TOTR}')

analysis = lens.Analysis(sys)
merit = lens.MeritZ(system=sys, samp_rays=args.SAMP_RAYS)
optimizer = torch.optim.Adam(merit.params_lr(args.LR_OPT), fused=True)
# %%
sys_id, cfg_id = 0, 2
analysis.plot_setup_with_trace(sys_id, cfg_id)
analysis.spot_diagram(sys_id, cfg_id)
rl = analysis.relative_illumination(sys_id, cfg_id, pupil_samp=63, field_samp=11)
distor = analysis.distortion(sys_id, cfg_id, pupil_samp=63, field_samp=11)
# %%
sys_id, cfg_id = 0, 2
view, azimuth = 1., 0.
pupil_samp, image_samp, image_delta = 129, 31, 1.
psf = analysis.psf(sys_id, cfg_id, pupil_samp, image_samp, image_delta, view, azimuth)
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_samp, image_delta, view, azimuth, freq_max=500.)
# %%
sys_id, cfg_id = 0, 2
views = [0., 0.5, 1.]
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_samp, image_delta, views, azimuth=0., freq_max=500.)
# %%
# set variables
match name:
    case 'zoom_3x':
        # hook the common variables        
        for i in range(1, len(sys.extract_surfs())-1):
            if sys.zoom_type[i-1][-1] != 'M':
                sys.avg_sys_para_grad(i)
            else:
                print(i)

        # set variables
        for i in range(1, len(sys.extract_surfs())-1):
            sys.freeze_sys_param(i, 'conic')
            
# %%
ang_num, azi_num = 2, 2
pupil_samp, image_samp, image_delta = 127, 31, 1.

norm_view = [0., 1.]
azimuth = (360. * torch.rand(azi_num)).tolist()

psfs = merit.psf_rs(pupil_samp, image_samp, image_delta, norm_view, azimuth, auto=False)
psfs = merit.psf_to_rgb(psfs, waveweights_rgb, True)
# %%
optimizer.zero_grad()
loss_opt = merit.forward_loss(args.MERIT, path='../results')
loss_opt[sys.valid].sum().backward()
optimizer.step()
# %%
sys_id, cfg_id = 0, 2
sys.print_sys_para(sys_id, cfg_id)
sys.print_sys_grad(sys_id, cfg_id)
# %%
merit.update_system(args.MAX_RADIUS, sys.zoom_type[-2][-1] == 'F')
# %%
sys_id, cfg_id = 0, 2
analysis.plot_setup_with_trace(sys_id, cfg_id)
analysis.spot_diagram(sys_id, cfg_id)
rl = analysis.relative_illumination(sys_id, cfg_id, pupil_samp=63, field_samp=11)
distor = analysis.distortion(sys_id, cfg_id, pupil_samp=63, field_samp=11)
# %%
