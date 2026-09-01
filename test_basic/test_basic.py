# %%
import torch
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=10)
# torch.autograd.set_detect_anomaly(True)

import sys as system
system.path.append('..')
import lens
# %%
name = 'gauss'
views = [0., 8., 14.]

args = lens.GetYaml(f'../lens_yaml/{name}.yaml')
lens.configure_material_catalog(getattr(args, 'MATERIAL_CATALOG', None))
torch.set_default_device(f'{args.DEVICE}')
lens.set_random_seed(args.SEED)

norm_views = [v / max(views) for v in views] if 'views' in vars() else args.NORM_VIEWS
vig = None if 'views' in vars() else args.VIG
waveweights_rgb = torch.tensor([args.WAVEWEIGHTS_R, args.WAVEWEIGHTS_G, args.WAVEWEIGHTS_B])

sys = f'../lens_json/{name}.json'
sys = lens.System(wavelengths=args.WAVELENGTHS, waveweights=args.WAVEWEIGHTS, p_wvl=args.P_WAVE, 
                  max_view=args.MAX_VIEW, 
                  sys_num=args.SYS_NUM, cfg_num=args.CFG_NUM, 
                  pre_samp=args.PRE_SAMP, fix_radius_surf=args.FIX_RADIUS_SURF, 
                  norm_views=norm_views, azimuths=args.AZIMUTHS, vig=vig,
                  file=sys)
print(f'EFFL: {sys.EFFL}')
print(f'FNO: {sys.FNO}')
print(f'ENPD: {sys.ENPD}')
print(f'ENPP: {sys.ENPP}')
print(f'EXPD: {sys.EXPD}')
print(f'EXPP: {sys.EXPP}')
print(f'TOTR: {sys.TOTR}')

analysis = lens.Analysis(sys)
merit = lens.Merit(sys, args.SAMP_RAYS)
optimizer = torch.optim.Adam(merit.params_lr(args.LR_OPT), fused=True)

sys_id, cfg_id = 0, 0
# %%
###############################! test analysis !###############################
analysis.plot_setup_with_trace(sys_id, cfg_id)
analysis.spot_diagram(sys_id, cfg_id, 31)
rl = analysis.relative_illumination(sys_id, cfg_id, pupil_samp=63, field_samp=11)
distor = analysis.distortion(sys_id, cfg_id, pupil_samp=63, field_samp=11)
# %%
pupil_samp = 127
image_size = 63
delta = 0.5
norm_view = 1.
azimuth = 0.
# %%
opd = analysis.wavefront(sys_id, cfg_id, pupil_samp, norm_view, azimuth, use_exit_pupil_shape=True)
# %%
psf = analysis.psf_spot(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth)
# %%
psf = analysis.psf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth, 480e-6)
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth, wavelength=480e-6, freq_max=500)
# %%
psf = analysis.psf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth, 550e-6)
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth, wavelength=550e-6, freq_max=500)
# %%
psf = analysis.psf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth, 650e-6)
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth, wavelength=650e-6, freq_max=500)
# %%
psf = analysis.psf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth)
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_view, azimuth, freq_max=500)
# %%
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_views, azimuth, 480e-6, 500)
# %%
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_views, azimuth, 550e-6, 500)
# %%
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_views, azimuth, 650e-6, 500)
# %%
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_views, azimuth, freq_max=500)
# %%
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, norm_views, [0., 90.], freq_max=500)
# %%
mtf = analysis.mtf(sys_id, cfg_id, pupil_samp, image_size, delta, azimuth=azimuth, freq_max=500)
# %%
###############################! test optim !###############################
# set variables
match name:
    case 'g_014':
        for i in range(1, len(sys.extract_surfs())-1):
            sys.freeze_sys_param(i, 'conic')
    case 'gauss' | 'wide_50':
        for i in range(1, len(sys.extract_surfs())-1):
            sys.freeze_sys_param(i, 'conic')
        sys.freeze_sys_param(sys.stop_id, 'roc')
    case 'phone':
        sys.freeze_sys_param(sys.stop_id, 'conic')
        sys.freeze_sys_param(sys.stop_id, 'roc')

sys.print_sys_para(sys_id, cfg_id)
# %%
merit.update_system(args.MAX_RADIUS, True)
sys.print_sys_para(sys_id, cfg_id)
# %%
analysis.plot_setup_with_trace(sys_id, cfg_id)
analysis.spot_diagram(sys_id, cfg_id)
rl = analysis.relative_illumination(sys_id, cfg_id, pupil_samp=63, field_samp=11)
distor = analysis.distortion(sys_id, cfg_id, pupil_samp=31, field_samp=11)
# %%
ang_num, azi_num = 2, 2
pupil_samp, image_samp, image_delta = 127, 31, 1.

angle = torch.rand(ang_num).tolist()
azimuth = (360. * torch.rand(azi_num)).tolist()

psfs = merit.psf_rs(pupil_samp, image_samp, image_delta, angle, azimuth, auto=False)
psfs = merit.psf_to_rgb(psfs, waveweights_rgb, True)
# %%
optimizer.zero_grad()
loss_opt = merit.forward_loss(args.MERIT, path='../results')
loss_opt[sys.valid].sum().backward()
optimizer.step()
# %%
sys.print_sys_para(sys_id, cfg_id)
sys.print_sys_grad(sys_id, cfg_id)
# %%
analysis.plot_setup_with_trace(sys_id, cfg_id)
analysis.spot_diagram(sys_id, cfg_id)
rl = analysis.relative_illumination(sys_id, cfg_id, pupil_samp=63, field_samp=11)
distor = analysis.distortion(sys_id, cfg_id, pupil_samp=31, field_samp=11)
# %%
