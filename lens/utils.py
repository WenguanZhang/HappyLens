import torch
import os
import yaml
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from scipy.spatial import ConvexHull
from scipy import stats
from scipy import special

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  

eps = 1e-16

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def rand_dropout(optimizer, prob=0.1):
    """
    Randomly drops out gradients with probability prob.
    
    :param optimizer: optimizer with the gradients to be dropped out
    :param prob: dropout probability, default 0.1
    """
    prob = 1. - prob
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                    if param.dim() == 1:
                        param.grad.data = torch.where(torch.rand(param.shape[0]) < prob, param.grad.data, torch.zeros_like(param.grad.data))
                    else:
                        param.grad.data = torch.where(torch.rand(param.shape[0])[:, None] < prob, param.grad.data, torch.zeros_like(param.grad.data))

fraunhofer = dict(   # http://en.wikipedia.org/wiki/Abbe_number
    i=365.01e-6,  # Hg UV
    h=404.66e-6,  # Hg violet
    g=435.84e-6,  # Hg blue
    Fp=479.99e-6,  # Cd blue
    F=486.1327e-6,  # H  blue
    e=546.07e-6,  # Hg green
    Gy=555.00e-6,  # greenish-yellow
    d=587.5618e-6,  # He yellow
    D=589.30e-6,  # Na yellow
    Cp=643.85e-6,  # Cd red
    C=656.2725e-6,  # H  red
    r=706.52e-6,  # He red
    Ap=768.20e-6,  # K  IR
    s=852.11e-6,  # Cs IR
    t=1013.98e-6,  # Hg IR
)  # unit: [mm]

lambda_F = fraunhofer["F"]
lambda_d = fraunhofer["d"]
lambda_C = fraunhofer["C"]

def _read_material_catalog(name, catalog_type):
    """Read a material catalog by name or explicit JSON path."""
    if not isinstance(name, str):
        raise TypeError(f'{catalog_type} catalog name must be a string, got {type(name).__name__}')

    catalog_dir = os.path.dirname(__file__)
    requested_name = name.strip()
    explicit_path = os.path.abspath(requested_name)
    if requested_name.lower().endswith('.json') and os.path.isfile(explicit_path):
        catalog_path = explicit_path
        normalized_name = os.path.splitext(os.path.basename(catalog_path))[0]
        if normalized_name.lower().startswith('glass_'):
            normalized_name = normalized_name[6:]
        normalized_name = normalized_name.lower()
    else:
        normalized_name = requested_name.lower()
        if normalized_name.startswith('glass_'):
            normalized_name = normalized_name[6:]
        if normalized_name.endswith('.json'):
            normalized_name = normalized_name[:-5]

        available_catalogs = {}
        for filename in sorted(os.listdir(catalog_dir), key=str.lower):
            filename_lower = filename.lower()
            if not filename_lower.endswith('.json'):
                continue
            stem = filename_lower[:-5]
            available_catalogs.setdefault(stem, filename)
            if stem.startswith('glass_'):
                available_catalogs.setdefault(stem[6:], filename)

        filename = available_catalogs.get(normalized_name)
        if filename is None:
            expected_paths = [
                os.path.join(catalog_dir, f'glass_{normalized_name}.json'),
                os.path.join(catalog_dir, f'{normalized_name}.json'),
            ]
            discovered = sorted(name for name in available_catalogs if not name.startswith('glass_'))
            raise FileNotFoundError(
                f'Material catalog not found: {catalog_type.upper()}={name!r}. '
                f'Searched for {expected_paths[0]} and {expected_paths[1]}. '
                f'Add one of these files or provide an existing JSON path. '
                f'Available catalogs: {", ".join(discovered) or "none"}.'
            )
        catalog_path = os.path.join(catalog_dir, filename)

    with open(catalog_path, 'r', encoding='utf-8') as file:
        catalog = json.load(file)
    if not isinstance(catalog, dict) or not catalog:
        raise ValueError(f'Material catalog {catalog_path} must contain a non-empty JSON object')
    return normalized_name, catalog


def _merge_material_catalogs(names, catalog_type):
    """Merge catalogs in order, preserving the first value for duplicate names."""
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)) or not names:
        raise TypeError(f'{catalog_type} catalog selection must be a name or a non-empty list')

    normalized_names = []
    merged_catalog = {}
    for name in names:
        normalized_name, catalog = _read_material_catalog(name, catalog_type)
        normalized_names.append(normalized_name)
        for material_name, material_data in catalog.items():
            if material_name not in merged_catalog:
                merged_catalog[material_name] = material_data
    return normalized_names, merged_catalog


def _catalog_parameter_transform(catalog):
    """Build the normalized n/v coordinates used by material optimization."""
    params_nv = torch.tensor(
        [[material['nd'], material['vd']] for material in catalog.values()],
        dtype=torch.get_default_dtype(),
        device='cpu',
    )
    lamb0 = stats.boxcox_normmax(params_nv[:, 0].numpy(), method='mle')
    lamb1 = stats.boxcox_normmax(params_nv[:, 1].numpy(), method='mle')
    lambs = torch.tensor([lamb0, lamb1], dtype=params_nv.dtype, device='cpu')
    params = (params_nv ** lambs - 1) / lambs
    mean = params.mean(dim=0)
    scale = params.amax(dim=0) - params.amin(dim=0)
    params = (params - mean) / scale
    return params_nv, lamb0, lamb1, lambs, params, mean, scale


# These containers are mutated in place when catalogs change. Other lens
# modules import them directly, so preserving object identity prevents stale
# references after a YAML-selected catalog is loaded.
glass_catalog = {}
glass_catalog_params = torch.empty((0, 2), device='cpu')
plastic_catalog = {}
plastic_catalog_params = torch.empty((0, 2), device='cpu')
active_material_catalogs = {}


def configure_material_catalog(config=None, *, glass='schott', plastic='plastic'):
    """Load the material catalogs selected by a YAML ``MATERIAL_CATALOG`` map.

    ``config`` accepts the mapping returned by ``GetYaml``, for example
    ``{'GLASS': ['schott', 'ohara'], 'PLASTIC': 'hoya'}``. A name resolves to
    ``glass_<name>.json`` in the ``lens`` directory; an explicit JSON path is
    also accepted. ``GLASS`` and ``PLASTIC`` describe how a catalog is used and
    do not restrict which catalog names may be loaded. Catalogs are merged in
    order and the first definition wins when material names repeat. Missing
    entries retain the default Schott and plastic catalogs.
    """
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError('MATERIAL_CATALOG must be a mapping')
        glass = config.get('GLASS', config.get('glass', glass))
        plastic = config.get('PLASTIC', config.get('plastic', plastic))

    glass_names, new_glass_catalog = _merge_material_catalogs(glass, 'glass')
    plastic_names, new_plastic_catalog = _merge_material_catalogs(plastic, 'plastic')
    glass_data = _catalog_parameter_transform(new_glass_catalog)
    plastic_data = _catalog_parameter_transform(new_plastic_catalog)

    glass_catalog.clear()
    glass_catalog.update(new_glass_catalog)
    # Modules import these tensors directly, so their object identity must stay
    # stable when catalogs are reconfigured. Replacing ``.data`` also allows
    # the backing dtype to follow a later ``torch.set_default_dtype()`` call;
    # ``Tensor.set_()`` cannot switch between float32 and float64 storage.
    glass_catalog_params.data = glass_data[4]

    plastic_catalog.clear()
    plastic_catalog.update(new_plastic_catalog)
    plastic_catalog_params.data = plastic_data[4]

    global glass_catalog_params_nv, gla_lamb0, gla_lamb1, gla_lambs, gla_mean, gla_scale
    global plastic_catalog_params_nv, pla_lamb0, pla_lamb1, pla_lambs, pla_mean, pla_scale
    glass_catalog_params_nv, gla_lamb0, gla_lamb1, gla_lambs, _, gla_mean, gla_scale = glass_data
    plastic_catalog_params_nv, pla_lamb0, pla_lamb1, pla_lambs, _, pla_mean, pla_scale = plastic_data

    active_material_catalogs.clear()
    active_material_catalogs.update({'GLASS': glass_names, 'PLASTIC': plastic_names})
    return active_material_catalogs.copy()


def _plot_material_catalogs():
    """Plot the active material transforms for internal diagnostics."""
    for params_nv, params, label in [
        (glass_catalog_params_nv, glass_catalog_params, 'glass'),
        (plastic_catalog_params_nv, plastic_catalog_params, 'plastic'),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, data, title in [
            (axes[0, 0], params_nv[:, 0], "Refractive index — n"),
            (axes[0, 1], params[:, 0], "Transformed refractive index — g"),
            (axes[1, 0], params_nv[:, 1], "Abbe number — v"),
            (axes[1, 1], params[:, 1], "Transformed Abbe number — g'"),
        ]:
            stats.probplot(data, dist='norm', plot=ax)
            ax.set_title(title)
            ax.get_lines()[0].set_markersize(3)
            ax.get_lines()[1].set_color('red')
        plt.tight_layout()
        plt.savefig(f'boxcox_qq_{label}.pdf', dpi=200, bbox_inches='tight')
        plt.show()

        hull = ConvexHull(params.cpu())
        hull_points = params[hull.vertices].cpu()
        plt.figure(figsize=(10, 10))
        sns.set_theme(style='white', font_scale=1.)
        sns.jointplot(
            x=params.cpu()[:, 0],
            y=params.cpu()[:, 1],
            label=f'{label.title()} Points',
            marginal_kws=dict(bins=15, kde=True, color='#a6c988'),
        )
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'r--', lw=2, label=f'{label.title()} Boundaries')
        plt.fill(hull_points[:, 0], hull_points[:, 1], 'r', alpha=0.2)
        plt.xlabel('Parameter g')
        plt.ylabel("Parameter g'")
        plt.legend()
        plt.savefig(f'./{label[:3]}_mat.svg')
        plt.show()


configure_material_catalog()

    
def nv_to_g1_g2(n, v, mat_cata='G'):
    if mat_cata == 'G':
        lambs = gla_lambs
        mean = gla_mean
        scale = gla_scale
    elif mat_cata == 'P':
        lambs = pla_lambs
        mean = pla_mean
        scale = pla_scale
    else:
        raise ValueError(f'Unknown material type {mat_cata}')
    n = (n ** lambs[0] - 1) / lambs[0]
    v = (v ** lambs[1] - 1) / lambs[1]
    g1 = (n - mean[0]) / scale[0]
    g2 = (v - mean[1]) / scale[1]
    return g1, g2
    
def g1_g2_to_n(g1, g2, wavelength, mat_cata='G'):
    # wavelength in [mm]
    if mat_cata == 'G':
        lambs = gla_lambs
        mean = gla_mean
        scale = gla_scale
    elif mat_cata == 'P':
        lambs = pla_lambs
        mean = pla_mean
        scale = pla_scale
    else:
        raise ValueError(f'Unknown material type {mat_cata}')
    g1 = g1 * scale[0] + mean[0]
    g2 = g2 * scale[1] + mean[1]
    n = (g1 * lambs[0] + 1) ** (1 / lambs[0])
    v = (g2 * lambs[1] + 1) ** (1 / lambs[1])
    
    B = (n - 1) / v / (lambda_F ** -2 - lambda_C ** -2) # [sys]
    A = n - B * lambda_d ** -2 # [sys]
    return A[None, ...] + B[None, ...] * (wavelength[..., None] / 1e3) ** -2

def fit_get_mat_id(params, method='M', mat_cata='G'):
    """
    params: [B, 2] (g1/g2)
    """
    if mat_cata == 'G':
        catalog_params = glass_catalog_params.to(params.device)
    elif mat_cata == 'P':
        catalog_params = plastic_catalog_params.to(params.device)
    else:
        raise ValueError(f'Unknown material type {mat_cata}')
                
    match method:
        case 'M': 
            catalog_params_ = (catalog_params - catalog_params.mean(dim=0, keepdim=True))
            cov_mat = torch.zeros((2, 2), device=params.device) # [2, 2]
            cov_mat[0, 0] = torch.sum(catalog_params_[:, 0] * catalog_params_[:, 0], dim=0) / (catalog_params_.shape[0] - 1)
            cov_mat[0, 1] = torch.sum(catalog_params_[:, 0] * catalog_params_[:, 1], dim=0) / (catalog_params_.shape[0] - 1)
            cov_mat[1, 0] = torch.sum(catalog_params_[:, 0] * catalog_params_[:, 1], dim=0) / (catalog_params_.shape[0] - 1)
            cov_mat[1, 1] = torch.sum(catalog_params_[:, 1] * catalog_params_[:, 1], dim=0) / (catalog_params_.shape[0] - 1)
            params_sub = params[None, :, None, :] - catalog_params[:, None, None, :] # [glass_num, B, 1, 2]
            catalog_res = torch.matmul(torch.matmul(params_sub, cov_mat[None, None, :, :].repeat(catalog_params.shape[0], params.shape[0], 1, 1)), params_sub.permute(0, 1, 3, 2))[:, :, 0, 0] # [glass_num, sys]
        case 'E':
            catalog_res = torch.sum((catalog_params[:, None, :] - params[None, :, :]) ** 2, dim=-1) # [glass_num, B]
    
    idx = torch.argmin(catalog_res, dim=0) # [B]
    return idx

vaccum_nd = 1.0
vaccum_vd = None

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    # print(a.shape)
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = ax * bw + aw * bx - az * by + ay * bz
    oy = ay * bw + az * bx + aw * by - ax * bz
    oz = az * bw - ay * bx + ax * by + aw * bz
    return torch.stack((ow, ox, oy, oz), -1)

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def normalize(d):
    return d / torch.sqrt(torch.sum(d ** 2, dim=-1).clip(eps))[..., None]

def length(d):
    return torch.sqrt(torch.sum(d ** 2, dim=-1).clip(eps))

def find_key(d, target, path=""):
    results = []
    for k, v in d.items():
        cur = f"{path}.{k}" if path else k
        if k == target:
            results.append((cur, v))
        if isinstance(v, dict):
            results.extend(find_key(v, target, cur))
    return results

def pupil_distribution(rays_h, rays_w, distribution):
    """
    Calculate the ray sample under the normalized pupil distribution, e.g., "suqare", "triangular" 
    Represent the ray_pupil with x and y coordinates. Normalized, so [-1, 1]
    Input Args:
        rays_h: (int) number of rays in h direction, e.g., 101, 1001, ...
        rays_w: (int) number of rays in w direction, e.g., 101, 1001, ...
        distribution: (str) name of distribution, e.g., 'square' 
    Returns Args:
        o_p: (2D Torch.tensor) sampled rays on pupil
        ref: (int) chief ray index
    """
    d = distribution
    h = rays_h
    w = rays_w
    if d == 'square':
        h_p = torch.linspace(-1., 1., h)
        w_p = torch.linspace(-1., 1., w)
        h_p, w_p = torch.meshgrid(h_p, w_p, indexing='ij')
        h_p, w_p = h_p.reshape(rays_h * rays_w, -1), w_p.reshape(rays_h * rays_w, -1)
        o_p = torch.hstack([h_p, w_p])
    elif d == 'hexapolar':
        h_p = torch.linspace(-1., 1., h)
        w_p = torch.linspace(-1., 1., w)
        h_p, w_p = torch.meshgrid(h_p, w_p, indexing='ij')
        theta = - torch.atan2(h_p, w_p)
        o_p = torch.stack((torch.sin(theta), torch.cos(theta)), dim=-1)
        o_p[..., 0] *= torch.max(torch.abs(h_p), torch.abs(w_p))
        o_p[..., 1] *= torch.max(torch.abs(h_p), torch.abs(w_p))
        o_p = o_p.reshape(rays_h * rays_w, -1)
    elif d == 'fibonacci':
        rays_num = rays_h * rays_w
        R = torch.sqrt(torch.linspace(1 / 2, rays_num - 1 / 2, rays_num)) / torch.sqrt(torch.tensor(rays_num - 1 / 2))
        T = 4 / (1 + torch.sqrt(torch.tensor(5))) * torch.pi * torch.linspace(1, rays_num, rays_num)
        x = R * torch.cos(T) * 1.
        y = R * torch.sin(T) * 1.
        o_p = torch.stack((x, y), dim=-1)
    elif d == 'ring':
        rings_num = rays_h
        x = [0.]
        y = [0.]
        rings = torch.linspace(0., 1., rings_num+1)
        for i, r in enumerate(rings[1:]):
            angle = torch.linspace(0, 2 * torch.pi, 6 * (i + 1) + 1)[1:]
            x.extend(r * torch.cos(angle) * 1.)
            y.extend(r * torch.sin(angle) * 1.)
        o_p = torch.stack((torch.tensor(x), torch.tensor(y)), dim=-1)
    elif d == 'line':
        h_p = torch.linspace(-1., 1., h)
        w_p = torch.zeros_like(h_p)
        o_p = torch.stack([w_p, h_p], dim=-1)
    else:
        raise ValueError('Unknown ray distribution', d)

    o_p = torch.cat((torch.tensor([[0., 0.]]), o_p)) # initial chief ray id = 0
    return o_p

def limit_var(var:torch.Tensor, var_min:float, var_max:float):
    res_min = torch.where(var < var_min, var_min - var, 0.)
    res_max = torch.where(var > var_max, var - var_max, 0.)
    return res_min + res_max

def list_convert(lst):
    """
    Convert a list with a single element or all identical elements to that element.
    If the list has more than one distinct element, return the list as is.
    
    :param lst: List to be converted
    :return: Single element if the list has only one element or all elements are the same, otherwise the original list
    """
    if isinstance(lst, list):
        if len(lst) == 1:
            return lst[0]
        elif len(set(lst)) == 1:
            return lst[0]
    return lst

def generate_normalized_numbers(n, vmin=-1, vmax=1):
    numbers = torch.zeros(n)
    remaining_sum = 1.0
    
    if n > 1:
        for i in range(n-1):
            max_val = min(vmax, remaining_sum + (n-i-1))
            min_val = max(vmin, remaining_sum - (n-i-1))
            numbers[i] = torch.rand(1) * (max_val - min_val) + min_val
            remaining_sum -= numbers[i]
        
        numbers[-1] = remaining_sum
        if numbers[-1] < vmin or numbers[-1] > vmax:
            return generate_normalized_numbers(n, vmin, vmax)
        else:
            return numbers
    else:
        return torch.tensor([1.0])


# ---------------------------------------------------------------------------
# Zernike polynomials: https://en.wikipedia.org/wiki/Zernike_polynomials
# ---------------------------------------------------------------------------

def zernike_radial(n, m, rho):
    """Radial polynomial R_n^m(rho) of the Zernike basis.

    Args:
        n: radial order (non-negative int)
        m: azimuthal order (non-negative int, n-m even)
        rho: radial coordinate tensor (arbitrary shape, 0 <= rho <= 1)

    Returns:
        R_n^m(rho) — same shape as rho
    """
    m = abs(m)
    if (n - m) % 2 != 0:
        raise ValueError(f"n-m must be even, got n={n}, m={m}")
    if m > n:
        raise ValueError(f"m must be <= n, got n={n}, m={m}")

    result = torch.zeros_like(rho)
    kmax = (n - m) // 2
    for k in range(kmax + 1):
        sign = 1.0 if k % 2 == 0 else -1.0
        num = factorial(n - k)
        den = factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        result = result + sign * (num / den) * rho ** (n - 2 * k)
    return result


def zernike_noll_to_nm(j):
    """Convert Noll index j (1-based) to (n, m).

    Returns:
        n: radial order
        m: azimuthal order (non-negative)
    """
    if j < 1:
        raise ValueError(f"Noll index j must be >= 1, got {j}")

    n = 0
    idx = 1  # current Noll index
    while idx <= max(j, 100):  # safety cap
        start_m = 0 if n % 2 == 0 else 1
        for m in range(start_m, n + 1, 2):
            if m == 0:
                if idx == j:
                    return n, 0
                idx += 1
            else:
                if idx == j:
                    return n, m * (-1) ** j
                idx += 1
                if idx == j:
                    return n, m * (-1) ** j
                idx += 1
        n += 1

    raise RuntimeError(f"Could not map Noll index j={j}")


def zernike_noll(j, rho, theta):
    """
    Args:
        j: Noll index (1-based)
        rho: radial coordinate tensor
        theta: azimuthal coordinate tensor

    Returns:
        Z_j(rho, theta) — same shape as rho
    """
    n, m = zernike_noll_to_nm(j)
    R = zernike_radial(n, m, rho)

    if m == 0:
        norm = (n + 1) ** 0.5
        return norm * R
    else:
        norm = (2 * (n + 1)) ** 0.5
        angular = torch.cos(abs(m) * theta) if m > 0 else torch.sin(abs(m) * theta)
        return norm * R * angular


def zernike_wavefront(rho, theta, coeffs):
    """Sum of weighted Zernike terms.

    Args:
        rho: radial coordinate tensor (any shape)
        theta: azimuthal coordinate tensor (same shape as rho)
        coeffs: dict of {noll_index: coefficient} or list of (noll_index, coeff)
                Coefficients have units of length (e.g. metres or waves).

    Returns:
        Superposition Σ c_j · Z_j(rho, theta) — same shape as rho
    """
    items = coeffs.items() if isinstance(coeffs, dict) else coeffs
    wf = torch.zeros_like(rho)
    for j, c in items:
        if abs(c) > 1e-30:  # skip numerically zero terms
            wf = wf + c * zernike_noll(j, rho, theta)
    return wf


# Convenience: names for common low-order Noll indices
ZERNIKE_NAME = {
    1:  "piston",
    2:  "tip X",
    3:  "tilt Y",
    4:  "defocus",
    5:  "astigmatism 0°",
    6:  "astigmatism 45°",
    7:  "coma X",
    8:  "coma Y",
    9:  "trefoil X",
    10: "trefoil Y",
    11: "primary spherical",
    12: "secondary astigmatism 0°",
    13: "secondary astigmatism 45°",
    14: "secondary coma X",
    15: "secondary coma Y",
}


class Ray(object):
    """
    Definition of a geometric ray.

    - o is the ray position
    - d is the ray direction (normalized)
    - t is the optical path (accumulated during propagate)
    """

    def __init__(self, o, d, wavelength):
        super(Ray, self).__init__()
        wavelength = wavelength if torch.is_tensor(wavelength) else torch.tensor(wavelength)
        wavelength = wavelength.unsqueeze(0) if wavelength.dim() == 0 else wavelength
        self.wavelength = wavelength
        
        self.o = o.unsqueeze(0).repeat(len(wavelength), *[1 for _ in range(len(o.shape))])
        self.d = d.unsqueeze(0).repeat(len(wavelength), *[1 for _ in range(len(d.shape))])
        self.t = torch.zeros(self.o.shape[0:-1])
        self.chief_id = torch.zeros_like(self.t[..., -1], dtype=torch.int64)
        self.valid = torch.ones((self.o.shape[0:-1])).bool()
        
        
class GetYaml():
    def __init__(self, file_path):
        if os.path.exists(file_path):
            self.file_path = file_path
            self.data = self.read_yaml()
            for key, value in yaml.load(self.data, Loader=yaml.FullLoader).items():
                setattr(self, key, value)

        else:
            print('%s not found!' % file_path)
        
    def read_yaml(self):
        with open(self.file_path, 'r', encoding='utf-8')as f:
            p = f.read()
            return p
        
        
class CoherentPsfOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, d, grid, opd, k):
        ctx.save_for_backward(o, d, grid, opd, k)
        r = grid[:, :, None, :] - o[None, None, :, :]
        dr = torch.einsum('...k,...k->...', d[None, None, :, :], r)
        amp = torch.einsum('...k->...', torch.exp(1j * k * (opd + dr)) * d[None, None, :, 2])
        psf = torch.abs(amp) ** 2
        return psf
    
    @staticmethod
    def backward(ctx, grad_output):
        o, d, grid, opd, k = ctx.saved_tensors
        grad_o = torch.empty_like(o) if ctx.needs_input_grad[0] else None
        grad_d = torch.empty_like(d) if ctx.needs_input_grad[1] else None
        grad_grid = torch.empty_like(grid) if ctx.needs_input_grad[2] else None
        grad_opd = torch.empty_like(opd) if ctx.needs_input_grad[3] else None

        phase = k * (opd + torch.einsum('...k,...k->...', d[None, None, :, :], grid[:, :, None, :] - o[None, None, :, :]))
        base_elec = torch.polar(phase.new_ones(phase.shape), phase)

        # shape [x,y,ray]
        dI_dEReal = 2 * torch.sum((base_elec * d[None, None, :, 2]).real, dim=-1, keepdim=True)
        dI_dEImag = 2 * torch.sum((base_elec * d[None, None, :, 2]).imag, dim=-1, keepdim=True)
        
        grad_phase = (dI_dEReal * (-base_elec.imag * d[None, None, :, 2]) + dI_dEImag * base_elec.real * d[None, None, :, 2]) * grad_output[..., None]
        grad_Dz = torch.sum((dI_dEReal * base_elec.real + dI_dEImag * base_elec.imag) * grad_output[..., None], dim=[0, 1])[:, None]
        grad_D = torch.cat([grad_Dz.new_zeros(grad_Dz.shape), grad_Dz.new_zeros(grad_Dz.shape), grad_Dz], dim=-1)

        if ctx.needs_input_grad[0]:
            grad_o = torch.sum((k * grad_phase)[..., None] * (-d[None, None, :, :]), dim=[0, 1])
        if ctx.needs_input_grad[1]:
            grad_d = torch.sum((k * grad_phase)[..., None] * (grid[:, :, None, :] - o[None, None, :, :]), dim=[0, 1]) + grad_D
        if ctx.needs_input_grad[2]:
            grad_grid = torch.sum((k * grad_phase)[..., None] * d[None, None, :, :], dim=[2])
        if ctx.needs_input_grad[3]:
            grad_opd = k * torch.sum(grad_phase, dim=[0, 1])
        
        return grad_o, grad_d, grad_grid, grad_opd, None
    
    
class RayleighSommerfeldPsfOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, o, t, grid, k, l):
        ctx.save_for_backward(o, t, grid, k, l)
        r = length(grid[:, :, None, :] - o[None, None, ...]).clip(min=eps)
        amp = torch.einsum('ijk->ij', l / r * (1j * k - 1 / r) * torch.exp(1j * k * (t + r)) / r)
        psf = torch.abs(amp) ** 2
        return psf
    
    @staticmethod
    def backward(ctx, grad_output):
        o, t, grid, k, l = ctx.saved_tensors
        grad_o = torch.empty_like(o) if ctx.needs_input_grad[0] else None
        grad_t = torch.empty_like(t) if ctx.needs_input_grad[1] else None
        grad_grid = torch.empty_like(grid) if ctx.needs_input_grad[2] else None
        
        r = length(grid[:, :, None, :] - o[None, None, ...]).clip(min=eps)
        baseE = (l / r) * (1j * k - 1 / r) * torch.exp(1j * k * (t + r)) / r
        E_grad = torch.sum(baseE, dim=-1, keepdim=True) * grad_output[..., None]
        
        dE_dt = baseE * k * 1j
        dE_dr = baseE * (-3 / r) - (k / r) ** 2 * l * torch.exp(1j * k * (t + r))
        grad_r = 2  * (E_grad.real * dE_dr.real + E_grad.imag * dE_dr.imag)
        
        if ctx.needs_input_grad[0]:
            grad_o = torch.sum(grad_r[..., None] * (o[None, None, ...] - grid[:, :, None, :]) / r[..., None], dim=[0, 1])
        if ctx.needs_input_grad[1]:
            grad_t = 2 * torch.sum(E_grad.real * dE_dt.real + E_grad.imag * dE_dt.imag, dim=[0, 1])
        if ctx.needs_input_grad[2]:
            grad_grid = torch.sum(grad_r[..., None] * (grid[:, :, None, :] - o[None, None, ...]) / r[..., None], dim=[2])

        return grad_o, grad_t, grad_grid, None, None

@torch.no_grad()
def plot_loss_pie(losses, labels, valids, path):
    # Calculate total loss
    if len(losses) != len(labels):
        raise ValueError("The length of losses and labels must be the same.")
    
    losses = [loss[valids].mean().item() for loss in losses]
    total_loss = sum(losses)
    
    # Calculate percentages
    percentages = [(loss / total_loss) * 100 for loss in losses]
    
    # Filter out very small percentages to avoid clutter
    threshold = 1.0  # 1% threshold
    filtered_percentages, filtered_labels = [], []
    for pct, label in zip(percentages, labels):
        if pct < threshold: continue
        else:
            filtered_percentages.append(pct)
            filtered_labels.append(label)
    
    # Add "Other" category for small percentages
    other_percentage = sum(percentages) - sum(filtered_percentages)
    if other_percentage > 0:
        filtered_percentages.append(other_percentage)
        filtered_labels.append('Other')
    
    # Calculate filtered losses based on filtered percentages
    filtered_losses = [(pct / 100 * total_loss) for pct in filtered_percentages]
    
    # Generate colors
    plt.figure(figsize=(6, 6))
    colors = cm.rainbow((torch.arange(len(filtered_labels)) / len(filtered_labels)).tolist())
    plt.pie(filtered_losses, labels=filtered_labels, autopct='%1.1f%%', textprops={'fontsize': 6}, colors=colors)
    plt.title(f'Loss Distribution')
    plt.tight_layout()
    plt.savefig(f'{path}/loss_pie.svg')
    plt.close()

def read_prime_json_to_zmx(json_file, zmx_file, wave, wt, p_wvl, norm_views, max_field=None, field='ang'):
    """
    field: 'ang' or 'imh'
    """
    with open(json_file) as file:
        lens_dict = json.load(file)
    file.close()
    
    if field == 'ang':
        field = 0
    elif field == 'imh':
        field = 3
        
    zmx = open(zmx_file, 'w')
    head_str = f"""
VERS 190513 25 123457 L123457
MODE SEQ
NAME
UNIT MM X W X CM MR CPMM
FLOA
GCAT SCHOTT
RAIM 0 2 1 1 0 0 0 0 0 1
FTYP {field} 0 {len(norm_views)} {len(wave)} 0 0 0 {len(norm_views)}"""
    zmx.writelines(head_str)

    lst = [0 for i in norm_views]
    result = ' '.join(map(str, lst))
    x_str = f"""
XFLN {result}"""
    zmx.writelines(x_str)

    lst = [max_field * view for view in norm_views]
    result = ' '.join(map(str, lst))
    y_str = f"""
YFLN {result}"""
    zmx.writelines(y_str)

    for i in range(len(wave)):
        wave_str = f"""
WAVM {i+1} {wave[i] * 1e3} {wt[i]}"""
        zmx.writelines(wave_str)
    wave_str = f"""
PWAV {p_wvl + 1}"""
    zmx.writelines(wave_str)
    
    distance = lens_dict["OBJECT"]['distance'][0] if isinstance(lens_dict["OBJECT"]['distance'], list) else lens_dict["OBJECT"]['distance']
    obj_str = f"""
SURF 0
    TYPE STANDARD
    FIMP
    CURV 0.0
    DISZ {'INFINITY' if distance is None else distance}
    DIAM 0"""
    zmx.writelines(obj_str)

    multi_thick = {}
    for i, item in enumerate(list(lens_dict)[1:-1]):
        if isinstance(lens_dict[item]['thick'], list):
            multi_thick[i+1] = lens_dict[item]['thick']
        if lens_dict[item]['stop']:
            surf_str = f"""
SURF {i+1}
    STOP"""
        else:
            surf_str = f"""
SURF {i+1}"""
        zmx.writelines(surf_str)
        
        roc = lens_dict[item]['roc']
        if lens_dict[item]['type'] == 'Standard':    
            surf_str = f"""
    TYPE STANDARD
    FIMP
    CURV {1 / roc if roc is not None else 0.0}
    DISZ {lens_dict[item]['thick'][0] if isinstance(lens_dict[item]['thick'], list) else lens_dict[item]['thick']}
    CONI {lens_dict[item]['conic']}
    DIAM {lens_dict[item]['radius'][0] if isinstance(lens_dict[item]['radius'], list) else lens_dict[item]['radius']} 1 0 0 1 "" """
            zmx.writelines(surf_str)
        elif lens_dict[item]['type'] == 'Asphere':
            surf_str = f"""
    TYPE EVENASPH
    FIMP
    CURV {1 / roc if roc is not None else 0.0}
    DISZ {lens_dict[item]['thick'][0] if isinstance(lens_dict[item]['thick'], list) else lens_dict[item]['thick']}
    CONI {lens_dict[item]['conic']}
    DIAM {lens_dict[item]['radius'][0] if isinstance(lens_dict[item]['radius'], list) else lens_dict[item]['radius']} 1 0 0 1 "" """
            zmx.writelines(surf_str)
            if len(lens_dict[item]['ai_list']) > 7:
                raise ValueError(f" Asphere surface {item} has more than 16th coefficients!")
            for ai in range(len(lens_dict[item]['ai_list'])):
                surf_str = f"""
    PARM {ai+2} {lens_dict[item]['ai_list'][ai]}"""
                zmx.writelines(surf_str)
        elif lens_dict[item]['type'] == 'Qcon' or lens_dict[item]['type'] == 'Qbfs':
            surf_str = f"""
    TYPE QED_TYPE
    FIMP
    CURV {1 / roc if roc is not None else 0.0}
    DISZ {lens_dict[item]['thick'][0] if isinstance(lens_dict[item]['thick'], list) else lens_dict[item]['thick']}
    CONI {lens_dict[item]['conic']}
    DIAM {lens_dict[item]['radius'][0] if isinstance(lens_dict[item]['radius'], list) else lens_dict[item]['radius']} 1 0 0 1 "" """
            zmx.writelines(surf_str)
            t = 1.0 if lens_dict[item]['type'] == 'Qcon' else 0.0
            surf_str = f"""
    XDAT 1 {t} 0 0 1.000000000000E+00 0.000000000000E+00 0
    XDAT 2 {len(lens_dict[item]['qi_list'])} 0 0 1.000000000000E+00 0.000000000000E+00 0
    XDAT 3 {lens_dict[item]['rnorm']} 0 0 1.000000000000E+00 0.000000000000E+00 0"""
            zmx.writelines(surf_str)
            for qi in range(len(lens_dict[item]['qi_list'])):
                surf_str = f"""
    XDAT {qi+4} {lens_dict[item]['qi_list'][qi]} 0 0 1.000000000000E+00 0.000000000000E+00 0"""
                zmx.writelines(surf_str)
            
        glass_name = lens_dict[item]['material']
        if glass_name != "VACUUM":
            zmx.writelines(f"""
    GLAS {glass_name}""")
                
    img_str = f"""
SURF {len(list(lens_dict))-1}
    TYPE STANDARD
    FIMP
    CURV 0.0
    DISZ 0
    DIAM {lens_dict["IMAGE"]['radius'][0] if isinstance(lens_dict["IMAGE"]['radius'], list) else lens_dict["IMAGE"]['radius']} 0 0 0 1 "" """
    zmx.writelines(img_str)
    
    if isinstance(lens_dict["OBJECT"]['distance'], list):
        multi_str = f"""
CONF 1 0 0 0 0 0 0 0 0 0
BLNK 
TOL TOFF   0   0 0.0000000000000000E+00 0.0000000000000000E+00   0 0 0 0 0
    """
        zmx.writelines(multi_str)
        cfg_num = len(lens_dict["OBJECT"]['distance'])
        zmx.writelines(f"""MNUM {cfg_num} {cfg_num}""")
        n = 1
        for i in range(cfg_num):
            n += 1
            obj_str = f"""
THIC 0 {i+1} {lens_dict["OBJECT"]['distance'][i]} 0 0 0 0 {i+1} {n} 1.000000000000E+00 0 "" 0"""
            zmx.writelines(obj_str)


        for k in multi_thick:
            n += 1
            for i in range(cfg_num):
                multi_str = f"""
THIC {k} {i+1} {multi_thick[k][i]} 0 0 0 0 {i+1} {n} 1.000000000000E+00 0 "" 0"""
                zmx.writelines(multi_str)

def read_zoom_json_to_zmx(json_file, zmx_file, wave, p_wvl, norm_views, max_angle):
    cfg_id = 0
    with open(json_file) as file:
        lens_dict = json.load(file)
    file.close()
    
    zmx = open(zmx_file, 'w')
    head_str = f"""
VERS 190513 25 123457 L123457
MODE SEQ
NAME
UNIT MM X W X CM MR CPMM
FLOA
GCAT SCHOTT
RAIM 0 2 1 1 0 0 0 0 0 1
FTYP 0 0 {len(norm_views)} {len(wave)} 0 0 0 {len(norm_views)}"""
    zmx.writelines(head_str)

    lst = [0 for i in norm_views]
    result = ' '.join(map(str, lst))
    x_str = f"""
XFLN {result}"""
    zmx.writelines(x_str)

    lst = [max_angle[cfg_id] * view for view in norm_views]
    result = ' '.join(map(str, lst))
    y_str = f"""
YFLN {result}"""
    zmx.writelines(y_str)

    for i in range(len(wave)):
        wave_str = f"""
WAVM {i+1} {wave[i] * 1e3} 1"""
        zmx.writelines(wave_str)
    wave_str = f"""
PWAV {p_wvl + 1}"""
    zmx.writelines(wave_str)
    
    distance = lens_dict["OBJECT"]['distance'][cfg_id] if isinstance(lens_dict["OBJECT"]['distance'], list) else lens_dict["OBJECT"]['distance']
    obj_str = f"""
SURF 0
    TYPE STANDARD
    FIMP
    CURV 0.0
    DISZ {'INFINITY' if distance is None else distance}
    DIAM 0"""
    zmx.writelines(obj_str)
    
    multi_thick = {}
    for i, item in enumerate(list(lens_dict)[1:-1]):
        if isinstance(lens_dict[item]['thick'], list):
            multi_thick[i+1] = lens_dict[item]['thick']
        
        if lens_dict[item]['stop']:
            surf_str = f"""
SURF {i+1}
    STOP"""
            diams = lens_dict[item]['radius']
        else:
            surf_str = f"""
SURF {i+1}"""
        zmx.writelines(surf_str)
        
        roc = lens_dict[item]['roc']
        if lens_dict[item]['type'] == 'Standard':    
            surf_str = f"""
    TYPE STANDARD
    FIMP
    CURV {1 / roc if roc is not None else 0.0}
    DISZ {lens_dict[item]['thick'][cfg_id] if isinstance(lens_dict[item]['thick'], list) else lens_dict[item]['thick']}
    CONI {lens_dict[item]['conic']}
    DIAM {lens_dict[item]['radius'][cfg_id] if isinstance(lens_dict[item]['radius'], list) else lens_dict[item]['radius']} 1 0 0 1 "" """
            zmx.writelines(surf_str)
        elif lens_dict[item]['type'] == 'Asphere':
            surf_str = f"""
    TYPE EVENASPH
    FIMP
    CURV {1 / roc if roc is not None else 0.0}
    DISZ {lens_dict[item]['thick'][cfg_id] if isinstance(lens_dict[item]['thick'], list) else lens_dict[item]['thick']}
    CONI {lens_dict[item]['conic']}
    DIAM {lens_dict[item]['radius'][cfg_id] if isinstance(lens_dict[item]['radius'], list) else lens_dict[item]['radius']} 1 0 0 1 "" """
            zmx.writelines(surf_str)
            if len(lens_dict[item]['ai_list']) > 7:
                raise ValueError(f" Asphere surface {item} has more than 16th coefficients!")
            for ai in range(len(lens_dict[item]['ai_list'])):
                surf_str = f"""
    PARM {ai+2} {lens_dict[item]['ai_list'][ai]}"""
                zmx.writelines(surf_str)
        elif lens_dict[item]['type'] == 'Qcon' or lens_dict[item]['type'] == 'Qbfs':
            surf_str = f"""
    TYPE QED_TYPE
    FIMP
    CURV {1 / roc if roc is not None else 0.0}
    DISZ {lens_dict[item]['thick'][cfg_id] if isinstance(lens_dict[item]['thick'], list) else lens_dict[item]['thick']}
    CONI {lens_dict[item]['conic']}
    DIAM {lens_dict[item]['radius'][cfg_id] if isinstance(lens_dict[item]['radius'], list) else lens_dict[item]['radius']} 1 0 0 1 "" """
            zmx.writelines(surf_str)
            t = 1.0 if lens_dict[item]['type'] == 'Qcon' else 0.0
            surf_str = f"""
    XDAT 1 {t} 0 0 1.000000000000E+00 0.000000000000E+00 0
    XDAT 2 {len(lens_dict[item]['qi_list'])} 0 0 1.000000000000E+00 0.000000000000E+00 0
    XDAT 3 {lens_dict[item]['rnorm']} 0 0 1.000000000000E+00 0.000000000000E+00 0"""
            zmx.writelines(surf_str)
            for qi in range(len(lens_dict[item]['qi_list'])):
                surf_str = f"""
    XDAT {qi+4} {lens_dict[item]['qi_list'][qi]} 0 0 1.000000000000E+00 0.000000000000E+00 0"""
                zmx.writelines(surf_str)
            
        glass_name = lens_dict[item]['material']
        if glass_name != "VACUUM":
            zmx.writelines(f"""
    GLAS {glass_name}""")
                
    img_str = f"""
SURF {len(list(lens_dict))-1}
    TYPE STANDARD
    FIMP
    CURV 0.0
    DISZ 0
    DIAM {lens_dict["IMAGE"]['radius'][cfg_id] if isinstance(lens_dict["IMAGE"]['radius'], list) else lens_dict["IMAGE"]['radius']} 0 0 0 1 "" """
    zmx.writelines(img_str)

    multi_str = f"""
CONF 1 0 0 0 0 0 0 0 0 0
BLNK 
TOL TOFF   0   0 0.0000000000000000E+00 0.0000000000000000E+00   0 0 0 0 0
    """
    zmx.writelines(multi_str)
    cfg_num = len(max_angle)
    zmx.writelines(f"""MNUM {cfg_num} {cfg_num}""")
    
    n = 1
    for j in range(len(norm_views)):
        n += 1
        for i in range(cfg_num):
            multi_str = f"""
YFIE {j+1} {i+1} {norm_views[j] * max_angle[i]} 0 0 0 0 1 1.000000000000E+00 1.000000000000E+00 0 "" 0"""
            zmx.writelines(multi_str)
        
    for k in multi_thick:
        n += 1
        for i in range(cfg_num):
            multi_str = f"""
THIC {k} {i+1} {multi_thick[k][i]} 0 0 0 0 {i+1} {n} 1.000000000000E+00 0 "" 0"""
            zmx.writelines(multi_str)
