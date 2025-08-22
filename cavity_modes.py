

import numpy as np
from typing import Callable, Tuple
from scipy.special import j0

from constants import PI, CHI_01


def rect_mode_u_TEmnp(dims, m: int, n: int, p: int):
    
    Lx, Ly, Lz = dims

    
    def sin_or_one(arg, order):
        if order == 0:
            return np.ones_like(arg)
        return np.sin(order * PI * arg)

    def u(pts: np.ndarray):
        
        x = (pts[:,0] + Lx/2) / Lx   
        y = (pts[:,1] + Ly/2) / Ly   
        z = (pts[:,2] + Lz/2) / Lz   

        val = np.abs(
            sin_or_one(x, m) *
            sin_or_one(y, n) *
            sin_or_one(z, p)
        )
        return val  

    vol = Lx * Ly * Lz
    bounds = (-Lx/2, Lx/2, -Ly/2, Ly/2, -Lz/2, Lz/2)
    return u, vol, bounds


def cyl_mode_u_TM010(dims):
    
    R, L = dims
    
    def u(pts: np.ndarray):
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        rho = np.sqrt(x**2 + y**2)
        
        
        inside = (rho <= R) & (np.abs(z) <= L/2)
        val = np.zeros_like(rho)
        val[inside] = np.abs(j0(CHI_01 * rho[inside] / R))
        return val
    
    vol = PI * R**2 * L
    bounds = (-R, R, -R, R, -L/2, L/2)
    return u, vol, bounds


def make_mode_u(cav):
    
    geometry = cav["geometry"] if isinstance(cav, dict) else cav.geometry
    dims = cav["dims"] if isinstance(cav, dict) else cav.dims

    if geometry == "rect":
        
        m, n, p = (1, 0, 1)
        if isinstance(cav, dict) and "rect_mode_indices" in cav:
            m, n, p = tuple(cav["rect_mode_indices"])
        elif hasattr(cav, "rect_mode_indices") and cav.rect_mode_indices is not None:
            m, n, p = cav.rect_mode_indices
        return rect_mode_u_TEmnp(dims, m, n, p)

    elif geometry == "cyl":
        return cyl_mode_u_TM010(dims)

    else:
        raise ValueError("geometry must be 'rect' or 'cyl'")


def numerical_Veff_from_u(u_func: Callable[[np.ndarray], np.ndarray],
                          volume: float, bounds: Tuple, N: int) -> float:
    
    x0, x1, y0, y1, z0, z1 = bounds
    xs = np.linspace(x0, x1, N)
    ys = np.linspace(y0, y1, N)
    zs = np.linspace(z0, z1, N)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='xy')
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    uu = u_func(pts)
    mean_u2 = float(np.mean(uu**2))
    return volume * mean_u2   
