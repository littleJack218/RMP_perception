"""Image-quality metrics (manuscript Eqs. 3 and 5-6).

PSNR : 10 log10(255^2 / MSE) on 8-bit grayscale (Eq. 3).
SSIM : luminance / contrast / structure decomposition (Eq. 5) with the standard
       11x11 Gaussian window (sigma = 1.5) and C1 = (0.01 L)^2, C2 = (0.03 L)^2
       with L = 255 (Eq. 6).
"""
import cv2
import numpy as np


def psnr8(f, g):
    """f, g: uint8-range arrays."""
    mse = float(np.mean((f.astype(np.float64) - g.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(f, g, data_range=255.0):
    f = f.astype(np.float64)
    g = g.astype(np.float64)
    win, sigma = 11, 1.5
    coords = np.arange(win) - (win - 1) / 2.0
    k = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    window = np.outer(k, k)

    mu_f = cv2.filter2D(f, -1, window, borderType=cv2.BORDER_REFLECT)
    mu_g = cv2.filter2D(g, -1, window, borderType=cv2.BORDER_REFLECT)
    mu_ff, mu_gg, mu_fg = mu_f * mu_f, mu_g * mu_g, mu_f * mu_g
    s_ff = cv2.filter2D(f * f, -1, window, borderType=cv2.BORDER_REFLECT) - mu_ff
    s_gg = cv2.filter2D(g * g, -1, window, borderType=cv2.BORDER_REFLECT) - mu_gg
    s_fg = cv2.filter2D(f * g, -1, window, borderType=cv2.BORDER_REFLECT) - mu_fg

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_fg + c1) * (2 * s_fg + c2)
    den = (mu_ff + mu_gg + c1) * (s_ff + s_gg + c2)
    return float(np.mean(num / den))


def masked_psnr(f, g, mask):
    """PSNR restricted to `mask` (used for the dynamic/static stratification)."""
    d = (f[mask].astype(np.float64) - g[mask].astype(np.float64)) ** 2
    return 10.0 * np.log10(255.0 ** 2 / max(float(d.mean()), 1e-10))
