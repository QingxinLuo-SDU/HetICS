"""step6x_methods.py — exposure-side and outcome-side knockoff candidates within a learned cluster."""
import time

import step6_methods as m6


def knockoff_candidates(F_k, Y_k, X_k, env_k, q):
    """Candidate sets S_x and S_y before invariance selection; one knockoff construction per cluster.

    Fixed-X knockoffs need n_k >= 2p; step6_methods.fixed_x_knockoff stops otherwise.
    """
    t0 = time.perf_counter()
    Fp, Yp = m6.prep(F_k, Y_k, env_k)
    _, Xp = m6.prep(F_k, X_k, env_k)        # Fp is identical; only the response is needed
    Ft = m6.fixed_x_knockoff(Fp)
    t_con = time.perf_counter() - t0

    t0 = time.perf_counter()
    S_x = m6.lcd_select(Fp, Ft, Xp, q=q)
    t_lx = time.perf_counter() - t0
    t0 = time.perf_counter()
    S_y = m6.lcd_select(Fp, Ft, Yp, q=q)
    t_ly = time.perf_counter() - t0
    return S_x, S_y, {"t_knockoff_construct": t_con, "t_lcd_x": t_lx, "t_lcd_y": t_ly}


# Knockoff timing components counted in each arm's selection time.
KNOCK_COST = {
    "xside_icp":  ("t_knockoff_construct", "t_lcd_x"),
    "yside_icp":  ("t_knockoff_construct", "t_lcd_y"),
    "dual_icp":   ("t_knockoff_construct", "t_lcd_x", "t_lcd_y"),
    "oracle_min": (),
}
