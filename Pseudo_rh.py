from numba import njit
import numpy as np
import math

NOISE = 0.0  # Noise threshold in dBZ.
PRECIP_BG = 25.0
# Background-reflectivity threshold (dBZ) used to identify spurious convection regions (SCRs).
# If the model background reflectivity exceeds PRECIP_BG while the observation indicates missing,
# the grid point is classified as SCRs (code 8888).

OBS_STRONG = 25.0
# Observed strong-echo threshold in dBZ.
# When obs > OBS_STRONG, the point is treated as a precipitating convective core region.
# These points are assigned to a saturation-based humidity adjustment
# (e.g., enforcing near-saturation above the LCL).
MISSING = -9999.0   # Missing flag in the remapped radar reflectivity volume (no valid signal)


@njit
def Process_data(ref_3d, dbz):
    """
    Classify each 3-D grid point by comparing observed and background composite reflectivity.

    Notes
    -----
    In this implementation, "null echo" refers to *missing/invalid radar returns* (obs == MISSING),
    rather than physically observed clear air (e.g., 0 dBZ). This distinction is critical to avoid
    mixing true clear-air observations with no-return regions caused by radar coverage limits,
    beam blockage, attenuation, or other artifacts.

    Inputs
    ------
    ref_3d : (K, J, I) array
        Observed reflectivity (dBZ) remapped onto the model grid.
        Missing/no-return values are encoded as MISSING (e.g., -9999).
    dbz : (K, J, I) array
        Background (model) reflectivity (dBZ).

    Outputs
    -------
    tet : (K, J, I) array
        Integer classification mask used to select the pseudo-humidity construction strategy.

    Codes
    -----
    6666 : obs > OBS_STRONG   (and obs is valid)
        Observed strong echo region. Intended treatment: saturation-based humidity adjustment
        (e.g., enforcing near-saturation above the LCL).

    5555 : obs <= OBS_STRONG  (and obs is valid)  and  bg > NOISE
        Observed weak-to-moderate echo with a non-clear-air background echo.
        Intended treatment: Bayesian pseudo-humidity retrieval using both observation and background.

    8888 : obs == MISSING  and  bg > PRECIP_BG
        Spurious convection region (SCR) under the null-echo criterion:
        background indicates strong precipitation echo, while the observation has no valid return.
        Intended treatment: suppress spurious convection via environmental neighborhood estimation
        and/or pseudo-humidity reduction within SCRs.

    0 : all other cases
        Not used for pseudo-humidity construction in this study.
    """
    K, J, I = ref_3d.shape
    tet = np.zeros_like(ref_3d)

    for k in range(K):
        for j in range(J):
            for i in range(I):
                obs = ref_3d[k, j, i]
                bg  = dbz[k, j, i]

                # --- Null-echo / missing observation: only these can become SCRs (8888) ---
                if obs == MISSING:
                    if bg > PRECIP_BG:
                        tet[k, j, i] = 8888
                    else:
                        tet[k, j, i] = 0
                    continue

                # --- Valid observation: strong echo (6666) ---
                if obs > OBS_STRONG:
                    tet[k, j, i] = 6666

                # --- Valid observation: weak-to-moderate echo (5555) if background has echo ---
                elif obs > NOISE and obs <= OBS_STRONG and bg > NOISE:
                    tet[k, j, i] = 5555

                else:
                    tet[k, j, i] = 0

    return tet

@njit
def OBS_and_BG(grid_value,
               BG_rh, BG_ref, OBS_ref,
               filtered_data,
               obs_missing,   # True = missing (e.g., -9999)
               sigma_z_obs,   # dBZ, controls Bayesian weights
               size, r):
    """
    Bayesian retrieval of pseudo–relative humidity (pseudo-RH) for echo-consistent points only.

    This routine is applied ONLY to grid points classified as 5555:
      - filtered_data == 5555
      - observation is valid (obs_missing == False)
      - typically represents OBS_ref <= OBS_STRONG AND BG_ref > NOISE (as defined in your classifier)

    Weight definition
    -----------------
      w(jj,ii) = exp( - ( BG_ref(k,jj,ii) - OBS_ref(k,j,i) )^2 / ( 2 * sigma_z_obs^2 ) )

    Parameters
    ----------
    grid_value : float[nlev, nj, ni]
        Output pseudo-RH field (can be pre-filled with BG_rh).
    BG_rh : float[nlev, nj, ni]
        Background relative humidity.
    BG_ref : float[nlev, nj, ni]
        Background reflectivity (dBZ).
    OBS_ref : float[nlev, nj, ni]
        Observed reflectivity (dBZ) on the model grid.
    filtered_data : int[nlev, nj, ni]
        Classification mask. Only code 5555 is used in this function.
    obs_missing : bool[nlev, nj, ni]
        True indicates missing/invalid observation at the grid point (e.g., -9999).
    sigma_z_obs : float
        Standard deviation (dBZ) controlling the decay of Bayesian weights.
    size : tuple(int nlev, int nj, int ni)
        Array dimensions.
    r : int
        Horizontal search radius (in grid points) for the moving window.

    Returns
    -------
    grid_value : float[nlev, nj, ni]
        Updated pseudo-RH field. If no effective weights are found, falls back to BG_rh at the center.
    """
    nlev, nj, ni = size
    scale = 2.0 * sigma_z_obs * sigma_z_obs

    for k in range(nlev):
        for j in range(nj):
            for i in range(ni):

                # Only process echo-consistent points with valid observations
                if filtered_data[k, j, i] != 5555:
                    continue
                if obs_missing[k, j, i]:
                    continue

                obs_center = OBS_ref[k, j, i]
                rh_center  = BG_rh[k, j, i]

                weighted_sum = 0.0
                weight_total = 0.0

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= nj:
                        continue
                    for di in range(-r, r + 1):
                        ii = i + di
                        if ii < 0 or ii >= ni:
                            continue

                        dz = BG_ref[k, jj, ii] - obs_center
                        w  = math.exp(-dz * dz / scale)

                        weighted_sum += w * BG_rh[k, jj, ii]
                        weight_total += w

                if weight_total > 1e-6:
                    grid_value[k, j, i] = weighted_sum / weight_total
                else:
                    grid_value[k, j, i] = rh_center

    return grid_value

@njit
def OBS_and_NBG(grid_value,
                filtered_data,
                OBS_ref,
                height_agl,   # float[nlev,nj,ni]  height above ground level (m)
                lcl_geo,      # float[nj,ni]       LCL height for each column (m)
                size):
    """
    Empirical saturation adjustment for strong observed echoes above the LCL.

    This routine follows the common "cloud analysis"-type treatment used in the WRF 3DVar indirect
    radar assimilation framework (e.g., Wang et al., 2013), where grid points associated with strong
    observed reflectivity are forced toward near-saturation *only above the lifting condensation level (LCL)*.
    The purpose is to provide a physically plausible moist thermodynamic environment for precipitating
    clouds without unrealistically moistening the sub-cloud layer.

    Target points
    -------------
    Only grid points with:
      - filtered_data == 6666  (strong observed echo category defined in this study)
      - height_agl(k,j,i) > lcl_geo(j,i)  (i.e., above the column LCL)

    Adjustment rule
    ---------------
    For those target points, the relative humidity (RH, in %) is set to an empirically specified value
    depending on the observed reflectivity magnitude at the same grid point:

      Z >= 50 dBZ  -> RH = 100%
      40 <= Z < 50 -> RH = 95%
      25 <= Z < 40 -> RH = 85%

    Grid points below the LCL are left unchanged to avoid introducing excessive moisture in the boundary layer.

    Parameters
    ----------
    grid_value : float[nlev, nj, ni]
        In/out relative humidity field (%). Values are updated in place.
    filtered_data : int[nlev, nj, ni]
        Classification mask. Code 6666 indicates strong observed echoes to which this saturation
        adjustment is applied (above LCL only).
    OBS_ref : float[nlev, nj, ni]
        Observed reflectivity (dBZ) on the model grid.
    height_agl : float[nlev, nj, ni]
        Height above ground level (m) at each model grid point and level.
    lcl_geo : float[nj, ni]
        LCL height (m) for each vertical column.
    size : tuple(int nlev, int nj, int ni)
        Array dimensions.

    Returns
    -------
    grid_value : float[nlev, nj, ni]
        Updated RH field with near-saturation imposed in strong-echo regions above LCL.
    """
    nlev, nj, ni = size

    for k in range(nlev):
        for j in range(nj):
            for i in range(ni):

                # Apply only to strong-echo points above the column LCL
                if filtered_data[k, j, i] == 6666 and height_agl[k, j, i] > lcl_geo[j, i]:
                    Z = OBS_ref[k, j, i]

                    # Empirical RH assignment as a function of reflectivity intensity (Wang et al., 2013-like)
                    if Z >= 50.0:
                        grid_value[k, j, i] = 100.0
                    elif Z >= 40.0:
                        grid_value[k, j, i] = 95.0
                    elif Z >= 25.0:
                        grid_value[k, j, i] = 85.0

    return grid_value

@njit
def NOBS_and_BG(grid_value,
                BG_rh,
                filtered_data,   # classification mask (SCRs coded as 8888)
                obs_missing,     # True = missing / null return (e.g., -9999)
                size,
                r,               # search radius (grid points)
                min_pts=3,
                rh_min=10.0):
    """
    Environmental pseudo-RH retrieval within spurious convection regions (SCRs).

    This routine is applied to grid points flagged as SCRs (filtered_data == 8888), where the model
    background exhibits strong reflectivity but radar returns are unavailable (null echo / missing).
    For each SCR point, the pseudo relative humidity (pseudo-RH) is estimated from the surrounding
    "environmental" background RH at the same model level.

    Method
    ------
    For an SCR grid point (k,j,i), search within a horizontal radius r at the same vertical level k,
    and collect neighboring points that represent non-convective environment (e.g., code 7777).
    The pseudo-RH is set to the mean of those neighboring background RH values, with two safeguards:

      1) Drying-only constraint: use only neighbors drier than the SCR center (RH_nei < RH_center),
         and apply the update only if the mean is smaller than RH_center (i.e., do not moisten).
      2) Lower bound: pseudo-RH is bounded below by rh_min to avoid unrealistically dry values.

    If fewer than min_pts valid environmental neighbors are found, the SCR point is left unchanged.

    Parameters
    ----------
    grid_value : float[nlev, nj, ni]
        In/out pseudo-RH field (%). Updated in place for SCR points.
    BG_rh : float[nlev, nj, ni]
        Background relative humidity (%).
    filtered_data : int[nlev, nj, ni]
        Classification mask. SCRs are coded as 8888; environmental non-convective points are coded as 7777.
    obs_missing : bool[nlev, nj, ni]
        Missing/invalid observation flag (True for null echo). Used as an additional safety gate.
    size : tuple(int nlev, int nj, int ni)
        Array dimensions.
    r : int
        Horizontal search radius (grid points).
    min_pts : int
        Minimum number of valid environmental neighbors required to form a pseudo-RH.
    rh_min : float
        Minimum allowed pseudo-RH (%).

    Returns
    -------
    grid_value : float[nlev, nj, ni]
        Updated pseudo-RH field in SCRs.
    """
    nlev, nj, ni = size

    for k in range(nlev):
        for j in range(nj):
            for i in range(ni):

                # --- Apply only inside SCRs (8888). obs_missing is an optional safety gate. ---
                if filtered_data[k, j, i] != 8888:
                    continue
                if not obs_missing[k, j, i]:
                    continue

                rh_center = BG_rh[k, j, i]

                sum_rh = 0.0
                count  = 0

                for dj in range(-r, r + 1):
                    jj = j + dj
                    if jj < 0 or jj >= nj:
                        continue
                    for di in range(-r, r + 1):
                        ii = i + di
                        if ii < 0 or ii >= ni:
                            continue
                        if dj == 0 and di == 0:
                            continue

                        # Use only "non-convective" points
                        if filtered_data[k, jj, ii] != 7777:
                            continue

                        rh_nei = BG_rh[k, jj, ii]

                        # Drying-only: exclude neighbors that are not drier than the SCR center
                        if rh_nei >= rh_center:
                            continue

                        sum_rh += rh_nei
                        count  += 1

                if count >= min_pts:
                    mean_rh = sum_rh / count

                    if mean_rh < rh_center:
                        new_rh = mean_rh
                        if new_rh < rh_min:
                            new_rh = rh_min
                        grid_value[k, j, i] = new_rh

    return grid_value



