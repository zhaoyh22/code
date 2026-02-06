import little_r as lr
import Pseudo_rh
import numpy as np
import xarray as xr
from wrf import getvar
from netCDF4 import Dataset
import sys
import os
import random
import time
import math

# --- 1. Read background (WRF) fields -----------------------------------------
wrf_path = "./fg"
wrfin  = Dataset(wrf_path)
xlat   = getvar(wrfin,'XLAT',0)
xlon   = getvar(wrfin,'XLONG',0)
height = getvar(wrfin,"height",0)      # geopotential height (m)
dbz    = getvar(wrfin,'dbz')           # model-simulated reflectivity (dBZ)
rh     = getvar(wrfin,'rh')            # model RH (%)
p      = getvar(wrfin,'p')             # pressure (Pa)
T      = getvar(wrfin,'tk')            # temperature (K)

# --- 2. Read radar reflectivity observations ---------------------------------
# Observations are provided on the same grid/levels (or pre-remapped to them).
raw = np.loadtxt('./reflectivity_YYYYMMDDHH00_new.txt')
date_str = 'YYYYMMDDHH'
date1 = '00'
ref_3d = raw.reshape((50,400,400))
# Missing/null-return flag: True means no valid radar return at this point
obs_missing = (ref_3d == -9999)

# --- 3. Classification (echo-consistency regimes) ----------------------------
# tet encodes the regimes (e.g., 5555 for echo-consistent weak/moderate echoes,
# 8888 for SCRs: background strong echoes collocated with missing/null radar returns).
tet = Pseudo_rh.Process_data(ref_3d, dbz.values)
nlev, nj, ni = ref_3d.shape
size = (nlev, nj, ni)

# --- 4. Compute AGL height and LCL (for saturation adjustment, if used) -------
z_ter  = getvar(wrfin,'HGT')        # terrain height (m)
z_geo  = height - z_ter             # height above ground level (m)

T2   = getvar(wrfin,'T2').values    # K
PSFC = getvar(wrfin,'PSFC').values  # Pa
Q2   = getvar(wrfin,'Q2').values    # kg/kg
cape_2d  = getvar(wrfin,'cape_2d')  # MCAPE/MCIN/LCL/LFC
lcl_geo  = cape_2d[2].values

# --- 5. Initialize pseudo-RH container ---------------------------------------
# grid_value is used as an output container for pseudo-RH retrieval.
grid_value = np.zeros(size, dtype=np.float64)

R          = 15
sigma_z    = 5

# --- 5.1 Bayesian pseudo-RH retrieval (ONLY for code 5555 with valid obs) -----
grid_value = Pseudo_rh.OBS_and_BG(grid_value,
                       BG_rh=rh.values,
                       BG_ref=dbz.values,
                       OBS_ref=ref_3d,
                       filtered_data=tet,
                       obs_missing=obs_missing,
                       sigma_z_obs=sigma_z,
                       size=size, r=R)

# --- 5.2 Environmental pseudo-RH for SCRs (code 8888 with missing obs) --------
R          = 25
grid_value = Pseudo_rh.NOBS_and_BG(grid_value,
                                   BG_rh=rh.values,
                                   filtered_data=tet,
                                   obs_missing=obs_missing,
                                   size=size,
                                   r=R,
                                   min_pts=3,
                                   rh_min=10.0)

pseudo_rh = grid_value

# --- 6. Quality control (QC) -------------------------------------------------
bg_ref = dbz.values
bg_rh  = rh.values

# For 5555 points (valid obs), require consistency between reflectivity innovation and RH increment:
# If obs reflectivity is smaller than background (delta_rz < 0), pseudo-RH should decrease (delta_rh < 0), etc.
delta_rz = ref_3d - bg_ref
delta_rh = pseudo_rh - bg_rh

mask_5555 = (tet == 5555) & (~obs_missing) & (pseudo_rh != 0.0)
mask_qc_5555 = mask_5555 & ((delta_rz * delta_rh) > 0.0)

# For SCR points (8888 with missing obs), do NOT use delta_rz because obs reflectivity is missing (e.g., -9999).
# A conservative QC is to keep only drying updates (pseudo_rh < bg_rh) and valid (non-zero) retrievals.
mask_8888 = (tet == 8888) & (obs_missing) & (pseudo_rh != 0.0)
mask_qc_8888 = mask_8888 & (pseudo_rh < bg_rh)

mask_qc = mask_qc_5555 | mask_qc_8888

# Output: keep QC-passed pseudo-RH; otherwise set to missing flag (-9999)
pseudo_rh_qc = np.where(mask_qc, pseudo_rh, -9999.0)

np.savetxt("pseudo_rh_QC_YYYYMMDDHH.txt", pseudo_rh_qc.reshape(-1), fmt="%.3f")

# --- 6. write Little_r file ----------------------------------------
indices = np.where(mask_qc)
grouped = {}
for k, j, i in zip(*indices):
    grouped.setdefault((j, i), set()).add(k)


stride = 1
sparse_keys = [(j,i) for (j,i) in grouped
               if (j % stride == 0 and i % stride == 0)]

if __name__ == "__main__":
    fout = f'obs.YYYYMMDDHH'
    if os.path.exists(fout):
        os.remove(fout)

    with open(fout, "a") as file:
        ID_start = 57350
        for idx, (south_north, west_east) in enumerate(sparse_keys):
            layers = sorted(grouped[(south_north, west_east)])

            lat = xlat.values[south_north, west_east]
            lon = xlon.values[south_north, west_east]
            ID  = ID_start + idx + 1
            FM  = 'FM-35 TEMP'
            elev = 100.0
            bogus = False
            seconds = str(random.randint(0, 59)).zfill(2)
            date    = date_str + date1 + seconds
            header_str = lr.header_record(lat, lon, ID, FM, elev, bogus, date)
            file.write(header_str + "\n")

            used_layers = 0
            for level in layers:
                h_agl = z_geo[level, south_north, west_east]
                pres   = p.values[level, south_north, west_east]
                h      = height.values[level, south_north, west_east]
                t      = T.values[level, south_north, west_east]
                td     = -888888.0
                wspd   = -888888.0
                wdir   = -888888.0
                u      = -888888.0
                v      = -888888.0
                ps_rh  = pseudo_rh_qc[level, south_north, west_east]
                tk     = -888888.0

                data_str = lr.data_record(
                    pres, h, t, td, wspd, wdir, u, v, ps_rh, tk
                )
                file.write(data_str + "\n")
                used_layers += 1

            file.write(lr.ending_record() + "\n")
            file.write(lr.tail_record(used_layers) + "\n")
