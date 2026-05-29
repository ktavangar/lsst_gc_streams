import numpy as np
from dl import authClient as ac, queryClient as qc
import pandas as pd
from io import StringIO
import dustmaps.planck
import dustmaps.sfd
import astropy.units as u
import astropy.coordinates as coord


## LSST extinction coefficicients taken from https://dp1.lsst.io/tutorials/notebook/309/notebook-309-1.html
R_G_LSST = 3.660566443989
R_R_LSST = 2.70136780871597


## Extinction coefficients for LSST filters (for SFD dust map) from Schlafly & Finkbeiner 2011
R_G_LSST_SFD = 3.237
R_R_LSST_SFD = 2.273


def generate_query(ra0, dec0, radius):
    '''
    Generate a SQL query to select data from the lsst_sim.simdr2 table
    within a specified radius of a given RA and Dec.

    We save positions, velocities, magnitudes, V-band extinction,
    and a couple other properties
    '''

    query = f"""
    SELECT l.ra, l.dec, l.pmracosd, l.pmdec, l.vrad, l.mu0,
           l.g_bpmag, l.g_rpmag, l.gaia_gmag,
           l.gmag, l.rmag, l.mass, l.av, l.mbolmag, l.logAge, l.logg
    FROM lsst_sim.simdr2 AS l
    WHERE q3c_radial_query(l.ra, l.dec, {ra0}, {dec0}, {radius})
    """
    return query




def gen_foreground_data(ra0, dec0, radius):
    '''
    Queries the lsst_sim.simdr2 table for stars within a specified radius of a given RA and Dec, and returns the results as a pandas DataFrame.
    '''

    query = generate_query(ra0, dec0, radius)

    df = qc.query(sql=query, fmt='table')

    coords = coord.SkyCoord(ra=df['ra']*u.deg, dec=df['dec']*u.deg, frame='icrs')

    query_xgal = dustmaps.planck.PlanckQuery(component='extragalactic')
    query_tau353 = dustmaps.planck.PlanckQuery(component='tau353')

    ebv_xgal = query_xgal(coords)
    ebv_tau353 = query_tau353(coords)

    large_ext_idx = np.where(ebv_xgal > 0.3)[0]

    ebv = ebv_xgal.copy()
    ebv[large_ext_idx] = ebv_tau353[large_ext_idx]

    A_g = R_G_LSST * ebv
    A_r = R_R_LSST * ebv

    df['gmag'] = df['gmag'] - A_g
    df['rmag'] = df['rmag'] - A_r


    ## Apply the SFD dustmap
    query_sfd = dustmaps.sfd.SFDQuery()
    ebv_sfd = query_sfd(coords)

    A_g_sfd = R_G_LSST_SFD * ebv_sfd
    A_r_sfd = R_R_LSST_SFD * ebv_sfd

    df['gmag'] = df['gmag'] + A_g_sfd
    df['rmag'] = df['rmag'] + A_r_sfd



    return df

