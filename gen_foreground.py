import numpy as np
from dl import authClient as ac, queryClient as qc
import pandas as pd
from io import StringIO




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

    return df

