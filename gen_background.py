import GCRCatalogs
import GCRCatalogs
from GCRCatalogs.helpers.tract_catalogs import tract_filter, sample_filter
from GCRCatalogs import GCRQuery
import numpy as np
from astropy.coordinates import angular_separation
from dustmaps.sfd import SFDQuery
from astropy import units as u
import astropy.coordinates as coord




## Approximate central location of DC2 sky footprint
DC2_MED_RA = 61.93478343665261
DC2_MED_DEC = -36.04770943348744

## Extinction coefficients for LSST filters (for SFD dust map) from Schlafly & Finkbeiner 2011
R_G_LSST = 3.237
R_R_LSST = 2.273






def _radec_to_unitvec(ra_deg, dec_deg):
    """
    Convert RA and DEC to vectors on the unit sphere
    """
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack((x, y, z))




def _unitvec_to_radec(vecs):
    """
    Convert vectors on the unit sphere to RA and DEC
    """

    x, y, z = vecs[:,0], vecs[:,1], vecs[:,2]
    dec = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    ra = np.rad2deg(np.arctan2(y, x)) % 360.0
    return ra, dec





def _rotate_vectors(vectors, axis, angle):
    """
    Rotate vectors by a given angle around a specified axis using Rodrigues' rotation formula.
    """

    k = axis / np.linalg.norm(axis)
    cosA = np.cos(angle)
    sinA = np.sin(angle)
    return (vectors * cosA
            + np.cross(k, vectors) * sinA
            + np.outer(np.dot(vectors, k), k) * (1 - cosA))








def _rotate_catalog_radec(ra, dec, old_center, new_center):
    """
    Performs a rotation of the input RA and DEC coordinates from the old_center to the new_center. 
    This is necessary to translate the DC2 catalog coordinates to be centered on the cluster of interest for our mock observations.
    This function takes into account the spherical geometry of the sky and performs a proper rotation rather than a simple shift in RA and DEC.

    Parameters:

    ra, dec: 
        arrays (deg) of original catalog

    old_center, new_center: 
        (ra_deg, dec_deg) of DC2 center and desired center

    Returns: 

        ra_new, dec_new (deg) rotated on the sphere
    """
    v = _radec_to_unitvec(ra, dec)
    v1 = _radec_to_unitvec([old_center[0]], [old_center[1]])[0]
    v2 = _radec_to_unitvec([new_center[0]], [new_center[1]])[0]

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if np.isclose(dot, 1.0):
        return ra.copy(), dec.copy()  # no rotation needed
    if np.isclose(dot, -1.0):
        # 180-degree rotation: pick any orthogonal axis
        axis = np.cross(v1, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(v1, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = np.cross(v1, v2)
        angle = np.arccos(dot)

    v_rot = _rotate_vectors(v, axis, angle)
    ra_new, dec_new = _unitvec_to_radec(v_rot)
    return ra_new, dec_new






def gen_background_data(cluster_icrs, search_radius):
    """
    Generates the DC2 background galaxy catalog, rotated to be centered on the cluster of interest and filtered to only include objects within the search radius.
    Works by first extracting the galaxies from the DC2 catalog, performing a rotation of the RA and DEC coordinates to be centered on the cluster, and then 
    filtering to only include objects within the search radius of the cluster center.

    Parameters:
    cluster_icrs: astropy.coordinates.SkyCoord
        The center of the cluster in ICRS coordinates

    search_radius: float
        The radius (in degrees) around the cluster center to include in the background catalog

    Returns:
    dict of arrays
        A dictionary containing the RA, DEC, and magnitudes of the background galaxies within the search
    """





    obj_cat = GCRCatalogs.load_catalog("desc_dc2_run2.2i_dr6_object")

    ## Change this line if you want to use more/different filters
    quantities = ['ra', 'dec', 'mag_g_cModel', 'mag_r_cModel', 'mag_i_cModel',
               'mag_z_cModel']

    ## extendedness == 1 corresponds to galaxies. Other filters are to ensure we have valid magnitudes and applies a magnitude cut to avoid 
    ## very faint objects. 'clean' removes objects with various quality issues (see DC2 documentation for details)
    filters = [GCRQuery('extendedness == 1'), 
            GCRQuery((np.isfinite, 'mag_g_cModel')), 
            GCRQuery((np.isfinite, 'mag_r_cModel')),
            GCRQuery((np.isfinite, 'mag_i_cModel')),
            GCRQuery((np.isfinite, 'mag_z_cModel')),
            GCRQuery('mag_g_cModel < 27'),
            GCRQuery('mag_r_cModel < 27'),
            GCRQuery('mag_i_cModel < 27'),
            GCRQuery('mag_z_cModel < 27'), ## assign some magnitude cut to avoid very faint objects
            GCRQuery('clean')]
    

    data_bkg =  obj_cat.get_quantities(quantities=quantities, filters=filters)

    # Query SFD dust map to un-extinguish sources in the original DC2 catalog for rotation
    coords = coord.SkyCoord(ra=data_bkg['ra']*u.degree, dec=data_bkg['dec']*u.degree, frame='icrs')
    sfd = SFDQuery()
    ebv = sfd(coords)

    A_g = ebv * R_G_LSST
    A_r = ebv * R_R_LSST

    data_bkg['mag_g_cModel'] -= A_g
    data_bkg['mag_r_cModel'] -= A_r




    ra_rot, dec_rot = _rotate_catalog_radec(data_bkg['ra'], data_bkg['dec'],
                                         old_center=(DC2_MED_RA, DC2_MED_DEC),
                                            new_center=(cluster_icrs.ra.degree, cluster_icrs.dec.degree))
    

    in_circle = np.degrees(angular_separation(np.radians(ra_rot), np.radians(dec_rot), 
                                np.radians(cluster_icrs.ra.degree), np.radians(cluster_icrs.dec.degree))) < search_radius



    data_bkg['ra'] = ra_rot
    data_bkg['dec'] = dec_rot

    newcoords = coord.SkyCoord(ra=data_bkg['ra']*u.degree, dec=data_bkg['dec']*u.degree, frame='icrs')

    ## re-apply extinction correction after rotation, can change the dust map used here later if desired
    sfdnew = SFDQuery() 
    ebv_newcoords = sfdnew(newcoords)

    data_bkg['mag_g_cModel'] += ebv_newcoords * R_G_LSST
    data_bkg['mag_r_cModel'] += ebv_newcoords * R_R_LSST



    data_in_circle = {key: val[in_circle] for key, val in data_bkg.items()}

    return data_in_circle