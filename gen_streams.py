import astropy.units as u
import astropy.coordinates as coord
from isochrones import get_ichrone
from isochrones.mist.bc import MISTBolometricCorrectionGrid
import h5py
import numpy as np
import dustmaps.sfd








def _convert_stream_icrs(stream):
    """
    Takes the native stream data structure from Holm-Hansen+2026 and returns it as an astropy SkyCoord object in the ICRS frame.

    Parameters:
    stream: h5py.Dataset
        The stream data structure from Holm-Hansen+2026, containing the positions and velocities of the stream particles. If the 
        Holm-Hansen+2026 .hdf5 file is loaded in as streams = h5py.File("path/to/streams.hdf5", "r"), then stream should be streams[stream_id], 
        where stream_id is the identifier for the stream of interest.

    Returns:
    astropy.coordinates.SkyCoord
        The positions and velocities of the stream particles in the ICRS frame
    """

    x = stream["x"][:]
    y = stream["y"][:]
    z = stream["z"][:]

    vx = stream["vx"][:]
    vy = stream["vy"][:]
    vz = stream["vz"][:]

    c_galcen = coord.SkyCoord(x=x*u.kpc, y=y*u.kpc, z=z*u.kpc, v_x=vx*u.km/u.s,
                              v_y=vy*u.km/u.s, v_z=vz*u.km/u.s, frame="galactocentric")

    return c_galcen.transform_to(coord.ICRS)



def _convert_cluster_icrs(cluster):
    """
    The same as _convert_stream_icrs, but for the progenitor cluster data structure.

    Parameters:
    cluster: h5py.Dataset
        The cluster data structure from Holm-Hansen+2026, containing the positions and velocities of the progenitor cluster. 
        If the Holm-Hansen+2026 .hdf5 file is loaded in as clusters = h5py.File("path/to/clusters.hdf5", "r"), then cluster 
        should be clusters[cluster_id], where cluster_id is the identifier for the cluster of interest.

    Returns:
    astropy.coordinates.SkyCoord
        The positions and velocities of the progenitor cluster in the ICRS frame
    """

    x = cluster["x"][-1]
    y = cluster["y"][-1]
    z = cluster["z"][-1]
    vx = cluster["vx"][-1]
    vy = cluster["vy"][-1]
    vz = cluster["vz"][-1]

    c_galcen = coord.SkyCoord(x=x*u.kpc, y=y*u.kpc, z=z*u.kpc, v_x=vx*u.km/u.s,
                              v_y=vy*u.km/u.s, v_z=vz*u.km/u.s, frame="galactocentric")

    return c_galcen.transform_to(coord.ICRS)



def _get_eep_scalar(mass, age, feh, mist):
    """
    A helper function to get the EEP value for a given mass, age, and metallicity from the MIST isochrones. Otherwise calling
    mist.get_eep(**params, accurate=True) returns an error for newer versions of numpy

    """
    result = mist.get_eep_accurate(float(mass), float(age), float(feh), return_object=True)
    return float(np.ravel(result.x)[0])






def _phot_stream(stream, cluster, bands=['LSST_g', 'LSST_r', 'Gaia_G_DR2Rev']):
    """
    Takes a given stream from Holm-Hansen+2026 and returns apparent magnitudes in the specified bands. No extinction is applied.

    Parameters:
    stream: h5py.Dataset
        The stream data structure from Holm-Hansen+2026, containing the positions and velocities of the stream particles. 
        If the Holm-Hansen+2026 .hdf5 file is loaded in as streams = h5py.File("path/to/streams.hdf5", "r"), then stream 
        should be streams[stream_id], where stream_id is the identifier for the stream of interest.

    cluster: h5py.Dataset
        The cluster data structure from Holm-Hansen+2026, containing the positions and velocities of the progenitor cluster. 
        If the Holm-Hansen+2026 .hdf5 file is loaded in as clusters = h5py.File("path/to/clusters.hdf5", "r"), then cluster should be 
        clusters[cluster_id], where cluster_id is the identifier for the cluster of interest.

    bands: list of str
        The photometric bands in which to calculate the apparent magnitudes. Must be compatible with the MIST bolometric correction grid. 
        Default is ['LSST_g', 'LSST_r', 'Gaia_G_DR2Rev']


    Returns:
    numpy.ndarray
        The apparent magnitudes of the stream particles in the specified bands, with shape (N_particles, N_bands)
    """

    mist = get_ichrone('mist')
    bc_grid = MISTBolometricCorrectionGrid(bands)

    x = stream['x'][:]
    y = stream['y'][:]
    z = stream['z'][:]

    coords = coord.SkyCoord(x=x*u.kpc, y=y*u.kpc, z=z*u.kpc,
                            frame='galactocentric')

    icrs = coords.transform_to(coord.ICRS)
    distances = icrs.distance.to(u.pc).value

    masses = stream['mass'][:]
    feh = float(cluster['feh'][()])
    age = float(np.log10(cluster['age'][()]*1e9))

    stream_icrs = _convert_stream_icrs(stream)
    coords = coord.SkyCoord(ra=stream_icrs.ra, dec=stream_icrs.dec, frame='icrs')


    query_sfd = dustmaps.sfd.SFDQuery()
    ebv_sfd = query_sfd(coords)
    Av = 3.1 * ebv_sfd

    eep = np.array([_get_eep_scalar(mass, age, feh, mist) for mass in masses])
    interp = mist.interp_value([eep, age, feh], ['Teff', 'logg', 'Mbol'])

    Teff = interp.T[0]
    logg = interp.T[1]
    Mbol = interp.T[2]
    mbol = Mbol + 5*np.log10(distances) - 5
    bolometric_corrections = bc_grid.interp([Teff, logg, feh, Av], bands).T

    return mbol - bolometric_corrections





def gen_stream_data(haloid, stream_id, streamcat_path):
    """
    Converts a given stream from the Holm-Hansen+2026 catalog as an LSST-like observational mock

    Parameters:

    haloid: str
        The identifier for the host halo in the Holm-Hansen+2026 catalog. Must be one of '523889', '519311', 'm12i', 'm12w'.

    stream_id: str
        The identifier for the stream of interest in the Holm-Hansen+2026 catalog. Must be a valid stream_id for the specified haloid. 

    streamcat_path: str
        The path to the directory containing the Holm-Hansen+2026 stream and cluster catalogs.

    Returns:
    tuple of (astropy.coordinates.SkyCoord, astropy.coordinates.SkyCoord, numpy.ndarray)
        A tuple containing the cluster SkyCoord, stream SkyCoord, and numpy array of apparent magnitudes (might change this later)
        
    """



    if haloid not in ["523889", "519311", "m12i", "m12w"]:
        raise ValueError("haloid must be one of '523889', '519311', 'm12i', 'm12w'")
     

    streams = h5py.File(f"{streamcat_path}/{haloid}/{haloid}_streams.hdf5", "r")
    clusters = h5py.File(f"{streamcat_path}/{haloid}/{haloid}_clusters.hdf5", "r")



    if stream_id not in streams.keys():
        raise ValueError(f"stream_id {stream_id} not found in stream catalog for haloid {haloid}")
    


    stream = streams[stream_id]
    cluster = clusters[stream_id]

    ## Change this line if you want to use more/different filters
    stream_filters = ['LSST_g', 'LSST_r', 'Gaia_G_DR2Rev']
    
    mags = _phot_stream(stream, cluster, bands=stream_filters)
    stream_icrs = _convert_stream_icrs(stream)
    cluster_icrs = _convert_cluster_icrs(cluster)

    return cluster_icrs, stream_icrs, mags