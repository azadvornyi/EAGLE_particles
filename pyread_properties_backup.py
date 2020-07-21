#!/bin/env python
#
# Read coordinates and IDs of gas particles in a specified
# region of an Eagle snapshot.
#


# from __future__ import print_function
import pyread_eagle as read_eagle
import numpy as np
import time
import sys
import h5py
from astropy import units as u
from astropy.constants import G
import matplotlib.pyplot as plt
import pandas as pd

# Location where tar file with snapshots was unpacked

with open('data_location.txt') as f:
    lines = [line.rstrip() for line in f]

sim12LR = lines[1]
sim25HD = lines[2]
sim50HD = lines[3]
sim100LR = lines[4]
sim100HD = lines[5]
sim100HD_sub = lines[6]
# ==============
sim = sim100HD_sub
# ==============

basedir = lines[0]

# The snapshot to read is identified by specifying the name of one of the snapshot files
# fname_sub reads subgroup number from group tables, since snapshot tables not always have subgroup numbers of particles


#
# Particle type to read. Particle types are:
#   0 = Gas
#   1 = Dark matter
#   4 = Stars
#   5 = Black holes
#

start = time.time()

fname = basedir + sim
fname_sub = basedir + sim100HD_sub

# Open the snapshot


path_to_snapshot_100_HD = '/L0100N1504/PE/Z0p10_W1p00_E_3p0_0p3_ALPHA1p0e6_rhogas1_reposlim3p0soft_100mgas_cosma/data/'

particledata_folders = np.array([
    'particledata_000_z020p000/',
    'particledata_001_z015p132/',
    'particledata_002_z009p993/',
    'particledata_003_z008p988/',
    'particledata_004_z008p075/',
    'particledata_005_z007p050/',
    'particledata_006_z005p971/',
    'particledata_007_z005p487/',
    'particledata_008_z005p037/',
    'particledata_009_z004p485/',
    'particledata_010_z003p984/',
    'particledata_011_z003p528/',
    'particledata_012_z003p017/',
    'particledata_013_z002p478/',
    'particledata_014_z002p237/',
    'particledata_015_z002p012/',
    'particledata_016_z001p737/',
    'particledata_017_z001p487/',
    'particledata_018_z001p259/',
    'particledata_019_z001p004/',
    'particledata_020_z000p865/',
    'particledata_021_z000p736/',
    'particledata_022_z000p615/',
    'particledata_023_z000p503/',
    'particledata_024_z000p366/',
    'particledata_025_z000p271/',
    'particledata_026_z000p183/',
    'particledata_027_z000p101/',
    'particledata_028_z000p000/', ])

particledata_files = np.array([
    'eagle_subfind_particles_000_z020p000.0.hdf5',
    'eagle_subfind_particles_001_z015p132.0.hdf5',
    'eagle_subfind_particles_002_z009p993.0.hdf5',
    'eagle_subfind_particles_003_z008p988.0.hdf5',
    'eagle_subfind_particles_004_z008p075.0.hdf5',
    'eagle_subfind_particles_005_z007p050.0.hdf5',
    'eagle_subfind_particles_006_z005p971.0.hdf5',
    'eagle_subfind_particles_007_z005p487.0.hdf5',
    'eagle_subfind_particles_008_z005p037.0.hdf5',
    'eagle_subfind_particles_009_z004p485.0.hdf5',
    'eagle_subfind_particles_010_z003p984.0.hdf5',
    'eagle_subfind_particles_011_z003p528.0.hdf5',
    'eagle_subfind_particles_012_z003p017.0.hdf5',
    'eagle_subfind_particles_013_z002p478.0.hdf5',
    'eagle_subfind_particles_014_z002p237.0.hdf5',
    'eagle_subfind_particles_015_z002p012.0.hdf5',
    'eagle_subfind_particles_016_z001p737.0.hdf5',
    'eagle_subfind_particles_017_z001p487.0.hdf5',
    'eagle_subfind_particles_018_z001p259.0.hdf5',
    'eagle_subfind_particles_019_z001p004.0.hdf5',
    'eagle_subfind_particles_020_z000p865.0.hdf5',
    'eagle_subfind_particles_021_z000p736.0.hdf5',
    'eagle_subfind_particles_022_z000p615.0.hdf5',
    'eagle_subfind_particles_023_z000p503.0.hdf5',
    'eagle_subfind_particles_024_z000p366.0.hdf5',
    'eagle_subfind_particles_025_z000p271.0.hdf5',
    'eagle_subfind_particles_026_z000p183.0.hdf5',
    'eagle_subfind_particles_027_z000p101.0.hdf5',
    'eagle_subfind_particles_028_z000p000.0.hdf5', ])

snapshot_folders = np.array([
    'snapshot_000_z020p000/',
    'snapshot_001_z015p132/',
    'snapshot_002_z009p993/',
    'snapshot_003_z008p988/',
    'snapshot_004_z008p075/',
    'snapshot_005_z007p050/',
    'snapshot_006_z005p971/',
    'snapshot_007_z005p487/',
    'snapshot_008_z005p037/',
    'snapshot_009_z004p485/',
    'snapshot_010_z003p984/',
    'snapshot_011_z003p528/',
    'snapshot_012_z003p017/',
    'snapshot_013_z002p478/',
    'snapshot_014_z002p237/',
    'snapshot_015_z002p012/',
    'snapshot_016_z001p737/',
    'snapshot_017_z001p487/',
    'snapshot_018_z001p259/',
    'snapshot_019_z001p004/',
    'snapshot_020_z000p865/',
    'snapshot_021_z000p736/',
    'snapshot_022_z000p615/',
    'snapshot_023_z000p503/',
    'snapshot_024_z000p366/',
    'snapshot_025_z000p271/',
    'snapshot_026_z000p183/',
    'snapshot_027_z000p101/',
    'snapshot_028_z000p000/', ])

snapshot_files = np.array([
    'snap_000_z020p000.0.hdf5',
    'snap_001_z015p132.0.hdf5',
    'snap_002_z009p993.0.hdf5',
    'snap_003_z008p988.0.hdf5',
    'snap_004_z008p075.0.hdf5',
    'snap_005_z007p050.0.hdf5',
    'snap_006_z005p971.0.hdf5',
    'snap_007_z005p487.0.hdf5',
    'snap_008_z005p037.0.hdf5',
    'snap_009_z004p485.0.hdf5',
    'snap_010_z003p984.0.hdf5',
    'snap_011_z003p528.0.hdf5',
    'snap_012_z003p017.0.hdf5',
    'snap_013_z002p478.0.hdf5',
    'snap_014_z002p237.0.hdf5',
    'snap_015_z002p012.0.hdf5',
    'snap_016_z001p737.0.hdf5',
    'snap_017_z001p487.0.hdf5',
    'snap_018_z001p259.0.hdf5',
    'snap_019_z001p004.0.hdf5',
    'snap_020_z000p865.0.hdf5',
    'snap_021_z000p736.0.hdf5',
    'snap_022_z000p615.0.hdf5',
    'snap_023_z000p503.0.hdf5',
    'snap_024_z000p366.0.hdf5',
    'snap_025_z000p271.0.hdf5',
    'snap_026_z000p183.0.hdf5',
    'snap_027_z000p101.0.hdf5',
    'snap_028_z000p000.0.hdf5', ])

fname_infall = basedir + path_to_snapshot_100_HD + particledata_folders[15] + particledata_files[15]
print(fname_infall)
fname_quench = basedir + path_to_snapshot_100_HD + particledata_folders[27] + particledata_files[27]

fname_final = basedir + path_to_snapshot_100_HD + particledata_folders[28] + particledata_files[28]
# snap = read_eagle.EagleSnapshot(fname)
snap_infall = read_eagle.EagleSnapshot(fname_infall)
snap_quench = read_eagle.EagleSnapshot(fname_quench)


# snap_final = read_eagle.EagleSnapshot(fname_final)
# print ("# Box size = %16.8e Mpc/h" % snap.boxsize)
# print ("#")
# print ("# Total number of gas  particles in snapshot = %d" % snap.numpart_total[0])
# gid	    fof	    sub	    x	            y	        z
# 37445	    4	    0	    2.816906	    9.714233	1.6791126

def read_header(fname):
    """

    :return: a, h and boxsize from a selected snapshot
    """
    """Read the header"""
    f = h5py.File(fname, 'r')
    a = f['Header'].attrs.get('Time')  # Scale factor
    h = f['Header'].attrs.get('HubbleParam')  # h
    boxsize = f['Header'].attrs.get('BoxSize')
    f.close()

    return a, h, boxsize


def read_dataset_dm_mass(fname):
    """Special treatment for dm mass"""
    f = h5py.File(fname, 'r')
    a = f['Header'].attrs.get('Time')  # Scale factor
    h = f['Header'].attrs.get('HubbleParam')  # h
    dm_mass = f['Header'].attrs.get('MassTable')[1]
    # n_particles = f['Header'].attrs.get('NumPart_Total')[1]

    # m = np.ones(n_particles, dtype='f8') * dm_mass

    cgs = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
    aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
    hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')

    f.close()

    # Convert t o p h y s i c a l .
    m = np.multiply(dm_mass, cgs * a ** aexp * h ** hexp, dtype='f8')

    return m


def convert_to_galactic(d):
    '''

    :param d: unitless data out of tables
    :return: M = [M_sun], v = [km/s], rho = [g/cm^3], T = [K]
    '''
    for key in d.keys():
        if key == 'Mass':
            d['Mass'] = d['Mass'] * u.g
            d['Mass'] = d['Mass'].to(u.solMass)
        if key == 'Velocity':
            d['Velocity'] = d['Velocity'] * (u.cm / u.s).to(u.km / u.s)
        if key == 'Density':
            d['Density'] = d['Density'] * u.g / (u.cm) ** 3
        if key == 'Temperature':
            d['Temperature'] = d['Temperature'] * u.K
        if key == 'Coordinates':
            d['Coordinates'] = d['Coordinates'] * u.cm.to(u.Mpc)
    return d


def get_properties(itype, property, snapshot, gn, sgn, cent, region_size, fname_read):
    """

    :param itype: type of particle 0, 1, 4 or 5
    :param property: list of names of the wanted attributes, should always start with (exclude 'Mass' for dm particles)
    ['GroupNumber','SubGroupnumber','property1',..] (GN and SGN are used to select particles of a galaxy)
    :param snapshot: snap.read_eagle
    :param gn: GroupNumber
    :param sgn: SubGroupNumber
    :param centre: centre of the galaxy
    :return: dict with all parameters for given gn and sgn + particle type
    """
    part = 0.5
    a, h, boxsize = read_header(fname_read)
    region = np.array([
        (cent[0] - part * region_size), (cent[0] + part * region_size),
        (cent[1] - part * region_size), (cent[1] + part * region_size),
        (cent[2] - part * region_size), (cent[2] + part * region_size)
    ])

    #region = np.array([0, 100, 0, 100, 0, 100])
    snapshot.select_region(*region)

    data = {}

    f = h5py.File(fname_read, 'r')
    for att in property:
        tmp = snapshot.read_dataset(itype, att)
        cgs = f['PartType%i/%s' % (itype, att)].attrs.get('CGSConversionFactor')
        aexp = f['PartType%i/%s' % (itype, att)].attrs.get('aexp-scale-exponent')
        hexp = f['PartType%i/%s' % (itype, att)].attrs.get('h-scale-exponent')
        data[att] = np.multiply(tmp, cgs * a ** aexp * h ** hexp, dtype='f8')
    f.close()

    mask = np.logical_and(data['GroupNumber'] == gn, data['SubGroupNumber'] == sgn)

    for att in data.keys():
        data[att] = data[att][mask]

    len_dict = len(data['GroupNumber'])

    if itype == 1:
        m_dm = read_dataset_dm_mass(fname_read)
        data['Mass'] = m_dm * np.ones(len_dict, dtype='f8')

    data['ParticleType'] = np.ones(len_dict, dtype='f8') * itype
    data = convert_to_galactic(data)

    return data


def rotation_curve(dat, c):
    r = np.linalg.norm(dat['Coordinates'] - c, axis=1)
    mask = np.argsort(r)
    r = r[mask]

    cmass = np.cumsum(dat['Mass'][mask])

    G_ = G.to(u.km ** 2 * u.Mpc * u.Msun ** -1 * u.s ** -2).value
    v = np.sqrt((G_ * cmass) / r)

    return v, r


prop_gas = ['GroupNumber', 'SubGroupNumber', 'Temperature', 'Density', 'Velocity', 'StarFormationRate', 'Mass',
            'Coordinates', 'ParticleIDs']
prop_star = ['GroupNumber', 'SubGroupNumber', 'Velocity', 'Mass', "Coordinates", 'ParticleIDs']
prop_dm = ['GroupNumber', 'SubGroupNumber', 'Velocity', "Coordinates"]  # + Mass

# 22.137348	3.5082026	15.748509

a_global_i, h_global_i, boxsize_global_i = read_header(fname_infall)
a_global_q, h_global_q, boxsize_global_q = read_header(fname_quench)
# centre = np.array([2.816906, 9.714233, 1.6791126])  # 0012 working
centre = np.array([72.447464, 28.807302, 26.802303])

centre_at_q_time = np.array([71.447205, 28.908436, 26.367676])
centre_at_i_time = np.array([74.311615, 22.76557, 26.967281])

# centre = np.array([12.08809, 4.474372, 1.4133347])  # 0012
# centre = np.array([1.9670323, 8.312414, 11.166461])  # 0012

# centre = np.array([17.040712, 24.555983, 18.038118])  # 0025
# centre = np.array([17.040712, 24.555983, 18.038118])  # 0025
# centre = np.array([38.055847, 45.362003, 34.04112])  # 0050

centre_at_f_time = np.array([71.72011, 28.95551, 26.62785])

# centre_h = centre * h_global

reading_region = 50

id_g = 0
id_dm = 1
id_s = 4
id_bh = 5

gn_inf = 626
sgn_inf = 0

gn_q = 11
sgn_q = 3

# data_gas_infall = get_properties(id_g, prop_gas, snap_infall, gn_inf, sgn_inf, centre_at_i_time * h_global_i, reading_region,
#                                  fname_infall)
# data_star_infall  = get_properties(id_s, prop_star, snap_infall, gn_inf, sgn_inf, centre_at_q_time*h_global_q, 0.5, fname_quench)

#data_gas_quench = get_properties(id_g, prop_gas, snap_quench, gn_q, sgn_q, centre_at_q_time * h_global_q, 0.5,
#                                 fname_quench)
# data_star_quench  = get_properties(id_s, prop_star, snap_quench, gn_q, sgn_q, centre_at_q_time*h_global_q, 0.5, fname_quench)

# data_star = get_properties(id_s, prop_star, snap, 4, 0, centre_h, 0.25)
# data_dm = get_properties(id_dm, prop_dm, snap, gn, sgn, centre_h, 0.5)
#
#
#
# total = {'Mass': np.concatenate((data_gas['Mass'], data_star['Mass'], data_dm['Mass'])),
#          'Coordinates': np.concatenate((data_gas['Coordinates'], data_star['Coordinates'], data_dm['Coordinates']))}

# # TODO: Particle IDs?
#
#
# print('a = ',a_global)
# print('data_gas ')
# print(data_gas_infall)

# M_gas_infall = np.sum(data_gas_infall['Mass'])
# M_gas_quench = np.sum(data_gas_quench['Mass'])

# M_star_infall = np.sum(data_star_infall['Mass'])


# print(len(data_gas_infall['ParticleIDs']))


# print(len(data_gas_quench['ParticleIDs']))


# print(len(data_star_quench['ParticleIDs']))


# print(data_gas_infall['ParticleIDs'])
# print(data_gas_quench['ParticleIDs'])
# print(data_star_quench['ParticleIDs'])

def converted_to_stars(d_g, d_s):
    # d_g = d_g.pop("Coordinates")
    # d_s = d_s.pop("Coordinates")

    d_g = {i: d_g[i] for i in d_g if i not in ["Coordinates", 'Velocity']}
    d_s = {i: d_s[i] for i in d_s if i not in ["Coordinates", 'Velocity']}

    d_g = pd.DataFrame.from_dict(d_g)
    d_s = pd.DataFrame.from_dict(d_s)

    dataframe = d_g.merge(d_s, left_on='ParticleIDs', right_on='ParticleIDs')
    print(type(dataframe))
    # =======uncommemt this for the total mass=========
    # table = pd.pivot_table(dataframe, values='Mass_y',
    #                        columns=['SubGroupNumber_y'], aggfunc=np.sum)

    # return table.values[0][0]
    # ================================================
    return dataframe

# M_star_tot = converted_to_stars(data_gas_infall, data_star_quench)
# print(M_star_tot)
# print(type(M_star_tot))
# M_star_tot = M_star_tot
# print(M_star_tot)
# print(np.log10(M_star_tot), 'log10 converted to stars')
# print(100* M_gas_quench/M_gas_infall.value, 'gas percentage at quenching time')
#
# print(100* M_star_tot/M_gas_infall.value,'M_star_from_gas/M_gas_infall star percentage the converted from gas')
#
#
# print(100* M_star_tot/np.sum(data_star_quench["Mass"].value),'M_star_from_gas/M_star_quench star percentage the converted from gas')
#
# print(np.log10(M_gas_infall.value ), 'log10 M_gas at infall')
# print(np.log10(np.sum(data_star_quench["Mass"].value) ), 'log10 M_star at quench')
# print(np.log10(np.sum(data_star_infall["Mass"].value) ), 'log10 M_star at infall')
print(time.time() - start, 'seconds to run the script')
# print('data_dm')
# print(data_dm)
#
# non_z_sfr_mask = np.where(data_gas['StarFormationRate'] > 0)
# z_sfr_mask = np.where(data_gas['StarFormationRate'] == 0)
#
# plt.scatter(np.log10(data_gas['Density'][non_z_sfr_mask].value), np.log10(data_gas['Temperature'][non_z_sfr_mask].value),
#             c='crimson', s=0.5)
# plt.scatter(np.log10(data_gas['Density'][z_sfr_mask].value), np.log10(data_gas['Temperature'][z_sfr_mask].value),
#             c='blue', s=0.5, alpha=0.5)
# plt.ylabel('log10(T/[K])')
# plt.xlabel('log10(Density/[g/cm^3])')
# # plt.savefig('rho-T_plot.png', format='png')
# plt.clf()
#
# for x, lab in zip([data_gas, data_dm, data_star, total], ['Gas', 'DM', 'Stars', 'Total']):
#     v, r = rotation_curve(x, centre)
#     plt.plot(r * 1000, v, label=lab)
# plt.xlim(0,80)
# plt.ylabel('v_rot [km/s]')
# plt.xlabel('r [kpc]')
# plt.legend()
# # plt.savefig('rotation_curve.png', format='png')
