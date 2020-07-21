#import pyread_properties as pp
import numpy as np
import queries as q
import eagleSqlTools as sql
from sys import argv
# import matplotlib as mpl
#
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import time
from scipy import interpolate
from sys import argv
import os
import pickle
import sys

con = sql.connect('twh176', password='tt408GS2')



# GalaxyID      GroupNumber    SubGN  x             y            z            r_vir
# 20943532      18             0      33.763016     66.14211     9.383969     857.9266
# 345925,       1,             0,     16.556612 ,   24.493837 ,  17.707695 ,  518.9658


import pickle
sim100 = 'RefL0100N1504'
sim25 = 'RefL0025N0376'
sim12 = 'RefL0012N0188'
# ============
sim = sim100  # set simulation here
# ============

# with open("host_dat", 'rb') as f11:
#     host_dat = pickle.load(f11)
if sim == sim100:
    box_size = 100
    host_mass = 1E13
elif sim == sim25:
    box_size = 25
    host_mass = 1E12
else:
    box_size = 12
    host_mass = 1E11


# # print(host_dat)


host_query = "SELECT \
             SH.GalaxyID as gid, \
             SH.GroupNumber as fof,\
             SH.SubGroupNumber as sub, \
             SH.CentreOfPotential_x as copx, \
             SH.CentreOfPotential_y as copy, \
             SH.CentreOfPotential_z as copz, \
             FOF.Group_R_Crit200 as r_vir \
             FROM\
             {0}_SubHalo as SH, \
             {0}_FOF as FOF\
             WHERE \
             FOF.GroupID = SH.GroupID \
             and SH.GalaxyID = 19634929 \
             and SH.SnapNum = 28 \
             and SH.MassType_DM > {1}".format(sim, host_mass)



# with open("host_dat", 'wb+') as f0:
#     pickle.dump(host_params[i], f0, pickle.HIGHEST_PROTOCOL)
i = 0   # index for host
j = 42   # index for sat
"""
host_params -> basic information about all hosts that satisfy SH.MassType_DM > host_mass (specified in lines 36 - 44)
i = 0, j=0 -> first host and its first satellite of the host_params list of hosts
"""
host_params = sql.execute_query(con, host_query)
my_host = host_params #[i]

print((my_host), my_host)

gid = int(my_host['gid'])
fof = int(my_host['fof'])
sub = int(my_host['sub'])
dx = float(my_host['copx'])
dy = float(my_host['copy'])
dz = float(my_host['copz'])
r_vir = my_host['r_vir']

# gid = 21730535
# fof = 4
# sub =  0
# dx =  52.799755
# dy =  4.9197164
# dz = 20.06351
# r_vir = 1422.0051



normal_sat_query = 'SELECT \
             H.GalaxyID as Hgid, \
             S.GalaxyID as Sgid, \
             S.GroupNumber as fof, \
             S.SubGroupNumber as sub \
             FROM \
             {0}_SubHalo as H, \
             {0}_SubHalo as S, \
             {0}_FOF as FOF \
             WHERE \
             H.GalaxyID = {1:.0f} \
             and H.GroupID = FOF.GroupID \
             and 0.0033*FOF.Group_R_Crit200 > ABS( H.CentreOfPotential_x  - (S.CentreOfPotential_x - FLOOR((S.CentreOfPotential_x+ {2:.0f})/ {5:.0f}))) \
             and 0.0033*FOF.Group_R_Crit200 > ABS( H.CentreOfPotential_y  - (S.CentreOfPotential_y - FLOOR((S.CentreOfPotential_y+ {3:.0f})/ {5:.0f}))) \
             and 0.0033*FOF.Group_R_Crit200 > ABS( H.CentreOfPotential_z  - (S.CentreOfPotential_z - FLOOR((S.CentreOfPotential_z+ {4:.0f})/ {5:.0f}))) \
             and S.Snapnum = 28 \
             and S.MassType_Star between 1E9 and 1E12'.format(sim, gid, dx, dy, dz, box_size)
#test_sat_index_1 = np.where(sats_info['Sgid'] == )
sats_info = sql.execute_query(con, normal_sat_query)

print(sats_info[j], "subinfo1")
sat_gid = int(sats_info['Sgid'][j])

sat_fof = int(sats_info['fof'][j])
sat_sub = int(sats_info['sub'][j])

# sat_fof = 1
# sat_sub = 42

# tree_host_query = 'SELECT \
#              DES.GalaxyID as gid, \
#              DES.TopLeafID as tlid, \
#              PROG.Redshift as z, \
#              PROG.MassType_DM as mdm, \
#              PROG.MassType_Star as ms,\
#              AP.SFR / (AP.Mass_Star+0.0001) as ssfr, \
#              PROG.CentreOfPotential_x as copx, \
#              PROG.CentreOfPotential_y as copy, \
#              PROG.CentreOfPotential_z as copz \
#          FROM \
#              {0}_Subhalo as PROG, \
#              {0}_Subhalo as DES, \
#              {0}_Aperture as AP \
#          WHERE \
#              DES.SnapNum = 28 \
#              and DES.GroupNumber = {1:.0f} \
#              and DES.SubGroupNumber = {2:.0f} \
#              and PROG.GalaxyID between DES.GalaxyID and DES.TopLeafID \
#              and AP.ApertureSize = 100 \
#              and AP.GalaxyID = PROG.GalaxyID \
#          ORDER BY \
#              DES.MassType_Star desc, \
#              PROG.Redshift asc, \
#              PROG.MassType_Star desc'.format(sim, fof, sub)
#
# tree_sat_query = 'SELECT \
#              DES.GalaxyID as gid, \
#              DES.TopLeafID as tlid,\
#              PROG.Redshift as z, \
#              PROG.MassType_DM as mdm, \
#              PROG.MassType_Star as ms, \
#              AP.SFR / (AP.Mass_Star+0.0001) as ssfr, \
#              PROG.CentreOfPotential_x as copx, \
#              PROG.CentreOfPotential_y as copy, \
#              PROG.CentreOfPotential_z as copz \
#          FROM \
#              {0}_Subhalo as PROG, \
#              {0}_Subhalo as DES, \
#              {0}_Aperture as AP \
#          WHERE \
#              DES.SnapNum = 28 \
#              and DES.GroupNumber = {1:.0f} \
#              and DES.SubGroupNumber = {2:.0f} \
#              and PROG.GalaxyID between DES.GalaxyID and DES.TopLeafID \
#              and AP.ApertureSize = 100 \
#              and AP.GalaxyID = PROG.GalaxyID \
#          ORDER BY \
#              DES.MassType_Star desc, \
#              PROG.Redshift asc, \
#              PROG.MassType_Star desc'.format(sim, sat_fof, sat_sub)
#

tree_host_query = 'SELECT \
             DES.GalaxyID as gid, \
             PROG.Redshift as z, \
        FOF.Group_R_Crit200 as r_200, \
             PROG.CentreOfPotential_x as copx, \
             PROG.CentreOfPotential_y as copy, \
             PROG.CentreOfPotential_z as copz \
         FROM \
             {0}_Subhalo as PROG, \
             {0}_Subhalo as DES, \
             {0}_Aperture as AP, \
            {0}_FOF as FOF \
         WHERE \
             DES.SnapNum = 28 \
             and DES.GroupNumber = {1:.0f} \
              and FOF.GroupID = PROG.GroupID \
            and DES.SubGroupNumber = {2:.0f} \
             and PROG.GalaxyID between DES.GalaxyID and DES.TopLeafID \
             and AP.ApertureSize = 100 \
             and AP.GalaxyID = PROG.GalaxyID \
         ORDER BY \
             DES.MassType_Star desc, \
             PROG.Redshift asc, \
             PROG.MassType_Star desc'.format(sim, fof, sub)

tree_sat_query = 'SELECT \
             DES.GalaxyID as gid, \
             PROG.Redshift as z, \
             AP.SFR / (AP.Mass_Star+0.0001) as ssfr, \
             PROG.CentreOfPotential_x as copx, \
             PROG.CentreOfPotential_y as copy, \
             PROG.CentreOfPotential_z as copz \
         FROM \
             {0}_Subhalo as PROG, \
             {0}_Subhalo as DES, \
             {0}_Aperture as AP \
         WHERE \
             DES.SnapNum = 28 \
             and DES.GroupNumber = {1:.0f} \
             and DES.SubGroupNumber = {2:.0f} \
             and PROG.GalaxyID between DES.GalaxyID and DES.TopLeafID \
             and AP.ApertureSize = 100 \
             and AP.GalaxyID = PROG.GalaxyID \
         ORDER BY \
             DES.MassType_Star desc, \
             PROG.Redshift asc, \
             PROG.MassType_Star desc'.format(sim, sat_fof, sat_sub)
#

tree_sat_query2 = 'SELECT \
             DES.GalaxyID as gid, \
             PROG.Redshift as z, \
             PROG.GroupNumber as fof, \
             PROG.SubGroupNumber as sub \
         FROM \
             {0}_Subhalo as PROG, \
             {0}_Subhalo as DES, \
             {0}_Aperture as AP \
         WHERE \
             DES.SnapNum = 28 \
             and DES.GroupNumber = {1:.0f} \
             and DES.SubGroupNumber = {2:.0f} \
             and PROG.GalaxyID between DES.GalaxyID and DES.TopLeafID \
             and AP.ApertureSize = 100 \
             and AP.GalaxyID = PROG.GalaxyID \
         ORDER BY \
             DES.MassType_Star desc, \
             PROG.Redshift asc, \
             PROG.MassType_Star desc'.format(sim, sat_fof, sat_sub)
tree_host = sql.execute_query(con, tree_host_query)
tree_sat = sql.execute_query(con, tree_sat_query)
tree_sat2 = sql.execute_query(con, tree_sat_query2)

print(tree_host)
print(tree_sat)

def times_Gyr(z):
    H0 = 67.77
    OmegaM = 0.307
    OmegaL = 0.693
    time_array = np.zeros(len(z))
    for i in range(len(z)):
        t = (2 / (3 * H0 * np.sqrt(OmegaL))) * np.log(
            (np.sqrt(OmegaL * ((1 + z[i]) ** (-3))) + np.sqrt(OmegaL * ((1 + z[i]) ** (-3)) + OmegaM)) / np.sqrt(
                OmegaM))
        time_array[i] =  t * 1000
    return time_array

def flag(scale_factor, box_,sat_pos,host_pos):
    coord = host_pos - sat_pos
    coord = coord * scale_factor
    half_box_ = box_*scale_factor/2
    if coord < -half_box_:
        coord = coord + box_
    elif coord > half_box_:
        coord = coord - box_
    return coord

# calculating distance between the host and the satellite
def moving_to_origin_sub(host, sat, box):
    distances = np.array([])
    time_ = np.array([])
    r_v = np.array([])
    # counter = 0
    # r_vir = host["r_200"] * 0.0033
    # for r in r_vir:
    #     r = r * (1 + host['z'][counter])
    #     counter += 1
    infall_z = np.array([-1])
    quench_z = np.array([-1])
    halfbox = box / 2
    if len(host) > len(sat):
        for i in reversed(range(len(sat))):
            for j in reversed(range(len(host))):

                if sat['z'][i] == host['z'][j]:
                    a = 1 / (1 + host['z'][j])
                    r_vir = host['r_200'][j] * 0.0033 / a
                    x = flag( a, box, sat['copx'][i], host['copx'][j])
                    y = flag( a, box, sat['copy'][i], host['copy'][j])
                    z = flag( a, box, sat['copz'][i], host['copz'][j])

                    r_v = np.append(r_v, r_vir)
                    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
                    if dist < r_vir:
                        infall = np.append(infall_z, sat['z'][i])

                    if sat['ssfr'][i] < 10**(-11):
                        quench_z = np.append(quench_z, sat['z'][i])

                    distances = np.append(distances, dist)
                    time_ = np.append(time_, sat['z'][i])
    else:
        for i in reversed(range(len(host))):
            for j in reversed(range(len(sat))):

                if host['z'][i] == sat['z'][j]:
                    a = 1 / (1 + sat['z'][j])
                    r_vir = host['r_200'][i] * 0.0033 / a
                    x = flag( a, box, sat['copx'][j], host['copx'][i])
                    y = flag( a, box, sat['copy'][j], host['copy'][i])
                    z = flag( a, box, sat['copz'][j], host['copz'][i])
                    r_v = np.append(r_v, r_vir)
                    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                    if dist < r_vir:
                        infall_z = np.append(infall_z, sat['z'][j])


                    if sat['ssfr'][j] < 10**(-11):
                        quench_z = np.append(quench_z, sat['z'][j])


                    distances = np.append(distances, dist)
                    time_ = np.append(time_, host['z'][i])

    time_=time_[::-1]
    Cdistances = distances[::-1]*(1+time_)
    distances = distances[::-1]
    quench_z = quench_z[::-1]
    infall_z = infall_z[::-1]
    r_v = r_v[::-1]

    return distances,Cdistances, time_, quench_z, infall_z, r_v


"""separation at available redshifts between the host and the satellite"""
separation_hos_sat,csep, time_z, quench, infall, r_vir_tot = moving_to_origin_sub(tree_host, tree_sat, box_size)

print(separation_hos_sat, 'separation cMpc')
print(time_z, 'time_z')
print(quench, 'quenching_z')
print(infall, 'infall_z')
print(tree_sat['ssfr'], 'ssfr')
print(tree_sat['copx'], "copx")
print(tree_sat['copy'], "copy")
print(tree_sat['copz'], "copz")
print(tree_sat['z'], 'z')
print(tree_sat2['fof'], 'groupnumber')
print(tree_sat2['sub'], 'subgroup')
plt.plot(times_Gyr(time_z), r_vir_tot, label="r_vir")
plt.plot(times_Gyr(time_z),separation_hos_sat, label="physicsl separation")
plt.plot(times_Gyr(time_z),csep, label="comoving separation")
plt.legend()
plt.savefig("separation.jpg", format = "jpg")

plt.clf()
plt.plot(times_Gyr(tree_sat['z']), tree_sat["ssfr"])
plt.hlines(0,14, 10**(-11))
plt.yscale('log')
plt.savefig("ssfr.jpg", format = "jpg")
def infall_and_quenching_times(host, sat, box, separ, z_sep):
    """

    :param host: host tree
    :param sat: satellite tree
    :param box: simulation box size
    :param separ: separation history between the host and the satellite
    :param z_sep: at which z separation is recorded
    :return: infall and quenching redshifts
    """
    counter = 0
    r_vir = host["r_200"]*0.0033
    for r in r_vir:
        r = r * (1 + host['z'][counter])
        counter += 1



