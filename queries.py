import pickle
sim100 = 'RefL0100N1504'
sim25 = 'RefL0025N0376'
sim12 = 'RefL0012N0188'
# ============
sim = sim25  # set simulation here
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

host_dat = 345925,  1, 0, 16.556612 , 24.493837 , 17.707695 , 518.9658
gid, fof, sub, dx, dy, dz, r_vir = host_dat
# # print(host_dat)
# gid = 1
# fof = 1
# sub = 0
# dx = 16.556612
# dy = 24.493837
# dz = 17.707695
# r_vir = 518.9658

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
             and SH.SnapNum = 28 \
             and SH.MassType_DM > {1}".format(sim, host_mass)

normal_sat_query = 'SELECT \
             S.GalaxyID as Sgid, \
             H.GalaxyID as Hgid \
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
