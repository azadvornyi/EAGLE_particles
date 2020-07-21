import pickle
import numpy as np
import pyread_properties as pp
import pandas as pd
import pyread_eagle as read_eagle
import h5py
import os
import sys
import dump_load as d
import time


# with open("{0}".format('0156260', 'rb')) as f5:
#     list_of_ids = pickle.load(f5)
# f = open("{0}".format('0156260'), 'rb')
# list_of_ids = pickle.load(f)
# f.close()

def dump(filename, *data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data if len(data) > 1 else data[0]
    else:
        return None

list_of_ids = load('0156260')

def make_label(row_to_convert):
    label = ''
    row_index = row_to_convert.values[0][0]
    for i in range(4):
        label = label + str(row_to_convert.values[row_index][i])
    return label


def make_fname_pt(snap):
    fname_ = str(pp.basedir) + str(pp.path_to_snapshot_100_HD) + str(pp.particledata_folders[snap]) + str(pp.particledata_files[snap])
    return fname_

def make_fname_st(snap):
    fname = pp.basedir + pp.path_to_snapshot_100_HD + pp.snapshot_folders[snap] + pp.snapshot_files[snap]
    return fname


def split_arrays(data):
    vel = data.pop('Velocity')
    coords = data.pop('Coordinates')
    data['v_x'] = vel[:,0]
    data['v_y'] = vel[:,1]
    data['v_z'] = vel[:,2]
    data['cop_x'] = coords[:,0]
    data['cop_y'] = coords[:,1]
    data['cop_z'] = coords[:,2]
    return data

def get_properties_table_3(itype, property, snapshot, gn, sgn, cent, region_size, fname_read):
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
    a, h, boxsize = pp.read_header(fname_read)
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

    #mask = np.logical_and(data['GroupNumber'] == gn, data['SubGroupNumber'] == sgn)

    # for att in data.keys():
    #     data[att] = data[att][mask]
    vel = data.pop('Velocity')
    coords = data.pop('Coordinates')
    data['v_x'] = vel[:,0]
    data['v_y'] = vel[:,1]
    data['v_z'] = vel[:,2]
    data['cop_x'] = coords[:,0]
    data['cop_y'] = coords[:,1]
    data['cop_z'] = coords[:,2]

    len_dict = len(data['GroupNumber'])

    if itype == 1:
        m_dm = pp.read_dataset_dm_mass(fname_read)
        data['Mass'] = m_dm * np.ones(len_dict, dtype='f8')

    data['ParticleType'] = np.ones(len_dict, dtype='f8') * itype
    data = pp.convert_to_galactic(data)


    return data


def get_properties_table_3_x(itype, property, snap):
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

    fname_pt = make_fname_pt(snap)
    fname_st = make_fname_st(snap)
    snap_pt = read_eagle.EagleSnapshot(fname_pt)
    snap_st = read_eagle.EagleSnapshot(fname_st)
    # part = 0.5
    a, h, boxsize = pp.read_header(fname_pt)
    # region = np.array([
    #     (cent[0] - part * region_size), (cent[0] + part * region_size),
    #     (cent[1] - part * region_size), (cent[1] + part * region_size),
    #     (cent[2] - part * region_size), (cent[2] + part * region_size)
    # ])

    #region = np.array([0, 100, 0, 100, 0, 100])
    region = (70,80,15,30,20,30)
    snap_pt.select_region(*region)

    t1 = time.time()
    itype_gas = 0
    itype_star = 4
    

    property_gas = ['GroupNumber', 'SubGroupNumber', 'Temperature', 'Density', 'Velocity', 'StarFormationRate', 'Mass',
            'Coordinates', 'ParticleIDs']
    property_star = ['GroupNumber', 'SubGroupNumber', 'Velocity', 'Mass', "Coordinates", 'ParticleIDs']

    data_gas = {}
    data_star = {}
    f = h5py.File(fname_pt, 'r')
    for att in property_gas:
        tmp = snap_pt.read_dataset(itype_gas, att)
        cgs = f['PartType%i/%s' % (itype_gas, att)].attrs.get('CGSConversionFactor')
        aexp = f['PartType%i/%s' % (itype_gas, att)].attrs.get('aexp-scale-exponent')
        hexp = f['PartType%i/%s' % (itype_gas, att)].attrs.get('h-scale-exponent')
        data_gas[att] = np.multiply(tmp, cgs * a ** aexp * h ** hexp, dtype='f8')

    for att in property_star:
        tmp = snap_pt.read_dataset(itype_star, att)
        cgs = f['PartType%i/%s' % (itype_star, att)].attrs.get('CGSConversionFactor')
        aexp = f['PartType%i/%s' % (itype_star, att)].attrs.get('aexp-scale-exponent')
        hexp = f['PartType%i/%s' % (itype_star, att)].attrs.get('h-scale-exponent')
        data_star[att] = np.multiply(tmp, cgs * a ** aexp * h ** hexp, dtype='f8')
    f.close()



    #mask = np.logical_and(data['GroupNumber'] == gn, data['SubGroupNumber'] == sgn)

    # for att in data.keys():
    #     data[att] = data[att][mask]
    data_gas  = split_arrays(data_gas)
    data_star = split_arrays(data_star)
    print(data_gas)
    print(data_star)
    
    len_g = len(data_gas['GroupNumber'])
    len_s = len(data_star['GroupNumber'])
    
    data_gas['ParticleType'] = np.ones(len_g, dtype='f8') * itype_gas
    data_star['ParticleType'] = np.ones(len_s, dtype='f8') * itype_star
    
    data_gas  = pp.convert_to_galactic(data_gas)
    data_star = pp.convert_to_galactic(data_star)

    data_gas  = pd.DataFrame.from_dict(data_gas)
    data_star = pd.DataFrame.from_dict(data_star)

    data = data_star.append(data_gas, ignore_index=True)

    

   

    if itype == 1:
        m_dm = pp.read_dataset_dm_mass(fname_read)
        data['Mass'] = m_dm * np.ones(len_dict, dtype='f8')

    

    directory = r'/cosma/home/durham/dc-zadv1/Data/Eagle_python/scripts/chopped_trees'
    path_to_table_2 = r'/cosma/home/durham/dc-zadv1/Data/Eagle_python/scripts/table_2/'
    for entry in os.scandir(directory):
        if entry.is_file():
            t = time.time()
            table = d.load(entry.path)

            row = table[table['snap']==snap]
            tree_id = int(row.values[0][0])
            gn = row.values[0][2]
            sgn = row.values[0][3]
            #row = [[   'id' , 'snap' ,   'gn' , 'sgn']]
            path_to_list_of_ids = path_to_table_2 + str(tree_id)
            list_of_ids = d.load(path_to_list_of_ids)
            
            list_of_ids  = pd.DataFrame.from_dict(list_of_ids)
            print(list_of_ids)
            #print(gn, type(gn))
            merged_tables = data.merge(list_of_ids, left_on='ParticleIDs', right_on='ParticleIDs')

            dirName = 'table_3/' + '1' + str(tree_id)
            try:
                os.makedirs(dirName)
                print("Directory ", dirName, " Created ")
                
            except FileExistsError:
                pass
            d.dump('%s/%s' % (dirName, snap), merged_tables)
            print('finished')
                    

    




tab_2 = d.load('table_2/1')

def write_to_table_3(entry_from_table_2, row_of_data):

    centre_at_i_time = np.array([71.447205, 28.908436, 26.367676])
    centre_at_i_time = np.array([74.311615, 22.76557, 26.967281])
    id_g = 0
    id_s = 4

    prop_gas_p = ['GroupNumber', 'SubGroupNumber', 'Temperature', 'Density', 'Velocity', 'StarFormationRate', 'Mass',
            'Coordinates', 'ParticleIDs']
    prop_star_p = ['GroupNumber', 'SubGroupNumber', 'Velocity', 'Mass', "Coordinates", 'ParticleIDs']

    prop_gas_s = ['GroupNumber', 'Temperature', 'Density', 'Velocity', 'StarFormationRate', 'Mass',
                'Coordinates', 'ParticleIDs']
    prop_star_s = ['GroupNumber', 'Velocity', 'Mass', "Coordinates", 'ParticleIDs']

    tree_id = int(row_of_data.values[0][0])
    snap = int(row_of_data.values[0][1])
    gn = row_of_data.values[0][2]
    sgn = row_of_data.values[0][3]
    # lab = make_label(entry_from_table_2)
    fname_ = make_fname_pt(snap)
    fname_s = make_fname_st(snap)
    snap_ = read_eagle.EagleSnapshot(fname_)
    snap_s = read_eagle.EagleSnapshot(fname_s)
    a_global_i, h_global_i, boxsize_global_i = pp.read_header(fname_)
    reading_region = 10
    # print(fname_infall)
    # fname_quench = pp.basedir + pp.path_to_snapshot_100_HD + pp.particledata_folders[27] + pp.particledata_files[27]

    data_gas_infall_p = get_properties_table_3_x(id_g, prop_gas_p, snap_, gn, sgn, centre_at_i_time * h_global_i,
                                        reading_region,
                                        fname_)
    data_star_infall_p = get_properties_table_3_x(id_s, prop_star_p, snap_, gn, sgn, centre_at_i_time * h_global_i,
                                             reading_region,
                                             fname_)
    # data_gas_infall_s = get_properties_table_3(id_g, prop_gas_s, snap_s, gn, sgn, centre_at_i_time * h_global_i,
    #                                          reading_region,
    #                                          fname_s)
    # data_star_infall_s = get_properties_table_3(id_s, prop_star_s, snap_s, gn, sgn, centre_at_i_time * h_global_i,
    #                                           reading_region,
    #                                           fname_s)


    table_3 = pp.converted_to_stars(entry_from_table_2, data_gas_infall_p)
    table_3_star = pp.converted_to_stars(entry_from_table_2, data_star_infall_p)
    print(len(entry_from_table_2['ParticleIDs']))
    print(table_3[['GroupNumber_x', 'SubGroupNumber_x', 'GroupNumber_y', 'SubGroupNumber_y']])
    print(table_3_star[['GroupNumber_x', 'SubGroupNumber_x', 'GroupNumber_y', 'SubGroupNumber_y']])

    # table_3_s = pp.converted_to_stars(entry_from_table_2, data_gas_infall_s)
    # table_3_star_s = pp.converted_to_stars(entry_from_table_2, data_star_infall_s)
    # print(len(entry_from_table_2['ParticleIDs']))
    # print(table_3_s[['GroupNumber_x', 'GroupNumber_y']])
    # print(table_3_star_s[['GroupNumber_x', 'GroupNumber_y']])

    dirName = 'table_3/' + '3' + str(tree_id)
    try:
        os.makedirs(dirName)
        print("Directory ", dirName, " Created ")
        
    except FileExistsError:
        pass
    d.dump('%s/%s' % (dirName, snap), table_3)
    print('finished')
    # with open("{0}".format(lab), 'wb+') as f0:
    #     pickle.dump(data_gas_infall, f0, pickle.HIGHEST_PROTOCOL)


def write_to_table_3_x(snap_num):


    id_g = 0
    id_s = 4

    prop_gas_p = ['GroupNumber', 'SubGroupNumber', 'Temperature', 'Density', 'Velocity', 'StarFormationRate', 'Mass',
            'Coordinates', 'ParticleIDs']
    prop_star_p = ['GroupNumber', 'SubGroupNumber', 'Velocity', 'Mass', "Coordinates", 'ParticleIDs']

    prop_gas_s = ['GroupNumber', 'Temperature', 'Density', 'Velocity', 'StarFormationRate', 'Mass',
                'Coordinates', 'ParticleIDs']
    prop_star_s = ['GroupNumber', 'Velocity', 'Mass', "Coordinates", 'ParticleIDs']

    
    # print(fname_infall)
    # fname_quench = pp.basedir + pp.path_to_snapshot_100_HD + pp.particledata_folders[27] + pp.particledata_files[27]

    get_properties_table_3_x(id_g, prop_gas_p, snap_num)
    # data_star_infall_p = get_properties_table_3_x(id_s, prop_star_p, snap_, gn, sgn, centre_at_i_time * h_global_i,
    #                                          reading_region,
    #                                          fname_)
    # data_gas_infall_s = get_properties_table_3(id_g, prop_gas_s, snap_s, gn, sgn, centre_at_i_time * h_global_i,
    #                                          reading_region,
    #                                          fname_s)
    # data_star_infall_s = get_properties_table_3(id_s, prop_star_s, snap_s, gn, sgn, centre_at_i_time * h_global_i,
    #                                           reading_region,
    #                                           fname_s)


    # table_3 = pp.converted_to_stars(entry_from_table_2, data_gas_infall_p)
    # table_3_star = pp.converted_to_stars(entry_from_table_2, data_star_infall_p)
    # print(len(entry_from_table_2['ParticleIDs']))
    # print(table_3[['GroupNumber_x', 'SubGroupNumber_x', 'GroupNumber_y', 'SubGroupNumber_y']])
    # print(table_3_star[['GroupNumber_x', 'SubGroupNumber_x', 'GroupNumber_y', 'SubGroupNumber_y']])

    # # table_3_s = pp.converted_to_stars(entry_from_table_2, data_gas_infall_s)
    # # table_3_star_s = pp.converted_to_stars(entry_from_table_2, data_star_infall_s)
    # # print(len(entry_from_table_2['ParticleIDs']))
    # # print(table_3_s[['GroupNumber_x', 'GroupNumber_y']])
    # # print(table_3_star_s[['GroupNumber_x', 'GroupNumber_y']])

    # dirName = 'table_3/' + '3' + str(tree_id)
    # try:
    #     os.makedirs(dirName)
    #     print("Directory ", dirName, " Created ")
        
    # except FileExistsError:
    #     pass
    # d.dump('%s/%s' % (dirName, snap), table_3)
    # print('finished')
    # with open("{0}".format(lab), 'wb+') as f0:
    #     pickle.dump(data_gas_infall, f0, pickle.HIGHEST_PROTOCOL)

write_to_table_3_x(15)

print('finished')
#write_to_table_3(list_of_ids, part_of_a_tree)
# fname_global = make_fname_pt(27)
# snap_global = read_eagle.EagleSnapshot(fname_global)
# centre_at_i_time = np.array([74.311615, 22.76557, 26.967281])
# region_x = (70,80,15,30,20,30)
# snap_global.select_region(*region_x)

# snap = 27
# t1 = time.time()
# directory = r'/cosma/home/durham/dc-zadv1/Data/Eagle_python/scripts/chopped_trees'
# path_to_table_2 = r'/cosma/home/durham/dc-zadv1/Data/Eagle_python/scripts/table_2/'
# for entry in os.scandir(directory):
#     if entry.is_file():
#         t = time.time()
#         table = d.load(entry.path)

#         row = table[table['snap']==snap]
#         tree_id = int(row.values[0][0])
#         gn = row.values[0][2]
#         sgn = row.values[0][3]
#         #row = [[   'id' , 'snap' ,   'gn' , 'sgn']]
#         path_to_list_of_ids = path_to_table_2 + str(tree_id)
#         list_of_ids = d.load(path_to_list_of_ids)
#         #print(gn, type(gn))
#         write_to_table_3(list_of_ids, row)
#         print(print(time.time() - t, "sec to run one loop "))


#print(time.time() - t1, "sec  to run the whole thing")





# try:
#     os.makedirs('table_3/{0}'.format(tree_id))
#     print("Directory ", tree_id, " Created ")
#
# except FileExistsError:
#     pass
#
# with open("table_3/{0}/{1}".format(tree_id, snap), 'wb+') as f0:
#     pickle.dump(tree_data, f0, pickle.HIGHEST_PROTOCOL)
