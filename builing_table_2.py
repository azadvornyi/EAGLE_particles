import pickle
import numpy as np
import pyread_properties as pp
import pandas as pd
import pyread_eagle as read_eagle
import os
import dump_load as d

"""table_1 contains the snapshot before infall time, group_number, sub_groupn_umber amd a unique ID, and integer >= 0
 to track the tree nodes later
 
 For now the table only contains one entry because the EAGLE databse is not working 
 """
# structure of table_1
# table_1 = id, snap, gn, sgn

table_1 = d.load('table_1/table_1')

ones = np.ones(5)

part_of_a_tree = {}
part_of_a_tree["id"] = [0,1,2,3,4]
part_of_a_tree["snap"] = [27]*ones
part_of_a_tree["gn"] = [11]*ones
part_of_a_tree["sgn"] = [3]*ones

centre_at_i_time = np.array([74.311615, 22.76557, 26.967281])

table_1 = pd.DataFrame.from_dict(table_1)

# table_2 contains list of particles (lsit of ParticleIDs) that are to be tracked


# new_list = [expression(i) for i in old_list if filter(i)]

# row index is equal to tree_id
def make_label(table):
    num_of_rows = len(table.values[:,1])
    label = np.empty(num_of_rows, dtype=str)
    print(label)
    for j in range(num_of_rows):
        l = ''
        for i in range(4):
            l = l + str(int(table.values[j][i]))
        print(l)

        print(label[j], 'label j')
        label[j] = l
    print(label)
    return label


def make_fname_pt(snap):
    fname = pp.basedir + pp.path_to_snapshot_100_HD + pp.particledata_folders[snap] + pp.particledata_files[snap]
    return fname

def make_fname_st(snap):
    fname = pp.basedir + pp.path_to_snapshot_100_HD + pp.snapshot_folders[snap] + pp.snapshot_files[snap]
    return fname


#lab = make_label(row, 0)

#print(lab)




def write_to_table_2(entry_from_table_1):

    print(entry_from_table_1)
    centre_at_i_time = np.array([74.311615, 22.76557, 26.967281])
    id_g = 0
    prop_gas = ['GroupNumber', 'SubGroupNumber','ParticleIDs']
    tree_id = np.array(entry_from_table_1.values[:,0], dtype=int)
    snap = np.array(entry_from_table_1.values[:,1], dtype=int)
    gn = np.array(entry_from_table_1.values[:,2], dtype=int)
    sgn = np.array(entry_from_table_1.values[:,3], dtype=int)
    lab = make_label(entry_from_table_1)
    print(snap)
    print(snap[0])
    fname_infall = make_fname_pt(snap[0])
    snap_infall = read_eagle.EagleSnapshot(fname_infall)
    a_global_i, h_global_i, boxsize_global_i = pp.read_header(fname_infall)
    reading_region = 10

    #print(fname_infall)
    #fname_quench = pp.basedir + pp.path_to_snapshot_100_HD + pp.particledata_folders[27] + pp.particledata_files[27]

    pp.get_properties(id_g, prop_gas, snap_infall, gn, sgn, centre_at_i_time * h_global_i, reading_region,
                                     fname_infall, tree_id)
    #print(data_gas_infall)
    #d.dump("table_2/%s" % lab, data_gas_infall)
    # with open("tabl".format(lab), 'wb+') as f0:
    #     pickle.dump(data_gas_infall, f0, pickle.HIGHEST_PROTOCOL)


# def write_to_table_2(entry_from_table_1):
#
#     print(entry_from_table_1)
#     centre_at_i_time = np.array([74.311615, 22.76557, 26.967281])
#     id_g = 0
#     prop_gas = ['GroupNumber', 'SubGroupNumber','ParticleIDs']
#     tree_id = entry_from_table_1.values[0][0]
#     snap = entry_from_table_1.values[0][1]
#     gn = entry_from_table_1.values[0][2]
#     sgn = entry_from_table_1.values[0][3]
#     lab = make_label(entry_from_table_1)
#     fname_infall = make_fname_pt(snap)
#     snap_infall = read_eagle.EagleSnapshot(fname_infall)
#     a_global_i, h_global_i, boxsize_global_i = pp.read_header(fname_infall)
#     reading_region = 10
#     #print(fname_infall)
#     #fname_quench = pp.basedir + pp.path_to_snapshot_100_HD + pp.particledata_folders[27] + pp.particledata_files[27]
#
#     data_gas_infall = pp.get_properties(id_g, prop_gas, snap_infall, gn, sgn, centre_at_i_time * h_global_i, reading_region,
#                                      fname_infall)
#     print(data_gas_infall)
#     d.dump("table_2/%s" % lab, data_gas_infall)
#     # with open("tabl".format(lab), 'wb+') as f0:
#     #     pickle.dump(data_gas_infall, f0, pickle.HIGHEST_PROTOCOL)



tree = pd.DataFrame()
directory = r'/cosma/home/durham/dc-zadv1/Data/Eagle_python/scripts/chopped_trees'

for entry in os.scandir(directory):
    if entry.is_file():
        table = d.load(entry.path)
        row = table.iloc[0]
        tree = tree.append(row, ignore_index=True)
            #write_to_table_2(table)
        #print(tree)
    tree = tree[[   'id' , 'snap'  ,   'gn' , 'sgn']]

print(tree.values[:,1])

#l = make_label(tree)

write_to_table_2(tree)
#
# for i in range(5):
#     row = table_1.loc[table_1['id'] == i]
#     write_to_table_2(row)

# file_pi = open('filename_pi.obj', 'w')
#
#
# pickle.dump(object_pi, file_pi)




