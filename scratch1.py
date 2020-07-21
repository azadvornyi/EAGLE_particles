import numpy as np
import pickle
import os
import pandas as pd

ones = np.ones(6)
one = np.ones(5)
table_1 = {}

table_1["id"] = [0,1,2,3,4,5]
table_1["snap"] = [15]*ones
table_1["gn"] = [626]*ones
table_1["sgn"] = [0]*ones
table_1["sg"] = [0]*ones


part_of_a_tree = {}
part_of_a_tree["id"] = [0,1,2,3,4]
part_of_a_tree["snap"] = [27]*one
part_of_a_tree["gn"] = [11]*one
part_of_a_tree["sgn"] = [3]*one

table_1 = pd.DataFrame.from_dict(table_1)

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


dump('table_1/table_1', table_1)

# table = load('table_1/table_1')
# tab_1  = load('table_2/0')
# print(tab_1)


def make_label(row_to_convert):
    label = ''
    #print(row_to_convert)
    for i in range(4):
        label = label + str(int(row_to_convert.values[0][i]))
    return label

def chop_trees(top,bot):
    #top = pd.DataFrame.from_dict(top)
    bot = pd.DataFrame.from_dict(bot)
    for i in range(5):
        top_row = top.loc[top['id'] == i]
        bot_row = bot.loc[bot['id'] == i]
        # print(top_row)
        # print(bot_row)
        label = make_label(top_row)
        tree = top_row.append(bot_row, ignore_index=True)
        print(label)
        dump('chopped_trees/%s' % (label), tree)


# chop_trees(table, part_of_a_tree)



# print(load('chopped_trees/0156260'))


directory = r'/cosma/home/durham/dc-zadv1/Data/Eagle_python/scripts/chopped_trees'

for entry in os.scandir(directory):
    if entry.is_file():
        tab = load(entry.path)
       # print(tab)


# df1 = pd.DataFrame.from_dict(table_1)


# df2 = pd.DataFrame.from_dict(part_of_a_tree)
# df3 = pd.concat([df1, df2], ignore_index=True,sort = False)
# print(df3[df3['id']==0])

#d_g = table_1.pop("snap")

# print(d_g)
# print(table_1)

tab_3 = load('table_2/0')
tab_3 =  pd.DataFrame.from_dict(tab_3)
print(tab_3, 'tab_3')