import pandas as pd
import numpy as np

df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                         "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two",
                         "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small",
                         "small", "large", "small", "small",
                         "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                   "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

table1 = pd.pivot_table(df, values='D', index=['A', 'B'],
                    columns=['C'], aggfunc=np.sum)


df1 = pd.DataFrame({'id': [1,2,3,4],
                    'mass': [10, 20, 30, 40],
                    'type':[0,0,0,0]})
df2 = pd.DataFrame({'id': [1,3,4],
                    'mass': [10, 30, 40],
                    'type': [1, 1, 0]})

df3 = df1.merge(df2, left_on='id', right_on='id')

print(df3)

table = pd.pivot_table(df3, values='mass_y',
                    columns=['type_y'], aggfunc=np.sum)

print(table)