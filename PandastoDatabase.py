from sqlalchemy import create_engine
import pandas as pd

price = pd.read_excel("BigData2.xlsx")
disk_engine = create_engine('sqlite:///ColorDatabse14000.db')
price.to_sql('color_table', disk_engine, if_exists='append')

# append_one = pd.read_excel("append_first.xlsx", index_col=0)
# append_two = pd.read_excel("append_second.xlsx", index_col=0)
# append_one['rank'] = 0
#
# bigdata = append_one.append(append_two, ignore_index=True)
#
#
# print(append_one)
# print(append_two)
# print(bigdata)
# bigdata.to_excel("BigData2.xlsx", index=False)