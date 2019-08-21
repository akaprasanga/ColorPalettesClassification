import pandas as pd

def get_group_dictionary():
    group_dictionary = {}
    with open("D:\Work\ColorRecommendation\ColorCategorization-Data\l2rcolors.txt") as f:
        content = f.readlines()
        for line in content:
            line = line.replace(" ", "")
            if len(line)>5:
                line = line.replace('newDataGroupColor', "")
                line = line.replace('Values=newList<ColorData>(new[]{newColorData{', '')
                line = line.replace('newColorData{', '')
                line = line.replace('Name=', '')
                line = line.replace('Value=', '')
                line = line.replace('{', '')
                line = line.replace('}', '')
                line = line.replace('"', '')
                line = line.replace(')', '')
                words = line.split(',')
                odd = words[1:len(words)-1:2]
                even = words[0:len(words)-1:2]
                # print(even)
                # print(len(even))
                group_dictionary[even[0]] = even[2:]

        return group_dictionary

def extract_colors_from_sql():
    """
CREATE TABLE `palettes` (
  `id` int(11) NOT NULL AUTO_INCREMENT,0
  `paletteId` int(11) DEFAULT NULL,1
  `username` varchar(255) DEFAULT NULL,2
  `numViews` int(11) DEFAULT NULL,3
  `numVotes` int(11) DEFAULT NULL,4
  `numHearts` int(11) DEFAULT NULL,5
  `paletteRank` int(11) DEFAULT NULL,6
  `dateCreated` date DEFAULT NULL,7
  `color1` varchar(255) DEFAULT NULL,8
  `color2` varchar(255) DEFAULT NULL,9
  `color3` varchar(255) DEFAULT NULL,10
  `color4` varchar(255) DEFAULT NULL,11
  `color5` varchar(255) DEFAULT NULL,12
  `colorWidths1` varchar(255) DEFAULT NULL,13
  `colorWidths2` varchar(255) DEFAULT NULL,14
  `colorWidths3` varchar(255) DEFAULT NULL,15
  `colorWidths4` varchar(255) DEFAULT NULL,16
  `colorWidths5` varchar(255) DEFAULT NULL,17
  `numColors` int(11) DEFAULT NULL,18
    :return:
    """
    import csv
    cols = ['id', 'paletteId', 'username', 'numViews', 'numVotes', 'numHearts', 'paletteRank', 'dateCreated', 'hex1', 'hex2', 'hex3', 'hex4', 'hex5', 'wid 1', 'wid 2', 'wid 3', 'wid 4', 'wid 5', 'numColors']
    df = pd.DataFrame(columns=cols)
    with open("D:\Work\ColorRecommendation\ColorCategorization-Data\\palettesdb.txt", encoding="utf8") as f:
        content = f.readlines()
        content = [x.replace("""INSERT INTO `palettes` VALUES """, """""") for x in content]
        content = [x.replace(";", "") for x in content]
        content = [x.replace("(", "") for x in content]
        content = [x.replace("),", "\n") for x in content]
        content = [x.replace(")", "") for x in content]
        counter = 0
        # content = content[:1]
        for data in content:
            data = data.replace(')', '')
            # print(data.replace(')', ''))
            lines = data.split('\n')
            for each_line in lines:
                each_line = each_line.replace("'", "")
                row = each_line.split(',')
                # print(row)
                if len(row)==19:
                    del row[2]
                    with open('Full.csv', 'a') as csvfile:
                        writer = csv.writer(csvfile, delimiter=",")
                        writer.writerow(row)

    print("Completed")
                #     df.loc[counter] = row
                #     counter = counter+1
        # for each in content:
        #     each= each.replace(')', '')
            # print(each.split(','))
    # df.to_excel('FullSqlDb.xlsx')

def sort_based_on_one_column(column_name, df, ascending_flag=True):
    return df.sort_values(column_name, ascending=ascending_flag)

def manage_large_df():
    df = pd.read_csv('Full.csv', encoding='ISO-8859-1')
    # df = df[1677:]
    df = df.drop(['id', 'palette_id', 'date_created', 'num_votes', 'num_views', 'num_hearts'], axis=1)
    df['palette_rank'] = df['palette_rank'].astype(int)
    df['num_colors'] = df['num_colors'].astype(int)
    #
    df = (df[df.palette_rank > 0])
    df = (df[df.num_colors > 4])

    required = sort_based_on_one_column('palette_rank', df)
    required = required.head(10_000)
    required.to_excel('10000ColorPalettes.xlsx')



filename = 'HugeSqlDb.xlsx'
# join_excels()
# extract_colors_from_sql()
# group_dictionary=get_group_dictionary()
# print(len(group_dictionary))
manage_large_df()
# extract_colors_from_sql()
