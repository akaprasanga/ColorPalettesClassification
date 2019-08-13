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
                # print(even[1:])
                group_dictionary[even[0]] = even[2:]

        return group_dictionary

def extract_colors_from_sql():
    """
    UNLOCK TABLES;

--
-- Table structure for table `palettes`
--

DROP TABLE IF EXISTS `palettes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
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
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=3955491 DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `palettes`
--

LOCK TABLES `palettes` WRITE;
/*!40000 ALTER TABLE `palettes` DISABLE KEYS */;
    :return:
    """

    with open("D:\Work\ColorRecommendation\ColorCategorization-Data\\firstline.txt") as f:
        content = f.readlines()
        line = content[0]
        colors = line.split('(')
        total_len = len(colors)
        pandas_col = ["rank", "hearts", "color 1", "color 2", "color 3", "color 4", "color 5", "wid 1", "wid 2", "wid 3", "wid 4", "wid 5", "num_colors"]
        df = pd.DataFrame(columns=pandas_col)
        for i, each in enumerate(colors):
            if i > 0:
                each = each[:-2].replace(')', '')
                each = each.replace("'", "")
                # print(each.split(','))
                each = each.split(',')
                df.loc[i] = [int(each[6]), int(each[5]), each[8], each[9], each[10], each[11], each[12], each[13], each[14], each[15], each[16], each[17],int(each[18])]
                # if i == total_len-1:
            #     list = each.split(',')
            #
            #     # print(list)
            # else:
            #     list = each.split(',')
            #     list = line[:-1]
            #     # print(list)
        df.to_excel("8128_3.xlsx")

def join_excels():
    df1 = pd.read_excel("8128.xlsx")
    df2 = pd.read_excel("8128_2.xlsx")
    df3 = pd.read_excel("8128_3.xlsx")

    df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

    df.to_excel("final.xlsx")

# join_excels()
# extract_colors_from_sql()
# group_dictionary=get_group_dictionary()
# print(group_dictionary)
