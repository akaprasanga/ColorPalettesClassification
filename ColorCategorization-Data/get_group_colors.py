

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

group_dictionary=get_group_dictionary()
print(group_dictionary)
# you may also want to remove whitespace characters like `\n` at the end of each line
# content = [x.strip() for x in content]