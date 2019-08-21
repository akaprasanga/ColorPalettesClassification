import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import colorsys
import colormath.color_conversions
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

class Visualization:

    def __init__(self):
        self.outer_color = {'0': 'white',
                      '1': 'blue',
                      '2': 'green',
                      '3': 'yellow',
                      '4': 'red',
                      '5': 'black',
                      '6': 'green',
                      '7': 'white',
                      '8': 'white',
                      '9': 'black'
                      }


    def rgb2hex(self, color):
        return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

    def hex2rgb(self, color):
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def hex_to_grayscale(self, hex_color):
        rgb_tuple = self.hex2rgb(hex_color)
        return 0.2126 * rgb_tuple[0] + 0.7152 * rgb_tuple[1] + 0.0722 * rgb_tuple[2]

    def hex2hls(self, color):
        rgb_tuple = self.hex2rgb(color)
        h, l, s = colorsys.rgb_to_hls(rgb_tuple[0]/255, rgb_tuple[1]/255, rgb_tuple[2]/255)
        return tuple((h*360, l*100, s*100))

    def hls2rgb(self, color):
        h, l, s = color[0]/360, color[1]/100, color[2]/100
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return tuple((int(r*255), int(g*255), int(b*255)))

    def hex2lab(self, color):
        rgb = self.hex2rgb(color)
        rgb = sRGBColor(rgb[0]/255, rgb[1]/255, rgb[2]/255)
        return convert_color(rgb, LabColor).get_value_tuple()

    def create_graph(self, hls_list, labels, hex_list):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for i, val in enumerate(hls_list):
            hex_color = hex_list[i]
            ax.scatter(hls_list[i][0], hls_list[i][1], hls_list[i][2], c=hex_color,
                       edgecolor=self.outer_color[str(labels[i])], linewidths=2, s=160, label=labels[i])
            ax.grid(color='#AFEEEE', linestyle='--', linewidth=0.5)
        plt.xlabel('HUE')
        plt.ylabel('Lightness')
        plt.show()
        # plt.savefig('Output/' + 'graph.png', orientation='landscape', dpi=200)

    def csv_to_dat(self):
        import pandas as pd
        df = pd.read_csv('D:\\GeoLite2-City-Blocks-IPv4.csv')
        df.to_csv("output.dat")

if __name__ == "__main__":
    Obj = Visualization()
    import get_group_colors
    names = ['BlacksandWhites', 'BluesandIndigos', 'GreensandTurquoise', 'GreensandEarth', 'YellowsandLightBrowns', 'OrangesandRusts', 'RedsandPinks', 'BurgundyandMaroons', 'BrownsandBeiges', 'PurplesandViolets']


    color_range = get_group_colors.get_group_dictionary()
    count = 0
    all_colors = []
    all_labels = []
    for k, v in color_range.items():
        if k == names[9]:
            label = [count]*len(v)
            count += 1
            all_colors = all_colors + v
            all_labels = all_labels + label

    # black = color_range[names[2]]
    black_hls = [Obj.hex2hls(x[1:]) for x in all_colors]
    # Obj.csv_to_dat()
    # Obj.create_graph(black_hls, all_labels, all_colors)