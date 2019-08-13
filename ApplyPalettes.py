from PIL import Image
import pyvips
import numpy as np


class ApplyPalettes:

    def __init__(self, filename):
        self.img_name = filename
        self.dtype_to_format = {
            'uint8': 'uchar',
            'int8': 'char',
            'uint16': 'ushort',
            'int16': 'short',
            'uint32': 'uint',
            'int32': 'int',
            'float32': 'float',
            'float64': 'double',
            'complex64': 'complex',
            'complex128': 'dpcomplex',
        }
        self.format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}
        self.img = Image.open(filename).convert('RGB')
        self.area = self.img.width * self.img.height
        self.colors = self.img.getcolors()
        if len(self.colors) > 5:
            self.img = self.img.convert('P', palette=Image.ADAPTIVE, colors=5)
            self.colors = self.img.getcolors()

    def rgb2hex(self, color):
        return '{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

    def hex2rgb(self, color):
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def np_to_vips(self, img):
        height, width, bands = img.shape
        linear = img.reshape(width * height * bands)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                          self.dtype_to_format[str(img.dtype)])
        return vi

    def vips_to_np(self, img):
        mem_img = img.write_to_memory()

        # then make a numpy array from that buffer object
        np_3d = np.ndarray(buffer=mem_img,
                           dtype=self.format_to_dtype[img.format],
                           shape=[img.height, img.width, img.bands])
        return np_3d

    def replace_colors(self, source_list, destination_list):
        img = self.np_to_vips(np.asarray(self.img))
        new_img = img
        for i, each in enumerate(source_list):
            new_img = (img == each).ifthenelse(destination_list[i], new_img)
        return new_img

    def sort_colors_according_to_width(self, destination_colors_with_wid):
        source_dist_list = [x[0]/self.area for x in self.colors]
        source_color_list = [x[1] for x in self.colors]
        np_source = np.array(source_color_list+source_dist_list).reshape(2, 5)
        np_destination = np.array(destination_colors_with_wid).reshape(2, 5)
        np_source = np_source.reshape(2, 5).T
        np_source = np_source[np_source[:, 1].argsort()]
        np_source = (np_source.T).ravel()

        np_destination = np_destination.reshape(2, 5).T
        np_destination = np_destination[np_destination[:, 1].argsort()]
        np_destination = (np_destination.T).ravel()

        adjusted_source_color_list = list(np_source[:5])
        adjusted_destination_color_list = list(np_destination[:5])

        # color_replaced_img = self.replace_colors(adjusted_source_color_list, adjusted_destination_color_list)
        # color_replaced_img.write_to_file('replace.jpg')
        return adjusted_source_color_list, adjusted_destination_color_list

    def handle_requests(self, selected_dataframe):
        recolored_img = []
        recolored_img.append(self.np_to_vips(np.asarray(self.img)))
        for index, row in selected_dataframe.iterrows():
            destination_list = [(row['r1'], row['g1'], row['b1']), (row['r2'], row['g2'], row['b2']),(row['r3'], row['g3'], row['b3']),(row['r4'], row['g4'], row['b4']),(row['r5'], row['g5'], row['b5']), row['wid 1'], row['wid 2'], row['wid 3'], row['wid 4'], row['wid 5']]
            sorted_source_colors, sorted_destination_colors = self.sort_colors_according_to_width(destination_list)
            recolored_img.append(self.replace_colors(sorted_source_colors, sorted_destination_colors))

        joined_image = pyvips.Image.arrayjoin(recolored_img, across=2, shim=10)
        joined_image = self.vips_to_np(joined_image)
        return joined_image

if __name__ == "__main__":
    Obj = ApplyPalettes("D:\Work\ColorRecommendation\images\\Ambilog.X.00.png")
    selected_colors = [(0,0,0), (255,0,0), (2,0,0), (150,0,0), (100,0,0)]
    selected_wid = [0.4, 0.2, 0.3, 0.05, 0.05]
    # selected_colors = [Obj.rgb2hex(x) for x in selected_colors]
    selected = selected_colors+selected_wid
    Obj.sort_colors_according_to_width(selected)