from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import QDateTime, Qt, QTimer, QThread, pyqtSignal, QRectF, QPoint, QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
from PIL import Image
import numpy as np
import glob
import os
import random


class WidgetGallery(QDialog):
    worker_thread_signal = pyqtSignal(list)
    recolor_thread_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)
        self.init_variables()

        parameter_groupbox = self.crete_parameter_groupbox()
        w = screen.size().width()

        parameter_groupbox.setFixedWidth(w // 5)

        image_groupbox = self.create_input_img_groupbox()
        list_view = self.create_image_listview()
        ordinary_groupbox = self.create_second_img_groupbox()

        main_layout = QGridLayout()
        main_layout.addWidget(parameter_groupbox, 0, 0)
        main_layout.addWidget(list_view, 0, 1)
        main_layout.addWidget(image_groupbox, 0, 2)
        # main_layout.addWidget(seam_groupbox, 1, 1)
        # main_layout.addWidget(ordinary_groupbox, 2, 1)

        self.timer_id = -1
        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())
        self.palettet_img_path = 'palettesimage.png'
        self.palettes_img = np.array(Image.open(self.palettet_img_path).convert('RGB'))
        self.setWindowTitle("ColorFilter")
        self.changeStyle('Fusion')
        self.render_image(self.palettes_img, self.category_scene)
        self.setLayout(main_layout)

        self.worker_process = WorkerThread()
        self.recolor_process = RecolorThread()
        self.connect_signals()

    def init_variables(self):
        self.list_of_files = []
        self.selected_color_category = "light_filter"
        self.selected_color_class = 2
        self.y_value = 170
        self.selected_dataframe = None
        self.named_colors = ['BlacksandWhites', 'BluesandIndigos', 'GreensandTurquoise', 'GreensandEarth', 'YellowsandLightBrowns',
                 'OrangesandRusts', 'RedsandPinks', 'BurgundyandMaroons', 'BrownsandBeiges', 'PurplesandViolets']

    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))

    def connect_signals(self):
        self.file_open_btn.clicked.connect(self.openfile_dialog)
        self.worker_process.worker_thread_signal.connect(self.worker_thread_complete)
        self.recolor_process.recolor_thread_signal.connect(self.recolor_thread_complete)
        self.colorful_rbtn.clicked.connect(self.call_sorting_algorithm)
        self.light_rbtn.clicked.connect(self.call_sorting_algorithm)
        self.dark_rbtn.clicked.connect(self.call_sorting_algorithm)
        self.tone_rbtn.clicked.connect(self.call_sorting_algorithm)
        self.muted_rbtn.clicked.connect(self.call_sorting_algorithm)
        self.next_btn.clicked.connect(self.recolor_design_next)
        self.ImageShowerList.currentItemChanged.connect(self.item_selection_changed)

    def create_image_listview(self):
        self.ImageShower = QGroupBox("Loaded Images")
        self.ImageShowerList = QListWidget()
        self.ImageShowerList.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout()
        layout.addWidget(self.ImageShowerList)
        #
        # layout.addStretch(1)
        self.ImageShower.setLayout(layout)
        w = app.primaryScreen().size().width()
        self.ImageShower.setFixedWidth(int(w*.075))
        return self.ImageShower

    def item_selection_changed(self, item):
        if item != None:
            location = int(item.text())
            self.current_img_path = self.list_of_files[location]
            self.current_img = np.asarray(Image.open(self.current_img_path).convert('RGB'))
            self.render_image(self.current_img, self.input_scene)

    def crete_parameter_groupbox(self):
        parameter_groupbox = QGroupBox("Parameters")

        self.file_open_btn = QPushButton('Open Folder')

        filter_group_box = QGroupBox("Apply Filters")
        vbox = QVBoxLayout()
        self.colorful_rbtn = QRadioButton("Colorful")
        self.light_rbtn = QRadioButton("Light")
        self.dark_rbtn = QRadioButton("Dark")
        self.tone_rbtn = QRadioButton("Tone on Tone")
        self.muted_rbtn = QRadioButton("Muted")

        self.selected_color_class_lbl = QLabel()
        self.selected_color_category_lbl = QLabel()
        self.selected_color_class_lbl.setText(str(self.selected_color_class))
        self.selected_color_category_lbl.setText(self.selected_color_category)
        self.light_rbtn.setChecked(True)
        vbox.addWidget(self.colorful_rbtn)
        vbox.addWidget(self.light_rbtn)
        vbox.addWidget(self.dark_rbtn)
        vbox.addWidget(self.tone_rbtn)
        vbox.addWidget(self.muted_rbtn)
        filter_group_box.setLayout(vbox)

        vbox_layout = QGridLayout()
        self.category_scene = QGraphicsScene()
        self.category_view = QGraphicsView(self.category_scene)
        self.category_view.mousePressEvent = self.get_coordinate

        self.recolor_btn = QPushButton('Recolor')
        self.next_btn = QPushButton('Next')

        vbox_layout.addWidget(self.file_open_btn)
        vbox_layout.addWidget(self.file_open_btn, 0 , 0, 1, 0)
        vbox_layout.addWidget(self.category_view, 1, 0)
        vbox_layout.addWidget(filter_group_box, 2, 0)
        vbox_layout.addWidget(self.recolor_btn, 3, 0)
        vbox_layout.addWidget(self.next_btn, 4, 0)
        vbox_layout.addWidget(self.selected_color_class_lbl, 5, 0)
        vbox_layout.addWidget(self.selected_color_category_lbl,6, 0)
        parameter_groupbox.setLayout(vbox_layout)

        return parameter_groupbox

    def create_input_img_groupbox(self):
        img_groupbox = QGroupBox('Image')
        vbox_layout = QVBoxLayout()

        self.input_scene = QGraphicsScene()
        self.input_view = QGraphicsView(self.input_scene)
        # self.input_view.mousePressEvent = self.get_coordinate


        vbox_layout.addWidget(self.input_view)
        img_groupbox.setLayout(vbox_layout)
        return img_groupbox

    def get_coordinate(self, event):
        pos = self.category_view.mapToScene(event.pos())
        x = int(pos.x())
        y = int(pos.y())
        self.y_value = y
        # print(x, y)
        if (0<x<self.palettes_img.shape[1]) and (0 < y < self.palettes_img.shape[0]):
            self.call_sorting_algorithm()

    def render_img_toQlbl(self, img, place_holder):
        pixmap = QPixmap(self.numpy_to_pixmap(img))
        # w = self.main_img_group_box.width()
        # mainsmaller_pixmap = pixmap.scaled(w-5, h-10,Qt.KeepAspectRatio, Qt.FastTransformation)
        place_holder.setPixmap(pixmap)

    def numpy_to_pixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def create_second_img_groupbox(self):
        img_groupbox = QGroupBox('Ordinary')
        vbox_layout = QVBoxLayout()

        self.ordinary_scene = QGraphicsScene()
        self.ordinary_view = QGraphicsView(self.ordinary_scene)
        # self.ordinary_view.mousePressEvent = self.get_coordinate

        vbox_layout.addWidget(self.ordinary_view)
        img_groupbox.setLayout(vbox_layout)
        return img_groupbox

    def create_seam_img_groupbox(self):
        img_groupbox = QGroupBox('Seam-Carving')
        vbox_layout = QVBoxLayout()

        self.seam_scene = QGraphicsScene()
        self.seam_view = QGraphicsView(self.seam_scene)
        # self.ordinary_view.mousePressEvent = self.get_coordinate

        vbox_layout.addWidget(self.seam_view)
        img_groupbox.setLayout(vbox_layout)
        return img_groupbox

    def openfile_dialog(self):
        # filename = QFileDialog.getOpenFileName(self, "Select Image")
        # if filename[0] != '':
        #     print(filename)
        #     self.palettet_img_path = filename[0]
        #     self.palettes_img = np.asarray(Image.open(self.palettet_img_path).convert('RGB'))
        #     self.render_image(self.palettes_img, self.input_scene)
        #
        self.ImageShowerList.blockSignals(True)
        folder_name = QFileDialog.getExistingDirectory(self, "Select Directory")
        files = self.list_files_inside_folder(folder_name)
        if files:
            self.ImageShowerList.blockSignals(True)
            self.list_of_files.clear()
            self.ImageShowerList.blockSignals(False)
            self.list_of_files = files
            self.current_img_path = files[0]
            self.current_img = np.asarray(Image.open(self.current_img_path).convert('RGB'))
            self.render_image(self.current_img, self.input_scene)

            self.ImageShowerList.clear()
            for i, each in enumerate(files):
                itm = QListWidgetItem(str(i))

                itm.setIcon(QIcon(each))

                self.ImageShowerList.blockSignals(True)
                self.ImageShowerList.addItem(itm)
                self.ImageShowerList.blockSignals(False)
            self.ImageShowerList.setIconSize(QSize(250, 250))

    def list_files_inside_folder(self, path_to_folder) -> list:
        files = []
        for ext in ('*.png', '*.jpg', '*JPG', '*JPEG'):
            files.extend(glob.glob(os.path.join(path_to_folder, ext)))

        list_of_files = []
        for each in files:
            each = each.replace("/", "\\")
            list_of_files.append(each)
        return files

    def convert_from_pil_to_numpy(self, img):
        return np.asarray(img)

    def numpy_to_pixmap(self, img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def render_image(self, img, place_holder, flag=False):
        place_holder.clear()
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        if flag:
            print(place_holder.height())
            pixmap = pixmap.scaled(place_holder.width(), place_holder.height(), Qt.KeepAspectRatio,Qt.FastTransformation)
        place_holder.addPixmap(pixmap)

    def value_change(self):
        if self.timer_id != -1:
            self.killTimer(self.timer_id)

        self.timer_id = self.startTimer(500)

    def timerEvent(self, event):
        self.killTimer(self.timer_id)
        self.timer_id = -1
        self.use_stepped_mixply()

    def get_filter_string(self):

        if self.colorful_rbtn.isChecked():
            return "colorful_filter"

        if self.light_rbtn.isChecked():
            return "light_filter"

        if self.dark_rbtn.isChecked():
            return "dark_filter"

        if self.tone_rbtn.isChecked():
            return "tone_filters"

        if self.muted_rbtn.isChecked():
            return "muted_filter"

    def define_category_from_position(self, y_value) -> int:
        if y_value < 75:
            return 0
        elif 75<y_value<155:
            return 1
        elif 155<y_value<239:
            return 2
        elif 239<y_value<323:
            return 3
        elif 323<y_value<406:
            return 4
        elif 406<y_value<490:
            return 5
        elif 490<y_value<576:
            return 6
        elif 576<y_value<655:
            return 7
        elif 655<y_value<740:
            return 8
        elif y_value>740:
            return 9

    def set_status_to_thread(self):
        self.worker_process.color_category = self.get_filter_string()
        self.worker_process.color_class = self.define_category_from_position(self.y_value)

    def set_status_to_recolor_thread(self):
        self.recolor_process.image_to_recolor = self.current_img_path
        self.recolor_process.color_dataframe = self.selected_dataframe.head(3)

    def recolor_design_next(self):
        self.recolor_process.image_to_recolor = self.current_img_path
        print(self.length_of_dataframe, len(self.selected_dataframe.index))
        random_num = random.sample(range(self.length_of_dataframe-1), 3)
        self.recolor_process.color_dataframe = self.selected_dataframe.iloc[[random_num[0], random_num[1], random_num[2]]]
        self.recolor_process.start()

    def call_sorting_algorithm(self):
        self.selected_color_category = self.get_filter_string()
        self.selected_color_class = self.define_category_from_position(self.y_value)
        self.selected_color_category_lbl.setText(self.selected_color_category)
        self.selected_color_class_lbl.setText(self.named_colors[self.selected_color_class])
        self.set_status_to_thread()
        self.worker_process.start()
        # from SortingDatabase import SortingDatabase
        # sorting_obj = SortingDatabase(filename="3000ColorValues.xlsx", c=tuple(self.selected_color_value), filter_string=self.get_filter_string())
        # sorting_obj.handle_requests()

    @QtCore.pyqtSlot(list)
    def worker_thread_complete(self, returned_list):
        print('color selection finished... Now applying it to a design')
        self.selected_dataframe = returned_list[0]
        self.length_of_dataframe = returned_list[1]
        self.set_status_to_recolor_thread()
        self.recolor_process.start()

    @QtCore.pyqtSlot(list)
    def recolor_thread_complete(self, returned_list):
        print('Recoloring Finished')
        self.render_image(returned_list[0], self.input_scene, True)
        # recolored_img = returned_list[0]

class WorkerThread(QThread):

    worker_thread_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(WorkerThread, self).__init__(parent)
        self.color_class = 2
        self.color_category ='light_filter'

    @QtCore.pyqtSlot()
    def run(self):
        return_list = []
        from SortingDatabase import SortingDatabase
        sorting_obj = SortingDatabase(filename="filters_aaded_2.xlsx", palette_filter=self.color_category, color_category=self.color_class)
        refined, count = sorting_obj.choose_from_two_category(self.color_class, self.color_category)
        return_list.append(refined)
        return_list.append(count)
        self.worker_thread_signal.emit(return_list)

class RecolorThread(QThread):

    recolor_thread_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super(RecolorThread, self).__init__(parent)
        self.image_to_recolor = None
        self.color_dataframe =None

    @QtCore.pyqtSlot()
    def run(self):
        return_list = []
        from ApplyPalettes import ApplyPalettes
        print('recolor thread started')
        print(self.image_to_recolor)
        recolorObj = ApplyPalettes(self.image_to_recolor)
        recolored_img = recolorObj.handle_requests(self.color_dataframe)
        return_list.append(recolored_img)
        self.recolor_thread_signal.emit(return_list)

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    h = screen.size().height()
    w = screen.size().width()
    gallery = WidgetGallery()
    gallery.setFixedSize(int(w - w * 0.025), int(h - h * 0.1))
    gallery.setWindowFlag(QtCore.Qt.WindowMinMaxButtonsHint)
    gallery.show()
    sys.exit(app.exec_())