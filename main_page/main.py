import csv
import math
import shutil
import tkinter as tk
import tkinter
from tkinter import ttk
import datetime

from tkinter import *

from PIL import Image, ImageTk
from tkinter import messagebox
import pandas as pd


from tkinter import ttk
from tkinter import filedialog
from tkinter.ttk import Label


import os
import sys

import cv2
import skimage.io
from matplotlib.ticker import MaxNLocator, MultipleLocator

from mrcnn.config import Config
from datetime import datetime

import mrcnn.model as modellib
from mrcnn import visualize

from matplotlib import pyplot as plt
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)
ScaleFactor=ctypes.windll.shcore.GetScaleFactorForDevice(0)

from ttkbootstrap import Style
from ttkbootstrap.constants import *
# Some basic setup:
# Setup detectron2 logger
import xml.etree.ElementTree as ET
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
import os, json, cv2, random
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer




LARGEFONT = ("Verdana", 35)
n = 1
listx = []
number = 0
process = 0
number_whole_mask=1
listprocessed_addr = []
df_selected_before_start = pd.DataFrame(
                columns=["Picture_ID", "Number of grain", "Max area", "Min area", "Average area", "Max ratio",
                         "Min ratio", "Average ratio", "Max angle", "Min angle", "Average angle"])
angles = []
ratios = []
sizes = []
heights = []
widths = []
ratio_mean = []
angle_mean = []
area_mean = []
unit = "nm"
min_x=0
min_y=0
max_x=0
max_y=0

ten_angles = [[]*1 for _ in range(10)]
ten_times = []
class tkinterApp(tk.Tk):

    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        self.tk.call('tk', 'scaling', ScaleFactor / 75)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Main):
            frame = F(container, self)

            # initializing frame of that object from
            # startpage, Main, Feedback respectively with
            # for loop
            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)
        # self.wm_attributes('-transparentcolor', self['bg'])

    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


# first window frame startpage
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        global icon1
        global array
        global icon2
        global array2
        global icon3
        global array3
        global icon4

        def measurement():
            global unit
            unit = "nm"
            window = tk.Toplevel(app)
            window.geometry('300x300')
            window.title("Choose nm/μm")
            window.attributes("-topmost", True)
            window.resizable(0,0)

            l = tk.Label(window, bg="yellow", width=35, text='Select a unit for grain',font=('Arial', 12))
            l.place(relx=0, rely=0.1)

            def print_selection():
                global unit
                l.config(text='You have select ' + r_value.get())
                unit = r_value.get()

            r_value = tk.StringVar()
            r_value.set('nm')
            unit = r_value.get()
            r1 = tk.Radiobutton(window, text='nm', variable=r_value, value='nm', command=print_selection, font=('Arial', 15))
            r1.place(relx=0.4, rely=0.3)

            r2 = tk.Radiobutton(window, text='μm', variable=r_value, value='μm', command=print_selection,font=('Arial', 15))
            r2.place(relx=0.4, rely=0.5)
            def next():
                window.destroy()
                controller.show_frame(Main)

            next_button = tk.Button(window, command=next, text="Next",font=('Arial', 15))
            next_button.place(relx=0.7, rely=0.7)

        title = tk.Label(self, text="Welcome to grain auto-detection system", bd=50, font=('Arial', 40))
        title.pack()

        subtitle = tk.Label(self, text="Easy and quick!", font=('Arial', 20))
        subtitle.pack()

        words = tk.Label(self, text="Nano-grain  |  Detection  |  Evaluation", bd=30, font=('Arial', 15))
        words.pack()

        icon1 = Image.open('../img/photo.png')
        resizeIcon1 = icon1.resize((60, 60))
        icon1 = ImageTk.PhotoImage(resizeIcon1)
        label1 = tk.Label(self, text="Import a image", image=icon1, compound="top", font=('Arial', 20))
        label1.place(relx=0.1, rely=0.4)

        array = Image.open('../img/array.png')
        resizearray = array.resize((60, 60))
        array = ImageTk.PhotoImage(resizearray)
        array1 = tk.Label(self, text="", image=array, compound="top", font=('Arial', 20))
        array1.place(relx=0.25, rely=0.41)

        icon2 = Image.open('../img/image.png')
        resizeIcon2 = icon2.resize((60, 60))
        icon2 = ImageTk.PhotoImage(resizeIcon2)
        label2 = tk.Label(self, text="Segmentation", image=icon2, compound="top", font=('Arial', 20))
        label2.place(relx=0.35, rely=0.4)

        array2 = tk.Label(self, text="", image=array, compound="top", font=('Arial', 20))
        array2.place(relx=0.485, rely=0.41)

        icon3 = Image.open('../img/idea.png')
        resizeIcon3 = icon3.resize((60, 60))
        icon3 = ImageTk.PhotoImage(resizeIcon3)
        label3 = tk.Label(self, text="Basic information", image=icon3, compound="top", font=('Arial', 20))
        label3.place(relx=0.55, rely=0.4)

        array3 = tk.Label(self, text="", image=array, compound="top", font=('Arial', 20))
        array3.place(relx=0.72, rely=0.41)

        icon4 = Image.open('../img/chart.png')
        resizeIcon4 = icon4.resize((60, 60))
        icon4 = ImageTk.PhotoImage(resizeIcon4)
        label4 = tk.Label(self, text="Save to CSV", image=icon4, compound="top", font=('Arial', 20))
        label4.place(relx=0.8, rely=0.4)

        start_button = tk.Button(self, text="Start", command=measurement, font=('Arial', 20))
        start_button.place(relx=0.48, rely=0.65)

# second window frame Main
class Main(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        global im_checked
        global im_unchecked
        cnt_mask = 1

        listaddr = []
        listid_selected = []
        listsave_selected = []
        listdetail_selected = []

        canvas_1 = tk.Canvas(self, width=760, height=50, bg="white")
        canvas_1.pack()
        canvas_1.place(x=100, y=90)

        canvas_2 = tk.Canvas(self, width=760, height=350, bg="white")
        canvas_2.pack()
        canvas_2.place(x=100, y=140)

        canvas_3 = tk.Canvas(self, width=760, height=50, bg="white")
        canvas_3.pack()
        canvas_3.place(x=1060, y=90)

        canvas_4 = tk.Canvas(self, width=760, height=50, bg="white")
        canvas_4.pack()
        canvas_4.place(x=1060, y=140)

        canvas_5 = tk.Canvas(self, width=760, height=50, bg="white")
        canvas_5.pack()
        canvas_5.place(x=100, y=770)

        canvas_6 = tk.Canvas(self, width=760, height=50, bg="white")
        canvas_6.pack()
        canvas_6.place(x=1060, y=770)

        def reference(degree, angles):
            ten_angles = [[]*1 for _ in range(10)]
            ten_times = []
            global reference_angle

            for angle in angles:
                index = angle / degree
                ten_angles[int(index)].append(angle)
            print(ten_angles)

            for i in range(degree):
                ten_times.append(len(ten_angles[i]))

            max_index = ten_times.index(max(ten_times))
            print(max_index)
            print(sum(ten_angles[max_index]) / len(ten_angles[max_index]))
            reference_angle = sum(ten_angles[max_index]) / len(ten_angles[max_index])

            referenceAngles = [reference_angle for _ in range(len(angles))]
            angles = (abs(np.array(angles) - np.array(referenceAngles))).astype(int).tolist()
            print(angles)

        def draw_angle(img, reference_angle, mask):
            img = cv2.imread(img, 1)

            if reference_angle == 90:
                x_final = 40
                y_final = 1000
            elif reference_angle == 0:
                x_final = 1000
                y_final = 25
            else:
                x_final = 40 + 1000
                print(x_final)
                y_final = int(25 + 1000 * math.tan(math.radians(reference_angle)))
                print(y_final)

            cv2.line(img, (40, 25), (x_final, y_final), (0, 0, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Reference axis', (10, 15), font, 0.4, (0, 0, 255), 1)

            cv2.imwrite(mask, img)

        def calInfo(imgPath):
            above_zero_count = 0
            below_zero_count = 0

            # calculate the midpoint of 2 points
            def midpoint(ptA, ptB):
                return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

            # load the image, convert it to grayscale, and blur it slightly
            image = cv2.imread(imgPath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            # perform edge detection, then perform a dilation + erosion to
            # close gaps in between object edges
            edged = cv2.Canny(gray, 50, 100)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            # find contours in the edge map
            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # sort the contours from left-to-right and initialize the
            # 'pixels per metric' calibration variable
            if cnts:
                (cnts, _) = contours.sort_contours(cnts)
                pixelsPerMetric = None

            # loop over the contours individually
            for c in cnts:
                # if the contour is not sufficiently large, ignore it
                if cv2.contourArea(c) < 1:
                    continue

                # compute the rotated bounding box of the contour
                orig = image.copy()
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")

                # order the points in the contour such that they appear
                # in top-left, top-right, bottom-right, and bottom-left
                # order, then draw the outline of the rotated bounding
                # box
                box = perspective.order_points(box)
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

                # loop over the original points and draw them
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

                # unpack the ordered bounding box, then compute the midpoint
                # between the top-left and top-right coordinates, followed by
                # the midpoint between bottom-left and bottom-right coordinates
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                # compute the midpoint between the top-left and top-right points,
                # followed by the midpoint between the top-righ and bottom-right
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)

                # draw the midpoints on the image
                cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

                # # draw lines between the midpoints
                # cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                #          (255, 0, 255), 2)
                # cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                #          (255, 0, 255), 2)

                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                if (dB < dA):
                    mid = dB
                    dB = dA
                    dA = mid

                    x1 = tltrX
                    x2 = blbrX
                    y1 = tltrY
                    y2 = blbrY
                else:
                    x1 = tlblX
                    x2 = trbrX
                    y1 = tlblY
                    y2 = trbrY

                if int(int(x1) - int(x2)) == 0:
                    Cobb = 90
                if int(int(y1) - int(y2)) == 0:
                    Cobb = 0

                if (int(int(x1) - int(x2)) != 0 and int(int(y1) - int(y2)) != 0):
                    k1 = (y2 - y1) / (x2 - x1)
                    k2 = 0

                    x = np.array([1, k1])
                    y = np.array([1, k2])

                    Lx = np.sqrt(x.dot(x))
                    Ly = np.sqrt(y.dot(y))

                    Cobb = int((np.arccos(x.dot(y) / (float(Lx * Ly))) * 180 / np.pi) + 0.5)

                    if k1 > 0:
                        above_zero_count += 1
                        Cobb = 90 - Cobb
                    elif k1 < 0:
                        below_zero_count += 1

                cv2.line(orig, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 0, 255), 2)
                # if the pixels per metric has not been initialized, then
                # compute it as the ratio of pixels to supplied metric
                # (in this case, inches)
                if pixelsPerMetric is None:
                    pixelsPerMetric = (500 * 2048) / (256 * 924)

                # compute the size of the object
                dimA = dA * pixelsPerMetric

                dimB = dB * pixelsPerMetric

                if dimA == dimB:
                    Cobb = 90
                angles.append(Cobb)
                heights.append(dimB)
                widths.append(dimA)
                if dimA == 0:
                    dimA = 1
                ratios.append(dimB / dimA)
                sizes.append(dimA * dimB)

        def detail(count):
            global angles
            global ratios
            global sizes
            global widths
            global heights
            global df_selected_before_start
            global ratio_mean
            global angle_mean
            global area_mean
            angles = []
            ratios = []
            sizes = []
            heights = []
            widths = []
            img_path = r'C:\Users\Comeonjkf\Desktop\Python GUI\main_page\save_mask'
            img_list = os.listdir(img_path)
            for img in img_list:
                img = os.path.join(r"save_mask\\", img)
                calInfo(img)

            reference(10, angles)

            # The mean, max, min values of angle
            angleMean = np.mean(angles)
            angleMax = max(angles)
            angleMin = min(angles)
            angle_mean.append(angleMean)

            # The mean, max, min values of ratio
            ratioMean = np.mean(ratios)
            ratioMax = max(ratios)
            ratioMin = min(ratios)
            ratio_mean.append(ratioMean)
            # The mean, max, min values of size(area)
            sizeMean = np.mean(sizes)
            sizeMax = max(sizes)
            sizeMin = min(sizes)
            area_mean.append(sizeMean)

            objectNum = len(angles)
            new = pd.DataFrame({'Picture_ID': listid_selected[count],
                                'Number of grain': objectNum,
                                'Max area': format(sizeMax,'.1f'),
                                'Min area': format(sizeMin,'.1f'),
                                'Average area': format(sizeMean,'.1f'),
                                'Max ratio': format(ratioMax,'.1f'),
                                'Min ratio': format(ratioMin,'.1f'),
                                'Average ratio': format(ratioMean,'.1f'),
                                'Max angle': format(angleMax,'.1f'),
                                'Min angle': format(angleMin,'.1f'),
                                'Average angle': format(angleMean,'.1f')
                                },
                               index=[1])
            df_selected_before_start = df_selected_before_start.append(new, ignore_index=True)

            # The code of "Long axis-short axis ratio histogram"
            plt.figure()
            plt.tick_params(labelsize=10)
            ax = plt.gca()
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ratios = [round(a)for a in ratios]
            for i in range(0,len(ratios)):
                if ratios[i]>15:
                    ratios[i]=15
            num, bins, patches = plt.hist(ratios, bins=np.arange(min(ratios) - 0.5, max(ratios) + 1.5),
                                          label="Length/width", edgecolor='black')
            plt.xticks(np.arange(min(ratios), max(ratios) + 1, 1))
            plt.xticks(fontsize=11)
            for num, bin in zip(num, bins):
                if num != 0:
                    plt.annotate(int(num), xy=(bin, num), xytext=(bin + 0.273, num + 0.03))

            plt.xlabel("Long axis-short axis ratio value", fontsize=13)
            plt.ylabel("Grains number", fontsize=13)
            plt.legend(loc="best")

            plt.savefig("save_graph/"+listid_selected[count]+"_RatioHist")

            # The code of "Deflection angle histogram"
            plt.figure()
            ax = plt.gca()
            plt.tick_params(labelsize=10)
            x_major_locator = MultipleLocator(5)
            ax.xaxis.set_major_locator(x_major_locator)
            plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
            num, bins, patches = plt.hist(angles, bins=np.arange(-0.5, 91.5), label="Angel Bias", edgecolor='black')
            plt.xlabel("Angle value (Degree)", fontsize=13)
            plt.ylabel("Grains number", fontsize=13)
            plt.legend(loc='best')
            for num, bin in zip(num, bins):
                if num != 0:
                    plt.annotate(int(num), xy=(bin, num), xytext=(bin - 0.3, num + 0.05))

            plt.savefig("save_graph/"+listid_selected[count]+"_AngleHist")

            from PIL import Image
            from PIL import ImageDraw

            pilim = Image.open("save_graph/" + listid_selected[count] + "_AngleHist.png")
            draw = ImageDraw.Draw(pilim)
            draw.rectangle([564.41, 434.53, 640, 480], fill=(255, 255, 255, 255))
            del draw
            draw = ImageDraw.Draw(pilim)
            draw.rectangle([0, 434.53, 95.75, 480], fill=(255, 255, 255, 255))
            del draw
            pilim.save('save_graph/'+listid_selected[count]+'_AngleHist.png')

            # The code of "Long axis and short axis scatter diagram"
            plt.figure()
            plt.xticks(range(0, int(max(widths)), 25))
            plt.yticks(range(0, int(max(heights)), 25))
            plt.scatter(widths, heights, label="Long axis & Short axis")
            plt.xlabel("Short axis "+"("+unit+")", fontsize=13)
            plt.ylabel("Long axis "+"("+unit+")", fontsize=13)
            plt.grid()
            plt.savefig("save_graph/"+listid_selected[count]+"_RatioScatter")

        def mask(address,count_of_selected):
            global number_whole_mask
            ROOT_DIR = os.getcwd()# Root directory of the project
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
            cfg.DATASETS.TRAIN = ("grain_train",)
            cfg.DATASETS.TEST = ()
            cfg.DATALOADER.NUM_WORKERS = 2
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
            cfg.SOLVER.IMS_PER_BATCH = 2
            cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
            cfg.SOLVER.MAX_ITER = 2000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
            cfg.SOLVER.STEPS = []  # do not decay learning rate
            cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
            # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

            cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                             "C:/Users/Comeonjkf/Desktop/Python GUI/main_page/model_final.pth")  # path to the model we just trained
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
            predictor = DefaultPredictor(cfg)


            # The "cv_imread" method enables the software to read the address which contains Chinese characters
            def cv_imread(filePath):
                import numpy as np
                cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
                return cv_img

            im = cv_imread(address)
            im = cv2.resize(im, (256, 256))
            outputs = predictor(im)
            result = outputs["instances"].to("cpu").pred_masks

            import numpy as np
            result = result + 0
            result = np.uint8(result)

            def devideMask(mask):
                for i in range(result.shape[0]):
                    save_path = 'save_mask/' + str(i) + '.png'

                    for j in range(result.shape[1]):
                        for q in range(result.shape[2]):
                            if mask[i, j, q] == 1:
                                mask[i, j, q] = 255

                    cv2.imwrite(save_path, mask[i, :, :])

            devideMask(result)

            detail(count_of_selected)

            mask_path = r"C:\Users\Comeonjkf\Desktop\Python GUI\main_page\save_mask"
            mask_set = os.listdir(mask_path)
            img_set = []
            maskAll = mask = np.zeros((256, 256, 3), dtype=np.uint8)

            i = 0
            for img in mask_set:
                img111 = cv2.imread(os.path.join('save_mask/', img))
                maskAll = maskAll + img111

                i = i + 1

            maskcnt=1
            savead=os.listdir(os.path.join(ROOT_DIR, "save_whole_mask"))
            for item in savead:
                maskcnt = maskcnt+1
            saveaddr = 'save_whole_mask/'+str(maskcnt)+'.png'
            cv2.imwrite(saveaddr,maskAll)
            draw_angle(saveaddr,reference_angle,saveaddr)

            listprocessed_addr.append(saveaddr)
            number_whole_mask=number_whole_mask+1

            #use the "shutil" to empty the "save_mask" folder when the whole mask is generated
            import shutil
            shutil.rmtree(os.path.join(ROOT_DIR, "save_mask"))
            os.mkdir(os.path.join(ROOT_DIR, "save_mask"))
            cv2.waitKey(0)

        def show(root1):
            root = Toplevel(root1)
            root.geometry('150x120')
            root.attributes("-topmost", True)
            percent = StringVar()
            progressbarOne = tk.ttk.Progressbar(root, length=200, mode='determinate', orient=tk.HORIZONTAL)
            progressbarOne.pack(pady=20)
            percentLabel = Label(root, textvariable=percent)
            percentLabel.pack()
            L = Label(root, text="in process")
            L.pack()
            progressbarOne['maximum'] = 100

        # the "resize" method can resize a pil_image object so it will fit into a box of size w_box times h_box, but retain aspect ratio
        def resize(w, h, w_box, h_box, pil_image):
            f1 = 1.0 * w_box / w  # 1.0 forces float division
            f2 = 1.0 * h_box / h
            factor = min([f1, f2])
            # use best down-sizing filter
            width = int(w * factor)
            height = int(h * factor)
            return pil_image.resize((width, height), Image.ANTIALIAS)

        # the "printcoords" method enables the user to choose a directory to upload all the images in it. Each image in this directory will be resized and be added as a row into the "tree".
        def printcoords():
            global n
            global filename
            global listx
            selectDirectory = tk.filedialog.askdirectory(title='Choose the directory.')
            files = os.listdir(selectDirectory)
            for filex in files:
                w, h = Image.open(selectDirectory + '/' + filex).size
                pil_image_resized = resize(w, h, 60, 60, Image.open(selectDirectory + '/' + filex))
                filename = ImageTk.PhotoImage(pil_image_resized)
                tree.insert("", "end", image=filename, values=("img" + str(n), selectDirectory + '/' + filex),
                            tags="unchecked")
                listx.append(filename)
                n = n + 1

        # the "printcoords2" method enables the user to choose a image to upload. This image will be resized and be added as a row into the "tree".
        def printcoords2():
            global n
            global filename
            global listx
            selectFileNames = tk.filedialog.askopenfilenames(title='Choose the images.')
            for selectFileName in selectFileNames:
                w, h = Image.open(selectFileName).size
                pil_image_resized = resize(w, h, 60, 60, Image.open(selectFileName))
                filename = ImageTk.PhotoImage(pil_image_resized)
                tree.insert("", "end", image=filename, values=("img" + str(n), selectFileName),
                            tags="unchecked")
                listx.append(filename)
                n = n + 1

        # the "toggleCheck" method is triggered when the user clicks the tags(checkboxes) in "tree". It can transfer the state of tag from "checked" to "unchecked" and from "unchecked" to "checked"
        def toggleCheck(event):
            rowid = tree.identify_row(event.y)
            tag = tree.item(rowid, "tags")[0]
            tags = list(tree.item(rowid, "tags"))
            tags.remove(tag)
            tree.item(rowid, tags=tags)
            if tag == "checked":
                tree.item(rowid, tags="unchecked")
            else:
                tree.item(rowid, tags="checked")

        # the "toggleCheck_right" method is triggered when the user clicks the tags(checkboxes) in "tree2". It can transfer the state of tag from "checked" to "unchecked" and from "unchecked" to "checked"
        def toggleCheck_right(event):
            rowid = tree2.identify_row(event.y)
            tag = tree2.item(rowid, "tags")[0]
            tags = list(tree2.item(rowid, "tags"))
            tags.remove(tag)
            tree2.item(rowid, tags=tags)
            if tag == "checked":
                tree2.item(rowid, tags="unchecked")
            else:
                tree2.item(rowid, tags="checked")

        def getaddr():
            global number
            global ratio_mean
            global angle_mean
            global area_mean
            number = 0
            ROOT_DIR = os.getcwd()
            shutil.rmtree(os.path.join(ROOT_DIR, "save_graph"))
            os.mkdir(os.path.join(ROOT_DIR, "save_graph"))
            shutil.rmtree(os.path.join(ROOT_DIR, "crop_img"))
            os.mkdir(os.path.join(ROOT_DIR, "crop_img"))
            listaddr.clear()
            listid_selected.clear()
            ratio_mean = []
            area_mean=[]
            angle_mean=[]

            check = 0
            for item in tree.get_children():
                if tree.item(item, "tags")[0] == "checked":
                    check = 1
            if check == 0:
                # LEVEL_WINDOW.attributes("-topmost", False)
                msg = 'Please choose at least one item!'
                messagebox.showwarning('Warning', msg)
                # LEVEL_WINDOW.attributes("-topmost", True)

            else:
                for item in tree.get_children():
                    if tree.item(item, "tags")[0] == "checked":
                        listaddr.append(tree.item(item, "values")[1])
                        listid_selected.append(tree.item(item, "values")[0])
                        number=number+1
                global process
                process = 0
                # show(self)
                showprocessed_image(self)


        def showprocessed_image(root1):
            root = Toplevel(root1)
            root.geometry('150x120')
            root.attributes("-topmost", True)
            percent = StringVar()
            progressbarOne = tk.ttk.Progressbar(root, length=200, mode='determinate', orient=tk.HORIZONTAL)
            progressbarOne.pack(pady=20)
            percentLabel = Label(root, textvariable=percent)
            percentLabel.pack()
            L = Label(root, text="in process")
            L.pack()
            progressbarOne['maximum'] = 100

            global listprocessed_addr
            global df_selected_before_start
            global ratio_mean
            global angle_mean
            global area_mean
            listprocessed_addr=[]
            df_selected_before_start = pd.DataFrame(
                columns=["Picture_ID", "Number of grain", "Max area", "Min area", "Average area", "Max ratio",
                         "Min ratio", "Average ratio", "Max angle", "Min angle", "Average angle"])
            j=0
            global process
            progressbarOne['value'] = process
            percent.set(str(int(process)) + "%")
            root.update()
            for item in listaddr:
                mask(item,j)
                j=j+1

                process += 100/number
                progressbarOne['value'] = process
                percent.set(str(int(process)) + "%")
                root.update()

                if progressbarOne['value'] >= progressbarOne['maximum']:
                    L['text'] = "The Process is done!"
                    root.destroy()
            cleantree(tree2)
            i = 0
            for item in listprocessed_addr:
                w, h = Image.open(item).size
                pil_image_resized = resize(w, h, 60, 60, Image.open(item))
                filename = ImageTk.PhotoImage(pil_image_resized)
                tree2.insert("", "end", image=filename, values=(listid_selected[i], os.path.realpath(item)), tags="unchecked")
                listx.append(filename)
                i = i + 1
            df_selected_before_start.to_csv('grain_test.csv', index=0)

            #generate there Analysis images:

            # The code of "Long axis and short axis scatter diagram"
            plt.figure()
            # yvalue = df_selected_before_start["Average ratio"].values
            # print(yvalue)
            # ratio_mean = [round(a) for a in ratio_mean]
            length= len(ratio_mean)
            print(ratio_mean)
            xvalue= np.arange(1,length+1,1)
            print(xvalue)
            plt.xticks(range(0, length+1, 1))
            # plt.yticks(np.arange(min(ratio_mean), max(ratio_mean) + 1, 0.5))
            print(min(ratio_mean))
            print(max(ratio_mean) + 1)
            plt.scatter(xvalue, ratio_mean, label="Average Long axis & Short axis")
            # for a, b in zip(xvalue, ratio_mean):
            #     plt.text(a, b, round(b,2))
            plt.xlabel("Number of img", fontsize=13)
            plt.ylabel("Average long-short axis ratio value", fontsize=13)
            plt.grid()

            plt.savefig("save_analysis/Average_long-short_axis_ratio")

            # The code of "Long axis and short axis scatter diagram"
            plt.figure()
            # yvalue = df_selected_before_start["Average ratio"].values
            # print(yvalue)
            # ratio_mean = [round(a) for a in ratio_mean]
            length = len(angle_mean)
            print(angle_mean)
            xvalue = np.arange(1, length + 1, 1)
            print(xvalue)
            plt.xticks(range(0, length + 1, 1))
            # plt.yticks(np.arange(min(angle_mean), max(angle_mean) + 1, 3))
            print(min(angle_mean))
            print(max(angle_mean) + 1)
            plt.scatter(xvalue, angle_mean, label="Average angle value")
            plt.xlabel("Number of img", fontsize=13)
            plt.ylabel("Average deflection angle value (Degree)", fontsize=13)
            plt.grid()

            plt.savefig("save_analysis/Average_angle")


            # The code of "Long axis and short axis scatter diagram"
            plt.figure()
            # yvalue = df_selected_before_start["Average ratio"].values
            # print(yvalue)
            # ratio_mean = [round(a) for a in ratio_mean]
            length = len(area_mean)
            print(area_mean)
            xvalue = np.arange(1, length + 1, 1)
            print(xvalue)
            plt.xticks(range(0, length + 1, 1))
            # plt.yticks(np.arange(min(area_mean), max(area_mean) + 1000, 1000))
            print(min(area_mean))
            print(max(area_mean) + 1000)
            plt.scatter(xvalue, area_mean, label="Average area")
            plt.xlabel("Number of img", fontsize=13)
            plt.ylabel("Average magnetic grain size value ("+unit+"\u00b2)", fontsize=13)
            plt.grid()

            plt.savefig("save_analysis/Average_size")



        def cleantree(tree):
            x = tree.get_children()
            for item in x:
                tree.delete(item)

        def delete():
            global n
            temp = 0
            temp2 = 1
            for item in tree.get_children():
                if tree.item(item, "tags")[0] == "checked":
                    tree.delete(item)
            for item in tree.get_children():
                temp = temp + 1
            n = temp + 1
            for item in tree.get_children():
                tempaddr = tree.item(item, "values")[1]
                tempid = "img" + str(temp2)
                tree.item(item, values=(tempid, tempaddr))
                temp2 = temp2 + 1

        def delete2():
            for item in tree2.get_children():
                if tree2.item(item, "tags")[0] == "checked":
                    tree2.delete(item)

        def select():
            a = entry1.get()
            b = entry2.get()
            a = int(a)
            b = int(b)
            i = 1
            for item in tree.get_children():
                tree.item(item, tags="unchecked")
            for item in tree.get_children():
                if a <= i <= b:
                    tree.item(item, tags="checked")
                i = i + 1

        def select_2():
            for item in tree2.get_children():
                tree2.item(item, tags="checked")

        def select_3():
            for item in tree.get_children():
                tree.item(item, tags="checked")

        def select_4():
            a = entry3.get()
            b = entry4.get()
            a = int(a)
            b = int(b)
            i = 1
            for item in tree2.get_children():
                tree2.item(item, tags="unchecked")
            for item in tree2.get_children():
                if a <= i <= b:
                    tree2.item(item, tags="checked")
                i = i + 1



        def save():
            global LEVEL_WINDOW
            check =0
            for item in tree2.get_children():
                if tree2.item(item, "tags")[0] == "checked":
                    check =1
            if check==0:
                # LEVEL_WINDOW.attributes("-topmost", False)
                msg = 'Please choose at least one item!'
                messagebox.showwarning('Warning', msg)
                # LEVEL_WINDOW.attributes("-topmost", True)

            else:

                df = pd.read_csv('grain_test.csv')
                df_selected = pd.DataFrame(
                    columns=["Picture_ID", "Number of grain", "Max area", "Min area", "Average area", "Max ratio",
                            "Min ratio", "Average ratio", "Max angle", "Min angle", "Average angle"])

                count = 0
                messagebox.showinfo("Hint", "Please select the folder to save CSV")
                folder_path = filedialog.askdirectory()
                filenames = os.listdir(folder_path)

                for i in filenames:
                    for num in range(1, 100):
                        str1 = i
                        str2 = "result_" + str(num) + ".csv"
                        if str1 == str2:
                            count = count + 1
                            break
                filename = "result_" + str(count + 1)
                listsave_selected.clear()
                for item in tree2.get_children():
                    if tree2.item(item, "tags")[0] == "checked":
                        listsave_selected.append(tree2.item(item, "values")[0])
                for i in listsave_selected:
                    temp = df.loc[df['Picture_ID'] == i]
                    df_selected = pd.concat([df_selected, temp])
                df_selected.to_csv(folder_path + "/" + filename + '.csv', index=0)

        def save_mask():
            check = 0
            for item in tree2.get_children():
                if tree2.item(item, "tags")[0] == "checked":
                    check = 1
            if check == 0:
                # LEVEL_WINDOW.attributes("-topmost", False)
                msg = 'Please choose at least one item!'
                messagebox.showwarning('Warning', msg)
                # LEVEL_WINDOW.attributes("-topmost", True)

            else:
                messagebox.showinfo("Hint", "Please select the folder to save masks")
                folder_path = filedialog.askdirectory()
                for item in tree2.get_children():
                    if tree2.item(item, "tags")[0] == "checked":
                        image = Image.open(tree2.item(item, "values")[1])
                        image.save(os.path.join(folder_path, tree2.item(item, "values")[0] + ".png"))
            # messagebox.showinfo("Hint", "Please select the folder to save masks")
            # folder_path = filedialog.askdirectory()
            # for item in tree2.get_children():
            #     if tree2.item(item, "tags")[0] == "checked":
            #         image = Image.open(tree2.item(item, "values")[1])
            #         image.save(os.path.join(folder_path, tree2.item(item, "values")[0] + ".png"))
        def Analysis():


            data_graphy1 = "save_analysis/Average_angle.png"
            data_graphy2 = "save_analysis/Average_long-short_axis_ratio.png"
            data_graphy3 = "save_analysis/Average_size.png"

            global data_photo1
            global data_photo2
            global data_photo3


            window = tk.Toplevel(app)

            MIN_WIDTH = 800
            sole_rectangle = None
            size = "media"
            window.title("Summary")
            screenwidth = window.winfo_screenwidth()
            screenheight = window.winfo_screenheight()
            window.geometry('{}x{}'.format(screenwidth, screenheight))
            window.attributes("-topmost", True)
            MIN_WIDTH = 800
            sole_rectangle = None
            size = "media"

            filename = None




            def middle_windows(window, width, height, reset=False):
                screenwidth = window.winfo_screenwidth()
                screenheight = window.winfo_screenheight()
                x = int((screenwidth - width) / 2)
                y = int((screenheight - height) / 2)
                window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
                if not reset:
                    window.resizable(0, 0)
                window.update()

            def now(fmt='%Y%m%d%H%M%S'):
                string = datetime.datetime.now().strftime(fmt)
                return string



            data_img1 = Image.open(data_graphy1)
            wd1, hd1 = data_img1.size
            resize_data_img1 = resize(wd1, hd1, 600, 600, data_img1)
            data_photo1 = ImageTk.PhotoImage(resize_data_img1)

            wd1_new, hd1_new = data_img1.size
            resize_data_img1_new = resize(wd1_new, hd1_new, 1000, 1000, data_img1)
            data_photo1_new = ImageTk.PhotoImage(resize_data_img1_new)

            data_img2 = Image.open(data_graphy2)
            wd2, hd2 = data_img2.size
            resize_data_img2 = resize(wd2, hd2, 600, 600, data_img2)
            data_photo2 = ImageTk.PhotoImage(resize_data_img2)

            wd2_new, hd2_new = data_img2.size
            resize_data_img2_new = resize(wd2_new, hd2_new, 1000, 1000, data_img2)
            data_photo2_new = ImageTk.PhotoImage(resize_data_img2_new)

            data_img3 = Image.open(data_graphy3)
            wd3, hd3 = data_img3.size
            resize_data_img3 = resize(wd3, hd3, 600, 600, data_img3)
            data_photo3 = ImageTk.PhotoImage(resize_data_img3)

            wd3_new, hd3_new = data_img3.size
            resize_data_img3_new = resize(wd3_new, hd3_new, 1000, 1000, data_img3)
            data_photo3_new = ImageTk.PhotoImage(resize_data_img3_new)

            # Length-width ratio histogram
            label_data1 = tkinter.Label(window, text="Average long-short axis ratio value scatter diagram: ", font=(None, 14, "bold"),
                                        height=2)
            label_data1.pack()
            label_data1.place(relx=0.03, rely=0.22)

            data_lable1 = tkinter.Label(window, image=data_photo2)
            data_lable1.pack()
            data_lable1.place(relx=0.02, rely=0.25)

            def graph1(event):  # the event when click the first graph
                global LEVEL_WINDOW1

                LEVEL_WINDOW1 = tk.Toplevel(window)
                LEVEL_WINDOW1.attributes('-topmost', True)
                middle_windows(LEVEL_WINDOW1, 1000, 1000, reset=True)

                photo_canvas = tk.Canvas(LEVEL_WINDOW1, width=1000, height=1000)
                photo_canvas.pack()
                photo_canvas.place(x=0, y=0)
                photo_canvas.create_image(0, 0, image=data_photo2_new, anchor='nw')

            data_lable1.bind('<Button-1>', graph1)

            # Deflection angle histogram
            label_data2 = tkinter.Label(window, text="Average deflection angle value scatter diagram:", font=(None, 14, "bold"), height=2)
            label_data2.pack()
            label_data2.place(relx=0.36, rely=0.22)

            data_lable2 = tkinter.Label(window, image=data_photo1)
            data_lable2.pack()
            data_lable2.place(relx=0.34, rely=0.25)

            def graph2(event):  # the event when click the second graph
                global LEVEL_WINDOW2

                LEVEL_WINDOW2 = tk.Toplevel(window)
                LEVEL_WINDOW2.attributes('-topmost', True)
                middle_windows(LEVEL_WINDOW2, 1000, 1000, reset=True)

                photo_canvas = tk.Canvas(LEVEL_WINDOW2, width=1000, height=1000)
                photo_canvas.pack()
                photo_canvas.place(x=0, y=0)
                photo_canvas.create_image(0, 0, image=data_photo1_new, anchor='nw')

            data_lable2.bind('<Button-1>', graph2)

            # Length and width scatter diagram
            label_data3 = tkinter.Label(window, text="Average magnetic grain size value scatter diagram:",
                                        font=(None, 14, "bold"), height=2)
            label_data3.pack()
            label_data3.place(relx=0.67, rely=0.22)

            data_lable3 = tkinter.Label(window, image=data_photo3)
            data_lable3.pack()
            data_lable3.place(relx=0.66, rely=0.25)

            def graph3(event):  # the event when click the third graph
                global LEVEL_WINDOW3

                LEVEL_WINDOW3 = tk.Toplevel(window)
                LEVEL_WINDOW3.attributes('-topmost', True)
                middle_windows(LEVEL_WINDOW3, 1000, 1000, reset=True)

                photo_canvas = tk.Canvas(LEVEL_WINDOW3, width=1000, height=1000)
                photo_canvas.pack()
                photo_canvas.place(x=0, y=0)
                photo_canvas.create_image(0, 0, image=data_photo3_new, anchor='nw')

            data_lable3.bind('<Button-1>', graph3)



        def newWindow():
            global detailImg_path
            global origin_image
            listdetail_selected.clear()
            for item in tree2.get_children():
                if tree2.item(item, "tags")[0] == "checked":
                    listdetail_selected.append(tree2.item(item, "values")[1])
                    listdetail_selected.append(tree2.item(item, "values")[0])
            for item in tree.get_children():
                if tree.item(item,"values")[0]==listdetail_selected[1].split("_")[0]:
                    i=listdetail_selected[1]
                    detailImg_path = listdetail_selected[0]
                    origin_image = tree.item(item,"values")[1]

            processed_image = detailImg_path

            data_graphy1 = "save_graph/"+i+"_AngleHist.png"
            data_graphy2 = "save_graph/"+i+"_RatioHist.png"
            data_graphy3 = "save_graph/"+i+"_RatioScatter.png"


            global photo0
            global photo_m
            global data_photo1
            global data_photo2
            global data_photo3
            global photor
            global sole_rectangle



            window = tk.Toplevel(app)



            MIN_WIDTH = 800
            sole_rectangle = None
            size = "media"
            window.title("detail image")
            screenwidth = window.winfo_screenwidth()
            screenheight = window.winfo_screenheight()
            window.geometry('{}x{}'.format(screenwidth, screenheight))
            window.attributes("-topmost", True)
            MIN_WIDTH = 800
            sole_rectangle = None
            size = "media"

            filename = None

            df2 = pd.read_csv('grain_test.csv')
            temp2 = df2.loc[df2['Picture_ID'] == i]
            AverGrain = temp2.iloc[0,4]
            MinGrain = temp2.iloc[0,3]
            MaxGrain = temp2.iloc[0,2]
            AverRation = temp2.iloc[0,7]
            MinRation = temp2.iloc[0,6]
            MaxRation = temp2.iloc[0,5]
            AverAngle = temp2.iloc[0,10]
            MinAngle = temp2.iloc[0,9]
            MaxAngle = temp2.iloc[0,8]


            datalist = [14, AverGrain, MinGrain, MaxGrain, AverRation, MinRation, MaxRation, AverAngle, MinAngle, MaxAngle]
            excelpath = "result_1.csv"
            corp_times = 1

            def create_folder(folder):
                folder = os.path.abspath(folder)
                if not os.path.exists(folder):
                    try:
                        os.makedirs(folder)
                    except Exception as e:
                        msg = 'Failed to create folder({}), exception is({})'.format(folder, e)

            img = Image.open(processed_image)
            source_path = os.path.realpath(processed_image)
            photo = ImageTk.PhotoImage(Image.open(source_path))

            if img.height > MIN_WIDTH:
                rate = img.width / MIN_WIDTH
                width = int(img.width / rate)
                height = int(img.height / rate)
                crop_img = img.resize((width, height), Image.ANTIALIAS)
                new_name = os.path.abspath('tmp/img_tmp.jpg')
                create_folder(os.path.dirname(new_name))
                crop_img.save(new_name)
                img_path = new_name
                photo = ImageTk.PhotoImage(Image.open(img_path))
                source_path = os.path.realpath(img_path)
            else:
                width = img.width
                height = img.height

            def middle_windows(window, width, height, reset=False):
                screenwidth = window.winfo_screenwidth()
                screenheight = window.winfo_screenheight()
                x = int((screenwidth - width) / 2)
                y = int((screenheight - height) / 2)
                window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
                if not reset:
                    window.resizable(0, 0)
                window.update()

            def now(fmt='%Y%m%d%H%M%S'):
                string = datetime.datetime.now().strftime(fmt)
                return string

            def crop_img(save_path, x_begin, y_begin, x_end, y_end):

                global corp_times
                global min_x
                global min_y
                global max_x
                global max_y
                corp_times = 1
                if x_begin < x_end:
                    min_x = x_begin
                    max_x = x_end
                else:
                    min_x = x_end
                    max_x = x_begin
                if y_begin < y_end:
                    min_y = y_begin
                    max_y = y_end
                else:
                    min_y = y_end
                    max_y = y_begin

                if (min_x == max_x) or (min_y == max_y):
                    LEVEL_WINDOW.attributes("-topmost", False)
                    msg = 'The crop img should be a rectangle, crop failed'
                    messagebox.showwarning('Warning', msg)
                    LEVEL_WINDOW.attributes("-topmost", True)
                else:
                    if os.path.isfile(source_path):
                        LEVEL_WINDOW.attributes("-topmost", False)
                        corp_image = Image.open(source_path)
                        region = corp_image.crop((min_x, min_y, max_x, max_y))
                        create_folder(os.path.dirname(save_path))
                        region.save(save_path)
                        namelist = ['img1' + '-' + 'crop' + str(corp_times)]
                        corp_times += 1
                        datalistnew = datalist = [14, AverGrain, MinGrain, MaxGrain, AverRation, MinRation, MaxRation, AverAngle, MinAngle, MaxAngle]
                        data = namelist + datalistnew
                        save_excel(data)
                        msg = 'The crop is completed and saved in:{}'.format(save_path)
                        messagebox.showinfo('result', msg)
                        LEVEL_WINDOW.attributes("-topmost", True)

                    else:
                        print('Cannot find the file:{}'.format(source_path))

            def save_excel(data_array):
                with open(excelpath, 'a+') as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(data_array)



            def left_mouse_down(event):
                global left_mouse_down_x
                global left_mouse_down_y
                left_mouse_down_x = event.x
                left_mouse_down_y = event.y

            def left_mouse_up(event):
                global corp_times
                global min_x
                global min_y
                global max_x
                global max_y

                global left_mouse_up_x, left_mouse_up_y
                left_mouse_up_x = event.x
                left_mouse_up_y = event.y

                count = 0
                filenames = os.listdir("C:/Users/Comeonjkf/Desktop/Python GUI/main_page/crop_img")

                for i in filenames:
                    for num in range(1, 100):
                        str1 = i
                        str2 = listdetail_selected[1]+'_new_'+str(num)+'.png'
                        if str1 == str2:
                            count = count + 1
                            break
                file = listdetail_selected[1]+'_new_'+str(count + 1)
                crop_img('crop_img/'+file+'.png', left_mouse_down_x, left_mouse_down_y, left_mouse_up_x,
                         left_mouse_up_y)
                w, h = Image.open('crop_img/'+file+'.png').size
                pil_image_resized = resize(w, h, 60, 60, Image.open('crop_img/'+file+'.png'))
                filename = ImageTk.PhotoImage(pil_image_resized)
                tree2.insert("", "end", image=filename, values=(file, os.path.realpath('crop_img/'+file+'.png')),
                             tags="unchecked")
                listx.append(filename)


                global angles
                global ratios
                global sizes
                global widths
                global heights
                angles = []
                ratios = []
                sizes = []
                heights = []
                widths = []

                ROOT_DIR = os.getcwd()
                cfg = get_cfg()
                cfg.merge_from_file(
                    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
                cfg.DATASETS.TRAIN = ("grain_train",)
                cfg.DATASETS.TEST = ()
                cfg.DATALOADER.NUM_WORKERS = 2
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
                cfg.SOLVER.IMS_PER_BATCH = 2
                cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
                cfg.SOLVER.MAX_ITER = 2000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
                cfg.SOLVER.STEPS = []  # do not decay learning rate
                cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
                # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

                cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
                                                 "C:/Users/Comeonjkf/Desktop/Python GUI/main_page/model_final.pth")  # path to the model we just trained
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
                predictor = DefaultPredictor(cfg)

                # The "cv_imread" method enables the software to read the address which contains Chinese characters
                def cv_imread(filePath):
                    import numpy as np
                    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
                    return cv_img

                im = cv_imread(origin_image)
                im = cv2.resize(im, (256, 256))
                outputs = predictor(im)
                result = outputs["instances"].to("cpu").pred_masks
                result = result[:, min_x:max_x, min_y:max_y]

                import numpy as np
                result = result + 0
                result = np.uint8(result)

                def devideMask(mask):
                    saveFlag = 1
                    for i in range(result.shape[0]):
                        save_path = 'save_screenshot/' + str(i) + '.png'
                        if (np.all(mask[i, :, :]==0)):
                            saveFlag = 0
                        else:
                            saveFlag = 1

                        for j in range(result.shape[1]):
                            for q in range(result.shape[2]):
                                if mask[i, j, q] == 1:
                                    mask[i, j, q] = 255

                        if(saveFlag == 1):
                            cv2.imwrite(save_path, mask[i, :, :])

                devideMask(result)

                img_path = r'C:\Users\Comeonjkf\Desktop\Python GUI\main_page\save_screenshot'
                img_list = os.listdir(img_path)
                for img in img_list:
                    img = os.path.join(r"save_screenshot\\", img)
                    calInfo(img)

                reference(10, angles)

                # The mean, max, min values of angle
                angleMean = np.mean(angles)
                angleMax = max(angles)
                angleMin = min(angles)


                # The mean, max, min values of ratio
                ratioMean = np.mean(ratios)
                ratioMax = max(ratios)
                ratioMin = min(ratios)


                # The mean, max, min values of size(area)
                sizeMean = np.mean(sizes)
                sizeMax = max(sizes)
                sizeMin = min(sizes)

                objectNum = len(angles)
                new = pd.DataFrame({'Picture_ID': file,
                                    'Number of grain': objectNum,
                                    'Max area': format(sizeMax, '.1f'),
                                    'Min area': format(sizeMin, '.1f'),
                                    'Average area': format(sizeMean, '.1f'),
                                    'Max ratio': format(ratioMax, '.1f'),
                                    'Min ratio': format(ratioMin, '.1f'),
                                    'Average ratio': format(ratioMean, '.1f'),
                                    'Max angle': format(angleMax, '.1f'),
                                    'Min angle': format(angleMin, '.1f'),
                                    'Average angle': format(angleMean, '.1f')
                                    },
                                   index=[1])
                df = pd.read_csv('grain_test.csv')
                df_new = df.append(new, ignore_index=True)

                df_new.to_csv('grain_test.csv',index=0)

                # The code of "Long axis-short axis ratio histogram"
                plt.figure()
                plt.tick_params(labelsize=10)
                ax = plt.gca()
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ratios = [round(a)for a in ratios]
                for i in range(0, len(ratios)):
                    if ratios[i] > 15:
                        ratios[i] = 15
                num, bins, patches = plt.hist(ratios, bins=np.arange(min(ratios) - 0.5, max(ratios) + 1.5),
                                              label="Long axis/short axis", edgecolor='black')
                plt.xticks(np.arange(min(ratios), max(ratios) + 1, 1))
                plt.xticks(fontsize=11)
                for num, bin in zip(num, bins):
                    if num != 0:
                        plt.annotate(int(num), xy=(bin, num), xytext=(bin + 0.273, num + 0.03))

                plt.xlabel("Long axis-short axis ratio value", fontsize=13)
                plt.ylabel("Grains number", fontsize=13)
                plt.legend(loc="best")

                plt.savefig("save_graph/" + file + "_RatioHist")

                # The code of "Deflection angle histogram"
                plt.figure()
                ax = plt.gca()
                plt.tick_params(labelsize=10)
                x_major_locator = MultipleLocator(5)
                ax.xaxis.set_major_locator(x_major_locator)
                plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                num, bins, patches = plt.hist(angles, bins=np.arange(-0.5, 91.5), label="Angel Bias", edgecolor='black')
                plt.xlabel("Angle value (Degree)", fontsize=13)
                plt.ylabel("Grains number", fontsize=13)
                plt.legend(loc='best')
                for num, bin in zip(num, bins):
                    if num != 0:
                        plt.annotate(int(num), xy=(bin, num), xytext=(bin - 0.3, num + 0.05))

                plt.savefig("save_graph/" + file + "_AngleHist")

                from PIL import ImageDraw

                pilim = Image.open("save_graph/" + file + "_AngleHist.png")
                draw = ImageDraw.Draw(pilim)
                draw.rectangle([564.41, 434.53, 640, 480], fill=(255, 255, 255, 255))
                del draw
                draw = ImageDraw.Draw(pilim)
                draw.rectangle([0, 434.53, 95.75, 480], fill=(255, 255, 255, 255))
                del draw
                pilim.save('save_graph/' + file + '_AngleHist.png')

                # The code of "Long axis and short axis scatter diagram"
                plt.figure()
                plt.xticks(range(0, int(max(widths)), 25))
                plt.yticks(range(0, int(max(heights)), 25))
                plt.scatter(widths, heights, label="Long axis & Short axis")
                plt.xlabel("Short axis " + "(" + unit + ")", fontsize=13)
                plt.ylabel("Long axis " + "(" + unit + ")", fontsize=13)
                plt.grid()

                plt.savefig("save_graph/" + file + "_RatioScatter")

                import shutil
                shutil.rmtree(os.path.join(ROOT_DIR, "save_screenshot"))
                os.mkdir(os.path.join(ROOT_DIR, "save_screenshot"))


            def moving_mouse(event):
                global sole_rectangle
                moving_mouse_x = event.x
                moving_mouse_y = event.y
                canvas = event.widget
                if sole_rectangle is not None:
                    canvas.delete(sole_rectangle)  # delete the last rectangle
                sole_rectangle = canvas.create_rectangle(left_mouse_down_x, left_mouse_down_y, moving_mouse_x,
                                                         moving_mouse_y, outline='red', width=3)

            def left_mouse_down1(event):  # the event when click the detail image
                global LEVEL_WINDOW

                LEVEL_WINDOW = tk.Toplevel(window)
                LEVEL_WINDOW.attributes('-topmost', True)
                middle_windows(LEVEL_WINDOW, width, height, reset=True)

                photo_canvas = tk.Canvas(LEVEL_WINDOW, width=width, height=height)
                photo_canvas.pack()
                photo_canvas.place(x=0, y=0)
                photo_canvas.create_image(0, 0, image=photo, anchor='nw')
                photo_canvas.bind('<B1-Motion>', moving_mouse)  # move the mouse
                photo_canvas.bind('<Button-1>', left_mouse_down)  # click the left mouse
                photo_canvas.bind('<ButtonRelease-1>', left_mouse_up)  # release the left mouse

            # the "resize" method can resize a pil_image object so it will fit into a box of size w_box times h_box, but retain aspect ratio
            def resize(w, h, w_box, h_box, pil_image):
                f1 = 1.0 * w_box / w  # 1.0 forces float division
                f2 = 1.0 * h_box / h
                factor = min([f1, f2])
                width = int(w * factor)
                height = int(h * factor)
                return pil_image.resize((width, height), Image.ANTIALIAS)

            # load the origin image
            img0 = Image.open(origin_image)
            w0, h0 = img0.size
            pil_image_resized0 = resize(w0, h0, 350, 350, img0)
            photo0 = ImageTk.PhotoImage(pil_image_resized0)
            photo_label = tkinter.Label(window, image=photo0)
            photo_label.pack()
            photo_label.place(relx=0.03, rely=0.02)


            imgr = Image.open("../img/array.png")
            wr, hr = imgr.size
            pil_image_resizedr = resize(wr, hr, 50, 50, imgr)
            photor = ImageTk.PhotoImage(pil_image_resizedr)
            photo_label_r1 = tkinter.Label(window, image=photor)
            photo_label_r1.pack()
            photo_label_r1.place(relx=0.23, rely=0.14)

            # load mask
            w, h = img.size
            pil_image_resized_m = resize(w, h, 350, 350, img)
            photo_m = ImageTk.PhotoImage(pil_image_resized_m)
            photo_label = tkinter.Label(window, image=photo_m)
            photo_label.pack()
            photo_label.place(relx=0.27, rely=0.02)
            photo_label.bind('<Button-1>', left_mouse_down1)


            photo_label_r2 = tkinter.Label(window, image=photor)
            photo_label_r2.pack()
            photo_label_r2.place(relx=0.47, rely=0.14)

            data_img1 = Image.open(data_graphy1)
            wd1, hd1 = data_img1.size
            resize_data_img1 = resize(wd1, hd1, 600, 600, data_img1)
            data_photo1 = ImageTk.PhotoImage(resize_data_img1)

            wd1_new, hd1_new = data_img1.size
            resize_data_img1_new = resize(wd1_new, hd1_new, 1000, 1000, data_img1)
            data_photo1_new = ImageTk.PhotoImage(resize_data_img1_new)

            data_img2 = Image.open(data_graphy2)
            wd2, hd2 = data_img2.size
            resize_data_img2 = resize(wd2, hd2, 600, 600, data_img2)
            data_photo2 = ImageTk.PhotoImage(resize_data_img2)

            wd2_new, hd2_new = data_img2.size
            resize_data_img2_new = resize(wd2_new, hd2_new, 1000, 1000, data_img2)
            data_photo2_new = ImageTk.PhotoImage(resize_data_img2_new)

            data_img3 = Image.open(data_graphy3)
            wd3, hd3 = data_img3.size
            resize_data_img3 = resize(wd3, hd3, 600, 600, data_img3)
            data_photo3 = ImageTk.PhotoImage(resize_data_img3)

            wd3_new, hd3_new = data_img3.size
            resize_data_img3_new = resize(wd3_new, hd3_new, 1000, 1000, data_img3)
            data_photo3_new = ImageTk.PhotoImage(resize_data_img3_new)

            # Length-width ratio histogram
            label_data1 = tkinter.Label(window, text="Long axis-short axis ratio histogram: ", font=(None, 14, "bold"), height=2)
            label_data1.pack()
            label_data1.place(relx=0.07, rely=0.42)

            data_lable1 = tkinter.Label(window, image=data_photo2)
            data_lable1.pack()
            data_lable1.place(relx=0.02, rely=0.45)

            def graph1(event):  # the event when click the first graph
                global LEVEL_WINDOW1

                LEVEL_WINDOW1 = tk.Toplevel(window)
                LEVEL_WINDOW1.attributes('-topmost', True)
                middle_windows(LEVEL_WINDOW1, 1000, 1000, reset=True)

                photo_canvas = tk.Canvas(LEVEL_WINDOW1, width=1000, height=1000)
                photo_canvas.pack()
                photo_canvas.place(x=0, y=0)
                photo_canvas.create_image(0, 0, image=data_photo2_new, anchor='nw')

            data_lable1.bind('<Button-1>', graph1)




            # Deflection angle histogram
            label_data2 = tkinter.Label(window, text="Deflection angle histogram:", font=(None, 14, "bold"), height=2)
            label_data2.pack()
            label_data2.place(relx=0.42, rely=0.42)

            data_lable2 = tkinter.Label(window, image=data_photo1)
            data_lable2.pack()
            data_lable2.place(relx=0.34, rely=0.45)

            def graph2(event):  # the event when click the second graph
                global LEVEL_WINDOW2

                LEVEL_WINDOW2 = tk.Toplevel(window)
                LEVEL_WINDOW2.attributes('-topmost', True)
                middle_windows(LEVEL_WINDOW2, 1000, 1000, reset=True)

                photo_canvas = tk.Canvas(LEVEL_WINDOW2, width=1000, height=1000)
                photo_canvas.pack()
                photo_canvas.place(x=0, y=0)
                photo_canvas.create_image(0, 0, image=data_photo1_new, anchor='nw')

            data_lable2.bind('<Button-1>', graph2)

            # Length and width scatter diagram
            label_data3 = tkinter.Label(window, text="Long axis and short axis scatter diagram:", font=(None, 14, "bold"), height=2)
            label_data3.pack()
            label_data3.place(relx=0.70, rely=0.42)

            data_lable3 = tkinter.Label(window, image=data_photo3)
            data_lable3.pack()
            data_lable3.place(relx=0.66, rely=0.45)

            def graph3(event):  # the event when click the third graph
                global LEVEL_WINDOW3

                LEVEL_WINDOW3 = tk.Toplevel(window)
                LEVEL_WINDOW3.attributes('-topmost', True)
                middle_windows(LEVEL_WINDOW3, 1000, 1000, reset=True)

                photo_canvas = tk.Canvas(LEVEL_WINDOW3, width=1000, height=1000)
                photo_canvas.pack()
                photo_canvas.place(x=0, y=0)
                photo_canvas.create_image(0, 0, image=data_photo3_new, anchor='nw')

            data_lable3.bind('<Button-1>', graph3)

            result_label1 = tkinter.Label(window, text="Long axis-short axis ratio", font=(None, 17, "bold"), height=2)
            result_label1.pack()
            result_label1.place(relx=0.50, rely=0.05)

            result_label11 = tkinter.Label(window, text="Average: ", font=(None, 15), height=2)
            result_label11.pack()
            result_label11.place(relx=0.7, rely=0.05)
            result_label11d = tkinter.Label(window, text=datalist[4], font=(None, 15), height=2)
            result_label11d.pack()
            result_label11d.place(relx=0.75, rely=0.05)

            result_label12 = tkinter.Label(window, text="Min:", font=(None, 15), height=2)
            result_label12.pack()
            result_label12.place(relx=0.80, rely=0.05)
            result_label12d = tkinter.Label(window, text=datalist[5], font=(None, 15), height=2)
            result_label12d.pack()
            result_label12d.place(relx=0.83, rely=0.05)

            result_label13 = tkinter.Label(window, text="Max:", font=(None, 15), height=2)
            result_label13.pack()
            result_label13.place(relx=0.88, rely=0.05)
            result_label13d = tkinter.Label(window, text=datalist[6], font=(None, 15), height=2)
            result_label13d.pack()
            result_label13d.place(relx=0.91, rely=0.05)

            result_label2 = tkinter.Label(window, text="Deflection angle", font=(None, 17, "bold"), height=2)
            result_label2.pack()
            result_label2.place(relx=0.53, rely=0.14)

            result_label21 = tkinter.Label(window, text="Average: ", font=(None, 15), height=2)
            result_label21.pack()
            result_label21.place(relx=0.7, rely=0.14)
            result_label21d = tkinter.Label(window, text=datalist[7], font=(None, 15), height=2)
            result_label21d.pack()
            result_label21d.place(relx=0.75, rely=0.14)

            result_label22 = tkinter.Label(window, text="Min: ", font=(None, 15), height=2)
            result_label22.pack()
            result_label22.place(relx=0.8, rely=0.14)
            result_label22d = tkinter.Label(window, text=datalist[8], font=(None, 15), height=2)
            result_label22d.pack()
            result_label22d.place(relx=0.83, rely=0.14)

            result_label23 = tkinter.Label(window, text="Max: ", font=(None, 15), height=2)
            result_label23.pack()
            result_label23.place(relx=0.88, rely=0.14)
            result_label23d = tkinter.Label(window, text=datalist[9], font=(None, 15), height=2)
            result_label23d.pack()
            result_label23d.place(relx=0.91, rely=0.14)

            result_label3 = tkinter.Label(window, text="Grain size ("+unit+"\u00b2)", font=(None, 17, "bold"), height=2)
            result_label3.pack()
            result_label3.place(relx=0.53, rely=0.23)

            result_label31 = tkinter.Label(window, text="Average: ", font=(None, 15), height=2)
            result_label31.pack()
            result_label31.place(relx=0.7, rely=0.23)
            result_label31d = tkinter.Label(window, text=datalist[1], font=(None, 15), height=2)
            result_label31d.pack()
            result_label31d.place(relx=0.75, rely=0.23)

            result_label32 = tkinter.Label(window, text="Min: ", font=(None, 15), height=2)
            result_label32.pack()
            result_label32.place(relx=0.8, rely=0.23)
            result_label32d = tkinter.Label(window, text=datalist[2], font=(None, 15), height=2)
            result_label32d.pack()
            result_label32d.place(relx=0.83, rely=0.23)

            result_label33 = tkinter.Label(window, text="Max: ", font=(None, 15), height=2)
            result_label33.pack()
            result_label33.place(relx=0.88, rely=0.23)
            result_label33d = tkinter.Label(window, text=datalist[3], font=(None, 15), height=2)
            result_label33d.pack()
            result_label33d.place(relx=0.91, rely=0.23)




        Add_button = tk.Button(canvas_1, text="Folder", bg="white", command=printcoords)
        Add_button.pack()
        Add_button.place(x=0, y=0, width=100, height=50)

        Add_button2 = tk.Button(canvas_1, text="Image", bg="white", command=printcoords2)
        Add_button2.pack()
        Add_button2.place(x=110, y=0, width=100, height=50)

        Delete_button = tk.Button(canvas_1, text="Delete", bg="white", command=delete)
        Delete_button.pack()
        Delete_button.place(x=220, y=0, width=70, height=50)


        Start_button = tk.Button(canvas_1, text="Start", bg="white", command=getaddr)
        Start_button.pack()
        Start_button.place(x=300, y=0, width=70, height=50)

        Select_button = tk.Button(canvas_5, text="Select", bg="white", command=select)
        Select_button.grid(row=0, column=4, padx=10, pady=5)

        Select_button = tk.Button(canvas_6, text="Select", bg="white", command=select_4)
        Select_button.grid(row=0, column=4, padx=10, pady=5)

        Select_all_button_1 = tk.Button(canvas_5, text="Select all", bg="white", command=select_3)
        Select_all_button_1.grid(row=0, column=5, padx=10, pady=5)

        Select_all_button_2 = tk.Button(canvas_6, text="Select all", bg="white", command=select_2)
        Select_all_button_2.grid(row=0, column=5, padx=10, pady=5)

        Save_CSV_button = tk.Button(canvas_3, text="Save CSV", bg="white", command=save)
        Save_CSV_button.pack()
        Save_CSV_button.place(x=0, y=0, width=70, height=50)

        Save_Mask_button = tk.Button(canvas_3, text="Save Mask", bg="white", command=save_mask)
        Save_Mask_button.pack()
        Save_Mask_button.place(x=90, y=0, width=70, height=50)

        Detail_button = tk.Button(canvas_3, text="Detail", bg="white", command=newWindow)
        Detail_button.pack()
        Detail_button.place(x=180, y=0, width=70, height=50)

        Analysis_button = tk.Button(canvas_3, text="Summary", bg="white", command=Analysis)
        Analysis_button.pack()
        Analysis_button.place(x=270, y=0, width=70, height=50)

        Delete2_button = tk.Button(canvas_3, text="Delete", bg="white", command=delete2)
        Delete2_button.pack()
        Delete2_button.place(x=360, y=0, width=70, height=50)

        back_button = tk.Button(self, text="<Back", bg="DodgerBlue", width=6,
                                command=lambda: controller.show_frame(StartPage))
        back_button.pack()
        back_button.place(x=0, y=0)

        im_checked = ImageTk.PhotoImage(Image.open("checked.png"))
        im_unchecked = ImageTk.PhotoImage(Image.open("unchecked.png"))

        tree = ttk.Treeview(master=canvas_2, columns=(1, 2))
        tree.column("#0", width=102, anchor="e")
        tree.column("#1", width=102, anchor="center")
        tree.column("#2", width=556, anchor="center")

        tree.heading("#0", text="Preview", anchor="center")
        tree.heading("#1", text="ID", anchor="center")
        tree.heading("#2", text="Detail", anchor="center")

        tree.tag_configure('checked', image=im_checked)
        tree.tag_configure('unchecked', image=im_unchecked)

        tree.bind('<Button 1>', toggleCheck)
        VScroll1 = tk.Scrollbar(canvas_2, orient='vertical', command=tree.yview)
        VScroll1.place(relx=0.971, rely=0.028, relwidth=0.024, relheight=0.958)
        tree.configure(yscrollcommand=VScroll1.set)

        tree.pack()

        tree2 = ttk.Treeview(master=canvas_4, columns=(1, 2))
        tree2.column("#0", width=102, anchor="e")
        tree2.column("#1", width=102, anchor="center")
        tree2.column("#2", width=556, anchor="center")

        tree2.heading("#0", text="Preview", anchor="center")
        tree2.heading("#1", text="ID", anchor="center")
        tree2.heading("#2", text="Detail", anchor="center")

        tree2.tag_configure('checked', image=im_checked)
        tree2.tag_configure('unchecked', image=im_unchecked)
        tree2.bind('<Button 1>', toggleCheck_right)

        tree2.pack()
        VScroll2 = tk.Scrollbar(canvas_4, orient='vertical', command=tree2.yview)
        VScroll2.place(relx=0.971, rely=0.028, relwidth=0.024, relheight=0.958)
        tree2.configure(yscrollcommand=VScroll2.set)

        tk.Label(canvas_5, text="select from img", bg="white").grid(row=0)
        tk.Label(canvas_5, text="to img", bg="white").grid(row=0, column=2)
        entry1 = tk.Entry(canvas_5, show=None)

        entry1.grid(row=0, column=1, padx=10, pady=5)

        entry2 = tk.Entry(canvas_5, show=None)

        entry2.grid(row=0, column=3, padx=10, pady=5)

        tk.Label(canvas_6, text="select from position", bg="white").grid(row=0)
        tk.Label(canvas_6, text="to position", bg="white").grid(row=0, column=2)
        entry3 = tk.Entry(canvas_6, show=None)

        entry3.grid(row=0, column=1, padx=10, pady=5)

        entry4 = tk.Entry(canvas_6, show=None)

        entry4.grid(row=0, column=3, padx=10, pady=5)

        s = ttk.Style()
        s.configure('Treeview', rowheight=60)
        s.layout('Treeview.Row',
                 [('Treeitem.row', {'sticky': 'nswe'}),
                  ('Treeitem.image', {'side': 'left', 'sticky': 'e'})])



# Driver Code
app = tkinterApp()
screenwidth = app.winfo_screenwidth()
screenheight = app.winfo_screenheight()
app.geometry('{}x{}+{}+{}'.format(screenwidth, screenheight, 0, 0))
app.attributes("-topmost", True)
app.mainloop()
