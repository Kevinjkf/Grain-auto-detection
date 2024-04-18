Our project's name is 'AI‑based Magnetic Nano‑grain Images Automatic Detection and Evaluation', I developed a system that can segment grains automatically and calculate relevant information accurately. In order to better understand our project and run my code, user guides are as below. If you want to get a general idea of what the software looks like and what it can do, go to the Youtube link: https://youtu.be/qXRdyfird6U to watch a short introductory video. 

## Quality assurance

The quality assurance can be measured by the abilities below.

- Maintainability: The software is designed to be easily updated in the future. For example, segmentation algorithm is encapsulated and therefore future replacement or improvement can be done easily. Another example is the unit selection at the start page, this is defined as a future functionality that adapts different grain size by users.
- Reliability: This can be measured by the number of error messages when run the software. To explain it, there are a few warnings and no error messages in console and software has been tested on a large number of test cases with no error.
- Usability & Robustness: The software is well-designed with user requirement with clear user manual is given below.
- Efficiency: The software is designed to provide high efficiency. For example, in order to speed up the process and save storage for computers, some useless information or intermediate steps are replaced or removed.

## A list of packages

| Name of packages | Version  | Explanations/Tips                          |
| ---------------- | -------- | ------------------------------------------ |
| openpyxl         | 3.0.9    | used to save grain information to csv      |
| pandas           | 1.1.5    | used to save grain information to csv      |
| matplotlib       | 3.5.1    | used to generate histograms and scatter    |
| tkinter          |          | Python GUI                                 |
| ttkbootstrap     | 1.7.3    | Python GUI                                 |
| pillow           | 9.0.1    | load images                                |
| numpy            | 1.21.5   | used to deal with matrixs                  |
| scipy            | 1.7.3    | used to deal with matrixs                  |
| imutils          | 0.5.4    | Image processing                           |
| opencv           | 4.5.5.62 | used to calculate information of grain     |
| detectron2       |          | used for object detection and segmentation |

## Environment requirements & Installation instructions

The core package of segmentation part is the “detectron2” ( [facebookresearch/detectron2: Detectron2 is a platform for object detection, segmentation and other visual recognition tasks. (github.com)](https://github.com/facebookresearch/detectron2))

Requirements of building detectron2:

1. Python == 3.7.11

2. GCC >=4.9

3. PyTorch 1.11.0 & [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation: You can install them together using the `conda install pytorch torchvision -c pytorch`.

4. fvcore:

   `pip install fvcore`

5. pycocotools:

`pip install git+https://github.com/philferriere/cocoapi.git #subdirectory=PythonAPI`

If you meet the error - ”Failed building wheel for pycocotools” : you can install the [Microsoft Visual C++ Build Tools 2015](http://go.microsoft.com/fwlink/?LinkId=691126) and try install the pycocotools using the installation commands above again.

- CUDA11.6 (no test in older version): You can download CUDA from the link: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
- Visual Studio 2022(no test in older version): You can download VS2022 from the link: https://visualstudio.microsoft.com/zh-hans/downloads/

6. Install the Detetron2:
   - `git clone https://github.com/facebookresearch/detectron2.git`
   - `cd detectron2 && pip install -e .`

7. Because the model_final.pth is too large to be pushed to github, if you want to run the code, you can contact me and I will send this file to you.

## User manual

### Hardware

This software is recommended to run on a computer(windows). Computers with
normal screen size are all accepted.

### Software

There are 3 pages in this project, which are home page, main page and detail page.

#### Homepage

<img src="img_readme\homepageum.png" alt="homepageum" style="zoom:60%;" />There will be a pop-up window to let you choose the grain size like the graph above.

In the middle of the home page, there is a simple process description of our software, which are image import, segmentation, basic information analysis and saving to csv file. If you click “Start” in the middle of the page, page will jump to the main page.

#### Main page

You can go to this page by clicking the "Start" button in the middle of the homepage. Functionalities implemented in this page is listed as the following:

1. See the mask of grains after importing original images/folders in main page. You can choose to add a package or add images manually. By clicking 'Add a package', you are asked to upload a package of images. By clicking the button 'Add images', you will then select one or a series of images per time for uploading to the software. After selecting images, you need to click the start button to begin the image processing, then a progress bar will appear. After that, the masks will appear on the right side of the software.
2. Empty or delete the working space. This function is realized by selecting images and clicking "Delete" button.
3. Automatically choose the image range by using the selector below.
4. Save the CSV file and masks to a specific local place. This function is realized by selecting images and clicking 'Save CSV' or 'Save mask' button.
5. Choose a specific image(you can only choose one image at a time) and click 'Detail' allows you to jump to detail page of this image.

![mainpageum](img_readme\mainpageum.png)

#### Detail page

The functionalities implemented in the detail page are listed below.

1.  Re-save the processed image and relevant information. You can click mask to make screenshots.

![screenshot1](img_readme\screenshot1.png)
![screenshot2](img_readme\screenshot2.png)

2. See the histograms of the deflection angle and the ratio of length and width.
3. See the scatter plot of the ratio of length and width.

4. See the average, max and min value of three types, which are long axis-short axis ratio, grain size, and deflection angle.

   ![detailpage](img_readme\detailpage.png)
