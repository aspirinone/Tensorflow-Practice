import cv2 as cv
import numpy as np
import os
import xml.etree.cElementTree as ET

root_dir = "D:/day_02"

def change_to_num():
    files = os.listdir(root_dir)
    index = 0
    for img_file in files:
        if os.path.isfile(os.path.join(root_dir,img_file)):
            index += 1
            image_path = os.path.join(root_dir,img_file)
            print(image_path)
            image = cv.imread(image_path)
            #print(image_path.replace("png","jpg"))
            cv.imwrite("D:/hand_data02/VOC2012/JPEGImages/"+ str(index) +".jpg",image)

def xml_modification():
    ann_dir = "D:/hand_data02/VOC2012/Annotations"
    files = os.listdir(ann_dir)
    for xml_file in files:
        if os.path.isfile(os.path.join(ann_dir,xml_file)):
            xml_path = os.path.join(ann_dir,xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for elem in root.iter('folder'):
                elem.text = 'VOC2012'

            #for elem in root.iter('name'):
                #name = elem.text
                #elem.text = name.replace(" ","")

            tree.write(xml_path)
            print("1")

def generate_classes_text():
    print("start to generate classes text:")
    ann_dir = "D:/hand_data02/VOC2012/Annotations/"

    aeroplane_train = open("D:/hand_data02/VOC2012/ImageSets/Main/aeroplane_train.txt",'w')
    aeroplane_val = open("D:/hand_data02/VOC2012/ImageSets/Main/aeroplane_val.txt", 'w')

    fist_train = open("D:/hand_data02/VOC2012/ImageSets/Main/fist_train.txt", 'w')
    fist_val = open("D:/hand_data02/VOC2012/ImageSets/Main/fist_val.txt", 'w')

    yeah_train = open("D:/hand_data02/VOC2012/ImageSets/Main/yeah_train.txt", 'w')
    yeah_val = open("D:/hand_data02/VOC2012/ImageSets/Main/yeah_val.txt", 'w')

    ok_train = open("D:/hand_data02/VOC2012/ImageSets/Main/ok_train.txt", 'w')
    ok_val = open("D:/hand_data02/VOC2012/ImageSets/Main/ok_val.txt", 'w')

    files = os.listdir(ann_dir)
    for xml_file in files:
        if os.path.isfile(os.path.join(ann_dir,xml_file)):
            xml_path = os.path.join(ann_dir,xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for elem in root.iter('filename'):
                filename = elem.text
            for elem in root.iter('name'):
                name = elem.text

            if name =="PalmsForward":
                aeroplane_train.write(filename.replace(".jpg"," ")+str(1)+"\n")
                aeroplane_val.write(filename.replace(".jpg", " ") + str(1) + "\n")

                fist_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                fist_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                yeah_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                yeah_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                ok_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                ok_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

            if name == "Fist":
                aeroplane_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                aeroplane_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                fist_train.write(filename.replace(".jpg", " ") + str(1) + "\n")
                fist_val.write(filename.replace(".jpg", " ") + str(1) + "\n")

                yeah_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                yeah_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                ok_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                ok_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

            if name == "Yeah":
                aeroplane_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                aeroplane_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                fist_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                fist_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                yeah_train.write(filename.replace(".jpg", " ") + str(1) + "\n")
                yeah_val.write(filename.replace(".jpg", " ") + str(1) + "\n")

                ok_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                ok_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

            if name == "OK":
                aeroplane_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                aeroplane_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                fist_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                fist_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                yeah_train.write(filename.replace(".jpg", " ") + str(-1) + "\n")
                yeah_val.write(filename.replace(".jpg", " ") + str(-1) + "\n")

                ok_train.write(filename.replace(".jpg", " ") + str(1) + "\n")
                ok_val.write(filename.replace(".jpg", " ") + str(1) + "\n")

    aeroplane_train.close()
    aeroplane_val.close()
    fist_train.close()
    fist_val.close()
    yeah_train.close()
    yeah_val.close()
    ok_train.close()
    ok_val.close()

generate_classes_text()
