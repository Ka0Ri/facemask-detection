import os
import cv2
import argparse
import xml.etree.ElementTree as ET


def create_folder(foldername):
    if os.path.exists(foldername):
        print('folder already exists:', foldername)
    else:
        os.makedirs(foldername)

def convert(root, output):
    create_folder(os.path.join(output, "JPEGImages"))
    create_folder(os.path.join(output, "ImageSets/Main"))
    create_folder(os.path.join(output, "Annotations"))
    with open(os.path.join(output, "ImageSets/Main", "default.txt"), "w") as f:
        for img_path in os.listdir(os.path.join(root, "Images")):
            img = cv2.imread(os.path.join(root, "Images", img_path))
            cv2.imwrite(os.path.join(output, "JPEGImages", img_path[:-4] +".jpg"), img)
            f.writelines(str(img_path[:-4]) + "\n")

            # tree = ET.parse(os.path.join(root, "Annotations", img_path[:-4] + ".xml"))
            # root = tree.getroot()
            # print(root[5][0].text)
            
            with open(os.path.join(root, "Annotations", img_path[:-4] + ".xml"), 'rb') as in_xml:
                xml = in_xml.read()
                with open(os.path.join(output, "Annotations", img_path[:-4] + ".xml"), "wb") as out_xml:
                    out_xml.write(xml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default="", help="path to Root folder contains Images folder for images and Annotation folder for folder.")
    parser.add_argument('--output', type=str, default="", help="path to output folder in VOC format")

    args = parser.parse_args() 
    print(args) 

    convert(args.root, args.output)

