import jetson.inference
import jetson.utils

import argparse
import sys

import cv2

# command: python detectnet.py --model=mb1-ssd.onnx --labels=labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes /dev/video0

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=jetson.inference.detectNet.Usage())
                                #  jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)



def mask_detection(img, net):

    # open_cvimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    jetson_img = jetson.utils.cudaFromNumpy(img)


    detections = net.Detect(jetson_img, overlay=opt.overlay)

    net.PrintProfilerTimes()
    # opencv_img = jetson.utils.cudaToNumpy(jetson_img)
    # opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
    # cv2.imshow("", opencv_img)
    # cv2.waitKey()
    print("detected {:d} objects in image".format(len(detections)))

    classID = [int(bbox.ClassID) for bbox in detections]

    if(sum(classID) == len(classID)):
        return 1
    else:
        return 0

if __name__ == "__main__":


    net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)
    img = cv2.imread("maksssksksss29.jpg")
    results = mask_detection(img, net)
    print(results)
