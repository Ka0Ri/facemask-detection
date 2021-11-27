# facemask-detection
This repository buils a custom facemask detection model for jetson nano (IE laboratory - CNU).

## Dataset  
The dataset used in this model is "Face Mask Detection", the original link can be found here: https://www.kaggle.com/andrewmvd/face-mask-detection. The annotation is in VOC format. 

Total images: `853 images`

Classification labels: `(1) with_mask, (2) without_mask, (3) mask_weared_incorrect` 

VOC folder format:
```
root:
-|Annotations:
---|file1.xml 
---|file2.xml 
---|file2.xml 
-|JPEGImages:
---|file1.jpg 
---|file2.jpg 
---|file2.jpg
-|ImageSets:
---|Main:
----|default.txt (or test.txt or trainval.txt)
--------file1
--------file2
--------file3
-labels.txt
---with_mask
---without_mask
---mask_weared_incorrect
```
Convert folder data into VOC folder structure
```
python VOCdataset_convert.py --root=<root folder contains Images and Annotations> --output=<VOC output folder >
```
Then, copy or create labels.txt file to `output` folder

## Model

Code for retrain model can be found in https://github.com/dusty-nv/pytorch-ssd/tree/8ed842a408f8c4a8812f430cf8063e0b93a56803

**Step 1**: Retrain SSD-mobilenet:
```
python train_ssd.py --dataset-type=voc --data=<path to data folder> --model-dir=<path to checkpoint model> --epochs=<number of epochs>
```
Please check other arguments for further optional training

Please check `vision/ssd/config` for other configurations

**Step 2**: Convert to ONNX model:
```
python onnx_export.py --labels=<path to labels.txt> --net=<model type, eg: mb1-ssd> --input=<path to checkpoint.pth> --output=<path to ckpt.onnx>
```
Please check other arguments for further options

**Step 3**: run a demo
```
python run_ssd_example.py <model-type> <path to checkpoint.pth> <path to labels.txt> <path to input image>
```
Please check other arguments for further options

![alt text]("https://github.com/Ka0Ri/facemask-detection/blob/main/run_ssd_example_output.jpg")

**Step 4**: jetson nano inference
```
python mask-detection --model=<path to ckpt.onnx> --labels=<path to labels.txt> --input-blob=input_0 --output-cvg=scores --output-bbox=boxes <camera ,e.g: /dev/video0>
```
