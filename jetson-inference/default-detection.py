import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput() # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	imgOutput = jetson.utils.cudaAllocMapped(width=480, height=320, format=img.format)
	jetson.utils.cudaResize(img, imgOutput)
	detections = net.Detect(imgOutput)
	display.Render(imgOutput)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))