# Personerkennung
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zmeSTP3J5zu2d5fHgsQC06DyYEYJFXq1?usp=sharing)

## Introduction
Im Bereich des Baustellenmanagements hat die intelligente Personenerkennung und -verfolgung einen hohen Anwendungswert, da sie die Zahl der Verkehrsunfälle erheblich verringern könnte.
Dieses Projekt konzentriert sich hauptsächlich auf YOLO für die Personenerkennung und DeepSORT für die Personenverfolgung. Der YOLO-Algorithmus wird häufig für die Erkennung mehrerer Personen in Echtzeit verwendet, wobei er eine gute durchschnittliche Genauigkeit beibehält. Für die Personenerkennung wird der Deep-Learning-Algorithmus YOLOv4 (wählbar zwischen den Versionen YOLOv3 und Tiny) implementiert, der nach einer Evaluierung optimiert werden soll. DeepSORT verwendet den Kalman-Filter für die Personenverfolgung und nutzt den ungarischen Matching-Algorithmus als Ankermechanismus. DeepSORT verwendet den Zieldetektor, um das Videobild zu verarbeiten, und extrahiert Merkmale in den Frames des erkannten Ziels, einschließlich der offensichtlichen Merkmale (für den Merkmalsvergleich, um einen ID-Wechsel zu vermeiden) und der Bewegungsmerkmale (Kalman-Filter, um sie vorherzusagen), und berechnet schließlich den Übereinstimmungsgrad vor den Zielen von zwei benachbarten Frames. DeepSort fügt ein vorab trainiertes neuronales Netz zur Erstellung von Objektmerkmalen hinzu, das eine Assoziation auf der Grundlage der Ähnlichkeit der Merkmale anstelle einer Überlappung ermöglicht.
Um den Detektor für den praktischen Einsatz anpassungsfähiger zu machen, insbesondere wenn die Personen klein oder verdeckt sind, wurde die Struktur des Detektors durch Hinzufügen von Aufmerksamkeitsmechanismen und Reduzieren von Parametern verbessert, um Personen mit relativ hoher Genauigkeit und geringem GPU-Speicherverbrauch zu erkennen. 

## Demo zur Personenerkennung
<p align="center"><img src="data/Fotos/personerekennung.gif"\></p>

## Demo zur Re-Identifizierung der Person
<p align="center"><img src="data/Fotos/re_ID.gif"\></p>

## Demo zur Merkmalsvergleich zur Vermeidung eines ID-Wechsels
<p align="center"><img src="data/Fotos/switch-ID.gif"\></p>

## Erste Schritte
Um loszulegen, installieren Sie die richtigen Abhängigkeiten entweder über Anaconda oder Pip. Ich empfehle die Anaconda-Route für Leute, die eine GPU verwenden, da sie die CUDA-Toolkit-Version für Sie konfiguriert.

### Conda (Empfohlen)

```bash
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip

(TensorFlow 2 Pakete benötigen eine pip Version >19.0.)
```bash
# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Der Nvidia-Driver (für die GPU, wenn Sie keine Conda Environment verwenden und CUDA noch nicht eingerichtet haben)
Stellen Sie sicher, dass Sie CUDA Toolkit Version 10.1 verwenden, da dies die richtige Version für die in diesem Repository verwendete TensorFlow Version ist.https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Herunterladen der offiziellen YOLOv4 Pre-trained Weights
Unser Objekt-Tracker verwendet YOLOv4 für die Objekterkennung, die Deep Sort dann zum Tracking verwendet. Es gibt ein offizielles vortrainiertes YOLOv4-Objektdetektormodell. Für einfache Demozwecke werden wir die vortrainierten Gewichte für unseren Tracker verwenden. Laden Sie die vortrainierte Datei yolov4.weights herunter:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

Kopieren Sie yolov4.weights aus Ihrem Download-Ordner und fügen Sie es in den Ordner "data" dieses Repositorys ein.

Wenn Sie yolov4-tiny.weights verwenden möchten, ein kleineres Modell, das schneller, aber weniger genau ist, können Sie die Datei hier herunterladen: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
Link zum Download von yolov3.weights:
https://pjreddie.com/media/files/yolov3.weights
Link zum Download von yolov3-tiny.weights:
https://pjreddie.com/media/files/yolov3-tiny.weights
Link zum Herunterladen der Personendetektion mit yolov3-608.weights, trainiert mit dem Open Images Dataset:
https://drive.google.com/file/d/1DEGM-DKt0D0XQfpuu-V01fc_3bNJ9n48/view?usp=sharing

## Running the Tracker with YOLOv4
To implement the object tracking using YOLOv4, first we convert the .weights into the corresponding TensorFlow model which will be saved to a checkpoints folder. Then all we need to do is run the object_tracker.py script to run our object tracker with YOLOv4, DeepSort and TensorFlow.
```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 

# Run yolov4 deep sort object tracker on video
python object_tracker.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# Run yolov4 deep sort object tracker on webcam (set video flag to 0)
python object_tracker.py --video 0 --output ./outputs/webcam.avi --model yolov4
```
The output flag allows you to save the resulting video of the object tracker running so that you can view it again later. Video will be saved to the path that you set. (outputs folder is where it will be if you run the above command!)

If you want to run yolov3 set the model flag to ``--model yolov3``, upload the yolov3.weights to the 'data' folder and adjust the weights flag in above commands. (see all the available command line flags and descriptions of them in a below section)

## Running the Tracker with YOLOv4-Tiny
The following commands will allow you to run yolov4-tiny model. Yolov4-tiny allows you to obtain a higher speed (FPS) for the tracker at a slight cost to accuracy. Make sure that you have downloaded the tiny weights file and added it to the 'data' folder in order for commands to work!
```
# save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
```

## Resulting Video
As mentioned above, the resulting video will save to wherever you set the ``--output`` command line flag path to. I always set it to save to the 'outputs' folder. You can also change the type of video saved by adjusting the ``--output_format`` flag, by default it is set to AVI codec which is XVID.

Example video showing tracking of all coco dataset classes:
<p align="center"><img src="data/helpers/all_classes.gif"\></p>

## Filter Classes that are Tracked by Object Tracker
By default the code is setup to track all 80 or so classes from the coco dataset, which is what the pre-trained YOLOv4 model is trained on. However, you can easily adjust a few lines of code in order to track any 1 or combination of the 80 classes. It is super easy to filter only the ``person`` class or only the ``car`` class which are most common.

To filter a custom selection of classes all you need to do is comment out line 159 and uncomment out line 162 of [object_tracker.py](https://github.com/theAIGuysCode/yolov4-deepsort/blob/master/object_tracker.py) Within the list ``allowed_classes`` just add whichever classes you want the tracker to track. The classes can be any of the 80 that the model is trained on, see which classes you can track in the file [data/classes/coco.names](https://github.com/theAIGuysCode/yolov4-deepsort/blob/master/data/classes/coco.names)

This example would allow the classes for person and car to be tracked.
<p align="center"><img src="data/helpers/filter_classes.PNG"\></p>

### Demo of Object Tracker set to only track the class 'person'
<p align="center"><img src="data/helpers/demo.gif"\></p>

### Demo of Object Tracker set to only track the class 'car'
<p align="center"><img src="data/helpers/cars.gif"\></p>

## Command Line Args Reference

```bash
save_model.py:
  --weights: path to weights file
    (default: './data/yolov4.weights')
  --output: path to output
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'False')
  --input_size: define input size of export model
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
    
 object_tracker.py:
  --video: path to input video (use 0 for webcam)
    (default: './data/video/test.mp4')
  --output: path to output video (remember to set right codec for given format. e.g. XVID for .avi)
    (default: None)
  --output_format: codec used in VideoWriter when saving video to file
    (default: 'XVID)
  --[no]tiny: yolov4 or yolov4-tiny
    (default: 'false')
  --weights: path to weights file
    (default: './checkpoints/yolov4-416')
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 or yolov4
    (default: yolov4)
  --size: resize images to
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: dont show video output
    (default: False)
  --info: print detailed info about tracked objects
    (default: False)
```

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
