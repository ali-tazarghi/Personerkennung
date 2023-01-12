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
Unser Person-Tracker verwendet YOLOv4 für die Personerkennung, die Deep Sort dann zum Tracking verwendet. Es gibt ein offizielles vortrainiertes YOLOv4-Objektdetektormodell. Für einfache Demozwecke werden wir die vortrainierten Gewichte für unseren Tracker verwenden. Laden Sie die vortrainierte Datei yolov4.weights herunter:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

Kopieren Sie yolov4.weights aus Ihrem Download-Ordner und fügen Sie es in den Ordner "data" dieses Repositorys ein.

Wenn Sie yolov4-tiny.weights verwenden möchten, ein kleineres Modell, das schneller, aber weniger genau ist, können Sie die Datei hier herunterladen: https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights

Link zum Download von yolov3.weights:
https://pjreddie.com/media/files/yolov3.weights

Link zum Download von yolov3-tiny.weights:
https://pjreddie.com/media/files/yolov3-tiny.weights

Link zum Herunterladen der Personendetektion mit yolov3-608.weights, trainiert mit dem Open Images Dataset:
https://drive.google.com/file/d/1DEGM-DKt0D0XQfpuu-V01fc_3bNJ9n48/view?usp=sharing

## Ausführen des Trackers mit YOLOv4
Um die Personenverfolgung mit YOLOv4 zu implementieren, konvertieren wir zunächst die .weights in das entsprechende TensorFlow-Modell, das in einem Checkpoints-Ordner gespeichert wird. Dann müssen wir nur noch das Skript main.py ausführen, um unseren Person-Tracker mit YOLOv4, DeepSort und TensorFlow zu starten.
```bash
# Darknet-weights in TensorFlow-Modell umwandeln
python convert_model.py --model yolov4 

# yolov4 Deep Sort Person Tracker auf Video ausführen
python main.py --video ./data/video/test.mp4 --output ./outputs/demo.avi --model yolov4

# yolov4 deep sort Person tracker auf Webcam ausführen (Video-Flag auf 0 setzen)
python main.py --video 0 --output ./outputs/webcam.avi --model yolov4
```
Mit dem Output-Flag können Sie das Video des Personentrackers speichern, damit Sie es später noch einmal ansehen können. Das Video wird in dem path gespeichert, den Sie angegeben haben. (Der Output-Ordner ist der Ort, an dem es sich befindet, wenn Sie den obigen Befehl ausführen!)

Wenn Sie yolov3 ausführen möchten, setzen Sie die Modellflagge auf ``--model yolov3``, laden Sie die yolov3.weights in den 'data'-Ordner und passen Sie die weights-Flagge in obigen Befehlen an. (siehe alle verfügbaren Befehlszeilen-Flags und deren Beschreibungen in einem der folgenden Abschnitte)

## Ausführen des Trackers mit YOLOv4-Tiny
Mit den folgenden Befehlen können Sie das yolov4-tiny-Modell ausführen. Mit Yolov4-Tiny können Sie eine höhere Geschwindigkeit (FPS) für den Tracker erzielen, was allerdings zu Lasten der Genauigkeit geht. Vergewissern Sie sich, dass Sie die Datei "tiny weights" heruntergeladen und in den Ordner "data" eingefügt haben, damit die Befehle funktionieren!
```
# save yolov4-tiny model
python convert_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny

# Run yolov4-tiny object tracker
python main.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/test.mp4 --output ./outputs/tiny.avi --tiny
```

Wie bereits erwähnt, wird das resultierende Video an dem Ort gespeichert, auf den Sie den path des ``--output`` -Befehlszeilenflags setzen. Ich habe es so eingestellt, dass es im Ordner "outputs" gespeichert wird. Sie können auch den Typ des gespeicherten Videos ändern, indem Sie das ``--output_format`` -Flag anpassen. standardmäßig ist es auf AVI-Codec eingestellt, was XVID ist.

## Command Line Args Reference

```bash
convert_model.py:
  --weights: Path zur weights-Datei
    (default: './data/yolov4.weights')
  --output: Path zur Ausgabe
    (default: './checkpoints/yolov4-416')
  --[no]tiny: yolov4 oder yolov4-tiny
    (default: 'False')
  --input_size: Eingabegröße des Exportmodells definieren
    (default: 416)
  --framework: what framework to use (tf, trt, tflite)
    (default: tf)
  --model: yolov3 oder yolov4
    (default: yolov4)
    
 main.py:
  --video: path zur input video (verwenden 0 für webcam)
    (default: './data/video/test.mp4')
  --output: path to output video ( achten beim Format auf den richtigen Codec, z.B. XVID für .avi)
    (default: None)
  --output_format: Codec, der in VideoWriter beim Speichern des Videos in einer Datei verwendet wird
    (default: 'XVID)
  --[no]tiny: yolov4 oder yolov4-tiny
    (default: 'false')
  --weights: Path zur Weights-Datei
    (default: './checkpoints/yolov4-416')
  --model: yolov3 oder yolov4
    (default: yolov4)
  --size: Ändern der Bildgröße auf
    (default: 416)
  --iou: iou threshold
    (default: 0.45)
  --score: confidence threshold
    (default: 0.50)
  --dont_show: Videoausgabe nicht anzeigen
    (default: False)
  --info: detaillierte Informationen über verfolgte personnen drucken
    (default: False)
```

### References  

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  * [darknet](https://github.com/AlexeyAB/darknet)
  * [DeepSort](https://github.com/zzh8829/yolov3-tf2)
