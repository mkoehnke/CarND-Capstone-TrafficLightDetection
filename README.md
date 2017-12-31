# Traffic Light Detection with Tensorflow Object Detection API

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

This repository was used to setup an environment to train/evaluate a pre-trained model to identify/classify traffic lights in pictures using tensorflow. It was part of my [final project](https://github.com/mkoehnke/CarND-Capstone) for the [Udacity Self-Driving Car Engineer Nano Degree program](https://eu.udacity.com/course/self-driving-car-engineer-nanodegree--nd013).

## Setup

#### Dependencies

```
sudo apt-get install protobuf-compiler
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

#### Protobuf Compilation

```
# From root directory
protoc object_detection/protos/*.proto --python_out=.
```

#### Add Libraries to PYTHONPATH

```
# From root directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

#### Download SSD MobileNet Model

```
# From root directory
curl http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz | tar -xv -C model/ --strip 1
```


### Creating TFRecord files

```
python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=data/sim_training_data/sim_data_capture --output_path=sim_data.record
```

```
python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=data/real_training_data/real_data_capture --output_path=real_data.record
```


## Training

### For Simulator Data

#### Training

```
python object_detection/train.py --pipeline_config_path=config/ssd_mobilenet_v1_coco_sim.config --train_dir=data/sim_training_data/sim_data_capture
```

#### Saving for Inference

```
python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_mobilenet_v1_coco_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-6000 --output_directory=model_frozen_sim/
```


### For Real Data

#### Training

```
python object_detection/train.py --pipeline_config_path=config/ssd_mobilenet_v1_coco_real.config --train_dir=data/real_training_data
```

#### Saving for Inference

```
python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_mobilenet_v1_coco_real.config --trained_checkpoint_prefix=data/real_training_data/real_data_capture/model.ckpt-25000 --output_directory=model_frozen_real/
```


## Attributions

- [https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62](https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62)
