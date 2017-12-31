# Traffic Light Detection with Tensorflow Object Detection API


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

#### Download Faster RCNN Model

```
# From root directory
curl http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz | tar -xv -C model/ --strip 1
```


### Creating TFRecord files

```
python data_conversion_udacity_sim.py --output_path sim_data.record
```

```
python data_conversion_udacity_real.py --output_path real_data.record
```


## Training

### For Simulator Data

#### Training

```
python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture
```

#### Saving for Inference

```
python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_sim/
```


### For Real Data

#### Training

```
python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --train_dir=data/real_training_data
```

#### Saving for Inference

```
python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_real/
```


## Attributions

- [https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62](https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62)
