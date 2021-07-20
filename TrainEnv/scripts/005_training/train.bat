@ECHO OFF
ECHO Activating virtual env
call activate tensorflow

set PIPELINE_CONFIG_PATH=../../workspace/training/pre-trained-models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config
set MODEL_DIR= ../../workspace/training/models/.
set NUM_TRAIN_STEPS= 100

python ../../models/research/object_detection/model_main_tf2.py ^
  --model_dir %MODEL_DIR% ^
  --num_train_steps %NUM_TRAIN_STEPS% ^
  --pipeline_config_path %PIPELINE_CONFIG_PATH% ^
  --alsologtostderr


call conda deactivate
PAUSE