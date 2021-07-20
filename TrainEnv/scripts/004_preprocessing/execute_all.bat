@ECHO OFF

ECHO - Create coco files from train and test data
python labelme2coco.py ../../workspace/training/images/train --output ../../workspace/training/annotations/train.json
python labelme2coco.py ../../workspace/training/images/test --output ../../workspace/training/annotations/test.json

ECHO - Create tf records
python create_coco_tf_record.py --logtostderr --train_image_dir=../../workspace/training/images/train/ --test_image_dir=../../workspace/training/images/test/ --train_annotations_file=../../workspace/training/annotations/train.json --test_annotations_file=../../workspace/training/annotations/test.json --include_masks=True --output_dir=../../workspace/training/annotations/

ECHO ----- Done
call conda deactivate
PAUSE