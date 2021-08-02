@ECHO OFF
ECHO Installing LabelMe

call activate tensorflow

pip install labelme

call conda deactivate
PAUSE