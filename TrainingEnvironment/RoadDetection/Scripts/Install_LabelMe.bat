@ECHO OFF
ECHO Installing LabelMe

call activate ConsciousCar

pip install labelme

call conda deactivate
PAUSE