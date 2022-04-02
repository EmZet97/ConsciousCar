@ECHO OFF
ECHO Starting LabelMe

call activate ConsciousCar
labelme
call conda deactivate
PAUSE