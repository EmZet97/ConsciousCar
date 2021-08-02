@ECHO OFF
ECHO Starting LabelMe

call activate tensorflow
labelme
call conda deactivate
PAUSE