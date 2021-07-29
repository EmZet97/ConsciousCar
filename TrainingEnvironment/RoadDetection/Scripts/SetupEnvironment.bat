@ECHO OFF
call activate ConsciousCar

ECHO Downloading packages
pip install cython
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
pip install torchvision

PAUSE

