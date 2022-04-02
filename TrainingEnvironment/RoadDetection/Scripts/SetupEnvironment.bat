@ECHO OFF
call activate ConsciousCar

ECHO Downloading packages
pip install cython
pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
pip install torchvision
pip install opencv-python
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install onnx

PAUSE

