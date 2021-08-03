call activate ConsciousCar

ECHO Creating masks...
python GenerateImageMasks.py
ECHO Preprocessing...
python PreprocessData.py
ECHO Testing...
python TestMaskFit.py

PAUSE