CALL conda create --name flt_rsk python=3.10
CALL activate flt_rsk
CALL conda install pandas=1.5
CALL conda install scikit-learn
CALL pip install --editable=utils
pause
