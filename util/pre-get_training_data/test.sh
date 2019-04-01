cd build
rm -rf ./*
cmake ..
make
cd ..
python setup_datautil.py build
# python test.py