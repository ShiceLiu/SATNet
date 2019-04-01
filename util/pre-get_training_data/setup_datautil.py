from distutils.core import setup, Extension  

module = Extension('DataProcess', sources = ['datautil.cpp'],
	library_dirs = ['/usr/local/cuda-8.0/lib64','./build'],
	libraries = ['cudart','datautil'],
	language = 'c')  

setup(name = 'DataProcess', version = '1.0', description = 'DataProcess', ext_modules = [module])
