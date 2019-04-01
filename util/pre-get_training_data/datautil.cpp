#include "Python.h"
#include <dlfcn.h>
#include <iostream>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL DataProcess_ARRAY_API
#include "numpy/arrayobject.h"

using namespace std;

static PyObject* TSDF(PyObject *self, PyObject *args)
{
	void *handle = dlopen("./build/libdatautil.so", RTLD_LAZY);
	if(!handle)
	{
	    printf("open lib error\n");
	    cout<<dlerror()<<endl;
	    return PyLong_FromLong(-2);
	}
	typedef PyObject* (*TSDF_t)(PyObject* args);
	TSDF_t _TSDF = (TSDF_t) dlsym(handle, "_TSDF");
	if(!_TSDF)
	{
		printf("open function error\n");
		cout<<dlerror()<<endl;
		dlclose(handle);
		return PyLong_FromLong(-3);
	}

	PyObject* returnList = _TSDF(args);
	dlclose(handle);
	return returnList;
}

static PyObject* TSDFTransform(PyObject *self, PyObject *args)
{
	void *handle = dlopen("./build/libdatautil.so", RTLD_LAZY);
	if(!handle)
	{
	    printf("open lib error\n");
	    cout<<dlerror()<<endl;
	    return PyLong_FromLong(-2);
	}
	typedef PyObject* (*TSDFTransform_t)(PyObject* args);
	TSDFTransform_t _TSDFTransform = (TSDFTransform_t) dlsym(handle, "_TSDFTransform");
	if(!_TSDFTransform)
	{
		printf("open function error\n");
		cout<<dlerror()<<endl;
		dlclose(handle);
		return PyLong_FromLong(-3);
	}

	PyObject* returnList = _TSDFTransform(args);
	dlclose(handle);
	return returnList;
}

static PyObject* DownSampleLabel(PyObject *self, PyObject *args)
{
	void *handle = dlopen("./build/libdatautil.so", RTLD_LAZY);
	if(!handle)
	{
	    printf("open lib error\n");
	    cout<<dlerror()<<endl;
	    return PyLong_FromLong(-2);
	}
	typedef PyObject* (*DownSampleLabel_t)(PyObject* args);
	DownSampleLabel_t _DownSampleLabel = (DownSampleLabel_t) dlsym(handle, "_DownSampleLabel");
	if(!_DownSampleLabel)
	{
		printf("open function error\n");
		cout<<dlerror()<<endl;
		dlclose(handle);
		return PyLong_FromLong(-3);
	}

	PyObject* returnList = _DownSampleLabel(args);
	dlclose(handle);
	return returnList;
}

static PyMethodDef MyMethods[] = {
	{"TSDF", TSDF, METH_VARARGS, "TSDF"},
	{"TSDFTransform", TSDFTransform, METH_VARARGS, "TSDFTransform"},
	{"DownSampleLabel", DownSampleLabel, METH_VARARGS, "DownSampleLabel"},
	{NULL, NULL, 0, NULL}
};

static PyObject* transError;

PyMODINIT_FUNC initDataProcess(void) 
{
	PyObject* m = Py_InitModule("DataProcess", MyMethods);
	if (m == NULL)  
	  return;  
	transError = PyErr_NewException("trans.error",NULL,NULL);  
	Py_INCREF(transError);  
	PyModule_AddObject(m,"error",transError);
}