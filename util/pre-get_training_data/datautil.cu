#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
// #include "boost/python.hpp"
#include "Python.h"
#define PY_ARRAY_UNIQUE_SYMBOL DataProcess_ARRAY_API
#include "numpy/arrayobject.h"
#include "malloc.h"

using namespace std;
// using namespace boost::python;

#define MIN(x,y) (((x)>(y)?(y):(x)))
#define MAX(x,y) (((x)<(y)?(y):(x)))
    
__global__ void depth2Grid__(int frame_width, int frame_height,
    float* camK, float* camPose, float vox_unit, float* voxSize, float* voxOrigin,
    float* depth_data_GPU, float* vox_binary_GPU, float* depth_mapping_3d) {

  float cam_K[9];
  for (int i = 0; i < 9; ++i)
    cam_K[i] = camK[i];
  float cam_pose[16];
  for (int i = 0; i < 16; ++i)
    cam_pose[i] = camPose[i];

  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = voxSize[i];
  float vox_origin[3];
  for (int i = 0; i < 3; ++i)
    vox_origin[i] = voxOrigin[i];

  int pixel_x = blockIdx.x;
  int pixel_y = threadIdx.x;

  float point_depth = depth_data_GPU[pixel_y * frame_width + pixel_x];

  float point_cam[3] = {0};
  point_cam[0] =  (pixel_x - cam_K[2])*point_depth/cam_K[0];
  point_cam[1] =  (pixel_y - cam_K[5])*point_depth/cam_K[4];
  point_cam[2] =  point_depth;

  float point_base[3] = {0};
  point_base[0] = cam_pose[0 * 4 + 0]* point_cam[0] + cam_pose[0 * 4 + 1]*  point_cam[1] + cam_pose[0 * 4 + 2]* point_cam[2];
  point_base[1] = cam_pose[1 * 4 + 0]* point_cam[0] + cam_pose[1 * 4 + 1]*  point_cam[1] + cam_pose[1 * 4 + 2]* point_cam[2];
  point_base[2] = cam_pose[2 * 4 + 0]* point_cam[0] + cam_pose[2 * 4 + 1]*  point_cam[1] + cam_pose[2 * 4 + 2]* point_cam[2];
  point_base[0] = point_base[0] + cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] + cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] + cam_pose[2 * 4 + 3];

  int z = (int)floor((point_base[0] - vox_origin[0])/vox_unit);
  int x = (int)floor((point_base[1] - vox_origin[1])/vox_unit);
  int y = (int)floor((point_base[2] - vox_origin[2])/vox_unit);

  if( x >= 0 && x < vox_size[0] && y >= 0 && y < vox_size[1] && z >= 0 && z < vox_size[2]){
    int vox_idx = z * vox_size[0] * vox_size[1] + y * vox_size[0] + x;
    vox_binary_GPU[vox_idx] = float(1.0);
    depth_mapping_3d[pixel_y * frame_width + pixel_x] = vox_idx;
  }
}
    
__global__ void SquaredDistanceTransform__(int frame_width, int frame_height,
    float vox_unit, float vox_margin, float* voxSize, float* voxOrigin,
    float* camK, float* camPose, float* vox_binary_GPU,
    float* depth_data_GPU, float* tsdf_data_GPU) {

  int vox_size[3];
  for (int i = 0; i < 3; ++i)
    vox_size[i] = voxSize[i];
  float vox_origin[3];
  for (int i = 0; i < 3; ++i)
    vox_origin[i] = voxOrigin[i];

  float cam_K[9];
  for (int i = 0; i < 9; ++i)
    cam_K[i] = camK[i];
  float cam_pose[16];
  for (int i = 0; i < 16; ++i)
    cam_pose[i] = camPose[i];

  int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vox_idx >= vox_size[0] * vox_size[1] * vox_size[2]){
    return;
  }

  int z = float((vox_idx / ( vox_size[0] * vox_size[1]))%vox_size[2]) ;
  int y = float((vox_idx / vox_size[0]) % vox_size[1]);
  int x = float(vox_idx % vox_size[0]);
  int search_region = (int)(vox_margin/vox_unit);

  if (vox_binary_GPU[vox_idx] >0 ){
    tsdf_data_GPU[vox_idx] = 0;
    return;
  }

  float point_base[3] = {0};
  point_base[0] = float(z) * vox_unit + vox_origin[0];
  point_base[1] = float(x) * vox_unit + vox_origin[1];
  point_base[2] = float(y) * vox_unit + vox_origin[2];

  float point_cam[3] = {0};
  point_base[0] = point_base[0] - cam_pose[0 * 4 + 3];
  point_base[1] = point_base[1] - cam_pose[1 * 4 + 3];
  point_base[2] = point_base[2] - cam_pose[2 * 4 + 3];
  point_cam[0] = cam_pose[0 * 4 + 0] * point_base[0] + cam_pose[1 * 4 + 0] * point_base[1] + cam_pose[2 * 4 + 0] * point_base[2];
  point_cam[1] = cam_pose[0 * 4 + 1] * point_base[0] + cam_pose[1 * 4 + 1] * point_base[1] + cam_pose[2 * 4 + 1] * point_base[2];
  point_cam[2] = cam_pose[0 * 4 + 2] * point_base[0] + cam_pose[1 * 4 + 2] * point_base[1] + cam_pose[2 * 4 + 2] * point_base[2];
  if (point_cam[2] <= 0){
    return;
  }

  int pixel_x = roundf(cam_K[0] * (point_cam[0] / point_cam[2]) + cam_K[2]);
  int pixel_y = roundf(cam_K[4] * (point_cam[1] / point_cam[2]) + cam_K[5]);
  if (pixel_x < 0 || pixel_x >= frame_width || pixel_y < 0 || pixel_y >= frame_height){ // outside FOV
    return;
  }

  float point_depth = depth_data_GPU[pixel_y * frame_width + pixel_x];
  if (point_depth < float(0.5f) || point_depth > float(8.0f)){
    return;
  }
  if (roundf(point_depth) == 0){ // mising depth
    tsdf_data_GPU[vox_idx] = float(-1.0);
    return;
  }

  float sign;
  if (abs(point_depth - point_cam[2]) < 0.0001){
    sign = 1; // avoid NaN
  }else{
    sign = (point_depth - point_cam[2])/abs(point_depth - point_cam[2]);
  }
  tsdf_data_GPU[vox_idx] = float(sign);
  for (int iix = MAX(0,x-search_region); iix < MIN((int)vox_size[0],x+search_region+1); iix++)
    for (int iiy = MAX(0,y-search_region); iiy < MIN((int)vox_size[1],y+search_region+1); iiy++)
      for (int iiz = MAX(0,z-search_region); iiz < MIN((int)vox_size[2],z+search_region+1); iiz++){
        int iidx = iiz * vox_size[0] * vox_size[1] + iiy * vox_size[0] + iix;
        if (vox_binary_GPU[iidx] > 0){
          float xd = abs(x - iix);
          float yd = abs(y - iiy);
          float zd = abs(z - iiz);
          float tsdf_value = sqrtf(xd * xd + yd * yd + zd * zd)/(float)search_region;
          if (tsdf_value < abs(tsdf_data_GPU[vox_idx])){
            tsdf_data_GPU[vox_idx] = float(tsdf_value*sign);
          }
        }
      }
}

__global__ void tsdfTransform__(int vox_Size, float* tsdf_data_GPU) {

  int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (vox_idx >= vox_Size){
    return;
  }
  float value = float(tsdf_data_GPU[vox_idx]);
  float sign = 1;
  if (abs(value) > 0.001)
    sign = value/abs(value);
  tsdf_data_GPU[vox_idx] = sign*(MAX(0.001,(1.0-abs(value))));
}

__global__ void downSample__(float* readin_bin, float* vox_size,
    int sampleRatio, float* label, float* tsdf,
    float* tsdf_downsample) {
      
  int xrange = vox_size[0]/sampleRatio;
  int yrange = vox_size[1]/sampleRatio;
  int zrange = vox_size[2]/sampleRatio;
  int vox_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(vox_idx >= xrange*yrange*zrange)
    return;
  int z = vox_idx / (xrange * yrange);
  int y = (vox_idx - z * xrange * yrange) / xrange;
  int x = vox_idx - z * xrange * yrange - y * xrange;
  int count[13] = {0};
  float sum = 0.0f;
  for(int i = z*sampleRatio; i < (z+1)*sampleRatio; i++) {
    for(int j = y*sampleRatio; j < (y+1)*sampleRatio; j++) {
      for(int k = x*sampleRatio; k < (x+1)*sampleRatio; k++) {
        int bin_index = i*vox_size[0]*vox_size[1]+j*vox_size[0]+k;
        sum += tsdf[bin_index];
        if((int)(readin_bin[bin_index])!=255)
          count[(int)(readin_bin[bin_index])]++;
        else
          count[12]++;
      }
    }
  }
  float sss = sampleRatio*sampleRatio*sampleRatio;
  tsdf_downsample[vox_idx] = sum / sss;
  if(count[0]+count[12] > 0.95 * sss) {
    int max = count[0];
    int maxindex = 0;
    for(int i = 1; i < 13; i++)
      if(max < count[i]) {
        max = count[i];
        maxindex = i;
      }
    if(maxindex == 12)
      label[vox_idx] = 255;
    else
      label[vox_idx] = maxindex;
  }
  else {
    int max = count[1];
    int maxindex = 1;
    for(int i = 2; i < 12; i++)
      if(max < count[i]) {
        max = count[i];
        maxindex = i;
      }
    label[vox_idx] = maxindex;
  }
}

static void init_numpy() {
  import_array();
}

extern "C" {

PyObject* _TSDF(PyObject *args)
{
  init_numpy();

	PyArrayObject* arr_depth = NULL;
	PyArrayObject* arr_camk = NULL;
	PyArrayObject* arr_camp = NULL;
	PyArrayObject* arr_voxo = NULL;
	PyArrayObject* arr_voxs = NULL;
  PyArrayObject* tsdf = NULL;
  PyArrayObject* mapping = NULL;
	float voxu, voxm;
	int h, w;
	if (!PyArg_ParseTuple(args, "O!O!O!O!fO!fiiO!O!", &PyArray_Type, &arr_depth,
    &PyArray_Type, &arr_camk, &PyArray_Type, &arr_camp, &PyArray_Type, &arr_voxo,
    &voxu, &PyArray_Type, &arr_voxs, &voxm, &h, &w, &PyArray_Type, &tsdf,
    &PyArray_Type, &mapping)) {
		printf("Fail Parsing Tuple!\n");
		return PyLong_FromLong(-1);
	}
	float* pdepth = (float*)PyArray_DATA(arr_depth);
	float* pcamk = (float*)PyArray_DATA(arr_camk);
	float* pcamp = (float*)PyArray_DATA(arr_camp);
	float* pvoxo = (float*)PyArray_DATA(arr_voxo);
	float* pvoxs = (float*)PyArray_DATA(arr_voxs);
	
	float* pvoxb = (float*)malloc((int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float));
	memset(pvoxb, 0, pvoxs[0]*pvoxs[1]*pvoxs[2]*sizeof(float));

	float* ptsdf = (float*)PyArray_DATA(tsdf);
	float* pmapping = (float*)PyArray_DATA(mapping);

	float *gpudepth, *gpucamk, *gpucamp, *gpuvoxo, *gpuvoxs, *gpuvoxb, *gpumapping, *gputsdf;
	cudaMalloc(&gpudepth, w*h*sizeof(float));
	cudaMalloc(&gpucamk, 9*sizeof(float));
	cudaMalloc(&gpucamp, 16*sizeof(float));
	cudaMalloc(&gpuvoxo, 3*sizeof(float));
	cudaMalloc(&gpuvoxs, 3*sizeof(float));
	cudaMalloc(&gpuvoxb, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float));
	cudaMalloc(&gpumapping, w*h*sizeof(float));
	cudaMalloc(&gputsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float));
	
	cudaMemcpy(gpudepth, pdepth, w*h*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpucamk, pcamk, 9*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpucamp, pcamp, 16*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuvoxo, pvoxo, 3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuvoxs, pvoxs, 3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuvoxb, pvoxb, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpumapping, pmapping, w*h*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gputsdf, ptsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float), cudaMemcpyHostToDevice);

	depth2Grid__<<<w,h>>>(w, h, gpucamk, gpucamp, voxu,
		gpuvoxs, gpuvoxo, gpudepth, gpuvoxb, gpumapping);

	cudaMemcpy(pmapping, gpumapping, w*h*sizeof(float), cudaMemcpyDeviceToHost);

	SquaredDistanceTransform__<<<((int)(pvoxs[0]*pvoxs[1]*pvoxs[2])+1023)/1024, 1024>>>(
		w, h, voxu, voxm, gpuvoxs, gpuvoxo, gpucamk, gpucamp, gpuvoxb,
		gpudepth, gputsdf);

	cudaMemcpy(ptsdf, gputsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(gpudepth);
	cudaFree(gpucamk);
	cudaFree(gpucamp);
	cudaFree(gpuvoxo);
	cudaFree(gpuvoxs);
	cudaFree(gpumapping);
	cudaFree(gpuvoxb);
	cudaFree(gputsdf);

	free(pvoxb);

	return PyLong_FromLong(0);
}

PyObject* _TSDFTransform(PyObject *args)
{
  init_numpy();

	PyArrayObject* arr_tsdf = NULL;
	PyArrayObject* arr_voxs = NULL;
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &arr_tsdf, &PyArray_Type, &arr_voxs)) {
		printf("Fail Parsing Tuple!\n");
		return PyLong_FromLong(-1);
	}
	float* ptsdf = (float*)PyArray_DATA(arr_tsdf);
	float* pvoxs = (float*)PyArray_DATA(arr_voxs);

	float *gputsdf;
	cudaMalloc(&gputsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float));
	
	cudaMemcpy(gputsdf, ptsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float), cudaMemcpyHostToDevice);
	
	tsdfTransform__<<<((int)(pvoxs[0]*pvoxs[1]*pvoxs[2])+1023)/1024, 1024>>>(
		(int)(pvoxs[0]*pvoxs[1]*pvoxs[2]), gputsdf);

	cudaMemcpy(ptsdf, gputsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(gputsdf);

	return PyLong_FromLong(0);
}

PyObject* _DownSampleLabel(PyObject *args)
{
  init_numpy();

	PyArrayObject* arr_readin = NULL;
	PyArrayObject* arr_voxs = NULL;
	PyArrayObject* arr_tsdf = NULL;
  PyArrayObject* label = NULL;
  PyArrayObject* tsdf_downsample = NULL;
	int sampleRatio;
	if (!PyArg_ParseTuple(args, "O!O!iO!O!O!", &PyArray_Type, &arr_readin, &PyArray_Type, &arr_voxs,
      &sampleRatio, &PyArray_Type, &arr_tsdf, &PyArray_Type, &label, &PyArray_Type, &tsdf_downsample)) {
		printf("Fail Parsing Tuple!\n");
		return PyLong_FromLong(-1);
	}
	float* preadin = (float*)PyArray_DATA(arr_readin);
	float* pvoxs = (float*)PyArray_DATA(arr_voxs);
	float* ptsdf = (float*)PyArray_DATA(arr_tsdf);

	int dims = (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])/(sampleRatio*sampleRatio*sampleRatio);
	
	float* plabel = (float*)PyArray_DATA(label);
	float* ptsdf_downsample = (float*)PyArray_DATA(tsdf_downsample);
	
	float *gpureadin, *gpuvoxs, *gpulabel, *gputsdf, *gputsdf_downsample;
	cudaMalloc(&gpureadin, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float));
	cudaMalloc(&gpuvoxs, 3*sizeof(float));
	cudaMalloc(&gpulabel, dims*sizeof(float));
	cudaMalloc(&gputsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float));
	cudaMalloc(&gputsdf_downsample, dims*sizeof(float));
	
	cudaMemcpy(gpureadin, preadin, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuvoxs, pvoxs, 3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gpulabel, plabel, dims*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gputsdf, ptsdf, (int)(pvoxs[0]*pvoxs[1]*pvoxs[2])*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(gputsdf_downsample, ptsdf_downsample, dims*sizeof(float), cudaMemcpyHostToDevice);
	
	downSample__<<<(dims+1023)/1024, 1024>>>(gpureadin, gpuvoxs,
		sampleRatio, gpulabel, gputsdf, gputsdf_downsample);

	cudaMemcpy(plabel, gpulabel, dims*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(ptsdf_downsample, gputsdf_downsample, dims*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(gpureadin);
	cudaFree(gpuvoxs);
	cudaFree(gpulabel);
	cudaFree(gputsdf);
	cudaFree(gputsdf_downsample);
	
	return PyLong_FromLong(0);
}

}