#ifndef _CNN_H
#define _CNN_H

#define CL_TARGET_OPENCL_VERSION 300

#include <time.h>
#include <stdio.h>
#include <CL/cl.h>
#include <math.h>

#pragma warning(disable:4996)


void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image);
void compare(const char* filename, int num_of_image);
void cnn_init(float* images, float* network, int num_images);
void cnn(float* images, float* network, int* labels, float* confidences, int num_images);

void enqueueConvolution(cl_mem input, cl_mem output, int index);
void enqueueSlidingWindow(cl_mem input, cl_mem output, int index);
void enqueueMaxPooling(cl_mem input, cl_mem output, int index);
void enqueueFullyConnectedLayer(cl_mem input, cl_mem output, int index);
void enqueueComparing(cl_mem layer, int i);
void softmax(float* input, int N);
int findmax(float* input, int classNum);
double get_etime_by_event(cl_event* event);

#endif 