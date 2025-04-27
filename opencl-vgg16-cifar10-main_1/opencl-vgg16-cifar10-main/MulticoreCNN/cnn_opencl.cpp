#include "cnn.h"
#include "utils.h"

#define NUM_OF_IMAGE 250

extern const int INPUT_DIM[];
extern const int OUTPUT_DIM[];
extern const int NBYN[];

cl_platform_id platform;
cl_program program;
cl_context context;
cl_device_id device;
cl_command_queue queue;
cl_int error;
cl_event event;
double elapsed_time[34] = { 0 };

const char* source_code;
size_t code_length;
cl_kernel conv;
cl_kernel im2col;
cl_kernel max_pooling;
cl_kernel fc_layer;

cl_mem inputs;
cl_mem layers[21];
cl_mem col_layer;
cl_mem network; 
float* result;

void cnn(float* images, float* networks, int* labels, float* confidences, int total_image_num) {

    cnn_init(images, networks, NUM_OF_IMAGE);

    time_t start, end;
    start = clock();

    //TODO

    /************************************************************/
    /*                      Enqueue Commands                    */
    /************************************************************/

    // 필터, 바이어스 버퍼는 한 번만 쓰고 계속 참조
    error = clEnqueueWriteBuffer(queue, network, CL_TRUE, 0, 60980520, networks, 0, NULL, NULL);
    CHECK_ERROR(error);


    int image_offset = 0;
    int offset;
    for (int i = 0; i < total_image_num / NUM_OF_IMAGE; i++) {
        error = clEnqueueWriteBuffer(queue, inputs, CL_TRUE, 0, sizeof(float) * 32 * 32 * 3 * NUM_OF_IMAGE, images + image_offset, 0, NULL, NULL);
        image_offset += 32 * 32 * 3 * NUM_OF_IMAGE;
        CHECK_ERROR(error);

        enqueueSlidingWindow(inputs, col_layer, 0);
        elapsed_time[0] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[0], 0);
        elapsed_time[1] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[0], col_layer, 1);
        elapsed_time[2] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[1], 1);
        elapsed_time[3] += get_etime_by_event(&event);

        enqueueMaxPooling(layers[1], layers[2], 2);
        elapsed_time[4] += get_etime_by_event(&event);

        enqueueSlidingWindow(layers[2], col_layer, 3);
        elapsed_time[5] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[3], 3);
        elapsed_time[6] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[3], col_layer, 4);
        elapsed_time[7] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[4], 4);
        elapsed_time[8] += get_etime_by_event(&event);

        enqueueMaxPooling(layers[4], layers[5], 5);
        elapsed_time[9] += get_etime_by_event(&event);

        enqueueSlidingWindow(layers[5], col_layer, 6);
        elapsed_time[10] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[6], 6);
        elapsed_time[11] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[6], col_layer, 7);
        elapsed_time[12] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[7], 7);
        elapsed_time[13] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[7], col_layer, 8);
        elapsed_time[14] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[8], 8);
        elapsed_time[15] += get_etime_by_event(&event);

        enqueueMaxPooling(layers[8], layers[9], 9);
        elapsed_time[16] += get_etime_by_event(&event);

        enqueueSlidingWindow(layers[9], col_layer, 10);
        elapsed_time[17] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[10], 10);
        elapsed_time[18] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[10], col_layer, 11);
        elapsed_time[19] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[11], 11);
        elapsed_time[20] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[11], col_layer, 12);
        elapsed_time[21] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[12], 12);
        elapsed_time[22] += get_etime_by_event(&event);

        enqueueMaxPooling(layers[12], layers[13], 13);
        elapsed_time[23] += get_etime_by_event(&event);

        enqueueSlidingWindow(layers[13], col_layer, 14);
        elapsed_time[24] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[14], 14);
        elapsed_time[25] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[14], col_layer, 15);
        elapsed_time[26] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[15], 15);
        elapsed_time[27] += get_etime_by_event(&event);
        enqueueSlidingWindow(layers[15], col_layer, 16);
        elapsed_time[28] += get_etime_by_event(&event);
        enqueueConvolution(col_layer, layers[16], 16);
        elapsed_time[29] += get_etime_by_event(&event);

        enqueueMaxPooling(layers[16], layers[17], 17);
        elapsed_time[30] += get_etime_by_event(&event);

        enqueueFullyConnectedLayer(layers[17], layers[18], 18);
        elapsed_time[31] += get_etime_by_event(&event);
        enqueueFullyConnectedLayer(layers[18], layers[19], 19);
        elapsed_time[32] += get_etime_by_event(&event);
        enqueueFullyConnectedLayer(layers[19], layers[20], 20);
        elapsed_time[33] += get_etime_by_event(&event);

        error = clEnqueueReadBuffer(queue, layers[20], CL_FALSE, 0, sizeof(float) * OUTPUT_DIM[20] * NUM_OF_IMAGE, result, 0, NULL, NULL);

        offset = 0;
        for (int j = 0; j < NUM_OF_IMAGE; j++) {
            softmax(result + offset, OUTPUT_DIM[20]);
            labels[j + i * NUM_OF_IMAGE] = findmax(result + offset, 10);
            confidences[j + i * NUM_OF_IMAGE] = result[offset + labels[j + i * NUM_OF_IMAGE]];
            offset += OUTPUT_DIM[20];
        }
    }

    for (int i = 0; i < 34; i++) {
        printf("%f\n", elapsed_time[i]);
    }
    
    end = clock();
    printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

    clReleaseMemObject(inputs);
    clReleaseMemObject(network);
    clReleaseKernel(im2col);
    clReleaseKernel(conv);
    clReleaseKernel(max_pooling);
    clReleaseKernel(fc_layer);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseDevice(device);
    clReleaseContext(context);
    clReleaseEvent(event);
    /*for (int i = 0; i < 22; i++) {
        clReleaseEvent(event[i]);
    }*/
}

void enqueueSlidingWindow(cl_mem input, cl_mem window , int i) {
    // NxN 배열을 슬라이딩 윈도우로 만들기
    error = clSetKernelArg(im2col, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(error);
    error = clSetKernelArg(im2col, 1, sizeof(cl_mem), &window);
    CHECK_ERROR(error);
    error = clSetKernelArg(im2col, 2, sizeof(cl_int), &NBYN[i]);
    CHECK_ERROR(error);

    size_t global_work_size[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(INPUT_DIM[i]),
        static_cast<size_t>(NUM_OF_IMAGE)
    };
    size_t local_work_size[3] = {
        //L_N[i] * L_N[i],
        NBYN[i] * NBYN[i],
        //L_DIM[i],
        1,
        1
    };

    error = clEnqueueNDRangeKernel(queue, im2col, 3, NULL, global_work_size, local_work_size, 0, NULL, &event);
    CHECK_ERROR(error);
}

void enqueueConvolution(cl_mem windows, cl_mem output, int i) {
    // 슬라이딩 윈도우가 적용된 레이어에 대해서 (1xN) * (Nx1) 행렬곱 수행
    error = clSetKernelArg(conv, 0, sizeof(cl_mem), &windows);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 1, sizeof(cl_mem), &network);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 2, sizeof(cl_mem), &output);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 3, sizeof(cl_float3) * 3 * L_N[i] * L_N[i] * L_DIM[i], NULL);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 4, sizeof(cl_int), &NETWORK_OFFSET[i]);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 5, sizeof(cl_int), &NBYN[i]);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 6, sizeof(cl_int), &INPUT_DIM[i]);
    CHECK_ERROR(error);
    error = clSetKernelArg(conv, 7, sizeof(cl_int), &L_N[i]);
    CHECK_ERROR(error);

    size_t global_work_size[3] = {
        NBYN[i] * NBYN[i],
        OUTPUT_DIM[i],
        NUM_OF_IMAGE
    };
    size_t local_work_size[3] = {
        //NBYN[i] * NBYN[i],
        L_N[i] * L_N[i],
        L_DIM[i],
        1
    };

    error = clEnqueueNDRangeKernel(queue, conv, 3, NULL, global_work_size, local_work_size, 0, NULL, &event);
    CHECK_ERROR(error);
}

void enqueueMaxPooling(cl_mem input, cl_mem output, int i) {
    error = clSetKernelArg(max_pooling, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 1, sizeof(cl_mem), &output);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 2, sizeof(cl_int), &NBYN[i-1]);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 3, sizeof(cl_int), &NBYN[i]);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 4, sizeof(cl_int), &INPUT_DIM[i]);
    CHECK_ERROR(error);
    error = clSetKernelArg(max_pooling, 5, sizeof(cl_int), &OUTPUT_DIM[i]);
    CHECK_ERROR(error);

    // 한 이미지를 처리하기 위한 출력채널 크기 * 이미지 개수(3000)
    size_t global_work_size[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(OUTPUT_DIM[i]),
        static_cast<size_t>(NUM_OF_IMAGE)
    };

    // 출력채널의 1개 채널
    size_t local_work_size[3] = {
        static_cast<size_t>(NBYN[i] * NBYN[i]),
        static_cast<size_t>(1),
        static_cast<size_t>(1)
    };

    error = clEnqueueNDRangeKernel(queue, max_pooling, 3, NULL, global_work_size, local_work_size, 0, NULL, &event);
    CHECK_ERROR(error);
}

void enqueueFullyConnectedLayer(cl_mem input, cl_mem output, int i) {
    error = clSetKernelArg(fc_layer, 0, sizeof(cl_mem), &input);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 1, sizeof(cl_mem), &network);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 2, sizeof(cl_mem), &output);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 3, sizeof(cl_int), &NETWORK_OFFSET[i]);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 4, sizeof(cl_int), &INPUT_DIM[i]);
    CHECK_ERROR(error);
    error = clSetKernelArg(fc_layer, 5, sizeof(cl_int), &OUTPUT_DIM[i]);
    CHECK_ERROR(error);

    // 한 이미지를 처리하기 위한 출력채널 크기 * 이미지 개수(3000)
    size_t global_work_size[2] = {
        static_cast<size_t>(OUTPUT_DIM[i]),
        static_cast<size_t>(NUM_OF_IMAGE)
    };

    // 출력채널의 크기
    size_t local_work_size[2] = {
        static_cast<size_t>(OUTPUT_DIM[i]),
        static_cast<size_t>(1)
    };

    error = clEnqueueNDRangeKernel(queue, fc_layer, 2, NULL, global_work_size, local_work_size, 0, NULL, &event);
    CHECK_ERROR(error);
}

double get_etime_by_event(cl_event* event) {
    clWaitForEvents(1, event);
    cl_ulong event_start, event_end;
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &event_start, NULL);
    clGetEventProfilingInfo(*event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &event_end, NULL);
    return (event_end - event_start) * 1.0E-9;
}
