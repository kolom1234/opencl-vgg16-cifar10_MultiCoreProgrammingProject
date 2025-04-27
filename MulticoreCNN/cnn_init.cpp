#include "cnn.h"
#include "utils.h"
#include <time.h>
#include <stdio.h>
#include <CL/cl.h>
#include <math.h>


extern cl_platform_id platform;
extern cl_program program;
extern cl_context context;
extern cl_device_id device;
extern cl_command_queue queue;
extern cl_int error;

extern const char* source_code;
extern size_t code_length;
extern cl_kernel conv;
extern cl_kernel im2col;
extern cl_kernel max_pooling;
extern cl_kernel fc_layer;

extern cl_mem inputs;
extern cl_mem ping_layer;
extern cl_mem layers[21];
extern cl_mem col_layer;
extern cl_mem pong_layer;
extern cl_mem network;
extern float* result;


void cnn_init(float* images, float* networks, int num_of_image) {
    /************************************************************/
    /*                      Initialization                      */
    /************************************************************/
    error = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(error);
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(error);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
    CHECK_ERROR(error);
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
    CHECK_ERROR(error);

    /************************************************************/
    /*                       Create Buffer                      */
    /************************************************************/

    // �̹��� ���� ����
    // �̹��� ���ۿ� �� ���� 3õ���� �̹����� �Ҵ��� ���� ������, 
    // �Ҵ簡���� �ִ� �޸𸮺��� �� Ŀ���� �ʱ� ���� �����ߴ� num_of_image��ŭ�� �Ҵ��Ѵ�.
    inputs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 32 * 32 * 3 * num_of_image, NULL, &error);
    CHECK_ERROR(error);
    
    // ����, ���̾ ���� ����
    // ���Ϳ� ���̾ ���۸� ������ ������� �� ���� ���۷� ����
    network = clCreateBuffer(context, CL_MEM_READ_ONLY, 60980520, NULL, &error);
    CHECK_ERROR(error);

    // �����̵� ������� ����� ���̾� ����, ũ��� �ִ� ũ��� ������
    col_layer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float3) * 32 * 32 * 64 * 3 * num_of_image, NULL, &error);
    CHECK_ERROR(error);

    // num_of_image�� ���� ���̾� ���� ����
    for (int i = 0; i < 21; i++) {
        layers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * NBYN[i] * NBYN[i] * OUTPUT_DIM[i] * num_of_image, NULL, &error);
        CHECK_ERROR(error);
    }
    // ������� �о�� ȣ��Ʈ ���� ����
    result = new float[OUTPUT_DIM[20] * num_of_image];

    /************************************************************/
    /*                       Create Kernel                      */
    /************************************************************/
    size_t len;
    source_code = get_source_code("kernel.cl", &len);

    program = clCreateProgramWithSource(context, 1, &source_code, &code_length, &error);
    CHECK_ERROR(error);

    error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    char buildResult[4096];
    size_t size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildResult), buildResult, &size);
    printf("%s\n", buildResult);
    CHECK_ERROR(error);

    conv = clCreateKernel(program, "conv", &error);
    CHECK_ERROR(error);

    im2col = clCreateKernel(program, "im2col", &error);
    CHECK_ERROR(error);

    max_pooling = clCreateKernel(program, "max_pooling", &error);
    CHECK_ERROR(error);

    fc_layer = clCreateKernel(program, "fc_layer", &error);
    CHECK_ERROR(error);
}


// softmax�� findmax�� �ݺ�Ƚ���� ũ�� �ʾ� ������ ���� ����.
// cnn_opencl.cpp���� �۾��� �� ���ذ� �Ǿ� �̰����� �ű�.
void softmax(float* input, int N) {
    int i;
    float max = input[0];
    for (i = 1; i < N; i++) {
        if (max < input[i]) max = input[i];
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += exp(input[i] - max);
    }
    for (i = 0; i < N; i++) {
        input[i] = exp(input[i] - max) / (sum + 1e-7f);
    }
}

int findmax(float* input, int classNum) {
    int i;
    int maxIndex = 0;
    float max = 0;
    for (i = 0; i < classNum; i++) {
        if (max < input[i]) {
            max = input[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

