#pragma once

#ifndef _UTILS_H_
#define _UTILS_H_

#pragma warning(disable:4996)

#include <stdlib.h>
#include <stdio.h>


#define CHECK_ERROR(error) \
    if (error != CL_SUCCESS) {\
        printf("ERROR [%s:%d]: %s\n", __FILE__, __LINE__, getErrorMessage(error)); \
        exit(EXIT_FAILURE); \
    }
const char* getErrorMessage(int error);

char* get_source_code(const char* file_name, size_t* len);

void compareLayerResult(float* layer, const char* path, int len);

const int INPUT_DIM[] = {
    3, 64,
    64,

    64,128,
    128,

    128, 256, 256,
    256,

    256, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    512
};

const int OUTPUT_DIM[] = {
    64, 64,
    64,

    128, 128,
    128,

    256, 256, 256,
    256,

    512, 512, 512,
    512,

    512, 512, 512,
    512,

    512,
    512,
    10
};

// CIFAR10을 사용하기 때문에 
// 이미지 사이즈를 의미하는 값이 
// 224x224가 아니라 32x32가 되었다.
const int NBYN[] = {
    32, 32,
    16,

    16, 16,
    8,

    8, 8, 8,
    4,

    4, 4, 4,
    2,

    2, 2, 2,
    1,

    1,
    1,
    1
};

// 각 컨볼루션 및 FC단계에 사용되는 가중치와 바이어스 배열의 시작 offset
// max_pooling단계에는 offset이 필요없으므로 0을 저장하고 있음.
// 각 커널에서 network에 offset값을 포인터 연산하면 
//  그 단계에서 필요한 가중치와 바이어스 배열의 시작 주소가 됨
const int NETWORK_OFFSET[] = {
    0,         // 0  -> 1
    1792,      // 1  -> 2
    0,     // 2  -> 3 (max_pooling)
    38720,     // 3  -> 4 9 * 64 * 128
    112576,    // 4  -> 5
    0,    // 5  -> 6 (max_pooling)
    260160,    // 6  -> 7
    555328,    // 7  -> 8
    1145408,   // 8  -> 9
    0,   // 9  -> 10 (max_pooling)
    1735488,   // 10 -> 11
    2915648,   // 11 -> 12
    5275456,   // 12 -> 13
    0,   // 13 -> 14 (max_pooling)
    7635264,   // 14 -> 15 
    9995072,   // 15 -> 16
    12354880,  // 16 -> 17 z
    0,  // 17 -> 18 (max_pooling)
    14714688,  // 18 -> 19
    14977344,  // 19 -> 20
    15240000   // 20 -> 21
};

const size_t L_DIM[] = {
    1, 16,
    0,          // max_pooling

    16, 16,
    0,          // max_pooling

    32, 32, 32,
    0,          // max_pooling

    32, 32, 32,
    0,          // max_pooling

    128, 128, 128,
    0,          // max_pooling
};

// local workitem들의 개수를 256으로 맞추려고 함.
const size_t L_N[] = {
    16, 4,
    0,          // max_pooling

    4, 4,
    0,          // max_pooling
    
    4, 4, 4,
    0,          // max_pooling

    4, 4, 4,
    0,          // max_pooling

    2, 2, 2,
    0,          // max_pooling
};
#endif