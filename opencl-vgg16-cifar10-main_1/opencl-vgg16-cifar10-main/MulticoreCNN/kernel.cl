


__kernel void im2col(
    __global float* inputs,
    __global float3* cols,
    const int N) {

    // 한 workitem당 한 개의 입력 셀에 대해 컬럼생성
    const int g_i = get_global_id(2);
    const int g_z = get_global_id(1);
    const int g_y = get_global_id(0) / N;
    const int g_x = get_global_id(0) % N;

    const int M = get_global_size(1); // 입력층 개수
    const int input_size = N * N * M; // 입력층 크기
    const int i_offset = input_size * g_i; // 현재 이미지 번호만큼의 오프셋
    __global float* input = inputs + i_offset;

    const int col_size = 3 * N * N * M; // 윈도우 크기
    const int c_offset = col_size * g_i; // 현재 이미지 번호만큼의 오프셋
    __global float3* col = cols + c_offset;

    float temp[9];

    for (int y = g_y - 1, j = 0; y <= g_y + 1; y++) {
        for (int x = g_x - 1; x <= g_x + 1; x++, j++) {
            temp[j] = (x >= 0 && x < N && y >= 0 && y < N) ? input[N * N * g_z + N * y + x] : 0;
        }
    }

    for (int i = 0; i < 9; i += 3) {
        col[3 * N * N * g_z + 3 * N * g_y + 3 * g_x + i / 3] = (float3)(temp[i], temp[i + 1], temp[i + 2]);
    }
}

__kernel void conv(
    __global float3* cols,
    __global float* network,
    __global float* outputs,
    __local float3* i_tile,
    const int N_OFFSET,
    const int N,
    const int inDim,
    const int L_N) {
    /*
     각 입력층마다 이렇게 윈도우가 생성되므로 여기에 필터를 곱함
      ex. 필터 예시
      -1 0 1
      -1 0 1
      -1 0 1
      => 벡터로 본다면 => -1 0 1 -1 0 1 -1 0 1

      행렬곱은 다음과 같음
                                                출력층(x, y)
      0  0  0  0  1  2  0  5  6    -1      ?    (0, 0)
      0  0  0  1  2  3  5  6  7     0      ?    (1, 0)
      0  0  0  2  3  4  6  7  8     1      ?    (2, 0)
      0  0  0  3  4  0  7  8  0    -1      ?    (3, 0)
      0  1  2  0  5  6  0  9 10  *  0   =  ?    (0, 1)
                                    1      ?    (1, 1)
                                                 ...
     이 때 행렬곱의 경우 타일링하기 수월하므로 타일링 적용하기
      */

    const int g_i = get_global_id(2);
    const int g_z = get_global_id(1);
    const int g_y = get_global_id(0) / N;
    const int g_x = get_global_id(0) % N;

    const int l_z = get_local_id(1);
    const int l_y = get_local_id(0) / L_N;
    const int l_x = get_local_id(0) % L_N;
    const int L_DIM = get_local_size(1);

    const int outDim = get_global_size(1);
    __global float* weights = network + N_OFFSET;
    __global float* biases = weights + 9 * inDim * outDim;
    __global float* output = outputs + N * N * outDim * g_i;

    const int one_col_size = 3 * N * N;
    const int w_offset = one_col_size * inDim * g_i;


    float sum = 0.0f;
    for (int i = 0; i < inDim; i += L_DIM) {
        // 타일링을 적용할 땐 실제로 계산을 수행할 영역만 로컬로 불러온 후 
        // 각자 개별적으로 접근하여 수행함
        const int i_offset = 3 * L_N * L_N * l_z + 3 * L_N * l_y + 3 * l_x;
        for (int j = 0; j < 3; j++) {
            i_tile[i_offset + j] = cols[w_offset + one_col_size * (i + l_z) + 3 * N * g_y + 3 * g_x + j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 캐싱한 부분에 대해서 행렬곱

        for (int j = l_z; j < L_DIM + l_z; j++) {
            float3* col = i_tile + 3 * L_N * L_N * (j % L_DIM) + 3 * L_N * l_y + 3 * l_x;

            //__global float3* col = cols + w_offset + one_col_size * (i + j % L_DIM) + 3 * N * g_y + 3 * g_x;
            float* weight = weights + 9 * inDim * g_z + 9 * (i + j % L_DIM);
            for (int k = 0; k < 3; k++) {
                sum += dot(*(col + k), (float3) (*(weight), *(weight + 1), *(weight + 2)));
                weight = weight + 3;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    sum += biases[g_z];
    if (sum < 0) {
        sum = 0;
    }
    output[N * N * g_z + N * g_y + g_x] = sum;
}

__kernel void max_pooling(
    __global float* inputs,
    __global float* outputs,
    const int in_nbyn,
    const int out_nbyn,
    const int inDim,
    const int outDim) {
    // 글로벌 사이즈 = (out_nbyn * out_nbyn, outDim, 이미지 개수)
    // 로컬 사이즈 = (out_nbyn * out_nbyn, 1, 1)
    const int image_id = get_global_id(2);
    const int g_z = get_global_id(1);
    const int l_y = get_local_id(0) / out_nbyn;
    const int l_x = get_local_id(0) % out_nbyn;

    __global float* input = inputs + image_id * in_nbyn * in_nbyn * inDim;
    __global float* output = outputs + image_id * out_nbyn * out_nbyn * outDim;

    int x = l_x << 1;
    int y = l_y << 1;

    const int i_offset = in_nbyn * in_nbyn * g_z;
    float max = input[i_offset + in_nbyn * y + x];
    if (max < input[i_offset + in_nbyn * y + ++x])
        max = input[i_offset + in_nbyn * y + x];
    if (max < input[i_offset + in_nbyn * ++y + x])
        max = input[i_offset + in_nbyn * y + x];
    if (max < input[i_offset + in_nbyn * y + --x])
        max = input[i_offset + in_nbyn * y + x];

    output[out_nbyn * out_nbyn * g_z + out_nbyn * l_y + l_x] = max;
}


__kernel void fc_layer(
    __global float* inputs,
    __global float* network,
    __global float* outputs,
    const int n_offset,
    const int inDim,
    const int outDim) {
    // 글로벌 사이즈 = (outDim, 이미지 번호)

    const int g_i = get_global_id(0);
    const int image_id = get_global_id(1);
    const int w_offset = inDim * outDim;
    __global float* weights = network + n_offset;
    __global float* biases = weights + w_offset;
    __global float* input = inputs + image_id * inDim;
    __global float* output = outputs + image_id * outDim;

    float sum = 0.0f;
    for (int i = 0; i < inDim; i++) {
        sum += input[i] * weights[inDim * g_i + i];
    }
    sum += biases[g_i];
    if (sum < 0.0f) {
        sum = 0.0f;
    }
    output[g_i] = sum;
}


//
// 이 주석을 지우면 에러 생길수도?
//

