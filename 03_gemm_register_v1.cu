// optimize sgemm
#include <stdio.h>
#include <stdlib.h>
// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// cal offset from row col and ld , in row-major matrix, ld is the width of the matrix
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// transfer float4
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define checkCudaErrors(func)				\
{									\
    cudaError_t e = (func);			\
    if(e != cudaSuccess)						                \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}


// K: ldA
// N: ldB
template <
    const int BLOCK_SIZE_M,  // 128 height of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // 8 width of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // 128 width of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // 8 height of block of C that each thread calculate
    const int THREAD_SIZE_X,  // 8 width of block of C that each thread calculate
    const bool ENABLE_DOUBLE_BUFFER // whether enable double buffering or not
    > 
__global__ void Sgemm( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int N,
    const int K) {

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;  // 0-16
    // the threads number in Block of X,Y
    const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;  // 16
    const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;  // 16
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;  // 256
    // thread id in cur Block
    const int tid = ty * THREAD_X_PER_BLOCK + tx;  // 0-256

    // shared memory
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];  // size=2* 8*128
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];  // size=2* 8*128
    // registers for A and B
    float frag_a[2][THREAD_SIZE_Y];  // size=2*8
    float frag_b[2][THREAD_SIZE_X];
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};  // size=8*8

    // A B 分块
    A = &A[(by * BLOCK_SIZE_M) * K];
    B = &B[bx * BLOCK_SIZE_N];

    
    int A_TILE_THREAD_PER_ROW = 2;
    int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;  // 0/4
    int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;  // 0-128
    int B_TILE_THREAD_PER_ROW = 32;  // 128 /4
    int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;  // 0-31 *4
    int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;  // 0-8

    // TODO: 大block循环: A 分成 128*8 大小的sharemem块
    for(int bki=0; bki < K; bki += BLOCK_SIZE_K){  // bki: 0, 8, 16, ..., K
        // 1 load A to share mem: 每个block读取128*8个数据，256个线程，每个线程读取4个float数
        As[0][A_TILE_COL+0][A_TILE_ROW_START] = A[A_TILE_ROW_START*K + bki + A_TILE_COL + 0];  // 这里A已经切分为 128*K 形状的块
        As[0][A_TILE_COL+1][A_TILE_ROW_START] = A[A_TILE_ROW_START*K + bki + A_TILE_COL + 1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START] = A[A_TILE_ROW_START*K + bki + A_TILE_COL + 2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START] = A[A_TILE_ROW_START*K + bki + A_TILE_COL + 3];
        // 2 load B to share mem
        Bs[0][B_TILE_ROW_START][B_TILE_COL+0] = B[(B_TILE_ROW_START+bki)*N + B_TILE_COL + 0];
        Bs[0][B_TILE_ROW_START][B_TILE_COL+1] = B[(B_TILE_ROW_START+bki)*N + B_TILE_COL + 1];
        Bs[0][B_TILE_ROW_START][B_TILE_COL+2] = B[(B_TILE_ROW_START+bki)*N + B_TILE_COL + 2];
        Bs[0][B_TILE_ROW_START][B_TILE_COL+3] = B[(B_TILE_ROW_START+bki)*N + B_TILE_COL + 3];
        __syncthreads();

        // TODO: 小循环：A 128*8的块再细分为 8*1大小的寄存器块
        for(int rki=0; rki<BLOCK_SIZE_K; ++rki){
            // 3 load share mem A to register 
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y){
                frag_a[0][thread_y] = As[0][rki][THREAD_SIZE_Y * ty + thread_y];
            }
            // 4 load share mem B to register 
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x){
                frag_b[0][thread_x] = Bs[0][rki][THREAD_SIZE_X * tx + thread_x];
            }
            // 5 计算8 * 8 次FMA
            for (int cal_y = 0; cal_y < THREAD_SIZE_Y; ++cal_y) {
                for (int cal_x = 0; cal_x < THREAD_SIZE_X; ++cal_x) {
                    accum[cal_y][cal_x] += frag_a[0][cal_y] * frag_b[0][cal_x];
                }
            }
        }
    }
    // C赋值
    for (int cal_y = 0; cal_y < THREAD_SIZE_Y; ++cal_y) {
        for (int cal_x = 0; cal_x < THREAD_SIZE_X; ++cal_x) {
            C[(BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + cal_y)*N + BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + cal_x] = accum[cal_y][cal_x];
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C1 = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;
    int k_block = K / BLOCK_SIZE_K;
    int stride = 2;

    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        int row = (i / K);
        int col = (i % K);
        int row_block = row / BLOCK_SIZE_M;
        int col_block = col / BLOCK_SIZE_K;
        if ((row_block * k_block + col_block) % stride == 0) h_A[i] = 1;
        else {
            h_A[i] = 0;
        }
    }

    // 生成B的数据
    for( int i = 0; i < K * N; i++ ) {
        if ( i >= K * N / 2) h_B[i] = 2;
        else {
            h_B[i] = 0;
        }
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            M, N, K, &alpha, 
            d_A, K, d_B, N, &beta, d_C, N
        );
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    cublasDestroy(blas_handle); 

    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        int row = i / N;
        int col = i % N;
        double abs_err = fabs(h_C[i] - h_C1[col * M + row]);
        double dot_length = M;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[col * M + row], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}
