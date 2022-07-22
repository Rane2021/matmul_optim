/*
 * @Description: 矩阵乘法
 * @Date: 2022-07-20 14:16:53
 * @LastEditTime: 2022-07-22 13:09:55
 * @FilePath: /01_gemm/02_gemm.cu
 */
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>


void initialData(float *ip, const int size)
{
    int i;
    for(i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

void printMetric(float *A, int size, std::string name){
    int nloop = 20;
    if(size < nloop)
        nloop = size;
    printf("%s: \n", name.c_str());
    for(int pi=0; pi < nloop; ++pi){
        printf("%.3f, ", A[pi]);
    }
    printf("\n");
}

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    // double epsilon = 1.0E-8;
    double epsilon = 1.0E-3;
    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("check metics size: %d\n", N);
            printf("error index %d, host %f gpu %f ", i, hostRef[i], gpuRef[i]);
            printf("Arrays do not match.\n\n");
            break;
        }
    }
}


/**
 * @description: 矩阵A(mxn) * 矩阵B(nxk)  = 矩阵C(mxk)  (M行K列)
 */
void gemmOnHost(float *A, float *B, float *C, const int M, const int N, const int K)
{
    for(int mi=0; mi<M; ++mi){
        for(int ki=0; ki<K; ++ki){
            // C[mi*M + ki] = 0;
            for(int ni=0; ni<N; ++ni){
                C[mi*K + ki] += A[mi*N + ni] * B[ni*K + ki];
            }
        }
    }
    return;
}

/**
 * @description: 矩阵A(mxn) * 矩阵B(nxk)  = 矩阵C(mxk)  (M行K列)
 */
__global__ void gemmOnGPU2D(float *A, float *B, float *C, int M, int N, int K)
{    
    unsigned int ixk = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iym = blockIdx.y * blockDim.y + threadIdx.y;
    if (ixk < K && iym < M){
        float sum = 0;
        for(int ni=0; ni<N; ++ni){
            sum += A[iym*N + ni] * B[ni*K + ixk];
        }
        C[iym*K + ixk] = sum;
    }
}

/**
 * @description: 矩阵A(mxn) * 矩阵B(nxk)  = 矩阵C(mxk)  (M行K列)
 * @event: 这个方法使用的共享内存有点多： block share mem size = blockDim.x*N + blockDim.y*N
 */
__global__ void gemmOnGPU2DShareMem(float *A, float *B, float *C, const int M, const int N, const int K)
{    
    // __shared__ float block_A[blockDim.y][N];  
    // __shared__ float block_B[N][blockDim.x];
    // 静态分配有问题，使用动态分配
    extern __shared__ float block_AB[];  
    // extern __shared__ float block_B[];

    unsigned int ixk = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iym = blockIdx.y * blockDim.y + threadIdx.y;

    if (ixk < K && iym < M){
        for(int ni=0; ni<N; ++ni){
            block_AB[threadIdx.y*N + ni] = A[iym*N + ni];  // A
            block_AB[blockDim.y*N + ni*blockDim.x + threadIdx.x] = B[ni*K + ixk];  // B
            // block_B[ni*blockDim.x + blockIdx.x] = B[ni*K + ixk];
        }
        __syncthreads();

        float sum = 0;
        for(int ni=0; ni<N; ++ni){
            // sum += block_A[blockIdx.y*N + ni] * block_B[ni*blockDim.x + blockIdx.x];
            sum += block_AB[threadIdx.y*N + ni] * block_AB[blockDim.y*N + ni*blockDim.x + threadIdx.x];
        }
        C[iym*K + ixk] = sum;
    }
}

/**
 * @description: 矩阵A(mxn) * 矩阵B(nxk)  = 矩阵C(mxk)  (M行K列)
 * @event: 每个块共享内存： block share mem size = blockDim.x*blockDim.y*2  (blockDim.x == blockDim.y)
 * 要求： blockDim.x == blockDim.y
 */
__global__ void gemmOnGPU2DShareMemV2(float *A, float *B, float *C, const int M, const int N, const int K)
{
	//分配共享内存: 不能这样申请两块
	// extern __shared__ float sharedM[];
	// extern __shared__ float sharedN[];
    extern __shared__ float smem[];
    float *sharedM = smem;                       
    float *sharedN = (float*)&smem[blockDim.x * blockDim.y]; 

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K){
        float Csub = 0.0;
        for (int i = 0; i < (int)(ceil((float)N / blockDim.x)); i++){
            if (i*blockDim.x + threadIdx.x < N) 
                sharedM[threadIdx.y*blockDim.x + threadIdx.x] = A[row*N + i * blockDim.x + threadIdx.x];
            else
                sharedM[threadIdx.y*blockDim.x + threadIdx.x] = 0.0;
            if (i*blockDim.y + threadIdx.y < N) 
                sharedN[threadIdx.y*blockDim.x + threadIdx.x] = B[(i*blockDim.y + threadIdx.y)*K + col];
            else
                sharedN[threadIdx.y*blockDim.x + threadIdx.x] = 0.0;
            __syncthreads();

            for (int j = 0; j < blockDim.x; j++)
                Csub += sharedM[threadIdx.y*blockDim.x + j] * sharedN[j*blockDim.y + threadIdx.x];
            __syncthreads();
        }
        C[row*K + col] = Csub;
    }
}

/**
 * @description: 矩阵A(mxn) * 矩阵B(nxk)  = 矩阵C(mxk)  (M行K列)
 * @event: 使用寄存器方式，进一步提高计算访存比
 * 下面的实现方式要求 M==N==K, BLOCK_SIZE = 32 blockDim.x=32,  blockDim.y = BLOCK_SIZE / 2
 */
__global__ void gemmOnGPU2DRegMem(float *A, float *B, float *C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y * 2  + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float val[2] = {0.0f};

    const int BLOCK_SIZE = 32;
    __shared__ float shTileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shTileB[BLOCK_SIZE][BLOCK_SIZE];

    // int width = N;
    int iter = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if(row < M && col < K){
        for(int i = 0; i < iter; i++){
            shTileA[threadIdx.y][threadIdx.x]=A[row * N+i*BLOCK_SIZE+threadIdx.x];
            shTileA[threadIdx.y+16][threadIdx.x]=A[(row+16)*N+i*BLOCK_SIZE+threadIdx.x];

            shTileB[threadIdx.y][threadIdx.x]=B[(i*BLOCK_SIZE+threadIdx.y)*N+col];
            shTileB[threadIdx.y+16][threadIdx.x]=B[(i*BLOCK_SIZE+threadIdx.y+16)*N+col];
            __syncthreads();

            for(int j = 0; j < BLOCK_SIZE; j++){
                val[0] += shTileA[threadIdx.y][j] * shTileB[j][threadIdx.x];
                val[1] += shTileA[threadIdx.y + 16][j] * shTileB[j][threadIdx.x];
            }
            __syncthreads();
        }
        
        C[row * N + col] = val[0];
        C[(row + 16) * N + col] = val[1];
    }
}


/**
 * @description: 矩阵A(mxn) * 矩阵B(nxk)  = 矩阵C(mxk)  (M行K列)
 * @event: 计算结果有问题，应该是那里
 */
__global__ void gemmOnGPU3D(float *A, float *B, float *C, int M, int N, int K)
{    
    unsigned int ixk = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iym = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int izn = blockIdx.z * blockDim.z + threadIdx.z;
    if (ixk < K && iym < M && izn < N){
        // float sum = 0;
        // for(int ni=0; ni<N; ++ni){
        C[iym*K + ixk] += A[iym*N + izn] * B[izn*K + ixk];
        // }
        // C[iym*K + ixk] = sum;
    }
}



int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    // int nm = 3;
    // int nn = 2;
    // int nk = 4;

    // int nm = 1 << 7;  // max for shear mem 
    // int nn = 1 << 7;
    // int nk = 1 << 7;

    int nm = 1 << 12;
    int nn = 1 << 12;
    int nk = 1 << 12;


    int nSizeA = nm*nn;
    int nSizeB = nn*nk;
    int nSizeC = nm*nk;
    int nBytesA = nSizeA * sizeof(float);
    int nBytesB = nSizeB * sizeof(float);
    int nBytesC = nSizeC * sizeof(float);

    // malloc host memory
    float *h_A, *h_B, *hostRef;
    h_A = (float *)malloc(nBytesA);
    h_B = (float *)malloc(nBytesB);
    hostRef = (float *)malloc(nBytesC);
    memset(hostRef, 0, nBytesC);

    float *gpuRef = (float *)malloc(nBytesC);
    memset(gpuRef, 0, nBytesC);
    float *gpuRefShareMem = (float *)malloc(nBytesC);
    memset(gpuRefShareMem, 0, nBytesC);
    float *gpuRefShareMemV2 = (float *)malloc(nBytesC);
    memset(gpuRefShareMemV2, 0, nBytesC);
    float *gpuRef3D = (float *)malloc(nBytesC);
    memset(gpuRef3D, 0, nBytesC);
    float *gpuRefRegMem = (float *)malloc(nBytesC);
    memset(gpuRefRegMem, 0, nBytesC);


    // initialize data at host side
    double iStart = seconds();
    initialData(h_A, nSizeA);
    initialData(h_B, nSizeB);
    double iElaps = seconds() - iStart;
    // printf("initialData elapsed %f ms\n", iElaps);
    // printMetric(h_A, nSizeA, "h_A");
    // printMetric(h_B, nSizeB, "h_B");


    // add matrix at host side for result checks
    iStart = seconds();
    // gemmOnHost (h_A, h_B, hostRef, nm, nn, nk);
    iElaps = seconds() - iStart;
    // printf("sumMatrixOnHost elapsed %f ms\n", iElaps);
    // printMetric(hostRef, nSizeC, "hostRef");


    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    CHECK(cudaMalloc((void **)&d_MatA, nBytesA));
    CHECK(cudaMalloc((void **)&d_MatB, nBytesB));
    CHECK(cudaMalloc((void **)&d_MatC, nBytesC));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_MatA, h_A, nBytesA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_MatB, h_B, nBytesB, cudaMemcpyHostToDevice));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    int dimz = 1;

    if(argc > 2){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
    }
    if(argc > 3){
        dimx = atoi(argv[1]);
        dimy = atoi(argv[2]);
        dimz = atoi(argv[3]);
    }

    dim3 block(dimx, dimy);
    dim3 grid((nk + block.x - 1) / block.x, (nm + block.y - 1) / block.y);
    dim3 block3D(dimx, dimy, dimz);
    dim3 grid3D((nk + block.x - 1) / block.x, (nm + block.y - 1) / block.y, (nn + block.z -1) / block.z);

    // method 1: 2D
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    gemmOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nm, nn, nk);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("gemmOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_MatC, nBytesC, cudaMemcpyDeviceToHost));
    // printMetric(gpuRef, nSizeC, "gpuRef");

    // method 2: share mem
    int shareMemSize = dimx*nn + dimy*nn;  // 动态共享内存大小
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    gemmOnGPU2DShareMem<<<grid, block, shareMemSize*sizeof(float)>>>(d_MatA, d_MatB, d_MatC, nm, nn, nk);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("gemmOnGPU2DShareMem <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRefShareMem, d_MatC, nBytesC, cudaMemcpyDeviceToHost));
    // printMetric(gpuRefShareMem, nSizeC, "gpuRefShareMem");

    // method 3: share mem
    int shareMemSizeV2 = dimx*dimy*2;  // 动态共享内存大小
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    gemmOnGPU2DShareMemV2<<<grid, block, shareMemSizeV2*sizeof(float)>>>(d_MatA, d_MatB, d_MatC, nm, nn, nk);
    // gemmOnGPU2DShareMemV2<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nm, nn, nk);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("gemmOnGPU2DShareMemV2 <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRefShareMemV2, d_MatC, nBytesC, cudaMemcpyDeviceToHost));
    // printMetric(gpuRefShareMemV2, nSizeC, "gpuRefShareMemV2");

    // method 4: 3D gemm
    // CHECK(cudaDeviceSynchronize());
    // iStart = seconds();
    // gemmOnGPU3D<<<grid3D, block3D>>>(d_MatA, d_MatB, d_MatC, nm, nn, nk);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = seconds() - iStart;
    // printf("gpuRef3D <<<(%d,%d,%d), (%d,%d,%d)>>> elapsed %f ms\n", grid3D.x, grid3D.y,  grid3D.z, block3D.x, block3D.y, block3D.z, iElaps);
    // CHECK(cudaGetLastError());
    // CHECK(cudaMemcpy(gpuRef3D, d_MatC, nBytesC, cudaMemcpyDeviceToHost));
    // printMetric(gpuRef3D, nSizeC, "gpuRef3D");

    // method 5: register optim
    dim3 block5(dimx, dimy/2);
    dim3 grid5((nk + block.x - 1) / block.x, (nm + block.y - 1) / block.y);

    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    gemmOnGPU2DRegMem<<<grid5, block5>>>(d_MatA, d_MatB, d_MatC, nm, nn, nk);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("gemmOnGPU2DRegMem <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, iElaps);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRefRegMem, d_MatC, nBytesC, cudaMemcpyDeviceToHost));
    // printMetric(gpuRefRegMem, nSizeC, "gemmOnGPU2DRegMem");
    

    // check device results
    // checkResult(hostRef, gpuRef, nSizeC);
    checkResult(gpuRef, gpuRefShareMem, nSizeC);
    checkResult(gpuRef, gpuRefShareMemV2, nSizeC);
    checkResult(gpuRef, gpuRefRegMem, nSizeC);


    // free device global memory
    CHECK(cudaFree(d_MatA));
    CHECK(cudaFree(d_MatB));
    CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    free(gpuRefShareMem);

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


/*常用命令：
线程束占用率
sudo /usr/local/cuda-11.6/bin/ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./test_cuda

全局内存加载吞吐量
sudo /usr/local/cuda-11.6/bin/ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second ./test_cuda 128 1

全局内存加载效率
sudo /usr/local/cuda-11.6/bin/ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct  ./test_cuda 128 1
*/


