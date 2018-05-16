// C libraries
#include <stdio.h>




/**
 * CUDA kernel for a strided batched transpose matrix operation (less than 32 rows) on a given strided square matrices.
 * @tparam T Template parameter type of strided matrices.
 * @tparam BLOCKSIZE CUDA blocksize used for kernel launch.
 * @param input Input non-transposed strided matrices of type T.
 * @param output Output transposed strided matrices of type T.
 * @param rows Number of rows in the input/output square matrices.
 * @param batchSize Number of strided matrices.
 * @return
 */
template <class T, unsigned int BLOCKSIZE>
__global__ void d_transposeSquare_strided_batched(T *input, T *output, int rows, int batchSize){
    
    if (threadIdx.x < rows) {
    
        // Batch ID
        unsigned int batchShift = blockIdx.x * rows * rows;
    
        // Initialize shared memory.
        __shared__
        T temp[BLOCKSIZE][BLOCKSIZE + 1];
    
        // Read elements from global memory and write them to shared memory.
        #pragma unroll
        for (int i = 0; i < rows; i++) {
        
            temp[i][threadIdx.x] = input[batchShift + i * rows + threadIdx.x];
        }
    
        // Read elements from shared memory and write them back to global memory.
        #pragma unroll
        for (int i = 0; i < rows; i++) {
        
            output[batchShift + i * rows + threadIdx.x] = temp[threadIdx.x][i];
        }
    }
}





/**
 * CUDA kernel for a strided batched transpose matrix operation (greater than 32 rows) on a given strided square matrices.
 * @tparam T Template parameter type of strided matrices.
 * @tparam BLOCKSIZE CUDA blocksize used for kernel launch.
 * @param input Input non-transposed strided matrices of type T.
 * @param output Output transposed strided matrices of type T.
 * @param rows Number of rows in the input/output square matrices.
 * @param batchSize Number of strided matrices.
 * @return
 */
template <class T, unsigned int BLOCKSIZE>
__global__ void d_transposeSquare_tiling_strided_batched(T *input, T *output, int rows, int batchSize){

    // Matrix Id = blockIdx.x;
    // Local linear block Id = blockIdx.y;

    // Tiles per matrix
    unsigned int tileDim = ceilf( rows / (float) blockDim.x );
    // Local block Id
    uint2 blockId, local_tid;
    blockId.x = blockIdx.y % tileDim;
    blockId.y = blockIdx.y / tileDim;
    // Local thread Id
    //local_tid.x = blockId.x * blockDim.x + threadIdx.x;
    local_tid.x = blockId.x * BLOCKSIZE + threadIdx.x;
    local_tid.y = blockId.y * BLOCKSIZE;
    // Global thread Id                matrix shift           +          local tile y-shift       +       local tile x-shift   +  warp-level thread Id
    unsigned int global_tid = ( blockIdx.x * rows * rows ) + ( blockId.y * blockDim.x * rows ) + ( blockId.x * blockDim.x ) + threadIdx.x;

    // Initialize shared memory.
    __shared__ T temp[BLOCKSIZE][BLOCKSIZE+1];

    // Check for overreach in x direction.
    if ( local_tid.x < rows ) {

        // Read elements from global memory and write them to shared memory.
        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; k++) {

            if (local_tid.y + k < rows)
                temp[threadIdx.x][k] = input[global_tid + k * rows];

        }
    }
    

    // Transposed indices.
    // Global thread Id:    matrix shift         +        local tile y-shift         +     local tile x-shift     +  warp-level thread Id
    global_tid = ( blockIdx.x * rows * rows ) + ( blockId.x * blockDim.x * rows ) + ( blockId.y * blockDim.x ) + threadIdx.x;

    // Update local thread Id transposed.
    local_tid.x = blockId.y * BLOCKSIZE + threadIdx.x;
    local_tid.y = blockId.x * BLOCKSIZE;

    // Check for overreach in x direction.
    if ( local_tid.x < rows ) {

        // Read elements from shared memory in a transposed fashion and write them back to global memory.
        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; k++) {

            if ( local_tid.y + k < rows ) {
                output[global_tid + k * rows] = temp[k][threadIdx.x];

            }
        }
    }
}







