#include "matrixOp_kernels.cuh"



# define CUDA_TRANSPOSE_TIMING

template <class T>
void transpose_strided_batched(T *input, T *output, int rows, int columns, int batchSize){

    // Check for square matrix.
    if ( rows == columns ) {

        // Launch kernel optimized by size.
        if (rows == 1) {
            return;
        } else if (rows == 2) {
            // Determine block and grid size.
            dim3 blockSize(2, 1, 1);
            dim3 gridSize(batchSize, 1);

            # ifdef CUDA_TRANSPOSE_TIMING
                // Initialize CUDA timers.
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
            
                // Start CUDA timer.
                cudaEventRecord(start);
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 2> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
                // End CUDA timer.
                cudaEventRecord(stop);
            
                // Calculate milliseconds elapsed.
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
            
                // Calculate performance metrics.
                float GFLOPS = ( 3 + (3*rows) + (3*rows) ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                float BANDWIDTH = ( rows + rows ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                printf("Time: %fms,\tGFLOPS: %fGB/s,\tBandwidth: %f GB/s\n", milliseconds, GFLOPS, BANDWIDTH);
            
            # else
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 2> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
            
            # endif



        } else if (rows <= 4) {
            // Determine block and grid size.
            dim3 blockSize(4, 1, 1);
            dim3 gridSize(batchSize, 1);
    
            # ifdef CUDA_TRANSPOSE_TIMING
                // Initialize CUDA timers.
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
        
                // Start CUDA timer.
                cudaEventRecord(start);
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 4> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
                // End CUDA timer.
                cudaEventRecord(stop);
        
                // Calculate milliseconds elapsed.
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
        
                // Calculate performance metrics.
                float GFLOPS = ( 3 + (3*rows) + (3*rows) ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                float BANDWIDTH = ( rows + rows ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                printf("Time: %fms,\tGFLOPS: %fGB/s,\tBandwidth: %f GB/s\n", milliseconds, GFLOPS, BANDWIDTH);
    
            # else
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 4> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
            
            # endif
            

        } else if (rows <= 8) {
            // Determine block and grid size.
            dim3 blockSize(8, 1, 1);
            dim3 gridSize(batchSize, 1);
    
            # ifdef CUDA_TRANSPOSE_TIMING
                // Initialize CUDA timers.
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
        
                // Start CUDA timer.
                cudaEventRecord(start);
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 8> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
                // End CUDA timer.
                cudaEventRecord(stop);
        
                // Calculate milliseconds elapsed.
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
        
                // Calculate performance metrics.
                float GFLOPS = ( 3 + (3*rows) + (3*rows) ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                float BANDWIDTH = ( rows + rows ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                printf("Time: %fms,\tGFLOPS: %fGB/s,\tBandwidth: %f GB/s\n", milliseconds, GFLOPS, BANDWIDTH);
    
            # else
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 8> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
            
            # endif
            
            

        } else if (rows <= 16) {
            // Determine block and grid size.
            dim3 blockSize(16, 1, 1);
            dim3 gridSize(batchSize, 1);
    
            # ifdef CUDA_TRANSPOSE_TIMING
                // Initialize CUDA timers.
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
        
                // Start CUDA timer.
                cudaEventRecord(start);
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 16> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
                // End CUDA timer.
                cudaEventRecord(stop);
        
                // Calculate milliseconds elapsed.
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
        
                // Calculate performance metrics.
                float GFLOPS = ( 3 + (3*rows) + (3*rows) ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                float BANDWIDTH = ( rows + rows ) * ( blockSize.x * batchSize )  / ( milliseconds / 1000 ) / ( 10e9 );
                printf("Time: %fms,\tGFLOPS: %fGB/s,\tBandwidth: %f GB/s\n", milliseconds, GFLOPS, BANDWIDTH);
    
            # else
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 16> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
            
            # endif
            
            

        } else if (rows <= 32) {
            // Determine block and grid size.
            dim3 blockSize(32, 1, 1);
            dim3 gridSize(batchSize, 1);
    
            # ifdef CUDA_TRANSPOSE_TIMING
                // Initialize CUDA timers.
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
        
                // Start CUDA timer.
                cudaEventRecord(start);
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 32> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
                // End CUDA timer.
                cudaEventRecord(stop);
        
                // Calculate milliseconds elapsed.
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
        
                // Calculate performance metrics.
                float GFLOPS = ( 3 + (3*rows) + (3*rows) ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                float BANDWIDTH = ( rows + rows ) * ( blockSize.x * batchSize ) / ( milliseconds / 1000 ) / ( 10e9 );
                printf("Time: %fms,\tGFLOPS: %fGB/s,\tBandwidth: %f GB/s\n", milliseconds, GFLOPS, BANDWIDTH);
    
            # else
                // Launch kernel.
                d_transposeSquare_strided_batched<T, 32> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
            
            # endif
            
            

        } else {
            // Determine BLOCKSIZE.
            const unsigned int BLOCKSIZE = 16;
            // Determine number of tiles horizontally (1 block of 32 per matrix) and vertically (columns / 32).
            unsigned int tiles_per_matrix = ceil(rows / (float) BLOCKSIZE) * ceil(columns / (float) BLOCKSIZE);
    
            // Determine block and grid size.
            dim3 blockSize(BLOCKSIZE, 1, 1);
            // grid.x = matrixId, grid.y = tileId
            dim3 gridSize(batchSize, tiles_per_matrix);
    
            # ifdef CUDA_TRANSPOSE_TIMING
                // Initialize CUDA timers.
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
        
                // Start CUDA timer.
                cudaEventRecord(start);
                // Launch kernel.
                d_transposeSquare_tiling_strided_batched<T, BLOCKSIZE> << < gridSize, blockSize >> >
                                                                                      (input, output, rows, batchSize);
                // End CUDA timer.
                cudaEventRecord(stop);
        
                // Calculate milliseconds elapsed.
                cudaEventSynchronize(stop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
        
                // Calculate performance metrics.
                float GFLOPS = (15 + (4 * rows) + 11 + (4 * rows)) * (blockSize.x * tiles_per_matrix * batchSize) /
                               (milliseconds / 1000) / (10e9);
                float BANDWIDTH =
                        (blockSize.x + blockSize.x) * (blockSize.x * tiles_per_matrix * batchSize) / (milliseconds / 1000) /
                        (10e9);
                printf("Time: %fms,\tGFLOPS: %fGB/s,\tBandwidth: %f GB/s\n", milliseconds, GFLOPS, BANDWIDTH);
    
            # else
                // Launch kernel.
                d_transposeSquare_tiling_strided_batched<T, BLOCKSIZE> <<< gridSize, blockSize >>> (input, output, rows, batchSize);
        
            # endif
    
        }
    }
}















