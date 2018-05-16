###########################################################

# USER SPECIFIC DIRECTORIES

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

# NVCC COMPILER OPTIONS

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS= -gencode arch=compute_30,code=sm_30
NVCC_LIBS=

# NVCC library directories:
NVCC_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
NVCC_INC_DIR= -I$(CUDA_ROOT_DIR)/include

##########################################################


# Compile test transpose.
all:
	$(NVCC) $(NVCC_FLAGS) src/test_transpose.cu -o TEST_TRANSPOSE


# Clean compilation files and run files.
clean:
	rm -f *o TEST_TRANSPOSE
