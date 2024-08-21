#compilers
NVCC = ~/miniconda3/envs/dasp/bin/nvcc
# /usr/local/cuda-12.0/bin/nvcc
GCC = ~/miniconda3/envs/dasp/bin/gcc
# /usr/local/gcc-12.2/bin/gcc
GENCODE_FLAGS = -gencode arch=compute_80,code=sm_80
NVCCVERSION = $(shell $(NVCC) --version | grep release | sed -n -E 's/.* ([0-9]+\.[0-9]+).*/\1/p')
GCC_BIN_DIR = $(shell dirname ${GCC})
GCCVERSION = $(shell ${GCC} --version | grep ^gcc | sed 's/^.* //g')

NVCC_FLAGS = -O3 -ccbin $(GCC_BIN_DIR) -m64 $(GENCODE_FLAGS)

CUDA_LIBS = -lcusparse -lcublas
LIBS =  -lineinfo $(CUDA_LIBS)

#options
OPTIONS = -Xcompiler -fopenmp-simd

REQUIRED_VERSION = 12.0

all: check_version double half # experiment

check_version:
	@if [ "$(REQUIRED_VERSION)" != "`echo -e "$(REQUIRED_VERSION)\n$(NVCCVERSION)" | sort -V | head -n1`" ]; then \
		echo "NVCC Version $(NVCCVERSION) is less than required version $(REQUIRED_VERSION). Exiting."; \
		exit 1; \
	fi
	@if [ "$(REQUIRED_VERSION)" != "`echo -e "$(REQUIRED_VERSION)\n$(GCCVERSION)" | sort -V | head -n1`" ]; then \
		echo "GCC Version $(GCCVERSION) is less than required version $(REQUIRED_VERSION). Exiting."; \
		exit 1; \
	fi

double:
	$(NVCC) $(NVCC_FLAGS) src/main_f64.cu -o spmv_double  -D f64 $(OPTIONS) $(LIBS) 

half:
	$(NVCC) $(NVCC_FLAGS) src/main_f16.cu -o spmv_half $(OPTIONS) $(LIBS)

experiment:
	$(NVCC) $(NVCC_FLAGS) src/main_experiment.cu -o experiment $(OPTIONS) $(LIBS)   

clean:
	rm -rf spmv_double
	rm -rf spmv_half
