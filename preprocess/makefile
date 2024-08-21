CXX       =g++
CXXFLAGS  =  -fpermissive -Wl,--no-as-needed -m64 -std=c++11
INCLUDE = -I include

OBJ_DIR = ./obj
APP_DIR = ./programs
SRC_DIR = ./src
TEST_DIR = ./test


#CUDA
CUDA_PATH ?= $(CUDA_HOME)
CUDA_CXXFLAGS = 
CUDA_CXXFLAGS += -std=c++11
CUDA_CXXFLAGS += -m64 
CUDA_CXXFLAGS += -O3 -arch=sm_70

NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
CUDA_INCLUDE  = -I include/cuda_inc
CUDA_INCLUDE += $(INCLUDE) 

CUDA_LDFLAGS = -lcublas -lcusparse -lcudart

#-------------------------------------------------------------------------
#GENERAL PROGRAMS

GEN_OBJ_DIR = $(OBJ_DIR)/general
GEN_APP_DIR = $(APP_DIR)/general
GEN_SRC_DIR = $(SRC_DIR)/general
GEN_TEST_DIR = $(TEST_DIR)/general

GEN_SRC = $(wildcard $(GEN_SRC_DIR)/*.cpp)
GEN_OBJECTS = $(GEN_SRC:$(GEN_SRC_DIR)/%.cpp=$(GEN_OBJ_DIR)/%.o)

$(GEN_OBJ_DIR)/%.o: $(GEN_SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(GEN_OBJ_DIR)/%.o: $(GEN_TEST_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

$(GEN_APP_DIR)/%: $(GEN_OBJ_DIR)/%.o $(GEN_OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ $?


#-------------------------------------------------------------------------
#CUDA PROGRAMS
CUDA_OBJ_DIR = $(OBJ_DIR)/cuda
CUDA_APP_DIR =$(APP_DIR)/cuda
CUDA_SRC_DIR = $(SRC_DIR)/cuda
CUDA_TEST_DIR = $(TEST_DIR)/cuda

CUDA_SRC = $(wildcard $(CUDA_SRC_DIR)/*.cpp)
CUDA_OBJECTS := $(CUDA_SRC: $(CUDA_SRC_DIR)/%.cpp=$(CUDA_OBJ_DIR)/%.o) $(GEN_OBJECTS)

$(CUDA_OBJ_DIR)/%.o : $(CUDA_SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) $(CUDA_LIBRARY) -o $@ -c $<

$(CUDA_OBJ_DIR)/%.o : $(CUDA_TEST_DIR)/%.cpp
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) $(CUDA_LIBRARY) -o $@ -c $<

$(CUDA_APP_DIR)/% : $(CUDA_OBJ_DIR)/%.o $(CUDA_OBJECTS)
	@mkdir -p $(@D)
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) $(CUDA_LDFLAGS) -o $@ $?


#-------------------------------------------------------------------------

.PHONY: all build clean

build: build_general build_cuda build_slurm

all: serial cuda build_slurm

build_general: 
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(GEN_APP_DIR)
	@mkdir -p $(GEN_OBJ_DIR)

build_cuda: build_general
	@mkdir -p $(CUDA_APP_DIR)
	@mkdir -p $(CUDA_OBJ_DIR)

build_slurm:
	@mkdir -p outputs_exps/
	@mkdir -p outputs_exps/slurm_runtime/

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*


serial : build_general $(GEN_APP_DIR)/Matrix_Blocking $(GEN_APP_DIR)/reorder_matrix $(GEN_APP_DIR)/TEST_matrices $(GEN_APP_DIR)/TEST_blocking_VBR_2D $(GEN_APP_DIR)/TEST_blocking_VBR $(GEN_APP_DIR)/TEST_similarities

cuda : build_cuda $(CUDA_APP_DIR)/TEST_cuda  $(CUDA_APP_DIR)/cuda_multiply
