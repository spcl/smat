#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

#define HGEMM_FUNC(name) void name(half *A, half *B, half *C, size_t M, size_t N, size_t K)
#define HGEMM_FUNC_SPARSE(name) void name(half *bcsrValuesA, half *B, half *C, size_t M, size_t N, size_t K, size_t nonzeroBlocks, int* blockInfo, int* relativeBlockIndexMapping)
#define HGEMM_FUNC_SPARSE2(name) void name(half *bcsrValuesA, int *bcsrRowPtrA, int *bcsrColIdxA, half *B, half *C, size_t M, size_t N, size_t K, size_t nonzeroBlocks, int* blockInfo, int* relativeBlockIndexMapping)


HGEMM_FUNC(cublasTensorOp);


HGEMM_FUNC_SPARSE(mmaNaiveKernel);
HGEMM_FUNC_SPARSE(mmaTKernel);


HGEMM_FUNC_SPARSE2(mmaBKernel);
HGEMM_FUNC_SPARSE2(mmaBTKernel);
HGEMM_FUNC_SPARSE2(mmaCBTKernel);


DEFINE_uint32(M, 512, "M");
DEFINE_uint32(N, 2048, "N");
DEFINE_uint32(K, 1024, "K");
DEFINE_bool(enable_wmma, true, "test WMMA API");
DEFINE_bool(enable_mma, true, "test MMA PTX instruction");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the cublas result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");
DEFINE_uint32(n_mult, 1, "n_mult * MMA_N = N");
DEFINE_string(filename, "/scratch/czox/sparse_matrices/reordering_tests/cop20k_A_reordered_transposed_reordered.mtx", "input .mtx file");


int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    omp_set_num_threads(FLAGS_cpu_procs);
    HGEMM_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

    cudaDeviceProp dev_prop;
    HGEMM_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    HLOG("CUDA HGEMM start with %u CPU processes on the %u-th GPU: %s", FLAGS_cpu_procs, FLAGS_gpu_rank, dev_prop.name);

    int driver_version = 0;
    int runtime_version = 0;
    HGEMM_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    HGEMM_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    HLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);
    HLOG("CUDA capability major/minor version number: %d.%d", dev_prop.major, dev_prop.minor);
    HLOG("%d multiprocessors, %d CUDA cores/MP: %d CUDA cores", dev_prop.multiProcessorCount,
         convert_SM_to_cores(dev_prop.major, dev_prop.minor),
         convert_SM_to_cores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
    HLOG("GPU max clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.clockRate) * 1e-3,
         static_cast<double>(dev_prop.clockRate) * 1e-6);
    HLOG("Memory clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.memoryClockRate) * 1e-3,
         static_cast<double>(dev_prop.memoryClockRate) * 1e-6);
    HLOG("Memory bus width: %d-bit", dev_prop.memoryBusWidth);
    HLOG("Total amount of global memory: %.0f MBytes (%zu Bytes)",
         static_cast<double>(dev_prop.totalGlobalMem) / 1048576, dev_prop.totalGlobalMem);
    HLOG("Total amount of constant memory: %.0f KBytes (%zu Bytes)", static_cast<double>(dev_prop.totalConstMem) / 1024,
         dev_prop.totalConstMem);
    HLOG("Total amount of shared memory per block: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerBlock) / 1024, dev_prop.sharedMemPerBlock);
    HLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerMultiprocessor) / 1024, dev_prop.sharedMemPerMultiprocessor);
    HLOG("L2 cache size: %.0f KBytes (%d Bytes)", static_cast<double>(dev_prop.l2CacheSize) / 1024,
         dev_prop.l2CacheSize);
    HLOG("Total number of registers available per block: %d", dev_prop.regsPerBlock);
    HLOG("Warp size: %d", dev_prop.warpSize);
    HLOG("Max number of threads per multiprocessor: %d", dev_prop.maxThreadsPerMultiProcessor);
    HLOG("Max number of threads per block: %d", dev_prop.maxThreadsPerBlock);
    HLOG("Max dimension size of a thread block (x,y,z): (%d, %d, %d)", dev_prop.maxThreadsDim[0],
         dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    HLOG("Max dimension size of a grid size (x,y,z): (%d, %d, %d)", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
         dev_prop.maxGridSize[2]);

    HLOG("A (%u x %u) * B (%u x %u) = C (%u x %u), N_MULT: %u", FLAGS_M, FLAGS_K, FLAGS_K, FLAGS_N, FLAGS_M, FLAGS_N, FLAGS_n_mult);
    HLOG(
        "Profiling: enable wmma: %d, enable mma: %d, warmup iterations: %u, profiling iterations: %u, sleep duration: "
        "%u ms, enable check: %d",
        FLAGS_enable_wmma, FLAGS_enable_mma, FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration,
        FLAGS_enable_check);
     

    std::string file (FLAGS_filename);
    HLOG("Input .mtx: %s", file.data());
    Tester tester(FLAGS_M, FLAGS_N, FLAGS_K, FLAGS_warmup_iterations, FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check, FLAGS_n_mult, file.data());
    //tester.evaluate(cublasTensorOp, "Cublas-Tensor-Op");

    tester.evaluateSparse(mmaNaiveKernel, "Mma-Naive-Kernel");
    tester.evaluateSparse(mmaTKernel, "Mma-T-Kernel");

    tester.evaluateSparse2(mmaBKernel, "Mma-B-Kernel");
    tester.evaluateSparse2(mmaBTKernel, "Mma-BT-Kernel");
    tester.evaluateSparse2(mmaCBTKernel, "Mma-CBT-Kernel");

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    HLOG("Done");

    return 0;
}
