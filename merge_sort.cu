#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "helper_timer.h"
#include "exception.h"
#include <cooperative_groups.h>


namespace cg = cooperative_groups;

#define SHARED_SIZE_LIMIT 1024U
#define SAMPLE_STRIDE 128
static uint *d_RanksA, *d_RanksB, *d_LimitsA, *d_LimitsB;
static const uint MAX_SAMPLE_COUNT = 32768;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define STRCASECMP _stricmp
#define STRNCASECMP _strnicmp
#else
#define STRCASECMP strcasecmp
#define STRNCASECMP strncasecmp
#endif

typedef enum cudaError_enum
{
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::cuEventQuery() and ::cuStreamQuery()).
     */
    CUDA_SUCCESS                              = 0,

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    CUDA_ERROR_INVALID_VALUE                  = 1,

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    CUDA_ERROR_OUT_OF_MEMORY                  = 2,

    /**
     * This indicates that the CUDA driver has not been initialized with
     * ::cuInit() or that initialization has failed.
     */
    CUDA_ERROR_NOT_INITIALIZED                = 3,

    /**
     * This indicates that the CUDA driver is in the process of shutting down.
     */
    CUDA_ERROR_DEINITIALIZED                  = 4,

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
    */
    CUDA_ERROR_PROFILER_DISABLED           = 5,
    /**
     * This indicates profiling has not been initialized for this context.
     * Call cuProfilerInitialize() to resolve this.
    */
    CUDA_ERROR_PROFILER_NOT_INITIALIZED       = 6,
    /**
     * This indicates profiler has already been started and probably
     * cuProfilerStart() is incorrectly called.
    */
    CUDA_ERROR_PROFILER_ALREADY_STARTED       = 7,
    /**
     * This indicates profiler has already been stopped and probably
     * cuProfilerStop() is incorrectly called.
    */
    CUDA_ERROR_PROFILER_ALREADY_STOPPED       = 8,
    /**
     * This indicates that no CUDA-capable devices were detected by the installed
     * CUDA driver.
     */
    CUDA_ERROR_NO_DEVICE                      = 100,

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    CUDA_ERROR_INVALID_DEVICE                 = 101,


    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid CUDA module.
     */
    CUDA_ERROR_INVALID_IMAGE                  = 200,

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::cuCtxDestroy() invoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::cuCtxGetApiVersion() for more details.
     */
    CUDA_ERROR_INVALID_CONTEXT                = 201,

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of CUDA 3.2. It is no longer an
     * error to attempt to push the active context via ::cuCtxPushCurrent().
     */
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT        = 202,

    /**
     * This indicates that a map or register operation has failed.
     */
    CUDA_ERROR_MAP_FAILED                     = 205,

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    CUDA_ERROR_UNMAP_FAILED                   = 206,

    /**
     * This indicates that the specified array is currently mapped and thus
     * cannot be destroyed.
     */
    CUDA_ERROR_ARRAY_IS_MAPPED                = 207,

    /**
     * This indicates that the resource is already mapped.
     */
    CUDA_ERROR_ALREADY_MAPPED                 = 208,

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular CUDA source file that do not include the
     * corresponding device configuration.
     */
    CUDA_ERROR_NO_BINARY_FOR_GPU              = 209,

    /**
     * This indicates that a resource has already been acquired.
     */
    CUDA_ERROR_ALREADY_ACQUIRED               = 210,

    /**
     * This indicates that a resource is not mapped.
     */
    CUDA_ERROR_NOT_MAPPED                     = 211,

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY            = 212,

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    CUDA_ERROR_NOT_MAPPED_AS_POINTER          = 213,

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * execution.
     */
    CUDA_ERROR_ECC_UNCORRECTABLE              = 214,

    /**
     * This indicates that the ::CUlimit passed to the API call is not
     * supported by the active device.
     */
    CUDA_ERROR_UNSUPPORTED_LIMIT              = 215,

    /**
     * This indicates that the ::CUcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE         = 216,

    /**
     * This indicates that the device kernel source is invalid.
     */
    CUDA_ERROR_INVALID_SOURCE                 = 300,

    /**
     * This indicates that the file specified was not found.
     */
    CUDA_ERROR_FILE_NOT_FOUND                 = 301,

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,

    /**
     * This indicates that initialization of a shared object failed.
     */
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      = 303,

    /**
     * This indicates that an OS call failed.
     */
    CUDA_ERROR_OPERATING_SYSTEM               = 304,


    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::CUstream and ::CUevent.
     */
    CUDA_ERROR_INVALID_HANDLE                 = 400,


    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names, and surface names.
     */
    CUDA_ERROR_NOT_FOUND                      = 500,


    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be indicated
     * differently than ::CUDA_SUCCESS (which indicates completion). Calls that
     * may return this value include ::cuEventQuery() and ::cuStreamQuery().
     */
    CUDA_ERROR_NOT_READY                      = 600,


    /**
     * An exception occurred on the device while executing a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    CUDA_ERROR_LAUNCH_FAILED                  = 700,

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        = 701,

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::CUDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using CUDA.
     */
    CUDA_ERROR_LAUNCH_TIMEOUT                 = 702,

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  = 703,

    /**
     * This error indicates that a call to ::cuCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register memory from a context which has not had peer access
     * enabled yet via ::cuCtxEnablePeerAccess(), or that
     * ::cuCtxDisablePeerAccess() is trying to disable peer access
     * which has not been enabled yet.
     */
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED    = 705,

    /**
     * This error indicates that a call to ::cuMemPeerRegister is trying to
     * register already-registered memory.
     */
    CUDA_ERROR_PEER_MEMORY_ALREADY_REGISTERED = 706,

    /**
     * This error indicates that a call to ::cuMemPeerUnregister is trying to
     * unregister memory that has not been registered.
     */
    CUDA_ERROR_PEER_MEMORY_NOT_REGISTERED     = 707,

    /**
     * This error indicates that ::cuCtxCreate was called with the flag
     * ::CU_CTX_PRIMARY on a device which already has initialized its
     * primary context.
     */
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         = 708,

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::cuCtxDestroy, or is a primary context which
     * has not yet been initialized.
     */
    CUDA_ERROR_CONTEXT_IS_DESTROYED           = 709,

    /**
     * A device-side assert triggered during kernel execution. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using CUDA.
     */
    CUDA_ERROR_ASSERT                         = 710,

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::cuCtxEnablePeerAccess().
     */
    CUDA_ERROR_TOO_MANY_PEERS                 = 711,

    /**
     * This error indicates that the memory range passed to ::cuMemHostRegister()
     * has already been registered.
     */
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,

    /**
     * This error indicates that the pointer passed to ::cuMemHostUnregister()
     * does not correspond to any currently registered memory region.
     */
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     = 713,

    /**
     * This indicates that an unknown internal error has occurred.
     */
    CUDA_ERROR_UNKNOWN                        = 999
} CUresult;


#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    //cuGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}


inline const char* _ConvertSMVer2ArchName(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the GPU Arch name)
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    const char* name;
  } sSMtoArchName;

  sSMtoArchName nGpuArchNameSM[] = {
      {0x30, "Kepler"},
      {0x32, "Kepler"},
      {0x35, "Kepler"},
      {0x37, "Kepler"},
      {0x50, "Maxwell"},
      {0x52, "Maxwell"},
      {0x53, "Maxwell"},
      {0x60, "Pascal"},
      {0x61, "Pascal"},
      {0x62, "Pascal"},
      {0x70, "Volta"},
      {0x72, "Xavier"},
      {0x75, "Turing"},
      {0x80, "Ampere"},
      {0x86, "Ampere"},
      {0x87, "Ampere"},
      {0x89, "Ada"},
      {0x90, "Hopper"},
      {-1, "Graphics Device"}};

  int index = 0;

  while (nGpuArchNameSM[index].SM != -1) {
    if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchNameSM[index].name;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoArchName for SM %d.%d is undefined."
      "  Default to use %s\n",
      major, minor, nGpuArchNameSM[index - 1].name);
  return nGpuArchNameSM[index - 1].name;
}
inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}

inline int gpuGetMaxGflopsDeviceId() {
  int current_device = 0, sm_per_multiproc = 0;
  int max_perf_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  uint64_t max_compute_perf = 0;
  //checkCudaErrors(cudaGetDeviceCount(&device_count));
  cudaGetDeviceCount(&device_count);

  if (device_count == 0) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }
  current_device = 0;

  while (current_device < device_count) {
    int computeMode = -1, major = 0, minor = 0;
    //checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
    //checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
    //checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

    cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device);

    // If this GPU is not running on Compute Mode prohibited,
    // then we can add it to the list
    if (computeMode != cudaComputeModeProhibited) {
      if (major == 9999 && minor == 9999) {
        sm_per_multiproc = 1;
      } else {
        sm_per_multiproc =
            _ConvertSMVer2Cores(major,  minor);
      }
      int multiProcessorCount = 0, clockRate = 0;
      //checkCudaErrors(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
      cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device);
      cudaError_t result = cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
      if (result != cudaSuccess) {
        // If cudaDevAttrClockRate attribute is not supported we
        // set clockRate as 1, to consider GPU with most SMs and CUDA Cores.
        if(result == cudaErrorInvalidValue) {
          clockRate = 1;
        }
        else {
          fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", __FILE__, __LINE__,
            /* --------+++++++++++++++----------------- */
            //static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
            static_cast<unsigned int>(result), 100);
          exit(EXIT_FAILURE);
        }
      }
      uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

      if (compute_perf > max_compute_perf) {
        max_compute_perf = compute_perf;
        max_perf_device = current_device;
      }
    } else {
      devices_prohibited++;
    }

    ++current_device;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error:"
            " all devices have compute mode prohibited.\n");
    exit(EXIT_FAILURE);
  }

  return max_perf_device;
}


inline int findCudaDevice() {
    int devID = 0;
    // Otherwise pick the device with highest Gflops/s
    devID = gpuGetMaxGflopsDeviceId();
    //checkCudaErrors(cudaSetDevice(devID));
    cudaSetDevice(devID);
    int major = 0, minor = 0;
    //checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
    //checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
           devID, _ConvertSMVer2ArchName(major, minor), major, minor);

    return devID;
}


inline int gpuDeviceInit(int devID) {
  int device_count;
  //checkCudaErrors(cudaGetDeviceCount(&device_count));
  cudaGetDeviceCount(&device_count);

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() CUDA error: "
            "no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  //checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
  //checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
  //checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
  cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID);
  if (computeMode == cudaComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use cudaSetDevice().\n");
    return -1;
  }

  if (major < 1) {
    fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
    exit(EXIT_FAILURE);
  }

  //checkCudaErrors(cudaSetDevice(devID));
  cudaSetDevice(devID);
  printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName(major, minor));

  return devID;
}

static inline __host__ __device__ uint iDivUp(uint a, uint b) {
  return ((a % b) == 0) ? (a / b) : (a / b + 1);
}
static inline __host__ __device__ uint getSampleCount(uint dividend) {
  return iDivUp(dividend, SAMPLE_STRIDE);
}

#define W (sizeof(uint) * 8)
static inline __device__ uint nextPowerOfTwo(uint x) {
  /*
      --x;
      x |= x >> 1;
      x |= x >> 2;
      x |= x >> 4;
      x |= x >> 8;
      x |= x >> 16;
      return ++x;
  */
  return 1U << (W - __clz(x - 1));
}
template <uint sortDir>
static inline __device__ uint binarySearchInclusive(uint val, uint *data,
                                                    uint L, uint stride) {
  if (L == 0) {
    return 0;
  }

  uint pos = 0;

  for (; stride > 0; stride >>= 1) {
    uint newPos = umin(pos + stride, L);

    if ((sortDir && (data[newPos - 1] <= val)) ||
        (!sortDir && (data[newPos - 1] >= val))) {
      pos = newPos;
    }
  }

  return pos;
}
template <uint sortDir>
static inline __device__ uint binarySearchExclusive(uint val, uint *data,
                                                    uint L, uint stride) {
  if (L == 0) {
    return 0;
  }

  uint pos = 0;

  for (; stride > 0; stride >>= 1) {
    uint newPos = umin(pos + stride, L);

    if ((sortDir && (data[newPos - 1] < val)) ||
        (!sortDir && (data[newPos - 1] > val))) {
      pos = newPos;
    }
  }

  return pos;
}


template <uint sortDir>
__global__ void mergeSortSharedKernel(uint *d_DstKey, uint *d_DstVal,
                                      uint *d_SrcKey, uint *d_SrcVal,
                                      uint arrayLength) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint s_key[SHARED_SIZE_LIMIT];
  __shared__ uint s_val[SHARED_SIZE_LIMIT];

  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x + 0] = d_SrcKey[0];
  s_val[threadIdx.x + 0] = d_SrcVal[0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

  for (uint stride = 1; stride < arrayLength; stride <<= 1) {
    uint lPos = threadIdx.x & (stride - 1);
    uint *baseKey = s_key + 2 * (threadIdx.x - lPos);
    uint *baseVal = s_val + 2 * (threadIdx.x - lPos);

    cg::sync(cta);
    uint keyA = baseKey[lPos + 0];
    uint valA = baseVal[lPos + 0];
    uint keyB = baseKey[lPos + stride];
    uint valB = baseVal[lPos + stride];
    uint posA =
        binarySearchExclusive<sortDir>(keyA, baseKey + stride, stride, stride) +
        lPos;
    uint posB =
        binarySearchInclusive<sortDir>(keyB, baseKey + 0, stride, stride) +
        lPos;

    cg::sync(cta);
    baseKey[posA] = keyA;
    baseVal[posA] = valA;
    baseKey[posB] = keyB;
    baseVal[posB] = valB;
  }

  cg::sync(cta);
  d_DstKey[0] = s_key[threadIdx.x + 0];
  d_DstVal[0] = s_val[threadIdx.x + 0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
      s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

static void mergeSortShared(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey,
                            uint *d_SrcVal, uint batchSize, uint arrayLength,
                            uint sortDir) {
  if (arrayLength < 2) {
    return;
  }

  assert(SHARED_SIZE_LIMIT % arrayLength == 0);
  assert(((batchSize * arrayLength) % SHARED_SIZE_LIMIT) == 0);
  uint blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
  uint threadCount = SHARED_SIZE_LIMIT / 2;

  if (sortDir) {
    mergeSortSharedKernel<1U><<<blockCount, threadCount>>>(
        d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
    //getLastCudaError("mergeSortShared<1><<<>>> failed\n");
    printf("mergeSortShared<1><<<>>> failed\n");
  } else {
    mergeSortSharedKernel<0U><<<blockCount, threadCount>>>(
        d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength);
    //getLastCudaError("mergeSortShared<0><<<>>> failed\n");
    printf("mergeSortShared<0><<<>>> failed\n");
  }
}





template <uint sortDir>
__global__ void generateSampleRanksKernel(uint *d_RanksA, uint *d_RanksB,
                                          uint *d_SrcKey, uint stride, uint N,
                                          uint threadCount) {
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= threadCount) {
    return;
  }

  const uint i = pos & ((stride / SAMPLE_STRIDE) - 1);
  const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
  d_SrcKey += segmentBase;
  d_RanksA += segmentBase / SAMPLE_STRIDE;
  d_RanksB += segmentBase / SAMPLE_STRIDE;

  const uint segmentElementsA = stride;
  const uint segmentElementsB = umin(stride, N - segmentBase - stride);
  const uint segmentSamplesA = getSampleCount(segmentElementsA);
  const uint segmentSamplesB = getSampleCount(segmentElementsB);

  if (i < segmentSamplesA) {
    d_RanksA[i] = i * SAMPLE_STRIDE;
    d_RanksB[i] = binarySearchExclusive<sortDir>(
        d_SrcKey[i * SAMPLE_STRIDE], d_SrcKey + stride, segmentElementsB,
        nextPowerOfTwo(segmentElementsB));
  }

  if (i < segmentSamplesB) {
    d_RanksB[(stride / SAMPLE_STRIDE) + i] = i * SAMPLE_STRIDE;
    d_RanksA[(stride / SAMPLE_STRIDE) + i] = binarySearchInclusive<sortDir>(
        d_SrcKey[stride + i * SAMPLE_STRIDE], d_SrcKey + 0, segmentElementsA,
        nextPowerOfTwo(segmentElementsA));
  }
}

static void generateSampleRanks(uint *d_RanksA, uint *d_RanksB, uint *d_SrcKey,
                                uint stride, uint N, uint sortDir) {
  uint lastSegmentElements = N % (2 * stride);
  uint threadCount =
      (lastSegmentElements > stride)
          ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
          : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  if (sortDir) {
    generateSampleRanksKernel<1U><<<iDivUp(threadCount, 256), 256>>>(
        d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
    //getLastCudaError("generateSampleRanksKernel<1U><<<>>> failed\n");
    printf("generateSampleRanksKernel<1U><<<>>> failed\n");

  } else {
    generateSampleRanksKernel<0U><<<iDivUp(threadCount, 256), 256>>>(
        d_RanksA, d_RanksB, d_SrcKey, stride, N, threadCount);
    //getLastCudaError("generateSampleRanksKernel<0U><<<>>> failed\n");
    printf("generateSampleRanksKernel<0U><<<>>> failed\n");
  }
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 2: generate sample ranks and indices
////////////////////////////////////////////////////////////////////////////////
__global__ void mergeRanksAndIndicesKernel(uint *d_Limits, uint *d_Ranks,
                                           uint stride, uint N,
                                           uint threadCount) {
  uint pos = blockIdx.x * blockDim.x + threadIdx.x;

  if (pos >= threadCount) {
    return;
  }

  const uint i = pos & ((stride / SAMPLE_STRIDE) - 1);
  const uint segmentBase = (pos - i) * (2 * SAMPLE_STRIDE);
  d_Ranks += (pos - i) * 2;
  d_Limits += (pos - i) * 2;

  const uint segmentElementsA = stride;
  const uint segmentElementsB = umin(stride, N - segmentBase - stride);
  const uint segmentSamplesA = getSampleCount(segmentElementsA);
  const uint segmentSamplesB = getSampleCount(segmentElementsB);

  if (i < segmentSamplesA) {
    uint dstPos = binarySearchExclusive<1U>(
                      d_Ranks[i], d_Ranks + segmentSamplesA, segmentSamplesB,
                      nextPowerOfTwo(segmentSamplesB)) +
                  i;
    d_Limits[dstPos] = d_Ranks[i];
  }

  if (i < segmentSamplesB) {
    uint dstPos = binarySearchInclusive<1U>(d_Ranks[segmentSamplesA + i],
                                            d_Ranks, segmentSamplesA,
                                            nextPowerOfTwo(segmentSamplesA)) +
                  i;
    d_Limits[dstPos] = d_Ranks[segmentSamplesA + i];
  }
}

static void mergeRanksAndIndices(uint *d_LimitsA, uint *d_LimitsB,
                                 uint *d_RanksA, uint *d_RanksB, uint stride,
                                 uint N) {
  uint lastSegmentElements = N % (2 * stride);
  uint threadCount =
      (lastSegmentElements > stride)
          ? (N + 2 * stride - lastSegmentElements) / (2 * SAMPLE_STRIDE)
          : (N - lastSegmentElements) / (2 * SAMPLE_STRIDE);

  mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
      d_LimitsA, d_RanksA, stride, N, threadCount);
  //getLastCudaError("mergeRanksAndIndicesKernel(A)<<<>>> failed\n");
    printf("mergeRanksAndIndicesKernel(A)<<<>>> failed\n");

  mergeRanksAndIndicesKernel<<<iDivUp(threadCount, 256), 256>>>(
      d_LimitsB, d_RanksB, stride, N, threadCount);
  //getLastCudaError("mergeRanksAndIndicesKernel(B)<<<>>> failed\n");
    printf("mergeRanksAndIndicesKernel(B)<<<>>> failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Merge step 3: merge elementary intervals
////////////////////////////////////////////////////////////////////////////////
template <uint sortDir>
inline __device__ void merge(uint *dstKey, uint *dstVal, uint *srcAKey,
                             uint *srcAVal, uint *srcBKey, uint *srcBVal,
                             uint lenA, uint nPowTwoLenA, uint lenB,
                             uint nPowTwoLenB, cg::thread_block cta) {
  uint keyA, valA, keyB, valB, dstPosA, dstPosB;

  if (threadIdx.x < lenA) {
    keyA = srcAKey[threadIdx.x];
    valA = srcAVal[threadIdx.x];
    dstPosA = binarySearchExclusive<sortDir>(keyA, srcBKey, lenB, nPowTwoLenB) +
              threadIdx.x;
  }

  if (threadIdx.x < lenB) {
    keyB = srcBKey[threadIdx.x];
    valB = srcBVal[threadIdx.x];
    dstPosB = binarySearchInclusive<sortDir>(keyB, srcAKey, lenA, nPowTwoLenA) +
              threadIdx.x;
  }

  cg::sync(cta);

  if (threadIdx.x < lenA) {
    dstKey[dstPosA] = keyA;
    dstVal[dstPosA] = valA;
  }

  if (threadIdx.x < lenB) {
    dstKey[dstPosB] = keyB;
    dstVal[dstPosB] = valB;
  }
}

template <uint sortDir>
__global__ void mergeElementaryIntervalsKernel(uint *d_DstKey, uint *d_DstVal,
                                               uint *d_SrcKey, uint *d_SrcVal,
                                               uint *d_LimitsA, uint *d_LimitsB,
                                               uint stride, uint N) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ uint s_key[2 * SAMPLE_STRIDE];
  __shared__ uint s_val[2 * SAMPLE_STRIDE];

  const uint intervalI = blockIdx.x & ((2 * stride) / SAMPLE_STRIDE - 1);
  const uint segmentBase = (blockIdx.x - intervalI) * SAMPLE_STRIDE;
  d_SrcKey += segmentBase;
  d_SrcVal += segmentBase;
  d_DstKey += segmentBase;
  d_DstVal += segmentBase;

  // Set up threadblock-wide parameters
  __shared__ uint startSrcA, startSrcB, lenSrcA, lenSrcB, startDstA, startDstB;

  if (threadIdx.x == 0) {
    uint segmentElementsA = stride;
    uint segmentElementsB = umin(stride, N - segmentBase - stride);
    uint segmentSamplesA = getSampleCount(segmentElementsA);
    uint segmentSamplesB = getSampleCount(segmentElementsB);
    uint segmentSamples = segmentSamplesA + segmentSamplesB;

    startSrcA = d_LimitsA[blockIdx.x];
    startSrcB = d_LimitsB[blockIdx.x];
    uint endSrcA = (intervalI + 1 < segmentSamples) ? d_LimitsA[blockIdx.x + 1]
                                                    : segmentElementsA;
    uint endSrcB = (intervalI + 1 < segmentSamples) ? d_LimitsB[blockIdx.x + 1]
                                                    : segmentElementsB;
    lenSrcA = endSrcA - startSrcA;
    lenSrcB = endSrcB - startSrcB;
    startDstA = startSrcA + startSrcB;
    startDstB = startDstA + lenSrcA;
  }

  // Load main input data
  cg::sync(cta);

  if (threadIdx.x < lenSrcA) {
    s_key[threadIdx.x + 0] = d_SrcKey[0 + startSrcA + threadIdx.x];
    s_val[threadIdx.x + 0] = d_SrcVal[0 + startSrcA + threadIdx.x];
  }

  if (threadIdx.x < lenSrcB) {
    s_key[threadIdx.x + SAMPLE_STRIDE] =
        d_SrcKey[stride + startSrcB + threadIdx.x];
    s_val[threadIdx.x + SAMPLE_STRIDE] =
        d_SrcVal[stride + startSrcB + threadIdx.x];
  }

  // Merge data in shared memory
  cg::sync(cta);
  merge<sortDir>(s_key, s_val, s_key + 0, s_val + 0, s_key + SAMPLE_STRIDE,
                 s_val + SAMPLE_STRIDE, lenSrcA, SAMPLE_STRIDE, lenSrcB,
                 SAMPLE_STRIDE, cta);

  // Store merged data
  cg::sync(cta);

  if (threadIdx.x < lenSrcA) {
    d_DstKey[startDstA + threadIdx.x] = s_key[threadIdx.x];
    d_DstVal[startDstA + threadIdx.x] = s_val[threadIdx.x];
  }

  if (threadIdx.x < lenSrcB) {
    d_DstKey[startDstB + threadIdx.x] = s_key[lenSrcA + threadIdx.x];
    d_DstVal[startDstB + threadIdx.x] = s_val[lenSrcA + threadIdx.x];
  }
}

static void mergeElementaryIntervals(uint *d_DstKey, uint *d_DstVal,
                                     uint *d_SrcKey, uint *d_SrcVal,
                                     uint *d_LimitsA, uint *d_LimitsB,
                                     uint stride, uint N, uint sortDir) {
  uint lastSegmentElements = N % (2 * stride);
  uint mergePairs = (lastSegmentElements > stride)
                        ? getSampleCount(N)
                        : (N - lastSegmentElements) / SAMPLE_STRIDE;

  if (sortDir) {
    mergeElementaryIntervalsKernel<1U><<<mergePairs, SAMPLE_STRIDE>>>(
        d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, d_LimitsA, d_LimitsB, stride,
        N);
    //getLastCudaError("mergeElementaryIntervalsKernel<1> failed\n");
    printf("mergeElementaryIntervalsKernel<1> failed\n");
  } else {
    mergeElementaryIntervalsKernel<0U><<<mergePairs, SAMPLE_STRIDE>>>(
        d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, d_LimitsA, d_LimitsB, stride,
        N);
    //getLastCudaError("mergeElementaryIntervalsKernel<0> failed\n");
    printf("mergeElementaryIntervalsKernel<0> failed\n");
  }
}


extern "C" void initMergeSort(void) {
  //checkCudaErrors(cudaMalloc((void **)&d_RanksA, MAX_SAMPLE_COUNT * sizeof(uint)));
  //checkCudaErrors(cudaMalloc((void **)&d_RanksB, MAX_SAMPLE_COUNT * sizeof(uint)));
  //checkCudaErrors(cudaMalloc((void **)&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(uint)));
  //checkCudaErrors(cudaMalloc((void **)&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(uint)));
  cudaMalloc((void **)&d_RanksA, MAX_SAMPLE_COUNT * sizeof(uint));
  cudaMalloc((void **)&d_RanksB, MAX_SAMPLE_COUNT * sizeof(uint));
  cudaMalloc((void **)&d_LimitsA, MAX_SAMPLE_COUNT * sizeof(uint));
  cudaMalloc((void **)&d_LimitsB, MAX_SAMPLE_COUNT * sizeof(uint));
}

extern "C" void mergeSort(uint *d_DstKey, uint *d_DstVal, uint *d_BufKey,
                          uint *d_BufVal, uint *d_SrcKey, uint *d_SrcVal,
                          uint N, uint sortDir) {
  uint stageCount = 0;
  printf("Entered mergeSort with N = %d\n", N);

  for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1, stageCount++);
  printf("stageCount = %d\n", stageCount);

  uint *ikey, *ival, *okey, *oval;

  if (stageCount & 1) {
    ikey = d_BufKey;
    ival = d_BufVal;
    okey = d_DstKey;
    oval = d_DstVal;
  } else {
    ikey = d_DstKey;
    ival = d_DstVal;
    okey = d_BufKey;
    oval = d_BufVal;
  }
    printf("\n**** pre assert ****\n");

    assert(N <= (SAMPLE_STRIDE * MAX_SAMPLE_COUNT));
    printf("\n**** post #1 assert ****\n");

    //pro gamer move 
    assert(N % SHARED_SIZE_LIMIT == 0);

    // 10000 % 1024U 
    printf("\n**** post #2 assert ****\n");
    mergeSortShared(ikey, ival, d_SrcKey, d_SrcVal, N / SHARED_SIZE_LIMIT, SHARED_SIZE_LIMIT, sortDir);

  for (uint stride = SHARED_SIZE_LIMIT; stride < N; stride <<= 1) {
    uint lastSegmentElements = N % (2 * stride);

    // Find sample ranks and prepare for limiters merge
    generateSampleRanks(d_RanksA, d_RanksB, ikey, stride, N, sortDir);

    // Merge ranks and indices
    mergeRanksAndIndices(d_LimitsA, d_LimitsB, d_RanksA, d_RanksB, stride, N);

    // Merge elementary intervals
    mergeElementaryIntervals(okey, oval, ikey, ival, d_LimitsA, d_LimitsB, stride, N, sortDir);

    if (lastSegmentElements <= stride) {
      // Last merge segment consists of a single array which just needs to be
      // passed through
      //checkCudaErrors(cudaMemcpy( okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements),
          //lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
      //checkCudaErrors(cudaMemcpy(
          //oval + (N - lastSegmentElements), ival + (N - lastSegmentElements),
          //lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice));
        cudaMemcpy( okey + (N - lastSegmentElements), ikey + (N - lastSegmentElements),
          lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice);
      cudaMemcpy(
          oval + (N - lastSegmentElements), ival + (N - lastSegmentElements),
          lastSegmentElements * sizeof(uint), cudaMemcpyDeviceToDevice);
    }

    uint *t;
    t = ikey;
    ikey = okey;
    okey = t;
    t = ival;
    ival = oval;
    oval = t;
  }
}

extern "C" int validateSortedValues(uint *resKey, uint *resVal, uint *srcKey,
                                    uint batchSize, uint arrayLength) {
  int correctFlag = 1, stableFlag = 1;

  printf("...inspecting keys and values array: ");

  for (uint i = 0; i < batchSize;
       i++, resKey += arrayLength, resVal += arrayLength) {
    for (uint j = 0; j < arrayLength; j++) {
      if (resKey[j] != srcKey[resVal[j]]) correctFlag = 0;

      if ((j < arrayLength - 1) && (resKey[j] == resKey[j + 1]) &&
          (resVal[j] > resVal[j + 1]))
        stableFlag = 0;
    }
  }

  printf(correctFlag ? "OK\n" : "***corrupted!!!***\n");
  printf(stableFlag ? "...stability property: stable!\n"
                    : "...stability property: NOT stable\n");

  return correctFlag;
}

extern "C" uint validateSortedKeys(uint *resKey, uint *srcKey, uint batchSize,
                                   uint arrayLength, uint numValues,
                                   uint sortDir) {
  uint *srcHist;
  uint *resHist;

  if (arrayLength < 2) {
    printf("validateSortedKeys(): arrays too short, exiting...\n");
    return 1;
  }

  printf("...inspecting keys array: ");
  srcHist = (uint *)malloc(numValues * sizeof(uint));
  resHist = (uint *)malloc(numValues * sizeof(uint));

  int flag = 1;

  for (uint j = 0; j < batchSize;
       j++, srcKey += arrayLength, resKey += arrayLength) {
    // Build histograms for keys arrays
    memset(srcHist, 0, numValues * sizeof(uint));
    memset(resHist, 0, numValues * sizeof(uint));

    for (uint i = 0; i < arrayLength; i++) {
      if ((srcKey[i] < numValues) && (resKey[i] < numValues)) {
        srcHist[srcKey[i]]++;
        resHist[resKey[i]]++;
      } else {
        fprintf(
            stderr,
            "***Set %u source/result key arrays are not limited properly***\n",
            j);
        flag = 0;
        goto brk;
      }
    }

    // Compare the histograms
    for (uint i = 0; i < numValues; i++)
      if (srcHist[i] != resHist[i]) {
        fprintf(stderr,
                "***Set %u source/result keys histograms do not match***\n", j);
        flag = 0;
        goto brk;
      }

    // Finally check the ordering
    for (uint i = 0; i < arrayLength - 1; i++)
      if ((sortDir && (resKey[i] > resKey[i + 1])) ||
          (!sortDir && (resKey[i] < resKey[i + 1]))) {
        fprintf(stderr,
                "***Set %u result key array is not ordered properly***\n", j);
        flag = 0;
        goto brk;
      }
  }

brk:
  free(resHist);
  free(srcHist);

  if (flag) printf("OK\n");

  return flag;
}

extern "C" void closeMergeSort(void) {
  //checkCudaErrors(cudaFree(d_RanksA));
  //checkCudaErrors(cudaFree(d_RanksB));
  //checkCudaErrors(cudaFree(d_LimitsB));
  //checkCudaErrors(cudaFree(d_LimitsA));
  cudaFree(d_RanksA);
  cudaFree(d_RanksB);
  cudaFree(d_LimitsB);
  cudaFree(d_LimitsA);
}

extern "C" {
    void merge_sort(uint* arr, uint N, uint numValues) {
        //print array
        printf("Array before sorting: \n");

        
        uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
        uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
        StopWatchInterface *hTimer = NULL;

        //const uint N = 4 * 1048576;
        const uint DIR = 1;
        //const uint numValues = 65536;

        int dev = findCudaDevice();

        if (dev == -1) {
            return;
        }

        printf("Allocating and initializing host arrays...\n\n");
        sdkCreateTimer(&hTimer);
        h_SrcKey = (uint *)malloc(N * sizeof(uint));
        h_SrcVal = (uint *)malloc(N * sizeof(uint));
        h_DstKey = (uint *)malloc(N * sizeof(uint));
        h_DstVal = (uint *)malloc(N * sizeof(uint));

        printf("Allocating and initializing CUDA arrays...\n\n");
        //checkCudaErrors(cudaMalloc((void **)&d_DstKey, N * sizeof(uint)));
        //checkCudaErrors(cudaMalloc((void **)&d_DstVal, N * sizeof(uint)));
        //checkCudaErrors(cudaMalloc((void **)&d_BufKey, N * sizeof(uint)));
        //checkCudaErrors(cudaMalloc((void **)&d_BufVal, N * sizeof(uint)));
        //checkCudaErrors(cudaMalloc((void **)&d_SrcKey, N * sizeof(uint)));
        //checkCudaErrors(cudaMalloc((void **)&d_SrcVal, N * sizeof(uint)));
        //checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice));
        //checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice));
        cudaMalloc((void **)&d_DstKey, N * sizeof(uint));
        cudaMalloc((void **)&d_DstVal, N * sizeof(uint));
        cudaMalloc((void **)&d_BufKey, N * sizeof(uint));
        cudaMalloc((void **)&d_BufVal, N * sizeof(uint));
        cudaMalloc((void **)&d_SrcKey, N * sizeof(uint));
        cudaMalloc((void **)&d_SrcVal, N * sizeof(uint));
        cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint), cudaMemcpyHostToDevice);
        cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint), cudaMemcpyHostToDevice);

        printf("Initializing GPU merge sort...\n");
        initMergeSort();

        printf("Running GPU merge sort...\n");
        //checkCudaErrors(cudaDeviceSynchronize());
        cudaDeviceSynchronize();
        sdkResetTimer(&hTimer);
        sdkStartTimer(&hTimer);
        mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N, DIR);
        //checkCudaErrors(cudaDeviceSynchronize());
        cudaDeviceSynchronize();
        sdkStopTimer(&hTimer);
        printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

        printf("Reading back GPU merge sort results...\n");
        //checkCudaErrors(cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost));
        //checkCudaErrors(cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost));
        cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint), cudaMemcpyDeviceToHost);

        uint keysFlag = validateSortedKeys(h_DstKey, h_SrcKey, 1, N, numValues, DIR);
        uint valuesFlag = validateSortedValues(h_DstKey, h_DstVal, h_SrcKey, 1, N);

        //print array

        printf("Shutting down...\n");
        closeMergeSort();
        sdkDeleteTimer(&hTimer);
        //checkCudaErrors(cudaFree(d_SrcVal));
        //checkCudaErrors(cudaFree(d_SrcKey));
        //checkCudaErrors(cudaFree(d_BufVal));
        //checkCudaErrors(cudaFree(d_BufKey));
        //checkCudaErrors(cudaFree(d_DstVal));
        //checkCudaErrors(cudaFree(d_DstKey));
        cudaFree(d_SrcVal);
        cudaFree(d_SrcKey);
        cudaFree(d_BufVal);
        cudaFree(d_BufKey);
        cudaFree(d_DstVal);
        cudaFree(d_DstKey);
        free(h_DstVal);
        free(h_DstKey);
        free(h_SrcVal);
        free(h_SrcKey);

        //exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
        return;
    }

}
