#ifndef _CULIB_H_
#define _CULIB_H_

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
        bool Success;
        int DeviceNum;
        char DeviceName[256];
        int CudaDriverVersion;
        int CudaDriverVersionMinor;
        int CudaRuntimeVersion;
        int CudaRuntimeVersionMinor;
        int CudaCapabilityMajor;
        int CudaCapabilityMinor;
        char TotalGlobalMemory[256];
        int Multiprocessors;
        int CudaCoresPerMultiprocessor;
        int GpuMaxClockRate;
	    int MemoryClockRate;
	    int MemoryBusWidth;
	    int L2CacheSize;
	    int MaxTextureDimensionSize[6];
	    int MaxLayered1DTextureSize[2];
	    int MaxLayered2DTextureSize[3];
	    int64_t TotalConstantMemory;
	    int64_t TotalSharedMemoryPerBlock;
	    int64_t TotalSharedMemoryPerMultiprocessor;
	    int TotalRegistersPerBlock;
	    int WarpSize;
	    int MaxThreadsPerMultiprocessor;
	    int MaxThreadsPerBlock;
	    int MaxDimensionSizeOfThreadBlock[3];
	    int MaxDimensionSizeOfGridSize[3];
	    int64_t MaxMemoryPitch;
	    int64_t TextureAlignment;
	    bool ConcurrentCopyAndKernelExecution;
	    bool RunTimeLimitOnKernels;
	    bool IntegratedGpuSharingHostMemory;
	    bool SupportHostPageLockedMemoryMapping;
	    bool AlignmentRequirementForSurfaces;
	    bool DeviceHasEccSupport;
	    bool DeviceSupportsUnifiedAddressing;
	    bool DeviceSupportsManagedMemory;
	    bool DeviceSupportsComputePreemption;
	    bool SupportsCooperativeKernelLaunch;
	    bool SupportsMultiDeviceCoopKernelLaunch;
	    int DevicePciDomainIdBusIdLocationId[3];
        } GpuInfo;

GpuInfo *run_query();

#endif