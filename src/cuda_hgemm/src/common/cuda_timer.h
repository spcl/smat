

#pragma once

#include "common.h"

class CudaTimer {
public:
    CudaTimer() {
        HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&m_start));
        HGEMM_CHECK(m_start);
        HGEMM_CHECK_CUDART_ERROR(cudaEventCreate(&m_end));
        HGEMM_CHECK(m_end);
    }

    ~CudaTimer() {
        if (m_start) {
            HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(m_start));
            m_start = nullptr;
        }

        if (m_end) {
            HGEMM_CHECK_CUDART_ERROR(cudaEventDestroy(m_end));
            m_end = nullptr;
        }
    }

    void start() {
        HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(m_start));
    }

    float end() {
        HGEMM_CHECK_CUDART_ERROR(cudaEventRecord(m_end));
        HGEMM_CHECK_CUDART_ERROR(cudaEventSynchronize(m_end));
        HGEMM_CHECK_CUDART_ERROR(cudaEventElapsedTime(&m_elapsed_time, m_start, m_end));

        return m_elapsed_time;
    }

private:
    cudaEvent_t m_start = nullptr;
    cudaEvent_t m_end = nullptr;
    float m_elapsed_time = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(CudaTimer);
};
