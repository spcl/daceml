#include "onnxruntime_c_api.h"
#include "cpu_provider_factory.h"
#ifndef __DACE_ONNX_H
#define __DACE_ONNX_H

const OrtApi* __ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

// helper function to check for status
void __ort_check_status(OrtStatus* status)
{
    if (status != NULL) {
        const char* msg = __ort_api->GetErrorMessage(status);
        fprintf(stderr, "%s\\n", msg);
        __ort_api->ReleaseStatus(status);
        exit(1);
    }
}
OrtEnv* __ort_env;
OrtKernelSession* __ort_session;
OrtSessionOptions* __ort_session_options;

OrtMemoryInfo* __ort_cpu_mem_info;

#endif  // __DACE_ONNX_H
