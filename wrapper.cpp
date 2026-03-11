/**
 * C wrapper implementation for MNN library
 */

#include "wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <cstring>
#include <vector>

/* ============ Interpreter API ============ */

MNNC_Interpreter* mnnc_interpreter_create_from_file(const char* model_path) {
    if (!model_path) return nullptr;
    auto* interpreter = MNN::Interpreter::createFromFile(model_path);
    return reinterpret_cast<MNNC_Interpreter*>(interpreter);
}

MNNC_Interpreter* mnnc_interpreter_create_from_buffer(const void* buffer, size_t size) {
    if (!buffer || size == 0) return nullptr;
    auto* interpreter = MNN::Interpreter::createFromBuffer(buffer, size);
    return reinterpret_cast<MNNC_Interpreter*>(interpreter);
}

void mnnc_interpreter_destroy(MNNC_Interpreter* interpreter) {
    if (interpreter) {
        MNN::Interpreter::destroy(reinterpret_cast<MNN::Interpreter*>(interpreter));
    }
}

void mnnc_interpreter_set_session_mode(MNNC_Interpreter* interpreter, MNNC_SessionMode mode) {
    if (!interpreter) return;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    interp->setSessionMode(static_cast<MNN::Interpreter::SessionMode>(mode));
}

void mnnc_interpreter_set_cache_file(MNNC_Interpreter* interpreter, const char* cache_path) {
    if (!interpreter || !cache_path) return;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    interp->setCacheFile(cache_path);
}

/* ============ Session API ============ */

MNNC_Session* mnnc_session_create(MNNC_Interpreter* interpreter, const MNNC_ScheduleConfig* config) {
    if (!interpreter) return nullptr;

    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    MNN::ScheduleConfig sconfig;

    if (config) {
        sconfig.type = static_cast<MNNForwardType>(config->forward_type);
        sconfig.numThread = config->num_threads;
    }

    auto* session = interp->createSession(sconfig);
    return reinterpret_cast<MNNC_Session*>(session);
}

MNNC_Session* mnnc_session_create_with_backend(
    MNNC_Interpreter* interpreter,
    const MNNC_ScheduleConfig* config,
    const MNNC_BackendConfig* backend
) {
    if (!interpreter) return nullptr;

    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    MNN::ScheduleConfig sconfig;
    MNN::BackendConfig bconfig;

    if (config) {
        sconfig.type = static_cast<MNNForwardType>(config->forward_type);
        sconfig.numThread = config->num_threads;
    }

    if (backend) {
        bconfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(backend->precision);
        bconfig.power = static_cast<MNN::BackendConfig::PowerMode>(backend->power);
        bconfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(backend->memory);
        sconfig.backendConfig = &bconfig;
    }

    auto* session = interp->createSession(sconfig);
    return reinterpret_cast<MNNC_Session*>(session);
}

void mnnc_session_release(MNNC_Interpreter* interpreter, MNNC_Session* session) {
    if (!interpreter || !session) return;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    interp->releaseSession(reinterpret_cast<MNN::Session*>(session));
}

MNNC_ErrorCode mnnc_session_resize(MNNC_Interpreter* interpreter, MNNC_Session* session) {
    if (!interpreter || !session) return MNNC_ERROR_INVALID_ARGUMENT;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    interp->resizeSession(reinterpret_cast<MNN::Session*>(session));
    return MNNC_OK;
}

MNNC_ErrorCode mnnc_session_resize_ex(MNNC_Interpreter* interpreter, MNNC_Session* session, int need_realloc) {
    if (!interpreter || !session) return MNNC_ERROR_INVALID_ARGUMENT;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    interp->resizeSession(reinterpret_cast<MNN::Session*>(session), need_realloc);
    return MNNC_OK;
}

MNNC_ErrorCode mnnc_session_run(MNNC_Interpreter* interpreter, MNNC_Session* session) {
    if (!interpreter || !session) return MNNC_ERROR_INVALID_ARGUMENT;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    auto code = interp->runSession(reinterpret_cast<MNN::Session*>(session));
    return static_cast<MNNC_ErrorCode>(code);
}

void mnnc_interpreter_resize_tensor(
    MNNC_Interpreter* interpreter,
    MNNC_Tensor* tensor,
    const int* dims,
    int dim_count
) {
    if (!interpreter || !tensor || !dims || dim_count <= 0) return;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    auto* t = reinterpret_cast<MNN::Tensor*>(tensor);
    std::vector<int> shape(dims, dims + dim_count);
    interp->resizeTensor(t, shape);
}

/* ============ Tensor API ============ */

MNNC_Tensor* mnnc_session_get_input(MNNC_Interpreter* interpreter, MNNC_Session* session, const char* name) {
    if (!interpreter || !session) return nullptr;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    auto* tensor = interp->getSessionInput(reinterpret_cast<MNN::Session*>(session), name);
    return reinterpret_cast<MNNC_Tensor*>(tensor);
}

MNNC_Tensor* mnnc_session_get_output(MNNC_Interpreter* interpreter, MNNC_Session* session, const char* name) {
    if (!interpreter || !session) return nullptr;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    auto* tensor = interp->getSessionOutput(reinterpret_cast<MNN::Session*>(session), name);
    return reinterpret_cast<MNNC_Tensor*>(tensor);
}

static halide_type_t dtype_to_halide(MNNC_DataType dtype) {
    switch (dtype) {
        case MNNC_DTYPE_FLOAT: return halide_type_of<float>();
        case MNNC_DTYPE_INT32: return halide_type_of<int32_t>();
        case MNNC_DTYPE_UINT8: return halide_type_of<uint8_t>();
        case MNNC_DTYPE_INT8: return halide_type_of<int8_t>();
        case MNNC_DTYPE_INT64: return halide_type_of<int64_t>();
        case MNNC_DTYPE_DOUBLE: return halide_type_of<double>();
        case MNNC_DTYPE_UINT32: return halide_type_of<uint32_t>();
        case MNNC_DTYPE_INT16: return halide_type_of<int16_t>();
        case MNNC_DTYPE_UINT16: return halide_type_of<uint16_t>();
        default: return halide_type_of<float>();
    }
}

MNNC_Tensor* mnnc_tensor_create(const int* dims, int dim_count, MNNC_DataType dtype, MNNC_DimensionType dim_type) {
    if (!dims || dim_count <= 0) return nullptr;

    std::vector<int> shape(dims, dims + dim_count);
    auto* tensor = MNN::Tensor::create(
        shape,
        dtype_to_halide(dtype),
        nullptr,
        static_cast<MNN::Tensor::DimensionType>(dim_type)
    );
    return reinterpret_cast<MNNC_Tensor*>(tensor);
}

MNNC_Tensor* mnnc_tensor_create_with_data(
    const int* dims,
    int dim_count,
    MNNC_DataType dtype,
    void* data,
    MNNC_DimensionType dim_type
) {
    if (!dims || dim_count <= 0) return nullptr;

    std::vector<int> shape(dims, dims + dim_count);
    auto* tensor = MNN::Tensor::create(
        shape,
        dtype_to_halide(dtype),
        data,
        static_cast<MNN::Tensor::DimensionType>(dim_type)
    );
    return reinterpret_cast<MNNC_Tensor*>(tensor);
}

void mnnc_tensor_destroy(MNNC_Tensor* tensor) {
    if (tensor) {
        delete reinterpret_cast<MNN::Tensor*>(tensor);
    }
}

int mnnc_tensor_get_dimensions(const MNNC_Tensor* tensor) {
    if (!tensor) return 0;
    return reinterpret_cast<const MNN::Tensor*>(tensor)->dimensions();
}

int mnnc_tensor_get_shape(const MNNC_Tensor* tensor, int* dims, int max_dims) {
    if (!tensor || !dims || max_dims <= 0) return 0;

    auto* t = reinterpret_cast<const MNN::Tensor*>(tensor);
    const auto& shape = t->shape();
    int count = std::min(static_cast<int>(shape.size()), max_dims);
    for (int i = 0; i < count; ++i) {
        dims[i] = shape[i];
    }
    return static_cast<int>(shape.size());
}

int mnnc_tensor_element_count(const MNNC_Tensor* tensor) {
    if (!tensor) return 0;
    return reinterpret_cast<const MNN::Tensor*>(tensor)->elementSize();
}

MNNC_DataType mnnc_tensor_get_dtype(const MNNC_Tensor* tensor) {
    if (!tensor) return MNNC_DTYPE_FLOAT;
    auto* t = reinterpret_cast<const MNN::Tensor*>(tensor);
    auto type = t->getType();

    if (type.code == halide_type_float) {
        if (type.bits == 32) return MNNC_DTYPE_FLOAT;
        if (type.bits == 64) return MNNC_DTYPE_DOUBLE;
        if (type.bits == 16) return MNNC_DTYPE_FLOAT16;
    } else if (type.code == halide_type_int) {
        if (type.bits == 32) return MNNC_DTYPE_INT32;
        if (type.bits == 64) return MNNC_DTYPE_INT64;
        if (type.bits == 16) return MNNC_DTYPE_INT16;
        if (type.bits == 8) return MNNC_DTYPE_INT8;
    } else if (type.code == halide_type_uint) {
        if (type.bits == 32) return MNNC_DTYPE_UINT32;
        if (type.bits == 16) return MNNC_DTYPE_UINT16;
        if (type.bits == 8) return MNNC_DTYPE_UINT8;
    }
    return MNNC_DTYPE_FLOAT;
}

MNNC_DimensionType mnnc_tensor_get_dimension_type(const MNNC_Tensor* tensor) {
    if (!tensor) return MNNC_NCHW;
    auto* t = reinterpret_cast<const MNN::Tensor*>(tensor);
    auto dim_type = t->getDimensionType();
    switch (dim_type) {
        case MNN::Tensor::CAFFE: return MNNC_NCHW;
        case MNN::Tensor::TENSORFLOW: return MNNC_NHWC;
        case MNN::Tensor::CAFFE_C4: return MNNC_NC4HW4;
        default: return MNNC_NCHW;
    }
}

MNNC_Tensor* mnnc_tensor_create_host_from_device(const MNNC_Tensor* device_tensor, int copy_data) {
    if (!device_tensor) return nullptr;
    auto* device = reinterpret_cast<const MNN::Tensor*>(device_tensor);
    auto* host = MNN::Tensor::createHostTensorFromDevice(device, copy_data != 0);
    return reinterpret_cast<MNNC_Tensor*>(host);
}

void* mnnc_tensor_get_host(MNNC_Tensor* tensor) {
    if (!tensor) return nullptr;
    return reinterpret_cast<MNN::Tensor*>(tensor)->host<void>();
}

const void* mnnc_tensor_get_host_const(const MNNC_Tensor* tensor) {
    if (!tensor) return nullptr;
    return reinterpret_cast<const MNN::Tensor*>(tensor)->host<void>();
}

int mnnc_tensor_copy_from_host_tensor(MNNC_Tensor* device, const MNNC_Tensor* host) {
    if (!device || !host) return 0;
    return reinterpret_cast<MNN::Tensor*>(device)->copyFromHostTensor(
        reinterpret_cast<const MNN::Tensor*>(host)
    ) ? 1 : 0;
}

int mnnc_tensor_copy_to_host_tensor(const MNNC_Tensor* device, MNNC_Tensor* host) {
    if (!device || !host) return 0;
    return reinterpret_cast<const MNN::Tensor*>(device)->copyToHostTensor(
        reinterpret_cast<MNN::Tensor*>(host)
    ) ? 1 : 0;
}

int mnnc_tensor_copy_from_host(MNNC_Tensor* tensor, const void* data, size_t size) {
    if (!tensor || !data) return 0;
    auto* t = reinterpret_cast<MNN::Tensor*>(tensor);
    void* host = t->host<void>();
    if (!host) return 0;

    size_t tensor_size = t->size();
    size_t copy_size = std::min(size, tensor_size);
    std::memcpy(host, data, copy_size);
    return 1;
}

int mnnc_tensor_copy_to_host(const MNNC_Tensor* tensor, void* data, size_t size) {
    if (!tensor || !data) return 0;
    auto* t = reinterpret_cast<const MNN::Tensor*>(tensor);
    const void* host = t->host<void>();
    if (!host) return 0;

    size_t tensor_size = t->size();
    size_t copy_size = std::min(size, tensor_size);
    std::memcpy(data, host, copy_size);
    return 1;
}

/* ============ Image Processing API ============ */

static MNN::CV::ImageFormat to_cv_format(MNNC_ImageFormat fmt) {
    switch (fmt) {
        case MNNC_IMAGE_RGBA: return MNN::CV::RGBA;
        case MNNC_IMAGE_RGB: return MNN::CV::RGB;
        case MNNC_IMAGE_BGR: return MNN::CV::BGR;
        case MNNC_IMAGE_GRAY: return MNN::CV::GRAY;
        case MNNC_IMAGE_BGRA: return MNN::CV::BGRA;
        case MNNC_IMAGE_YUV_NV21: return MNN::CV::YUV_NV21;
        case MNNC_IMAGE_YUV_NV12: return MNN::CV::YUV_NV12;
        case MNNC_IMAGE_YUV_I420: return MNN::CV::YUV_I420;
        default: return MNN::CV::RGBA;
    }
}

MNNC_ImageProcess* mnnc_image_process_create(const MNNC_ImageProcessConfig* config) {
    if (!config) return nullptr;

    MNN::CV::ImageProcess::Config cvconfig;
    cvconfig.filterType = static_cast<MNN::CV::Filter>(config->filter_type);
    cvconfig.sourceFormat = to_cv_format(config->source_format);
    cvconfig.destFormat = to_cv_format(config->dest_format);
    cvconfig.wrap = static_cast<MNN::CV::Wrap>(config->wrap);

    for (int i = 0; i < 4; ++i) {
        cvconfig.mean[i] = config->mean[i];
        cvconfig.normal[i] = config->normal[i];
    }

    auto* processor = MNN::CV::ImageProcess::create(cvconfig);
    return reinterpret_cast<MNNC_ImageProcess*>(processor);
}

void mnnc_image_process_destroy(MNNC_ImageProcess* processor) {
    if (processor) {
        delete reinterpret_cast<MNN::CV::ImageProcess*>(processor);
    }
}

void mnnc_image_process_set_matrix(MNNC_ImageProcess* processor, const float* matrix) {
    if (!processor || !matrix) return;

    MNN::CV::Matrix m;
    m.setAll(matrix[0], matrix[1], matrix[2],
             matrix[3], matrix[4], matrix[5],
             matrix[6], matrix[7], matrix[8]);

    reinterpret_cast<MNN::CV::ImageProcess*>(processor)->setMatrix(m);
}

MNNC_ErrorCode mnnc_image_process_convert(
    MNNC_ImageProcess* processor,
    const uint8_t* src,
    int src_width,
    int src_height,
    int src_stride,
    MNNC_Tensor* dst
) {
    if (!processor || !src || !dst) return MNNC_ERROR_INVALID_ARGUMENT;

    auto* proc = reinterpret_cast<MNN::CV::ImageProcess*>(processor);
    auto* tensor = reinterpret_cast<MNN::Tensor*>(dst);

    int stride = src_stride > 0 ? src_stride : src_width;
    auto code = proc->convert(src, src_width, src_height, stride, tensor);

    return static_cast<MNNC_ErrorCode>(code);
}

/* ============ Utility API ============ */

const char* mnnc_get_version(void) {
    return MNN_VERSION;
}

void mnnc_debug_print_io(MNNC_Interpreter* interpreter, MNNC_Session* session) {
    if (!interpreter || !session) return;
    auto* interp = reinterpret_cast<MNN::Interpreter*>(interpreter);
    auto* sess = reinterpret_cast<MNN::Session*>(session);

    auto inputs = interp->getSessionInputAll(sess);
    for (const auto& pair : inputs) {
        auto* tensor = pair.second;
        const auto& shape = tensor->shape();
        MNN_PRINT("Input '%s': dims=%d, shape=[", pair.first.c_str(), tensor->dimensions());
        for (int i = 0; i < tensor->dimensions(); i++) {
            MNN_PRINT("%d%s", shape[i], i < tensor->dimensions()-1 ? "," : "");
        }
        MNN_PRINT("]\n");
    }

    auto outputs = interp->getSessionOutputAll(sess);
    for (const auto& pair : outputs) {
        auto* tensor = pair.second;
        const auto& shape = tensor->shape();
        MNN_PRINT("Output '%s': dims=%d, shape=[", pair.first.c_str(), tensor->dimensions());
        for (int i = 0; i < tensor->dimensions(); i++) {
            MNN_PRINT("%d%s", shape[i], i < tensor->dimensions()-1 ? "," : "");
        }
        MNN_PRINT("]\n");
    }
}

/* ============ Module API ============ */

MNNC_Module* mnnc_module_load(
    const void* buffer,
    size_t size,
    const char** input_names,
    int input_count,
    const char** output_names,
    int output_count,
    const MNNC_ModuleConfig* config
) {
    if (!buffer || size == 0) return nullptr;

    std::vector<std::string> inputs;
    for (int i = 0; i < input_count; i++) {
        inputs.push_back(input_names[i]);
    }

    std::vector<std::string> outputs;
    for (int i = 0; i < output_count; i++) {
        outputs.push_back(output_names[i]);
    }

    MNN::Express::Module::Config modConfig;
    modConfig.shapeMutable = true;  // Enable dynamic shapes
    modConfig.rearrange = false;

    MNN::Express::Module::BackendInfo backend;
    MNN::BackendConfig backendConfig;
    if (config) {
        backend.type = static_cast<MNNForwardType>(config->forward_type);
        backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(config->precision);
        backendConfig.power = static_cast<MNN::BackendConfig::PowerMode>(config->power);
        backendConfig.memory = static_cast<MNN::BackendConfig::MemoryMode>(config->memory);
        backend.config = &backendConfig;
        modConfig.backend = &backend;
    }

    auto* module = MNN::Express::Module::load(
        inputs, outputs,
        reinterpret_cast<const uint8_t*>(buffer), size,
        &modConfig
    );

    return reinterpret_cast<MNNC_Module*>(module);
}

void mnnc_module_destroy(MNNC_Module* module) {
    if (module) {
        MNN::Express::Module::destroy(reinterpret_cast<MNN::Express::Module*>(module));
    }
}

// Helper to convert Tensor DimensionType to Express Dimensionformat
static MNN::Express::Dimensionformat toFormat(MNN::Tensor::DimensionType type) {
    switch (type) {
        case MNN::Tensor::CAFFE: return MNN::Express::NCHW;
        case MNN::Tensor::TENSORFLOW: return MNN::Express::NHWC;
        case MNN::Tensor::CAFFE_C4: return MNN::Express::NC4HW4;
        default: return MNN::Express::NCHW;
    }
}

MNNC_ErrorCode mnnc_module_forward(
    MNNC_Module* module,
    MNNC_Tensor** inputs,
    int input_count,
    MNNC_Tensor** outputs,
    int output_count
) {
    if (!module || !inputs || input_count <= 0) {
        return MNNC_ERROR_INVALID_ARGUMENT;
    }

    auto* mod = reinterpret_cast<MNN::Express::Module*>(module);

    // Convert input tensors to VARP
    std::vector<MNN::Express::VARP> inputVars;
    for (int i = 0; i < input_count; i++) {
        auto* tensor = reinterpret_cast<MNN::Tensor*>(inputs[i]);
        if (!tensor) {
            return MNNC_ERROR_INVALID_ARGUMENT;
        }

        auto shape = tensor->shape();
        auto format = toFormat(tensor->getDimensionType());
        auto dtype = tensor->getType();

        // Create VARP with shape
        auto var = MNN::Express::_Input(shape, format, dtype);
        auto ptr = var->writeMap<void>();
        if (ptr && tensor->host<void>()) {
            memcpy(ptr, tensor->host<void>(), tensor->size());
        }
        inputVars.push_back(var);
    }

    // Run forward
    auto outputVars = mod->onForward(inputVars);

    if (outputVars.empty()) {
        return MNNC_ERROR_COMPUTE;
    }

    // Copy outputs back
    for (size_t i = 0; i < outputVars.size() && i < (size_t)output_count; i++) {
        auto& var = outputVars[i];
        auto info = var->getInfo();
        if (!info) {
            outputs[i] = nullptr;
            continue;
        }

        // Create output tensor with correct shape
        auto* outTensor = MNN::Tensor::create(info->dim, info->type);
        auto srcPtr = var->readMap<void>();
        if (srcPtr && outTensor->host<void>()) {
            memcpy(outTensor->host<void>(), srcPtr, outTensor->size());
        }
        outputs[i] = reinterpret_cast<MNNC_Tensor*>(outTensor);
    }

    return MNNC_OK;
}
