/**
 * C-compatible wrapper for MNN library
 * This header provides a C interface to the MNN C++ library
 */

#ifndef MNN_C_WRAPPER_H
#define MNN_C_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations */
typedef struct MNNC_Interpreter MNNC_Interpreter;
typedef struct MNNC_Session MNNC_Session;
typedef struct MNNC_Tensor MNNC_Tensor;
typedef struct MNNC_ImageProcess MNNC_ImageProcess;
typedef struct MNNC_Module MNNC_Module;

/* Error codes */
typedef enum {
    MNNC_OK = 0,
    MNNC_ERROR_INVALID_ARGUMENT = 1,
    MNNC_ERROR_OUT_OF_MEMORY = 2,
    MNNC_ERROR_NOT_SUPPORT = 3,
    MNNC_ERROR_COMPUTE = 4,
    MNNC_ERROR_TENSOR_NOT_SUPPORT = 5,
    MNNC_ERROR_TENSOR_NEED_DIVIDE = 6,
    MNNC_ERROR_UNKNOWN = 100,
} MNNC_ErrorCode;

/* Forward types */
typedef enum {
    MNNC_FORWARD_CPU = 0,
    MNNC_FORWARD_METAL = 1,
    MNNC_FORWARD_CUDA = 2,
    MNNC_FORWARD_OPENCL = 3,
    MNNC_FORWARD_AUTO = 4,
    MNNC_FORWARD_OPENGL = 6,
    MNNC_FORWARD_VULKAN = 7,
    MNNC_FORWARD_NN = 8,
    MNNC_FORWARD_USER_0 = 9,
    MNNC_FORWARD_USER_1 = 10,
    MNNC_FORWARD_USER_2 = 11,
    MNNC_FORWARD_USER_3 = 12,
    MNNC_FORWARD_ALL = 13,
} MNNC_ForwardType;

/* Data types for tensors */
typedef enum {
    MNNC_DTYPE_FLOAT = 0,
    MNNC_DTYPE_INT32 = 1,
    MNNC_DTYPE_UINT8 = 2,
    MNNC_DTYPE_INT8 = 3,
    MNNC_DTYPE_FLOAT16 = 4,
    MNNC_DTYPE_INT64 = 5,
    MNNC_DTYPE_DOUBLE = 6,
    MNNC_DTYPE_UINT32 = 7,
    MNNC_DTYPE_INT16 = 8,
    MNNC_DTYPE_UINT16 = 9,
    MNNC_DTYPE_BFLOAT16 = 10,
} MNNC_DataType;

/* Dimension format */
typedef enum {
    MNNC_NCHW = 0,
    MNNC_NHWC = 1,
    MNNC_NC4HW4 = 2,
    MNNC_NHWC4 = 3,
} MNNC_DimensionType;

/* Session mode */
typedef enum {
    MNNC_SESSION_RELEASE = 0,
    MNNC_SESSION_KEEP_NETWORK = 1,
} MNNC_SessionMode;

/* Schedule config */
typedef struct {
    MNNC_ForwardType forward_type;
    int num_threads;
} MNNC_ScheduleConfig;

/* Backend config */
typedef struct {
    int precision;  /* 0: normal, 1: high, 2: low, 3: low with BF16 */
    int power;      /* 0: normal, 1: high, 2: low */
    int memory;     /* 0: normal, 1: high, 2: low */
} MNNC_BackendConfig;

/* Image formats */
typedef enum {
    MNNC_IMAGE_RGBA = 0,
    MNNC_IMAGE_RGB = 1,
    MNNC_IMAGE_BGR = 2,
    MNNC_IMAGE_GRAY = 3,
    MNNC_IMAGE_BGRA = 4,
    MNNC_IMAGE_YUV_NV21 = 11,
    MNNC_IMAGE_YUV_NV12 = 12,
    MNNC_IMAGE_YUV_I420 = 13,
} MNNC_ImageFormat;

/* Image process config */
typedef struct {
    int filter_type;     /* 0: nearest, 1: bilinear, 2: bicubic */
    MNNC_ImageFormat source_format;
    MNNC_ImageFormat dest_format;
    float mean[4];       /* Mean values for normalization */
    float normal[4];     /* Normalization scale values */
    int wrap;            /* Wrap mode */
} MNNC_ImageProcessConfig;

/* ============ Interpreter API ============ */

/**
 * Create an interpreter from a model file
 * @param model_path Path to the .mnn model file
 * @return Interpreter handle or NULL on failure
 */
MNNC_Interpreter* mnnc_interpreter_create_from_file(const char* model_path);

/**
 * Create an interpreter from a memory buffer
 * @param buffer Pointer to model data
 * @param size Size of model data in bytes
 * @return Interpreter handle or NULL on failure
 */
MNNC_Interpreter* mnnc_interpreter_create_from_buffer(const void* buffer, size_t size);

/**
 * Destroy an interpreter and free its resources
 * @param interpreter Interpreter handle
 */
void mnnc_interpreter_destroy(MNNC_Interpreter* interpreter);

/**
 * Set the session mode
 * @param interpreter Interpreter handle
 * @param mode Session mode
 */
void mnnc_interpreter_set_session_mode(MNNC_Interpreter* interpreter, MNNC_SessionMode mode);

/**
 * Set cache file path for compiled kernels
 * @param interpreter Interpreter handle
 * @param cache_path Path to cache file
 */
void mnnc_interpreter_set_cache_file(MNNC_Interpreter* interpreter, const char* cache_path);

/* ============ Session API ============ */

/**
 * Create a session with the given configuration
 * @param interpreter Interpreter handle
 * @param config Schedule configuration
 * @return Session handle or NULL on failure
 */
MNNC_Session* mnnc_session_create(MNNC_Interpreter* interpreter, const MNNC_ScheduleConfig* config);

/**
 * Create a session with backend config
 * @param interpreter Interpreter handle
 * @param config Schedule configuration
 * @param backend Backend configuration
 * @return Session handle or NULL on failure
 */
MNNC_Session* mnnc_session_create_with_backend(
    MNNC_Interpreter* interpreter,
    const MNNC_ScheduleConfig* config,
    const MNNC_BackendConfig* backend
);

/**
 * Release a session
 * @param interpreter Interpreter handle
 * @param session Session handle
 */
void mnnc_session_release(MNNC_Interpreter* interpreter, MNNC_Session* session);

/**
 * Resize the session (call after changing input dimensions)
 * @param interpreter Interpreter handle
 * @param session Session handle
 * @return Error code
 */
MNNC_ErrorCode mnnc_session_resize(MNNC_Interpreter* interpreter, MNNC_Session* session);

/**
 * Resize the session with explicit reallocation control
 * @param interpreter Interpreter handle
 * @param session Session handle
 * @param need_realloc If non-zero, force memory reallocation (required for MemoryMode::Low with dynamic shapes)
 * @return Error code
 */
MNNC_ErrorCode mnnc_session_resize_ex(MNNC_Interpreter* interpreter, MNNC_Session* session, int need_realloc);

/**
 * Run inference on the session
 * @param interpreter Interpreter handle
 * @param session Session handle
 * @return Error code
 */
MNNC_ErrorCode mnnc_session_run(MNNC_Interpreter* interpreter, MNNC_Session* session);

/**
 * Resize an input tensor to a new shape
 * Must call mnnc_session_resize after resizing tensors and before running inference.
 * @param interpreter Interpreter handle
 * @param tensor Tensor handle (must be a session input tensor)
 * @param dims Array of new dimensions
 * @param dim_count Number of dimensions
 */
void mnnc_interpreter_resize_tensor(
    MNNC_Interpreter* interpreter,
    MNNC_Tensor* tensor,
    const int* dims,
    int dim_count
);

/* ============ Tensor API ============ */

/**
 * Get an input tensor by name
 * @param interpreter Interpreter handle
 * @param session Session handle
 * @param name Tensor name (NULL for default input)
 * @return Tensor handle or NULL if not found
 */
MNNC_Tensor* mnnc_session_get_input(MNNC_Interpreter* interpreter, MNNC_Session* session, const char* name);

/**
 * Get an output tensor by name
 * @param interpreter Interpreter handle
 * @param session Session handle
 * @param name Tensor name (NULL for default output)
 * @return Tensor handle or NULL if not found
 */
MNNC_Tensor* mnnc_session_get_output(MNNC_Interpreter* interpreter, MNNC_Session* session, const char* name);

/**
 * Create a tensor with the given shape
 * @param dims Array of dimensions
 * @param dim_count Number of dimensions
 * @param dtype Data type
 * @param dim_type Dimension format
 * @return Tensor handle or NULL on failure
 */
MNNC_Tensor* mnnc_tensor_create(const int* dims, int dim_count, MNNC_DataType dtype, MNNC_DimensionType dim_type);

/**
 * Create a tensor that wraps existing host data
 * @param dims Array of dimensions
 * @param dim_count Number of dimensions
 * @param dtype Data type
 * @param data Pointer to host data
 * @param dim_type Dimension format
 * @return Tensor handle or NULL on failure
 */
MNNC_Tensor* mnnc_tensor_create_with_data(
    const int* dims,
    int dim_count,
    MNNC_DataType dtype,
    void* data,
    MNNC_DimensionType dim_type
);

/**
 * Destroy a tensor created by mnnc_tensor_create
 * NOTE: Do not call this on tensors returned by mnnc_session_get_input/output
 * @param tensor Tensor handle
 */
void mnnc_tensor_destroy(MNNC_Tensor* tensor);

/**
 * Get the number of dimensions
 * @param tensor Tensor handle
 * @return Number of dimensions
 */
int mnnc_tensor_get_dimensions(const MNNC_Tensor* tensor);

/**
 * Get the shape of the tensor
 * @param tensor Tensor handle
 * @param dims Output array for dimensions (must be pre-allocated)
 * @param max_dims Maximum number of dimensions to copy
 * @return Actual number of dimensions
 */
int mnnc_tensor_get_shape(const MNNC_Tensor* tensor, int* dims, int max_dims);

/**
 * Get the total element count
 * @param tensor Tensor handle
 * @return Total number of elements
 */
int mnnc_tensor_element_count(const MNNC_Tensor* tensor);

/**
 * Get the data type
 * @param tensor Tensor handle
 * @return Data type
 */
MNNC_DataType mnnc_tensor_get_dtype(const MNNC_Tensor* tensor);

/**
 * Get the dimension type (layout format)
 * @param tensor Tensor handle
 * @return Dimension type (NCHW, NHWC, NC4HW4, etc.)
 */
MNNC_DimensionType mnnc_tensor_get_dimension_type(const MNNC_Tensor* tensor);

/**
 * Create a host tensor with matching shape/type from a device tensor
 * This is the equivalent of createHostTensorFromDevice in MNN
 * @param device_tensor The device tensor to match
 * @param copy_data If true, copy data from device to the new host tensor
 * @return New host tensor or NULL on failure
 */
MNNC_Tensor* mnnc_tensor_create_host_from_device(const MNNC_Tensor* device_tensor, int copy_data);

/**
 * Get raw pointer to host data
 * @param tensor Tensor handle
 * @return Pointer to data or NULL
 */
void* mnnc_tensor_get_host(MNNC_Tensor* tensor);

/**
 * Get raw pointer to host data (const version)
 * @param tensor Tensor handle
 * @return Pointer to data or NULL
 */
const void* mnnc_tensor_get_host_const(const MNNC_Tensor* tensor);

/**
 * Copy data from a host tensor to a device tensor (device.copyFromHostTensor(host))
 * @param device Destination device tensor
 * @param host Source host tensor
 * @return 1 on success, 0 on failure
 */
int mnnc_tensor_copy_from_host_tensor(MNNC_Tensor* device, const MNNC_Tensor* host);

/**
 * Copy data from a device tensor to a host tensor (device.copyToHostTensor(host))
 * @param device Source device tensor
 * @param host Destination host tensor
 * @return 1 on success, 0 on failure
 */
int mnnc_tensor_copy_to_host_tensor(const MNNC_Tensor* device, MNNC_Tensor* host);

/**
 * Copy data from host memory to tensor
 * @param tensor Destination tensor
 * @param data Source data
 * @param size Size in bytes
 * @return 1 on success, 0 on failure
 */
int mnnc_tensor_copy_from_host(MNNC_Tensor* tensor, const void* data, size_t size);

/**
 * Copy data from tensor to host memory
 * @param tensor Source tensor
 * @param data Destination buffer
 * @param size Size in bytes
 * @return 1 on success, 0 on failure
 */
int mnnc_tensor_copy_to_host(const MNNC_Tensor* tensor, void* data, size_t size);

/* ============ Image Processing API ============ */

/**
 * Create an image processor
 * @param config Image process configuration
 * @return Image processor handle or NULL on failure
 */
MNNC_ImageProcess* mnnc_image_process_create(const MNNC_ImageProcessConfig* config);

/**
 * Destroy an image processor
 * @param processor Image processor handle
 */
void mnnc_image_process_destroy(MNNC_ImageProcess* processor);

/**
 * Set the transformation matrix
 * @param processor Image processor handle
 * @param matrix 3x3 transformation matrix (row-major, 9 floats)
 */
void mnnc_image_process_set_matrix(MNNC_ImageProcess* processor, const float* matrix);

/**
 * Convert image data to a tensor
 * @param processor Image processor handle
 * @param src Source image data
 * @param src_width Source image width
 * @param src_height Source image height
 * @param src_stride Source image stride (bytes per row, 0 for tight packing)
 * @param dst Destination tensor
 * @return Error code
 */
MNNC_ErrorCode mnnc_image_process_convert(
    MNNC_ImageProcess* processor,
    const uint8_t* src,
    int src_width,
    int src_height,
    int src_stride,
    MNNC_Tensor* dst
);

/* ============ Utility API ============ */

/**
 * Get MNN version string
 * @return Version string
 */
const char* mnnc_get_version(void);

/**
 * Debug: print all session inputs and outputs with their shapes
 * @param interpreter Interpreter handle
 * @param session Session handle
 */
void mnnc_debug_print_io(MNNC_Interpreter* interpreter, MNNC_Session* session);

/* ============ Module API (better dynamic shape support) ============ */

/**
 * Module config for loading
 */
typedef struct {
    int shape_mutable;    /* 1 for dynamic shapes, 0 for static */
    MNNC_ForwardType forward_type;
    int num_threads;
    int precision;  /* 0: normal, 1: high, 2: low, 3: low with BF16 */
    int power;      /* 0: normal, 1: high, 2: low */
    int memory;     /* 0: normal, 1: high, 2: low */
} MNNC_ModuleConfig;

/**
 * Load a module from buffer with input/output names
 * @param buffer Model data
 * @param size Model size
 * @param input_names Array of input names
 * @param input_count Number of inputs
 * @param output_names Array of output names
 * @param output_count Number of outputs
 * @param config Module config (can be NULL for defaults)
 * @return Module handle or NULL on failure
 */
MNNC_Module* mnnc_module_load(
    const void* buffer,
    size_t size,
    const char** input_names,
    int input_count,
    const char** output_names,
    int output_count,
    const MNNC_ModuleConfig* config
);

/**
 * Destroy a module
 * @param module Module handle
 */
void mnnc_module_destroy(MNNC_Module* module);

/**
 * Run module inference
 * @param module Module handle
 * @param inputs Array of input tensors (VARP wrapped)
 * @param input_count Number of inputs
 * @param outputs Output array to fill (pre-allocated)
 * @param output_count Number of outputs expected
 * @return Error code
 */
MNNC_ErrorCode mnnc_module_forward(
    MNNC_Module* module,
    MNNC_Tensor** inputs,
    int input_count,
    MNNC_Tensor** outputs,
    int output_count
);

#ifdef __cplusplus
}
#endif

#endif /* MNN_C_WRAPPER_H */
