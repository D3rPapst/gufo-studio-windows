# Real-ESRGAN SRVGGNetCompact upscaling (realesr-animevideov3,
# realesr-general-x4v3) of videos and images; FFmpeg I/O from gufo_rife.
# gufo-studio-windows: a library extension (no upstream capability for
# upscaling yet); `gufo serve upscale` and the "gufo.upscale" post-step of
# video jobs link these targets directly.
# Written with AI assistance (Claude Code).

# `gufo serve upscale --device npu|cpu`: ONNX Runtime is loaded at run time
# (Windows ML on Windows); only its MIT C API header is vendored, so there is
# no link dependency. Off elsewhere until a Linux NPU path is tested.
if(WIN32)
  set(GUFO_ENABLE_ONNXRUNTIME_DEFAULT ON)
else()
  set(GUFO_ENABLE_ONNXRUNTIME_DEFAULT OFF)
endif()
option(GUFO_ENABLE_ONNXRUNTIME
  "Load ONNX Runtime at run time for the NPU / CPU upscaler"
  ${GUFO_ENABLE_ONNXRUNTIME_DEFAULT})

# Host side of the NPU / CPU path (`gufo serve upscale --device npu|cpu`):
# tiling, pixel shuffle, residual and OpenCV-exact INTER_AREA fitting
# (cpu_post), the safetensors header / architecture check of the converted
# weights (srvgg_weights), and the run-time loaded ONNX Runtime sessions
# (ort_upscaler).
# cpu_post is its own library: the HIP upscaler's fit "crop" uses the same
# INTER_AREA port without pulling in ONNX Runtime's loader.
gufo_extension_target(
  NAME gufo_realesrgan_post
  SOURCES
    cpu_post.cpp
)
if(TARGET gufo_realesrgan_post)
  target_include_directories(gufo_realesrgan_post PUBLIC ${PROJECT_SOURCE_DIR})
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(gufo_realesrgan_post PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

gufo_extension_target(
  NAME gufo_realesrgan_cpu
  SOURCES
    srvgg_weights.cpp
)
if(TARGET gufo_realesrgan_cpu)
  target_include_directories(gufo_realesrgan_cpu PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_realesrgan_cpu PUBLIC gufo_realesrgan_post)
  if(GUFO_ENABLE_ONNXRUNTIME)
    target_sources(gufo_realesrgan_cpu PRIVATE
      ${GUFO_EXTENSION_BUILD_ROOT}/ort_upscaler.cpp)
    target_include_directories(gufo_realesrgan_cpu PRIVATE
      ${PROJECT_SOURCE_DIR}/third_party/onnxruntime/include)
    target_compile_definitions(gufo_realesrgan_cpu PUBLIC GUFO_HAVE_ONNXRUNTIME=1)
    if(WIN32)
      target_link_libraries(gufo_realesrgan_cpu PRIVATE ole32)
    else()
      target_link_libraries(gufo_realesrgan_cpu PRIVATE ${CMAKE_DL_LIBS})
    endif()

    # NPU / CPU probe: one frame through a body ONNX, PPM out, optional
    # comparison with a reference PNG (Take One's output). Not named gufo-*.
    add_executable(npu-upscale-probe EXCLUDE_FROM_ALL
      ${GUFO_EXTENSION_BUILD_ROOT}/npu_probe_main.cpp)
    set_target_properties(npu-upscale-probe PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_link_libraries(npu-upscale-probe PRIVATE gufo_realesrgan_cpu
      gufo_core)
  endif()

  if(BUILD_TESTING)
    add_executable(realesrgan_cpu_post_test
      ${PROJECT_SOURCE_DIR}/tests/models/realesrgan/cpu_post_test.cpp)
    target_link_libraries(realesrgan_cpu_post_test PRIVATE gufo_realesrgan_cpu)
    add_test(NAME realesrgan_cpu_post_test COMMAND realesrgan_cpu_post_test)
    set_tests_properties(realesrgan_cpu_post_test PROPERTIES
      LABELS "cpu;models;realesrgan")
    # Tile plan of the HIP upscaler (header only) and the safetensors header /
    # SRVGGNetCompact architecture check of the server's start-up.
    add_executable(realesrgan_tile_plan_test
      ${PROJECT_SOURCE_DIR}/tests/models/realesrgan/tile_plan_test.cpp)
    target_include_directories(realesrgan_tile_plan_test PRIVATE
      ${PROJECT_SOURCE_DIR})
    add_test(NAME realesrgan_tile_plan_test COMMAND realesrgan_tile_plan_test)
    set_tests_properties(realesrgan_tile_plan_test PROPERTIES
      LABELS "cpu;models;realesrgan")
    add_executable(realesrgan_srvgg_weights_test
      ${PROJECT_SOURCE_DIR}/tests/models/realesrgan/srvgg_weights_test.cpp)
    target_link_libraries(realesrgan_srvgg_weights_test PRIVATE
      gufo_realesrgan_cpu)
    add_test(NAME realesrgan_srvgg_weights_test
      COMMAND realesrgan_srvgg_weights_test)
    set_tests_properties(realesrgan_srvgg_weights_test PROPERTIES
      LABELS "cpu;models;realesrgan")
  endif()
endif()

gufo_extension_target(
  NAME gufo_realesrgan
  SOURCES
    upscaler.hip
    upscale_video.cpp
)
if(TARGET gufo_realesrgan)
  set_target_properties(gufo_realesrgan PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_include_directories(gufo_realesrgan PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_realesrgan PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
  target_link_libraries(gufo_realesrgan PUBLIC gufo_rife gufo_realesrgan_post
    hip::host)

  # Standalone driver with the same options as `gufo video-upscale`.
  add_executable(gufo-upscale EXCLUDE_FROM_ALL
    ${PROJECT_SOURCE_DIR}/src/cli/video/upscale_main.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/video/video_upscale.cpp)
  set_target_properties(gufo-upscale PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-upscale PRIVATE gufo_realesrgan)
  gufo_stage_hip_runtime(gufo-upscale)
endif()
