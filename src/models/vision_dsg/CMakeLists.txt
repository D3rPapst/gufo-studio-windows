# Depth Anything V2, Grounding DINO and SAM for `gufo serve vision`
# (POST /v1/vision/depth, /v1/vision/find, /v1/vision/segment): FP32 on the
# GPU, weights read from the Hugging Face snapshots. CPU side: the
# transformers image processors (Pillow-exact resizing from the ccip
# extension) and the request-independent post-processing.
# Written with AI assistance (Claude Code).

add_library(gufo_vision_dsg_cpu STATIC depth_preprocess.cpp grounding_cpu.cpp
  sam_cpu.cpp)
target_include_directories(gufo_vision_dsg_cpu PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_vision_dsg_cpu PUBLIC cxx_std_20)
target_link_libraries(gufo_vision_dsg_cpu PUBLIC gufo_ccip)

if (ENGINE_ENABLE_HIP)
  add_library(gufo_vision_dsg STATIC EXCLUDE_FROM_ALL
    hip_ops.hip
    depth_anything.hip
    grounding_dino.hip
    sam.hip)
  target_include_directories(gufo_vision_dsg PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_options(gufo_vision_dsg PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
  target_link_libraries(gufo_vision_dsg PUBLIC gufo_vision_dsg_cpu gufo_core
    hip::host roc::rocblas)

  # Oracle probe: runs one model on one image and writes the raw outputs as
  # little-endian float32 files for reference/compare_*.py. Not named gufo-*.
  add_executable(vision-dsg-probe EXCLUDE_FROM_ALL probe_main.cpp)
  set_target_properties(vision-dsg-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(vision-dsg-probe PRIVATE gufo_vision_dsg)
  gufo_stage_hip_runtime(vision-dsg-probe)
endif()
