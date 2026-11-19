# Qwen3-Omni-30B-A3B-Captioner: audio in, caption out. GGUF loading, the F32
# CPU reference and the HIP runtime; gufo-omni is the driver for all of it.
# The audio tower reuses Qwen3-ASR's kernels (extension target
# gufo_qwen3_asr_hip) and the thinker the vendored llama.cpp K-quant matmuls
# of DeepSeek V4 Flash (gufo_deepseek_v4_flash_mmq), both linked read-only.
# gufo-studio-windows: a library extension behind `gufo serve omni` (no
# upstream capability takes audio into text generation yet).
# Written with AI assistance (Claude Code).
#
# GGUF loading and the F32 reference are plain C++: built everywhere, so the
# CPU model test runs in the CPU test build too.
gufo_extension_target(
  NAME gufo_qwen3_omni_model
  SOURCES
    model.cpp
    parallel_read.cpp
    reference.cpp
)
if(TARGET gufo_qwen3_omni_model)
  target_include_directories(gufo_qwen3_omni_model PUBLIC ${PROJECT_SOURCE_DIR})
  target_link_libraries(gufo_qwen3_omni_model PUBLIC gufo_core Threads::Threads)

  if(BUILD_TESTING)
    # CPU only: reads weight bytes from a GGUF fixture file.
    add_executable(qwen3_omni_model_test
      ${PROJECT_SOURCE_DIR}/tests/models/qwen3_omni/model_test.cpp)
    target_link_libraries(qwen3_omni_model_test PRIVATE gufo_qwen3_omni_model)
    add_test(NAME qwen3_omni_model_test COMMAND qwen3_omni_model_test)
    set_tests_properties(qwen3_omni_model_test PROPERTIES LABELS "cpu;models;omni")
  endif()
endif()

# HIP only; manifest dependencies build qwen3_asr and deepseek_v4_flash first.
gufo_extension_target(
  NAME gufo_qwen3_omni
  SOURCES
    captioner.cpp
    hip/audio_encoder.cpp
    hip/thinker.cpp
    hip/thinker_kernels.hip
)

if(TARGET gufo_qwen3_omni)
  set_target_properties(gufo_qwen3_omni PROPERTIES EXCLUDE_FROM_ALL TRUE)
  set_source_files_properties(
    ${GUFO_EXTENSION_BUILD_ROOT}/hip/thinker_kernels.hip
    PROPERTIES LANGUAGE HIP)
  target_include_directories(gufo_qwen3_omni PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_definitions(gufo_qwen3_omni PUBLIC ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_qwen3_omni PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_qwen3_omni PUBLIC
    gufo_qwen3_omni_model gufo_core gufo_qwen_text gufo_qwen3_asr_hip
    gufo_deepseek_v4_flash_mmq hip::host roc::hipblas Threads::Threads)

  add_executable(gufo-omni EXCLUDE_FROM_ALL
    ${GUFO_EXTENSION_BUILD_ROOT}/main.cpp
    ${GUFO_EXTENSION_BUILD_ROOT}/ds4_hooks.cpp)
  # Next to gufo.exe, where build.ps1 stages the ROCm and vcpkg DLLs.
  set_target_properties(gufo-omni PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_compile_options(gufo-omni PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O2>)
  target_link_libraries(gufo-omni PRIVATE gufo_qwen3_omni)
endif()
