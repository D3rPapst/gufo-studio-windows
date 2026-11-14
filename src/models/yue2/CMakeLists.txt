# YuE2 (m-a-p/YuE2-3B + YuE2-Vae, songs with vocals): the qwen.tiktoken
# tokenizer, the yue2-native-v1 prompts and phase sampler (CPU), the AR / NAR
# mixture-of-transformers model and the song pipeline (HIP; the Qwen3 stack,
# GEMM, attention and the Oobleck VAE decoder come from ace_step). Weights:
# CC BY-NC 4.0 plus the creator permission in MODEL_LICENSE (docs/models/
# yue2.md). gufo-studio-windows only; added from src/cli/serve/media/
# media.cmake. Written with AI assistance (Claude Code).
add_library(gufo_yue2 STATIC EXCLUDE_FROM_ALL tokenizer.cpp protocol.cpp)
target_include_directories(gufo_yue2 PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_yue2 PUBLIC gufo_ace_step ICU::i18n)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_yue2 PRIVATE model.hip pipeline.cpp)
  target_compile_options(gufo_yue2 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffp-contract=off>)

  add_executable(gufo-yue2-check EXCLUDE_FROM_ALL check_main.cpp)
  set_target_properties(gufo-yue2-check PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-yue2-check PRIVATE gufo_yue2)
  gufo_stage_hip_runtime(gufo-yue2-check)

  add_executable(gufo-yue2 EXCLUDE_FROM_ALL song_main.cpp)
  set_target_properties(gufo-yue2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  target_link_libraries(gufo-yue2 PRIVATE gufo_yue2)
  gufo_stage_hip_runtime(gufo-yue2)
endif()

if (BUILD_TESTING)
  # Prompts, chunking and the phase sampler; with GUFO_YUE2_TIKTOKEN (the
  # model's qwen.tiktoken) and GUFO_YUE2_ORACLE (tokens.json of the oracle)
  # also the token ids against the reference tokenizer.
  add_executable(yue2_protocol_test
    ${PROJECT_SOURCE_DIR}/tests/models/yue2/protocol_test.cpp)
  target_link_libraries(yue2_protocol_test PRIVATE gufo_yue2)
  add_test(NAME yue2_protocol_test COMMAND yue2_protocol_test)
  set_tests_properties(yue2_protocol_test PROPERTIES
    LABELS "cpu;models;yue2" TIMEOUT 60)
endif()
