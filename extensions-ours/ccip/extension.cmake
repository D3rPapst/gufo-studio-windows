# CCIP character fingerprints (deepghs CAFormer) + anime face detector
# (YOLOv8n) as a gufo library extension: native CPU fp32 runtime, no Python
# or ONNX at run time. Callers (router /v1/figuren, CLI) link gufo_ccip and
# pass decoded RGB8 pixels. Written with AI assistance (Claude Code).
#
# Headers are included as "ccip/...": the include root is the folder that
# holds the extension folders.

cmake_path(GET GUFO_EXTENSION_BUILD_ROOT PARENT_PATH _ccip_include_root)

find_package(OpenMP)
function(ccip_use_openmp target)
  if(NOT TARGET ${target} OR NOT OpenMP_CXX_FOUND)
    return()
  endif()
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/clang:-fopenmp>)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/openmp>)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>)
  endif()
  target_link_libraries(${target} PRIVATE ${OpenMP_CXX_LIBRARIES})
endfunction()

gufo_extension_target(
  NAME gufo_ccip
  SOURCES
    tensors.cpp
    image_ops.cpp
    cpu/kernels.cpp
    cpu/caformer.cpp
    cpu/yolo.cpp
    cpu/swinv2.cpp
    cpu/ocr_net.cpp
    ocr.cpp
    ocr_post.cpp
    cpu/nn.cpp
    cpu/rtdetr.cpp
    cpu/vitpose.cpp
    cpu/dwpose.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    Threads::Threads
)

if(TARGET gufo_ccip)
  target_include_directories(gufo_ccip PUBLIC
    "$<BUILD_INTERFACE:${_ccip_include_root}>"
  )
  ccip_use_openmp(gufo_ccip)
  # kernels.cpp and the WD14 tagger's packed GEMM (cpu/swinv2.cpp) use AVX2/FMA.
  set(_ccip_simd "${GUFO_EXTENSION_BUILD_ROOT}/cpu/kernels.cpp" "${GUFO_EXTENSION_BUILD_ROOT}/cpu/swinv2.cpp")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      set_source_files_properties(${_ccip_simd} PROPERTIES COMPILE_OPTIONS "/clang:-mavx2;/clang:-mfma")
    else()
      set_source_files_properties(${_ccip_simd} PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
  elseif(MSVC)
    set_source_files_properties(${_ccip_simd} PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
  # The OCR box geometry repeats OpenCV's and Clipper's float/double steps
  # exactly (imgutils parity): no FMA contraction there.
  set(_ccip_ocr_post "${GUFO_EXTENSION_BUILD_ROOT}/ocr_post.cpp")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
      set_source_files_properties("${_ccip_ocr_post}" PROPERTIES COMPILE_OPTIONS "/clang:-ffp-contract=off")
    else()
      set_source_files_properties("${_ccip_ocr_post}" PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    endif()
  elseif(MSVC)
    set_source_files_properties("${_ccip_ocr_post}" PROPERTIES COMPILE_OPTIONS "/fp:precise")
  endif()
endif()

# Every stage against onnxruntime (reference/make_oracle.py). Needs
# GUFO_CCIP_MODEL (ccip_caformer.*), GUFO_CCIP_FACE (face_yolov8n.*) and
# GUFO_CCIP_ORACLE (manifest.json + *.safetensors); skips (77) otherwise.
gufo_extension_target(
  NAME ccip_oracle_test
  SOURCES
    tests/oracle_test.cpp
  LINK_LIBRARIES
    gufo_ccip
    gufo_ext_runtime
)

# /v1/vision/detect detectors against dghs-imgutils (reference/make_vision_oracle.py).
# Needs GUFO_VISION_DIR (the four converted default detectors) and
# GUFO_VISION_ORACLE (manifest.json + vision_*.safetensors); skips (77) otherwise.
gufo_extension_target(
  NAME vision_oracle_test
  SOURCES
    tests/vision_oracle_test.cpp
  LINK_LIBRARIES
    gufo_ccip
    gufo_ext_runtime
)

# /v1/vision/detect "wd14_tags" against dghs-imgutils (reference/make_wd14_oracle.py).
# Needs GUFO_VISION_DIR (wd-swinv2-tagger-v3.* from reference/convert_wd14.py)
# and GUFO_WD14_ORACLE (manifest.json + wd14_*.safetensors); skips (77) otherwise.
gufo_extension_target(
  NAME wd14_oracle_test
  SOURCES
    tests/wd14_oracle_test.cpp
  LINK_LIBRARIES
    gufo_ccip
    gufo_ext_runtime
)

if(TARGET wd14_oracle_test)
  ccip_use_openmp(wd14_oracle_test)
  add_test(NAME ccip.wd14_oracle COMMAND wd14_oracle_test)
  set_tests_properties(ccip.wd14_oracle PROPERTIES
    LABELS "ccip;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 1200)
endif()

# /v1/vision/detect "ocr" against dghs-imgutils (reference/make_ocr_oracle.py).
# Needs GUFO_OCR_DIR (ch_PP-OCRv4_det/rec from reference/convert_ocr.py) and
# GUFO_OCR_ORACLE (manifest.json + ocr_*.safetensors); skips (77) otherwise.
gufo_extension_target(
  NAME ocr_oracle_test
  SOURCES
    tests/ocr_oracle_test.cpp
  LINK_LIBRARIES
    gufo_ccip
    gufo_ext_runtime
)

if(TARGET ocr_oracle_test)
  ccip_use_openmp(ocr_oracle_test)
  add_test(NAME ccip.ocr_oracle COMMAND ocr_oracle_test)
  set_tests_properties(ccip.ocr_oracle PROPERTIES
    LABELS "ccip;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 1200)
endif()

if(TARGET vision_oracle_test)
  ccip_use_openmp(vision_oracle_test)
  add_test(NAME ccip.vision_oracle COMMAND vision_oracle_test)
  set_tests_properties(ccip.vision_oracle PROPERTIES
    LABELS "ccip;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 1200)
endif()

# "pose" (RT-DETR + ViTPose) against the transformers reference
# (reference/make_pose_oracle.py). Needs GUFO_VISION_DIR (converted with
# reference/convert_pose.py) and GUFO_POSE_ORACLE; skips (77) otherwise.
gufo_extension_target(
  NAME pose_oracle_test
  SOURCES
    tests/pose_oracle_test.cpp
  LINK_LIBRARIES
    gufo_ccip
    gufo_ext_runtime
)

if(TARGET pose_oracle_test)
  ccip_use_openmp(pose_oracle_test)
  add_test(NAME ccip.pose_oracle COMMAND pose_oracle_test)
  set_tests_properties(ccip.pose_oracle PROPERTIES
    LABELS "ccip;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 1800)
endif()

if(TARGET ccip_oracle_test)
  ccip_use_openmp(ccip_oracle_test)
  add_test(NAME ccip.oracle COMMAND ccip_oracle_test)
  set_tests_properties(ccip.oracle PROPERTIES
    LABELS "ccip;cpu;quality;external-model"
    SKIP_RETURN_CODE 77
    TIMEOUT 1200)
endif()
