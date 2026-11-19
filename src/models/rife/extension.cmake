# RIFE 4.7 video frame interpolation (rife47 / rife49 IFNet) with FFmpeg I/O.
# gufo-studio-windows: a library extension (no upstream capability for frame
# interpolation yet); `gufo serve video --rife-model` and the realesrgan /
# mmaudio extensions link gufo_rife directly.
# Written with AI assistance (Claude Code).

if(WIN32)
  set(gufo_rife_video_io video_io_windows.cpp)
else()
  set(gufo_rife_video_io video_io_posix.cpp)
endif()

gufo_extension_target(
  NAME gufo_rife
  SOURCES
    interpolate.cpp
    ${gufo_rife_video_io}
)

if(TARGET gufo_rife)
  set_target_properties(gufo_rife PROPERTIES EXCLUDE_FROM_ALL TRUE)
  target_include_directories(gufo_rife PUBLIC ${PROJECT_SOURCE_DIR})
  target_compile_definitions(gufo_rife PRIVATE
    GUFO_FFMPEG_EXECUTABLE="${GUFO_FFMPEG_EXECUTABLE}"
    GUFO_FFPROBE_EXECUTABLE="${GUFO_FFPROBE_EXECUTABLE}")

  if(ENGINE_ENABLE_HIP)
    target_sources(gufo_rife PRIVATE ${GUFO_EXTENSION_BUILD_ROOT}/ifnet.hip)
    target_compile_options(gufo_rife PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3>)
    target_link_libraries(gufo_rife PUBLIC hip::host)

    # Standalone driver with the same options as `gufo video-interpolate`.
    add_executable(gufo-rife EXCLUDE_FROM_ALL
      ${PROJECT_SOURCE_DIR}/src/cli/video/rife_main.cpp
      ${PROJECT_SOURCE_DIR}/src/cli/video/video_interpolate.cpp)
    set_target_properties(gufo-rife PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
    target_link_libraries(gufo-rife PRIVATE gufo_rife)
    gufo_stage_hip_runtime(gufo-rife)
  endif()
endif()
