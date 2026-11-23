# Print-ready meshes ("druckfertig") for gufo serve objekt (TRELLIS.2, TripoSG): voxel closing,
# marching cubes, snap onto the original surface, Taubin, colours, millimetre scale, STL / GLB.
# The TripoSG marching cubes (src/models/triposg/marching_cubes.cpp) lives in this library so
# both the generators and the print step share it without pulling in the model code.
# CPU only, no dependencies. AI-assisted (Claude Code), 2026-11-23.

add_library(gufo_mesh_print STATIC
  print_ready.cpp
  ${PROJECT_SOURCE_DIR}/src/models/triposg/marching_cubes.cpp)
target_include_directories(gufo_mesh_print PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(gufo_mesh_print PUBLIC cxx_std_20)
find_package(Threads)
if (Threads_FOUND)
  target_link_libraries(gufo_mesh_print PUBLIC Threads::Threads)
endif()

# gufo-mesh-print in.glb out-prefix [--height-mm H | --eye-mm E] ...: the print step on a
# GLB file (measurements, comparisons with the Python reference).
add_executable(gufo-mesh-print EXCLUDE_FROM_ALL print_cli.cpp)
set_target_properties(gufo-mesh-print PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_link_libraries(gufo-mesh-print PRIVATE gufo_mesh_print)

if (BUILD_TESTING)
  add_executable(mesh_print_ready_test ${PROJECT_SOURCE_DIR}/tests/core/mesh_print_ready_test.cpp)
  target_link_libraries(mesh_print_ready_test PRIVATE gufo_mesh_print)
  if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(mesh_print_ready_test PRIVATE /UNDEBUG)
  else()
    target_compile_options(mesh_print_ready_test PRIVATE -UNDEBUG)
  endif()
  add_test(NAME mesh_print_ready_test COMMAND mesh_print_ready_test)
  set_tests_properties(mesh_print_ready_test PROPERTIES LABELS "cpu;mesh;objekt" TIMEOUT 300)
endif()
