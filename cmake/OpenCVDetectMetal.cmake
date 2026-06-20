set(HAVE_METAL OFF)

if(APPLE)
  find_library(METAL_LIBRARY Metal)
  find_library(FOUNDATION_LIBRARY Foundation)

  if(METAL_LIBRARY AND FOUNDATION_LIBRARY)
    set(HAVE_METAL ON)
    set(METAL_LIBRARIES "${METAL_LIBRARY}" "${FOUNDATION_LIBRARY}")

    # MetalPerformanceShaders / MetalPerformanceShadersGraph provide the NN op primitives used
    # by the experimental dnn Metal executor. They are detected here but kept OUT of
    # METAL_LIBRARIES (core does not need them); the dnn module links them itself when present.
    # On macOS 11+ they always ship alongside Metal.
    find_library(MPS_LIBRARY MetalPerformanceShaders)
    find_library(MPSGRAPH_LIBRARY MetalPerformanceShadersGraph)
  endif()
endif()

mark_as_advanced(METAL_LIBRARY FOUNDATION_LIBRARY MPS_LIBRARY MPSGRAPH_LIBRARY)
