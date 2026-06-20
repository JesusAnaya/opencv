// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_METAL_HPP
#define OPENCV_CORE_METAL_HPP

#include "opencv2/core/cvdef.h"
#include "opencv2/core/mat.hpp"

namespace cv {
/** @addtogroup core_metal
@{
*/
namespace metal {

/** @brief Returns true if the Metal UMat backend is available at runtime.

The result is true only when OpenCV was built with Metal support and a default
Metal device and command queue can be created.
*/
CV_EXPORTS bool haveMetal();

//! @cond INTERNAL
CV_EXPORTS bool threshold(const UMat& src, UMat& dst, double thresh, double maxval, int thresholdType);

// Experimental device-interop accessors. They let out-of-core device backends (for example
// the experimental dnn Metal executor) reach the raw Metal objects behind a Metal-backed
// UMat without pulling in core-private headers. The returned pointers are the underlying
// id<MTLBuffer> / id<MTLDevice> / id<MTLCommandQueue> handed back as void*; they are owned by
// OpenCV and must not be released by the caller. Each returns NULL when Metal is unavailable
// or the UMat is not Metal-backed.
CV_EXPORTS void* getMTLBuffer(const UMat& m);
CV_EXPORTS void* getMTLBufferContents(const UMat& m); // host pointer, or NULL if not host-visible
CV_EXPORTS void* getMTLDevice();
CV_EXPORTS void* getMTLCommandQueue();
//! @endcond

} // namespace metal
/** @} */
} // namespace cv

#endif // OPENCV_CORE_METAL_HPP
