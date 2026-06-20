// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "metal_private.hpp"

#ifdef HAVE_METAL

namespace cv {
namespace metal {

bool haveMetal()
{
    return getMetalContext()->valid();
}

MatAllocator* getMetalAllocator()
{
    CV_SINGLETON_LAZY_INIT(MatAllocator, getMetalAllocator_())
}

// Experimental device-interop accessors (declared in opencv2/core/metal.hpp). Manual
// reference counting is in effect here (no ARC), so the id<-> void* casts are plain C casts
// that do not transfer ownership: the returned handles stay owned by the UMat / MetalContext.
void* getMTLBuffer(const UMat& m)
{
    if (!m.u)
        return NULL;
    MetalBuffer* b = getBuffer(m.u);
    return b ? (void*)b->buffer : NULL;
}

void* getMTLBufferContents(const UMat& m)
{
    return m.u ? (void*)getContents(m.u) : NULL;
}

void* getMTLDevice()
{
    std::shared_ptr<MetalContext> ctx = getMetalContext();
    return (ctx && ctx->valid()) ? (void*)ctx->device() : NULL;
}

void* getMTLCommandQueue()
{
    std::shared_ptr<MetalContext> ctx = getMetalContext();
    return (ctx && ctx->valid()) ? (void*)ctx->queue() : NULL;
}

} // namespace metal
} // namespace cv

#endif // HAVE_METAL
