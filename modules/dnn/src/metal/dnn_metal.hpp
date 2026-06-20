// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Experimental Metal + MetalPerformanceShadersGraph (MPSGraph) executor for the NEW dnn
// engine. It is a per-op device executor dispatched from the single execution seam in
// net_impl2.cpp (forwardGraph), keyed on Layer::type. It is OFF by default and falls back
// transparently to the CPU layer->forward() for any op or configuration it does not handle.
//
// This is a proof-of-concept: tensors are bridged to Metal UMats per op (host<->device
// copies). Keeping tensors device-resident across ops is a planned follow-up.

#ifndef OPENCV_DNN_METAL_DNN_METAL_HPP
#define OPENCV_DNN_METAL_DNN_METAL_HPP

#ifdef HAVE_METAL

#include "opencv2/core.hpp"
#include "opencv2/dnn/dnn.hpp"

namespace cv {
namespace dnn {
namespace metal {

// True when OpenCV was built with Metal and a Metal device is available at runtime.
// (CV_EXPORTS on the control surface so the separate test binary can drive it.)
CV_EXPORTS bool dnnMetalAvailable();

// Master enable for the experimental path. Default comes from the OPENCV_DNN_METAL
// environment variable (read once); tests flip it programmatically.
CV_EXPORTS void setEnabled(bool enabled);
CV_EXPORTS bool isEnabled();

// Instrumentation so a test can prove the Metal path actually executed an op rather than
// silently falling back to the CPU.
CV_EXPORTS void resetCounters();
CV_EXPORTS int  opsExecuted();      // total ops run on the GPU since the last reset
CV_EXPORTS int  opsDeclined();      // ops seen but declined (-> CPU fallback)

// The dispatch seam. Returns true if the op was fully executed on the GPU (outputs filled);
// false means the caller must run the normal CPU layer->forward(). Never throws across the
// boundary - any failure returns false.
bool tryForward(const Ptr<Layer>& layer,
                std::vector<Mat>& inputs,
                std::vector<Mat>& outputs,
                std::vector<Mat>& internals);

// Op bodies (implemented in mps_ops.mm). Each returns true on success, false to decline.
bool mpsConv(const Ptr<Layer>& layer, std::vector<Mat>& inputs, std::vector<Mat>& outputs);
// Unary elementwise activations (ReLU, Sigmoid, TanH, Exp, AbsVal), dispatched by layer->type.
bool mpsUnary(const Ptr<Layer>& layer, std::vector<Mat>& inputs, std::vector<Mat>& outputs);

// Single source of truth for "can the Metal conv executor run this Conv2?". It encodes EXACTLY
// the constraints mpsConv enforces, expressed over statically-known facts (dtype, weight rank,
// const-ness, and the spatial config). The claim pass (metal_claim.cpp) calls it before the CPU
// lowering passes; mpsConv calls it again at run time. Because both use this one function, a
// claimed conv is guaranteed to be accepted at the seam, so it never falls through to the CPU
// Conv2::forward (which asserts a block layout the claimed conv no longer has).
// autoPad is the Conv2Layer::AutoPadding value as an int (0 == AUTO_PAD_NONE).
bool metalConvSupported(int actType,
                        int weightType, int weightDims, bool weightConst,
                        bool hasBias, int biasType, bool biasConst,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& pads,
                        int autoPad);

// Verbose logging gate (OPENCV_DNN_METAL_VERBOSE), shared by the op bodies.
bool verbose();

} // namespace metal
} // namespace dnn
} // namespace cv

#endif // HAVE_METAL
#endif // OPENCV_DNN_METAL_DNN_METAL_HPP
