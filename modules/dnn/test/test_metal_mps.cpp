// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Real end-to-end test for the experimental Metal/MPSGraph executor in the NEW dnn engine.
// It loads tiny self-contained ONNX models (embedded below, no opencv_extra needed), runs
// them through ENGINE_NEW once with the Metal path disabled (CPU reference) and once enabled,
// asserts the Metal path actually executed the ops (not a silent CPU fallback), and asserts
// the GPU result matches the CPU reference within tolerance.

#include "test_precomp.hpp"
#include "opencv2/core/metal.hpp"

#ifdef HAVE_METAL

#include "opencv2/core/utils/configuration.private.hpp"

// Embedded ONNX fixtures (conv_onnx / relu_onnx / conv_relu_onnx and their _len), generated
// offline so the test needs no external test data. See the header of test_metal_mps_models.inc
// for how to regenerate them.
#include "test_metal_mps_models.inc"

namespace cv { namespace dnn { namespace metal {
// Control surface implemented in modules/dnn/src/metal/dnn_metal.mm (CV_EXPORTS there).
CV_EXPORTS void setEnabled(bool);
CV_EXPORTS void resetCounters();
CV_EXPORTS int  opsExecuted();
}}} // namespace cv::dnn::metal

namespace opencv_test { namespace {

using namespace cv;
using namespace cv::dnn;

static bool classicEngineForced()
{
    return static_cast<EngineType>(cv::utils::getConfigurationParameterSizeT(
               "OPENCV_FORCE_DNN_ENGINE", ENGINE_AUTO)) == ENGINE_CLASSIC;
}

// Restores the Metal enable flag to OFF when the test scope exits (even on failure).
struct MetalGuard
{
    ~MetalGuard() { cv::dnn::metal::setEnabled(false); }
};

// Runs the model through the NEW engine once with the Metal executor disabled (CPU reference)
// and once enabled, asserts enabling Metal never changes the result (correct GPU compute when an
// op runs, transparent fallback when it declines), and returns how many ops actually ran on the
// GPU. Returns -1 when the test was skipped (classic engine forced).
static int runAndCheckParity(const unsigned char* blob, size_t len,
                             const std::vector<int>& inShape, const char* name)
{
    if (!cv::metal::haveMetal())
        throw SkipTestException("Metal backend is not available on this machine");
    if (classicEngineForced())
    {
        applyTestTag(CV_TEST_TAG_DNN_SKIP_PARSER);
        return -1;
    }

    MetalGuard guard;

    std::vector<uchar> model(blob, blob + len);
    Net net = readNetFromONNX(model, ENGINE_NEW);
    EXPECT_FALSE(net.empty());

    Mat input(inShape, CV_32F);
    RNG rng(0x42);
    rng.fill(input, RNG::UNIFORM, -1.0, 1.0);

    cv::dnn::metal::setEnabled(false);
    cv::dnn::metal::resetCounters();
    net.setInput(input);
    Mat ref = net.forward().clone();

    cv::dnn::metal::setEnabled(true);
    cv::dnn::metal::resetCounters();
    net.setInput(input);
    Mat out = net.forward().clone();
    const int executed = cv::dnn::metal::opsExecuted();

    normAssert(ref, out, name, /*l1*/ 1e-4, /*lInf*/ 1e-3);
    return executed;
}

// ReLU is elementwise, so the new engine does not repack it into a blocked layout: it runs
// end-to-end on the Apple GPU via MPSGraph and must match the CPU result. This is the primary
// Phase 0 proof that a real NN op executes on Metal through the new dnn engine.
TEST(Test_Metal_MPS, relu)
{
    int executed = runAndCheckParity(relu_onnx, relu_onnx_len, {1, 4, 8, 8}, "relu");
    if (executed < 0)
        return; // skipped (classic engine)
    EXPECT_GE(executed, 1) << "ReLU did not execute on the Metal/MPSGraph path";
}

// More layout-neutral unary activations that reach Metal the same way as ReLU. The random
// input is uniform[-1,1], a domain on which all four ops are well defined (Sqrt is omitted
// because it is NaN on negatives).
TEST(Test_Metal_MPS, sigmoid)
{
    int executed = runAndCheckParity(sigmoid_onnx, sigmoid_onnx_len, {1, 4, 8, 8}, "sigmoid");
    if (executed < 0)
        return;
    EXPECT_GE(executed, 1) << "Sigmoid did not execute on the Metal/MPSGraph path";
}

TEST(Test_Metal_MPS, tanh)
{
    int executed = runAndCheckParity(tanh_onnx, tanh_onnx_len, {1, 4, 8, 8}, "tanh");
    if (executed < 0)
        return;
    EXPECT_GE(executed, 1) << "TanH did not execute on the Metal/MPSGraph path";
}

TEST(Test_Metal_MPS, exp)
{
    int executed = runAndCheckParity(exp_onnx, exp_onnx_len, {1, 4, 8, 8}, "exp");
    if (executed < 0)
        return;
    EXPECT_GE(executed, 1) << "Exp did not execute on the Metal/MPSGraph path";
}

TEST(Test_Metal_MPS, abs)
{
    int executed = runAndCheckParity(abs_onnx, abs_onnx_len, {1, 4, 8, 8}, "abs");
    if (executed < 0)
        return;
    EXPECT_GE(executed, 1) << "AbsVal did not execute on the Metal/MPSGraph path";
}

// The new engine repacks conv activations into a blocked layout (useBlockLayout) and fuses
// Conv+activation before execution, so the plain-NCHW Metal conv executor currently declines and
// the op runs on CPU. These tests assert that the decline is SAFE (enabling Metal does not change
// the output). Running conv on Metal requires claiming the op before useBlockLayout (Phase 1).
TEST(Test_Metal_MPS, conv_block_layout_declines)
{
    int executed = runAndCheckParity(conv_onnx, conv_onnx_len, {1, 3, 8, 8}, "conv");
    if (executed < 0)
        return;
    EXPECT_EQ(executed, 0) << "conv unexpectedly ran on Metal - block-layout handling changed";
}

TEST(Test_Metal_MPS, conv_relu_block_layout_declines)
{
    int executed = runAndCheckParity(conv_relu_onnx, conv_relu_onnx_len, {1, 3, 8, 8}, "conv_relu");
    if (executed < 0)
        return;
    EXPECT_EQ(executed, 0) << "fused conv+relu unexpectedly ran on Metal";
}

}} // namespace

#endif // HAVE_METAL
