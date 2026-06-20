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

// Conv needs a different harness than the elementwise ops. Running a conv on Metal requires the
// claim pass (in prepareForInference) to mark it BEFORE the block-layout lowering, and that pass
// runs only once per Net (cached by the 'prepared' flag) and only when Metal is enabled. So we
// cannot toggle Metal on a single already-prepared Net; instead we use two fresh Nets: a
// reference Net that lives entirely with Metal disabled (pure CPU block-layout path), and a test
// Net that has Metal enabled before its first forward (so the claim pass engages). The results
// must match, and the returned counter says how many ops actually ran on the GPU. Returns -1 when
// skipped (classic engine forced).
static int runAndCheckParityTwoNets(const unsigned char* blob, size_t len,
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
    Mat input(inShape, CV_32F);
    RNG rng(0x42);
    rng.fill(input, RNG::UNIFORM, -1.0, 1.0);

    // Reference: Metal disabled for the whole life of this Net (CPU block-layout path).
    cv::dnn::metal::setEnabled(false);
    Net refNet = readNetFromONNX(model, ENGINE_NEW);
    EXPECT_FALSE(refNet.empty());
    refNet.setInput(input);
    Mat ref = refNet.forward().clone();

    // Test: Metal enabled before the first forward, so the claim pass marks supported convs.
    cv::dnn::metal::setEnabled(true);
    cv::dnn::metal::resetCounters();
    Net testNet = readNetFromONNX(model, ENGINE_NEW);
    EXPECT_FALSE(testNet.empty());
    testNet.setInput(input);
    Mat out = testNet.forward().clone();
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

// A supported convolution (explicit padding, constant fp32 OIHW weights) is claimed before the
// block-layout lowering, so it stays in plain NCHW and runs on the Apple GPU via MPSGraph. The
// result must match the CPU reference. This is the Phase 1 headline: a real conv on Metal.
TEST(Test_Metal_MPS, conv_runs_on_metal)
{
    int executed = runAndCheckParityTwoNets(conv_onnx, conv_onnx_len, {1, 3, 8, 8}, "conv");
    if (executed < 0)
        return; // skipped (classic engine)
    EXPECT_GE(executed, 1) << "conv did not execute on the Metal/MPSGraph path";
}

// With the conv claimed, the activation is no longer fused into it (fusion is skipped for claimed
// convs), so both the conv and the following ReLU run as separate ops on Metal: the conv via
// mpsConv, the ReLU via mpsUnary. Hence at least two ops execute on the GPU.
TEST(Test_Metal_MPS, conv_relu_runs_on_metal)
{
    int executed = runAndCheckParityTwoNets(conv_relu_onnx, conv_relu_onnx_len, {1, 3, 8, 8}, "conv_relu");
    if (executed < 0)
        return;
    EXPECT_GE(executed, 2) << "conv+relu did not both execute on the Metal/MPSGraph path";
}

// A conv with auto_pad=SAME_UPPER is outside what the Metal executor handles in this phase, so the
// claim predicate declines it and it keeps the ordinary CPU block-layout path. This pins the
// safety invariant: an unsupported conv is never claimed (executed == 0) and still produces the
// correct result with no crash, even with Metal enabled before the first forward.
TEST(Test_Metal_MPS, conv_declined_falls_back)
{
    int executed = runAndCheckParityTwoNets(conv_declined_onnx, conv_declined_onnx_len,
                                            {1, 3, 8, 8}, "conv_declined");
    if (executed < 0)
        return;
    EXPECT_EQ(executed, 0) << "a SAME_UPPER conv must not run on Metal in this phase";
}

}} // namespace

#endif // HAVE_METAL
