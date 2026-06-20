// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// MPSGraph op bodies for the experimental dnn Metal executor. Manual reference counting
// (no ARC), mirroring the core Metal layer. Each op bridges the host Mats to Metal-backed
// UMats (shared storage on unified memory), runs an MPSGraph, and copies the result back.
// On any unsupported configuration the op returns false so the caller runs the CPU path.

#include "../precomp.hpp"
#include "dnn_metal.hpp"

#ifdef HAVE_METAL

#include "opencv2/core/metal.hpp"
#include "opencv2/core/utils/logger.hpp"
#include "opencv2/dnn/all_layers.hpp"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace cv {
namespace dnn {
namespace metal {

namespace {

// A tensor materialized as a Metal-backed UMat (the device-resident object MPS operates on).
struct MetalTensor
{
    UMat umat;
    id<MTLBuffer> buffer = nil;
    void* contents = NULL;
};

bool makeTensor(MetalTensor& t, int dims, const int* sizes)
{
    // Force SHARED storage: this machine's iGPU is not truly unified, so the default (MANAGED)
    // buffer would need an explicit didModifyRange flush after a CPU write before the GPU sees
    // it. SHARED makes [buffer contents] genuine shared memory, so the memcpy bridge is coherent
    // in both directions without manual synchronization.
    t.umat = UMat(dims, sizes, CV_32F, USAGE_ALLOCATE_SHARED_MEMORY);
    t.buffer = (id<MTLBuffer>)cv::metal::getMTLBuffer(t.umat);
    t.contents = cv::metal::getMTLBufferContents(t.umat);
    return t.buffer != nil && t.contents != NULL;
}

// Upload a host Mat's bytes into a freshly created Metal tensor of matching shape.
bool uploadTensor(MetalTensor& t, const Mat& src)
{
    Mat m = src.isContinuous() ? src : src.clone();
    if (!makeTensor(t, m.dims, m.size.p))
        return false;
    memcpy(t.contents, m.data, m.total() * m.elemSize());
    return true;
}

NSArray<NSNumber*>* shapeOf(int dims, const int* sizes)
{
    NSMutableArray<NSNumber*>* a = [NSMutableArray arrayWithCapacity:dims];
    for (int i = 0; i < dims; i++)
        [a addObject:@(sizes[i])];
    return a;
}

MPSGraphTensorData* tensorData(const MetalTensor& t, NSArray<NSNumber*>* shape)
{
    return [[[MPSGraphTensorData alloc] initWithMTLBuffer:t.buffer
                                                    shape:shape
                                                 dataType:MPSDataTypeFloat32] autorelease];
}

// Resolve a row/col pair from an OpenCV size_t vector (2 -> [0],[1]; 1 -> both; else default).
bool pair2(const std::vector<size_t>& v, size_t def, size_t& row, size_t& col)
{
    if (v.empty())        { row = col = def; return true; }
    if (v.size() == 1)    { row = col = v[0]; return true; }
    if (v.size() == 2)    { row = v[0]; col = v[1]; return true; }
    return false; // 1D/3D conv not handled here
}

// Same for a Conv2Layer int vector (strides/dilations). Sizes are pre-validated by
// metalConvSupported (empty or 2), so this never has to reject.
void intPair(const std::vector<int>& v, size_t def, size_t& row, size_t& col)
{
    if (v.size() == 2)      { row = (size_t)v[0]; col = (size_t)v[1]; }
    else if (v.size() == 1) { row = col = (size_t)v[0]; }
    else                    { row = col = def; }
}

// Read a graph result back into a (continuous) host Mat. strideBytes:nil means tightly packed
// row-major, which matches a continuous NCHW float Mat.
bool readBack(MPSGraphTensorData* data, Mat& dst)
{
    if (data == nil)
        return false;
    MPSNDArray* nd = [data mpsndarray];
    if (nd == nil)
        return false;
    [nd readBytes:dst.data strideBytes:nil];
    return true;
}

} // namespace

// Build and run a single 2D NCHW convolution (weights OIHW, optional per-output-channel bias)
// on MPSGraph, reading the result back into dst. All callers funnel through here so the GPU op
// itself is written once. Returns false only on an infrastructure failure (no device, MPS throw).
static bool runMpsConv2D(const Mat& src, const Mat& weights, const Mat& bias,
                         size_t sH, size_t sW, size_t dH, size_t dW,
                         size_t pT, size_t pL, size_t pB, size_t pR,
                         int groups, Mat& dst)
{
    const int N = src.size[0], C = src.size[1], H = src.size[2], W = src.size[3];
    const int O = weights.size[0], Ig = weights.size[1], kH = weights.size[2], kW = weights.size[3];
    if (Ig <= 0 || C % Ig != 0)
        return false;

    // Expected output spatial size with explicit padding; must match the engine-allocated dst.
    const int outH = (int)((H + pT + pB - dH * (kH - 1) - 1) / sH) + 1;
    const int outW = (int)((W + pL + pR - dW * (kW - 1) - 1) / sW) + 1;
    if (dst.size[0] != N || dst.size[1] != O || dst.size[2] != outH || dst.size[3] != outW)
        return false;

    id<MTLDevice> device = (id<MTLDevice>)cv::metal::getMTLDevice();
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)cv::metal::getMTLCommandQueue();
    if (device == nil || queue == nil)
        return false;

    MPSGraph* graph = [[[MPSGraph alloc] init] autorelease];

    MPSGraphTensor* srcT = [graph placeholderWithShape:@[@(N), @(C), @(H), @(W)]
                                              dataType:MPSDataTypeFloat32 name:@"src"];
    MPSGraphTensor* wT = [graph placeholderWithShape:@[@(O), @(Ig), @(kH), @(kW)]
                                            dataType:MPSDataTypeFloat32 name:@"w"];

    MPSGraphConvolution2DOpDescriptor* desc =
        [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:sW strideInY:sH
                                                   dilationRateInX:dW dilationRateInY:dH
                                                            groups:groups
                                                       paddingLeft:pL paddingRight:pR
                                                        paddingTop:pT paddingBottom:pB
                                                      paddingStyle:MPSGraphPaddingStyleExplicit
                                                        dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                     weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    if (desc == nil)
        return false;

    MPSGraphTensor* outT = [graph convolution2DWithSourceTensor:srcT weightsTensor:wT
                                                     descriptor:desc name:@"conv"];

    MPSGraphTensor* bT = nil;
    if (!bias.empty())
    {
        bT = [graph placeholderWithShape:@[@1, @(O), @1, @1]
                                dataType:MPSDataTypeFloat32 name:@"b"];
        outT = [graph additionWithPrimaryTensor:outT secondaryTensor:bT name:@"bias"];
    }

    MetalTensor srcMt, wMt, bMt;
    if (!uploadTensor(srcMt, src) || !uploadTensor(wMt, weights))
        return false;
    int biasSizes[4] = {1, O, 1, 1};
    if (!bias.empty())
    {
        if (!makeTensor(bMt, 4, biasSizes))
            return false;
        Mat bc = bias.isContinuous() ? bias : bias.clone();
        memcpy(bMt.contents, bc.data, (size_t)O * sizeof(float));
    }

    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        [NSMutableDictionary dictionary];
    feeds[srcT] = tensorData(srcMt, @[@(N), @(C), @(H), @(W)]);
    feeds[wT] = tensorData(wMt, @[@(O), @(Ig), @(kH), @(kW)]);
    if (bT != nil)
        feeds[bT] = tensorData(bMt, @[@1, @(O), @1, @1]);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = nil;
    @try
    {
        results = [graph runWithMTLCommandQueue:queue
                                          feeds:feeds
                                  targetTensors:@[outT]
                               targetOperations:nil];
    }
    @catch (NSException* e)
    {
        if (verbose())
            CV_LOG_WARNING(NULL, "dnn/metal: MPSGraph conv threw: " << [[e reason] UTF8String]);
        return false;
    }

    return readBack(results[outT], dst);
}

bool metalConvSupported(int actType,
                        int weightType, int weightDims, bool weightConst,
                        bool hasBias, int biasType, bool biasConst,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& pads,
                        int autoPad)
{
    if (actType != CV_32F)
        return false;
    if (!weightConst || weightType != CV_32F || weightDims != 4)
        return false;
    if (hasBias && (!biasConst || biasType != CV_32F))
        return false;
    if (!(strides.empty()   || strides.size()   == 2))
        return false;
    if (!(dilations.empty() || dilations.size() == 2))
        return false;
    // SAME_UPPER/SAME_LOWER/VALID need the runtime input shape to resolve padding; decline them
    // in this phase and keep only explicit (NOTSET) padding, given as [padT, padL, padB, padR].
    if (autoPad != (int)AUTO_PAD_NONE)
        return false;
    if (!(pads.empty() || pads.size() == 4))
        return false;
    return true;
}

bool mpsConv(const Ptr<Layer>& layer, std::vector<Mat>& inputs, std::vector<Mat>& outputs)
{
    if (inputs.empty() || outputs.empty())
        return false;
    Mat& dst = outputs[0];
    if (dst.type() != CV_32F || dst.dims != 4)
        return false;

    // New-engine convolution (type "Conv2"): weights/bias are plain OIHW constant inputs
    // (inputs[1]/inputs[2], left unfolded for claimed convs); the spatial config lives on the
    // public Conv2Layer fields. This path is only reached for a conv the claim pass marked as
    // deviceClaimed, and metalConvSupported() here mirrors the claim predicate exactly, so a
    // claimed conv is always accepted (never falls through to the block-layout CPU assert).
    Conv2Layer* conv2 = dynamic_cast<Conv2Layer*>(layer.get());
    if (conv2)
    {
        Mat& src = inputs[0];
        if (src.type() != CV_32F || src.dims != 4 || inputs.size() < 2)
            return false;
        const Mat& weights = inputs[1];
        Mat bias = inputs.size() > 2 ? inputs[2] : Mat();
        const bool hasBias = !bias.empty();
        if (!metalConvSupported(src.type(),
                                weights.type(), weights.dims, /*weightConst*/ true,
                                hasBias, hasBias ? bias.type() : 0, /*biasConst*/ true,
                                conv2->strides, conv2->dilations, conv2->pads,
                                (int)conv2->auto_pad))
            return false;

        size_t sH, sW, dH, dW;
        intPair(conv2->strides, 1, sH, sW);
        intPair(conv2->dilations, 1, dH, dW);
        size_t pT = 0, pL = 0, pB = 0, pR = 0;
        if (conv2->pads.size() == 4)
        {
            pT = (size_t)conv2->pads[0]; pL = (size_t)conv2->pads[1];
            pB = (size_t)conv2->pads[2]; pR = (size_t)conv2->pads[3];
        }
        const int groups = conv2->ngroups > 0 ? conv2->ngroups : 1;
        return runMpsConv2D(src, weights, hasBias ? bias : Mat(),
                            sH, sW, dH, dW, pT, pL, pB, pR, groups, dst);
    }

    // Legacy "Convolution" layer (classic engine): constant weights/bias live in blobs.
    Mat& src = inputs[0];
    if (src.type() != CV_32F || src.dims != 4)
        return false;

    Ptr<BaseConvolutionLayer> conv = layer.dynamicCast<BaseConvolutionLayer>();
    if (conv.empty() || layer->blobs.empty())
        return false;
    Mat weights = layer->blobs[0];
    Mat bias = layer->blobs.size() > 1 ? layer->blobs[1] : Mat();
    if (weights.type() != CV_32F || weights.dims != 4)
        return false;
    if (!bias.empty() && bias.type() != CV_32F)
        return false;

    const int C = src.size[1], Ig = weights.size[1];
    if (Ig <= 0 || C % Ig != 0)
        return false;
    const int groups = C / Ig;

    size_t sH, sW, dH, dW, pT, pL, pB, pR;
    if (!pair2(conv->strides, 1, sH, sW) ||
        !pair2(conv->dilations, 1, dH, dW) ||
        !pair2(conv->pads_begin, 0, pT, pL) ||
        !pair2(conv->pads_end, 0, pB, pR))
        return false;

    return runMpsConv2D(src, weights, bias, sH, sW, dH, dW, pT, pL, pB, pR, groups, dst);
}

// Pick the MPSGraph op for a dnn unary-activation type. Returns nil to decline.
static MPSGraphTensor* applyUnary(MPSGraph* graph, MPSGraphTensor* in, const String& type)
{
    if (type == "ReLU")    return [graph reLUWithTensor:in name:nil];
    if (type == "Sigmoid") return [graph sigmoidWithTensor:in name:nil];
    if (type == "TanH")    return [graph tanhWithTensor:in name:nil];
    if (type == "Exp")     return [graph exponentWithTensor:in name:nil];
    if (type == "AbsVal")  return [graph absoluteWithTensor:in name:nil];
    return nil;
}

bool mpsUnary(const Ptr<Layer>& layer, std::vector<Mat>& inputs, std::vector<Mat>& outputs)
{
    if (inputs.empty() || outputs.empty())
        return false;
    Mat& src = inputs[0];
    Mat& dst = outputs[0];
    if (src.type() != CV_32F || dst.type() != CV_32F)
        return false;
    if (src.dims != dst.dims || src.total() != dst.total())
        return false;

    // ReLU is the only unary here with a parameter: handle plain ReLU (negativeSlope == 0)
    // and decline leaky ReLU. The others are parameter-free.
    if (layer->type == "ReLU")
    {
        Ptr<ReLULayer> relu = layer.dynamicCast<ReLULayer>();
        if (!relu.empty() && relu->negativeSlope != 0.f)
            return false;
    }

    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)cv::metal::getMTLCommandQueue();
    if (queue == nil)
        return false;

    NSArray<NSNumber*>* shape = shapeOf(src.dims, src.size.p);

    MPSGraph* graph = [[[MPSGraph alloc] init] autorelease];
    MPSGraphTensor* srcT = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"src"];
    MPSGraphTensor* outT = applyUnary(graph, srcT, layer->type);
    if (outT == nil)
        return false;

    MetalTensor srcMt;
    if (!uploadTensor(srcMt, src))
        return false;

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{srcT: tensorData(srcMt, shape)};

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = nil;
    @try
    {
        results = [graph runWithMTLCommandQueue:queue
                                          feeds:feeds
                                  targetTensors:@[outT]
                               targetOperations:nil];
    }
    @catch (NSException* e)
    {
        if (verbose())
            CV_LOG_WARNING(NULL, "dnn/metal: MPSGraph unary '" << layer->type << "' threw: "
                           << [[e reason] UTF8String]);
        return false;
    }

    return readBack(results[outT], dst);
}

} // namespace metal
} // namespace dnn
} // namespace cv

#endif // HAVE_METAL
