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

bool mpsConv(const Ptr<Layer>& layer, std::vector<Mat>& inputs, std::vector<Mat>& outputs)
{
    if (inputs.empty() || outputs.empty())
        return false;
    Mat& src = inputs[0];
    Mat& dst = outputs[0];
    if (src.type() != CV_32F || dst.type() != CV_32F)
        return false;
    if (src.dims != 4 || dst.dims != 4)             // 2D conv (NCHW) only for the PoC
        return false;

    Ptr<BaseConvolutionLayer> conv = layer.dynamicCast<BaseConvolutionLayer>();
    if (conv.empty())
        return false;

    // Constant weights/bias only (the common folded-initializer case). Variable weights
    // (inputs[1]/inputs[2]) are declined for the PoC.
    if (layer->blobs.empty())
        return false;
    Mat weights = layer->blobs[0];
    Mat bias = layer->blobs.size() > 1 ? layer->blobs[1] : Mat();
    if (weights.type() != CV_32F || weights.dims != 4)
        return false;
    if (!bias.empty() && bias.type() != CV_32F)
        return false;

    const int N = src.size[0], C = src.size[1], H = src.size[2], W = src.size[3];
    const int O = weights.size[0], Ig = weights.size[1], kH = weights.size[2], kW = weights.size[3];
    if (Ig <= 0 || C % Ig != 0)
        return false;
    const int groups = C / Ig;

    size_t sH, sW, dH, dW, pT, pL, pB, pR;
    if (!pair2(conv->strides, 1, sH, sW) ||
        !pair2(conv->dilations, 1, dH, dW) ||
        !pair2(conv->pads_begin, 0, pT, pL) ||
        !pair2(conv->pads_end, 0, pB, pR))
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

bool mpsRelu(const Ptr<Layer>& layer, std::vector<Mat>& inputs, std::vector<Mat>& outputs)
{
    if (inputs.empty() || outputs.empty())
        return false;
    Mat& src = inputs[0];
    Mat& dst = outputs[0];
    if (src.type() != CV_32F || dst.type() != CV_32F)
        return false;
    if (src.dims != dst.dims || src.total() != dst.total())
        return false;

    Ptr<ReLULayer> relu = layer.dynamicCast<ReLULayer>();
    const float slope = relu.empty() ? 0.f : relu->negativeSlope;
    if (slope != 0.f)                               // plain ReLU only for the PoC
        return false;

    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)cv::metal::getMTLCommandQueue();
    if (queue == nil)
        return false;

    NSArray<NSNumber*>* shape = shapeOf(src.dims, src.size.p);

    MPSGraph* graph = [[[MPSGraph alloc] init] autorelease];
    MPSGraphTensor* srcT = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"src"];
    MPSGraphTensor* outT = [graph reLUWithTensor:srcT name:@"relu"];

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
            CV_LOG_WARNING(NULL, "dnn/metal: MPSGraph relu threw: " << [[e reason] UTF8String]);
        return false;
    }

    return readBack(results[outT], dst);
}

} // namespace metal
} // namespace dnn
} // namespace cv

#endif // HAVE_METAL
