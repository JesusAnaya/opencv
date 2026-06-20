// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Experimental claim pass for the dnn Metal/MPSGraph executor. It runs inside
// prepareForInference, BEFORE the CPU-lowering passes (constArgs / fuseBasic / useBlockLayout),
// and marks every Conv2 op the Metal executor can run (metalConvSupported) by setting the
// backend-agnostic Conv2Layer::deviceClaimed. The downstream passes consult that flag to keep
// a claimed conv on
// the plain-NCHW path with unfolded OIHW weights and no activation fusion, so the execution seam
// can dispatch it to MPSGraph. The marking predicate is a strict subset of what the seam's
// mpsConv accepts (both call metalConvSupported), so a claimed conv is always handled on the GPU
// and never reaches Conv2::forward, which asserts a block layout the claimed conv no longer has.
//
// Without HAVE_METAL the whole pass is a no-op stub.

#include "../precomp.hpp"
#include "../net_impl.hpp"

#ifdef HAVE_METAL
#include "opencv2/dnn/all_layers.hpp"
#include "dnn_metal.hpp"
#endif

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_METAL

// Walk a graph (and any subgraphs) and mark the Metal-runnable Conv2 ops. At this point in the
// pipeline a convolution's weights/bias are still plain OIHW constant args (inputs[1]/inputs[2]);
// constArgs() has not yet folded them into the layer.
static void claimMetalConvsInGraph(Net::Impl* netimpl, const Ptr<Graph>& graph)
{
    if (!graph)
        return;
    const std::vector<Ptr<Layer> >& prog = graph->prog();
    for (const Ptr<Layer>& layer : prog)
    {
        if (!layer)
            continue;
        if (std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs())
        {
            for (const Ptr<Graph>& g : *subgraphs)
                claimMetalConvsInGraph(netimpl, g);
        }

        Conv2Layer* conv = dynamic_cast<Conv2Layer*>(layer.get());
        if (!conv)
            continue;

        const std::vector<Arg>& inputs = layer->inputs;
        if (inputs.size() < 2)              // need at least data + weights
            continue;

        // Activation type is read from the arg metadata, not from a materialized tensor: if it is
        // not a known float32 (e.g. shape/type not inferred yet, or fp16), the conv is simply not
        // claimed and keeps its ordinary CPU path. Declining is always safe.
        const ArgData& actData = netimpl->args[inputs[0].idx];
        const ArgData& wData   = netimpl->args[inputs[1].idx];
        const bool weightConst = netimpl->isConstArg(inputs[1]);

        const bool hasBias = inputs.size() > 2 && inputs[2].idx > 0;
        int  biasType  = 0;
        bool biasConst = false;
        if (hasBias)
        {
            biasType  = netimpl->args[inputs[2].idx].type;
            biasConst = netimpl->isConstArg(inputs[2]);
        }

        if (metal::metalConvSupported(actData.type,
                                      wData.type, wData.shape.dims, weightConst,
                                      hasBias, biasType, biasConst,
                                      conv->strides, conv->dilations, conv->pads,
                                      (int)conv->auto_pad))
        {
            conv->deviceClaimed = true;
        }
    }
}

void Net::Impl::claimMetalConvs()
{
    if (!metal::isEnabled() || !metal::dnnMetalAvailable())
        return;
    claimMetalConvsInGraph(this, mainGraph);
}

#else

void Net::Impl::claimMetalConvs() {}

#endif // HAVE_METAL

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
