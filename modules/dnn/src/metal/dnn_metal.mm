// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "dnn_metal.hpp"

#ifdef HAVE_METAL

#include "opencv2/core/metal.hpp"
#include "opencv2/core/utils/configuration.private.hpp"
#include "opencv2/core/utils/logger.hpp"

namespace cv {
namespace dnn {
namespace metal {

namespace {

struct State
{
    bool initialized = false;
    bool enabled = false;
    int executed = 0;
    int declined = 0;
};

State& state()
{
    static State s;
    if (!s.initialized)
    {
        s.enabled = utils::getConfigurationParameterBool("OPENCV_DNN_METAL", false);
        s.initialized = true;
    }
    return s;
}

} // namespace

bool verbose()
{
    static bool v = utils::getConfigurationParameterBool("OPENCV_DNN_METAL_VERBOSE", false);
    return v;
}

bool dnnMetalAvailable()
{
    return cv::metal::haveMetal();
}

void setEnabled(bool enabled)
{
    state().enabled = enabled;
}

bool isEnabled()
{
    return state().enabled;
}

void resetCounters()
{
    state().executed = 0;
    state().declined = 0;
}

int opsExecuted()
{
    return state().executed;
}

int opsDeclined()
{
    return state().declined;
}

bool tryForward(const Ptr<Layer>& layer,
                std::vector<Mat>& inputs,
                std::vector<Mat>& outputs,
                std::vector<Mat>& internals)
{
    CV_UNUSED(internals);

    if (!isEnabled() || !dnnMetalAvailable() || layer.empty())
        return false;

    bool handled = false;
    @autoreleasepool
    {
        const String& type = layer->type;
        if (type == "Conv2" || type == "Convolution")
            handled = mpsConv(layer, inputs, outputs);
        else if (type == "ReLU")
            handled = mpsRelu(layer, inputs, outputs);
    }

    if (handled)
        state().executed++;
    else
        state().declined++;

    if (verbose())
    {
        CV_LOG_INFO(NULL, "dnn/metal: op '" << layer->type << "' ("
                    << layer->name << ") " << (handled ? "executed on GPU" : "declined -> CPU"));
    }
    return handled;
}

} // namespace metal
} // namespace dnn
} // namespace cv

#endif // HAVE_METAL
