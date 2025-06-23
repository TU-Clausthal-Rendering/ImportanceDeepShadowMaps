/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
// #include "Core/API/Shared/D3D12Handles.h"
#include "FSRPass.h"
#include "Core/API/NativeHandleTraits.h"
#include <ffx_api/dx12/ffx_api_dx12.hpp>
#include <ffx_api/ffx_upscale.hpp>
#include <ffx_api/ffx_api_types.h>
#include "Utils/Math/FalcorMath.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, FSRPass>();
}

namespace
{
const std::string kColorIn = "color";
const std::string kDepth = "depth";
const std::string kMotion = "mvec";
// const std::string kExposure = "exposure"; // Optional resource containing a 1x1 exposure value.
// const std::string kReactive = "reactive"; // Optional resource containing alpha value of reactive objects in the scene.
const std::string kOutput = "output";
} // namespace

static void ffxApiMessageFunc(uint32_t type, const wchar_t* message)
{
    auto wstr = std::wstring(message);
    auto cstr = std::string(wstr.begin(), wstr.end());
    logWarningOnce("FFX API message: " + cstr);
}

FSRPass::FSRPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice), mContext(nullptr) {}

Properties FSRPass::getProperties() const
{
    return {};
}

RenderPassReflection FSRPass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    reflector.addInput(kColorIn, "Color buffer for the current frame (at render resolution)");
    reflector.addInput(kDepth, "32bit depth values for the current frame (at render resolution)");
    reflector.addInput(kMotion, "2-dimensional motion vectors");

    reflector.addOutput(kOutput, "Output texture").format(ResourceFormat::RGBA32Float).bindFlags(Resource::BindFlags::AllColorViews);
    return reflector;
}

void FSRPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    auto pNative = mpDevice->getNativeHandle().as<ID3D12Device*>();

    ffx::CreateBackendDX12Desc backendDesc{};
    backendDesc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12;
    backendDesc.device = pNative;

    ffx::CreateContextDescUpscale contextDesc{};
    contextDesc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE; // | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
    contextDesc.maxRenderSize = {compileData.defaultTexDims.x, compileData.defaultTexDims.y};
    contextDesc.maxUpscaleSize = {compileData.defaultTexDims.x, compileData.defaultTexDims.y};
    // #if defined(_DEBUG)
    contextDesc.flags |= FFX_UPSCALE_ENABLE_DEBUG_CHECKING;
    contextDesc.fpMessage = ffxApiMessageFunc;
    // #endif

    auto ret = ffx::CreateContext(mContext, nullptr, contextDesc, backendDesc);
    if (ret != ffx::ReturnCode::Ok)
        throw std::runtime_error("Failed to create FFX context");

    FfxApiEffectMemoryUsage gpuMemoryUsageUpscaler;
    ffx::QueryDescUpscaleGetGPUMemoryUsage upscalerGetGPUMemoryUsage{};
    upscalerGetGPUMemoryUsage.gpuMemoryUsageUpscaler = &gpuMemoryUsageUpscaler;

    ffx::Query(mContext, upscalerGetGPUMemoryUsage);
    std::cerr << "FSR: Upscaler Context VRAM totalUsageInBytes " << gpuMemoryUsageUpscaler.totalUsageInBytes / 1048576.f << " MB"
              << std::endl;
    std::cerr << "                       aliasableUsageInBytes " << gpuMemoryUsageUpscaler.aliasableUsageInBytes / 1048576.f << " MB"
              << std::endl;
}

static FfxApiResource ffxGetResourceApi(RenderContext* pRenderContext, Texture* pTexture)
{
    if (!pTexture)
    {
        return ffxApiGetResourceDX12(nullptr);
    }

    // transition state to shader resource
    pRenderContext->resourceBarrier(pTexture, Resource::State::ShaderResource);

    ID3D12Resource* pResource = pTexture->getNativeHandle().as<ID3D12Resource*>();
    FfxApiResource apiRes = ffxApiGetResourceDX12(pResource, FFX_API_RESOURCE_STATE_COMPUTE_READ, 0);
    return apiRes;
}

void FSRPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    mTimer.update();
    auto pColorIn = renderData.getTexture(kColorIn);
    auto pDepth = renderData.getTexture(kDepth);
    auto pMotion = renderData.getTexture(kMotion);
    auto pOut = renderData.getTexture(kOutput);

    if (!mpScene || !mEnabled)
    {
        pRenderContext->blit(pColorIn->getSRV(), pOut->getRTV());
        return;
    }

    auto pCamera = mpScene->getCamera();

    ffx::DispatchDescUpscale dispatchUpscale{};
    ID3D12GraphicsCommandList* pCommandList =
        pRenderContext->getLowLevelData()->getCommandBufferNativeHandle().as<ID3D12GraphicsCommandList*>();
    dispatchUpscale.commandList = pCommandList;
    dispatchUpscale.color = ffxGetResourceApi(pRenderContext, pColorIn.get());
    dispatchUpscale.depth = ffxGetResourceApi(pRenderContext, pDepth.get());
    dispatchUpscale.motionVectors = ffxGetResourceApi(pRenderContext, pMotion.get());
    dispatchUpscale.exposure = ffxGetResourceApi(pRenderContext, nullptr);
    dispatchUpscale.output = ffxGetResourceApi(pRenderContext, pOut.get());
    dispatchUpscale.reactive = ffxGetResourceApi(pRenderContext, nullptr);
    ;
    dispatchUpscale.transparencyAndComposition = ffxGetResourceApi(pRenderContext, nullptr);
    float2 jitterOffset = float2(pCamera->getJitterX(), -pCamera->getJitterY()) * float2(pColorIn->getWidth(), pColorIn->getHeight());
    dispatchUpscale.jitterOffset.x = jitterOffset.x; //-pCamera->getJitterX(); // TODO jitter ([-0.5, 0.5] range?)
    dispatchUpscale.jitterOffset.y = jitterOffset.y; //-pCamera->getJitterY();
    
    dispatchUpscale.motionVectorScale.x = float(pColorIn->getWidth());
    dispatchUpscale.motionVectorScale.y = float(pColorIn->getHeight());
    dispatchUpscale.reset = mReset;
    mReset = false;
    dispatchUpscale.enableSharpening = mSharpness > 0.0f;
    dispatchUpscale.sharpness = mSharpness;
    dispatchUpscale.frameTimeDelta = float(mTimer.delta() * 1000.0); // FSR expects milliseconds
    dispatchUpscale.preExposure = 1.0f;
    dispatchUpscale.renderSize.width = pColorIn->getWidth();
    dispatchUpscale.renderSize.height = pColorIn->getHeight();
    dispatchUpscale.upscaleSize.width = pOut->getWidth();
    dispatchUpscale.upscaleSize.height = pOut->getHeight();
    dispatchUpscale.cameraFovAngleVertical = focalLengthToFovY(pCamera->getFocalLength(), Camera::kDefaultFrameHeight);
    dispatchUpscale.cameraFar = pCamera->getFarPlane();
    dispatchUpscale.cameraNear = pCamera->getNearPlane();
    dispatchUpscale.flags = 0;
    // dispatchUpscale.flags = FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW;
    ffx::ReturnCode retCode = ffx::Dispatch(mContext, dispatchUpscale);
    if (retCode != ffx::ReturnCode::Ok)
    {
        logWarning("Failed ffx::Dispatch");
    }

    pRenderContext->setPendingCommands(true);
    pRenderContext->uavBarrier(pOut.get());
    pRenderContext->flush();
}

void FSRPass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Enabled", mEnabled);
    if (!mEnabled)
        return;

    widget.slider("Sharpness", mSharpness, 0.0f, 1.0f);

    if (widget.button("Reset"))
    {
        mReset = true;
    }
}
