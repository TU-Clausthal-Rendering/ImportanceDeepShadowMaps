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
#include "IDSMLinkedList.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/SampleGenerators/HaltonSamplePattern.h"

namespace
{
    //Shader Paths
    const std::string kShaderFolder = "RenderPasses/IDSMRenderer/IDSMLinkedList/";
    const std::string kGenShader = kShaderFolder + "GenerateIDSMLinkedList.rt.slang";
    const std::string kShaderShowImportanceMap = kShaderFolder + "DebugShowImportance.cs.slang";

    //UI
    const Gui::DropdownList kAccelDataFormat = {{1, "Uint"}, {2, "Uint2"}, {4, "Uint4"}};

}; // namespace

IDSMLinkedList::IDSMLinkedList(ref<Device> pDevice, ref<Scene> pScene) : DeepShadowMapMethod(pDevice, pScene)
{
    mpFence = GpuFence::create(mpDevice);
    FALCOR_ASSERT(mpFence);
    Sampler::Desc samplerDesc = {};
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
    mpLinearSampler = Sampler::create(mpDevice, samplerDesc);
    FALCOR_ASSERT(mpLinearSampler);
    samplerDesc.setFilterMode(Sampler::Filter::Point, Sampler::Filter::Point, Sampler::Filter::Point);
    mpPointSampler = Sampler::create(mpDevice, samplerDesc);
    FALCOR_ASSERT(mpPointSampler);
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);

    mpImportanceMapHelper = std::make_unique<ImportanceMapHelper>(mpDevice, mpScene->getLightCount(), mResolution, true);
}

void IDSMLinkedList::prepareResources(RenderContext* pRenderContext)
{

    //This is triggered if either the resolution or number of lights changed
    if (mResolutionChanged)
    {
        mpImportanceMapHelper->updateResolution(mResolution);
        //The following buffers need to be cleared when light count changes
        mLinkedListCounter.clear();
        mLinkedListCounterCPU.clear();
        mCounterFenceWaitValues.clear();
        mUIElementCounter.clear();
    }

    if (mTransparencyBufferUsesColor != mUseColoredTransparency || mResolutionChanged)
    {
        mLinkedListData.clear();
        mTransparencyBufferUsesColor = mUseColoredTransparency;
    }

    mResolutionChanged = false;

    updateSMMatrices();

    // Create AVSM trace program
    if (!mGenLinkedListShadowPip.pProgram)
    {
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kGenShader);
        desc.setMaxPayloadSize(32u);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1u);

        mGenLinkedListShadowPip.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mGenLinkedListShadowPip.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        }

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");
        defines.add("ACCEL_BOXES_PIXEL_OFFSET", mAccelUsePCF ? "1.0" : "0.5");

        mGenLinkedListShadowPip.pProgram = RtProgram::create(mpDevice, desc, defines);
    }

    auto& lights = mpScene->getLights();

    // Create / Destroy resources
    {
        const uint numBuffers = lights.size();
        const uint numAccelBuffers = lights.size();
        mLinkedListNodeBufferSize = mResolution.x * mResolution.y * mApproxNumElementsPerPixel;
        // Counter
        if (mLinkedListCounter.empty())
        {
            mLinkedListCounter.resize(kFramesInFlight);
            mLinkedListCounterCPU.resize(kFramesInFlight);
            mCounterFenceWaitValues.resize(kFramesInFlight);
            mUIElementCounter.resize(numAccelBuffers);

            std::vector<uint> initData(numAccelBuffers, 0);
            for (uint i = 0; i < kFramesInFlight; i++)
            {
                mLinkedListCounter[i] = Buffer::createStructured(
                    mpDevice, sizeof(uint), numAccelBuffers, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    Buffer::CpuAccess::None, initData.data(), false
                );
                mLinkedListCounter[i]->setName("LinkedListElementCounter_" + std::to_string(i));

                mLinkedListCounterCPU[i] = Buffer::createStructured(
                    mpDevice, sizeof(uint), numAccelBuffers, ResourceBindFlags::None, Buffer::CpuAccess::Read, &initData, false
                );
                mLinkedListCounterCPU[i]->setName("LinkedListElementCounterCPU_" + std::to_string(i));

                mCounterFenceWaitValues[i] = 0;
            }

            for (uint i = 0; i < numAccelBuffers; i++)
                mUIElementCounter[i] = mResolution.x * mResolution.y * mApproxNumElementsPerPixel;
        }
        if (mLinkedListData.empty())
        {
            mLinkedListData.resize(numAccelBuffers);
            uint dataStructSize = mUseColoredTransparency ? 5 : 3;
            for (uint i = 0; i < numAccelBuffers; i++)
            {
                mLinkedListData[i] = Buffer::createStructured(
                    mpDevice, sizeof(uint) * dataStructSize, mLinkedListNodeBufferSize,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false
                );
                mLinkedListData[i]->setName("LinkedListIrrShadowNodes" + std::to_string(i));
            }
        }
    }
}

void IDSMLinkedList::dummyProfileGeneration(RenderContext* pRenderContext)
{
    mpImportanceMapHelper->dummyRenderPassProfile(pRenderContext);

    auto& lights = mpScene->getLights();
    for (uint i = 0; i < lights.size(); i++)
    {
        if (!lights[i]->isActive())
            break;
        FALCOR_PROFILE(pRenderContext, "Trace Light:" + lights[i]->getName());
        {
            FALCOR_PROFILE(pRenderContext, "raytraceScene");
        }
    }
}

void IDSMLinkedList::generate(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Generate_IDSM_LL");

    // Abort early if disabled
    bool skipGeneration = (mSkipFrameCount % mSkipGenerationFrameCount) != 0;
    mSkipFrameCount++;
    if (skipGeneration)
    {
        dummyProfileGeneration(pRenderContext);
        return;
    }

    prepareResources(pRenderContext);

    // Handle light MVP for directional lights
    if (mHasDirectionalLight)
    {
        if (mDirectionalLightIndex < 0 || mDirectionalLightIndex < (int(mpScene->getLightCount())-1))
        {
            for (uint i = 0; i < mpScene->getLightCount(); i++)
            {
                if (mpScene->getLight(i)->getType() == LightType::Directional)
                {
                    mDirectionalLightIndex = i;
                    break;
                }
            }
        }        

        LightMVP tmp = mShadowMapMVP[mDirectionalLightIndex];
        mShadowMapMVP[mDirectionalLightIndex] = mStaggeredDirectionalLightMVP;
        mStaggeredDirectionalLightMVP = tmp;
    }

    auto& lights = mpScene->getLights();
    int frameInFlight = mStagingCount; //Counter GPU CPU sync

    int lastFrameInFlight = 0;
    lastFrameInFlight = frameInFlight - 1;
    lastFrameInFlight = lastFrameInFlight < 0 ? kFramesInFlight - 1 : lastFrameInFlight;
    uint maxRayBudget = mResolution.x * mResolution.y * mSampleOverestimate * mSampleOverestimate;
    uint maxNodeSize = mResolution.x * mResolution.y * mApproxNumElementsPerPixel;

    // Get the sample distribution ready
    mpImportanceMapHelper->generateSampleDistribution(
        pRenderContext, mLinkedListCounter[lastFrameInFlight], maxNodeSize, maxRayBudget, mLinkedListCounter[frameInFlight], false
    );

    // Defines
    mGenLinkedListShadowPip.pProgram->addDefine("MAX_IDX", std::to_string(mResolution.x * mResolution.y * mApproxNumElementsPerPixel));
    mGenLinkedListShadowPip.pProgram->addDefine("MIDPOINT_PERCENTAGE", std::to_string(mMidpointPercentage));
    mGenLinkedListShadowPip.pProgram->addDefine("MIDPOINT_DEPTH_BIAS", std::to_string(mMidpointDepthBias));
    mGenLinkedListShadowPip.pProgram->addDefine("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");
    mGenLinkedListShadowPip.pProgram->addDefine("ACCEL_BOXES_PIXEL_OFFSET", mAccelUsePCF ? "1.0" : "0.5");
    mGenLinkedListShadowPip.pProgram->addDefine("SAMPLE_DIST_MIPS", std::to_string(mpImportanceMapHelper->getSampleDistribution(0)->getMipCount()));
    mGenLinkedListShadowPip.pProgram->addDefine("USE_HALTON_SAMPLE_PATTERN", mpHaltonBuffer ? "1" : "0");
    mGenLinkedListShadowPip.pProgram->addDefine("NUM_HALTON_SAMPLES", std::to_string(mJitterSampleCount));
    mGenLinkedListShadowPip.pProgram->addDefine("USE_RANDOM_RANDOM_SOFT_SHADOWS", mEnableRandomSoftShadows ? "1" : "0");
    mGenLinkedListShadowPip.pProgram->addDefine("RANDOM_SOFT_SHADOWS_POS_RADIUS", std::to_string(mRandomSoftShadowsPositionRadius));
    mGenLinkedListShadowPip.pProgram->addDefine("RANDOM_SOFT_SHADOWS_DIR_SPREAD", std::to_string(mRandomSoftShadowsDirSpread));
    mGenLinkedListShadowPip.pProgram->addDefine("TRACE_NON_OPAQUE_ONLY", mUseMask ? "1" : "0"); // Trace non-opaque only if mask is used
    mGenLinkedListShadowPip.pProgram->addDefine("INCLUDE_CAST_SHADOW_INSTANCE_MASK_BIT", mEnableBlacklistWithShadowMaterialFlag ? "0" : "1"); //Determines if the castShadow instance mask bit is used
    
    // Create Program Vars
    if (!mGenLinkedListShadowPip.pVars)
    {
        mGenLinkedListShadowPip.pProgram->addDefines(mpSampleGenerator->getDefines());
        mGenLinkedListShadowPip.pProgram->setTypeConformances(mpScene->getTypeConformances());
        mGenLinkedListShadowPip.pVars = RtProgramVars::create(mpDevice, mGenLinkedListShadowPip.pProgram, mGenLinkedListShadowPip.pBindingTable);
        mpSampleGenerator->setShaderData(mGenLinkedListShadowPip.pVars->getRootVar());
    }

    FALCOR_ASSERT(mGenLinkedListShadowPip.pVars);
    auto var = mGenLinkedListShadowPip.pVars->getRootVar();

    // Trace the pass for every light
    for (uint i = 0; i < lights.size(); i++)
    {
        if (!lights[i]->isActive())
            break;
        FALCOR_PROFILE(pRenderContext, "Trace Light:" + lights[i]->getName());
        // Bind Utility
        bool isDirectional = lights[i]->getType() == LightType::Directional;

        var["CB"]["gFrameCount"] = mFrameCount;
        var["CB"]["gLightPos"] = mShadowMapMVP[i].pos;
        var["CB"]["gIsDirectional"] = isDirectional;
        var["CB"]["gLightDir"] = lights[i]->getData().dirW;
        var["CB"]["gFar"] = mNearFar.y;
        var["CB"]["gLightIdx"] = i;
        var["CB"]["gMipCount"] = mpImportanceMapHelper->getSampleDistribution(i)->getMipCount();
        var["CB"]["gSMRes"] = mResolution;
        var["CB"]["gViewProj"] = mShadowMapMVP[i].viewProjection;
        var["CB"]["gInvViewProj"] = mShadowMapMVP[i].invViewProjection;
        var["CB"]["gSpreadAngle"] = mShadowMapMVP[i].spreadAngle;

        var["gCounter"] = mLinkedListCounter[frameInFlight];
        var["gData"] = mLinkedListData[i];
        var["gImportanceMap"] = mpImportanceMapHelper->getImportanceMap(i);
        var["gSampleDistribution"] = mpImportanceMapHelper->getSampleDistribution(i);
        var["gHaltonSamples"] = mpHaltonBuffer;

        // Get dimensions of ray dispatch.
        uint2 targetDim = uint2(float2(mResolution) * mSampleOverestimate);

               
        FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

        // Spawn the rays.
        mpScene->raytrace(pRenderContext, mGenLinkedListShadowPip.pProgram.get(), mGenLinkedListShadowPip.pVars, uint3(targetDim, 1));

        mFrameCount++;
    }

    const uint numAABBs = lights.size();

    //Copy data from GPU to CPU counter
    {
        // Copy to CPU
        pRenderContext->copyBufferRegion(
            mLinkedListCounterCPU[mStagingCount].get(), 0, mLinkedListCounter[mStagingCount].get(), 0, sizeof(uint32_t) * numAABBs
        );
        pRenderContext->flush();
        // Frame in flight for the counter
        mCounterFenceWaitValues[mStagingCount] = mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        mStagingCount = (mStagingCount + 1) % kFramesInFlight;

        uint64_t& fenceWaitVal = mCounterFenceWaitValues[mStagingCount];
        // Wait for the GPU to finish the frame
        mpFence->syncCpu(fenceWaitVal);

        void* data = mLinkedListCounterCPU[mStagingCount]->map(Buffer::MapType::Read);
        std::memcpy(mUIElementCounter.data(), data, sizeof(uint) * numAABBs);
        mLinkedListCounterCPU[mStagingCount]->unmap();
    }
}

DefineList IDSMLinkedList::getDefines()
{
    DefineList defines = {};
    defines.add(DeepShadowMapMethod::getDefines());
    defines.add("SHADOW_ACCEL_PCF", mAccelUsePCF ? "1" : "0");
    defines.add("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");
    return defines;
}

void IDSMLinkedList::setShaderData(const ShaderVar& var)
{
    auto shadowVar = var["gIDSMLinkedList"];

    shadowVar["SMCB"]["gSMSize"] = mResolution;
    shadowVar["SMCB"]["gNear"] = mNearFar.x;
    shadowVar["SMCB"]["gFar"] = mNearFar.y;
    shadowVar["SMCB"]["gMipCount"] = mpImportanceMapHelper->getSampleDistribution(0)->getMipCount();
    shadowVar["SMCB"]["gMaxBufferSize"] = mLinkedListNodeBufferSize;
    uint2 opaqueSMMaxDispatch = getShaderDispatchSize();
    shadowVar["SMCB"]["gISMMaxSize"] = opaqueSMMaxDispatch.x * opaqueSMMaxDispatch.y;
    

    auto& lights = mpScene->getLights();
    for (uint i = 0; i < lights.size(); i++)
    {
        shadowVar["ShadowVPs"]["gShadowMapVP"][i] = mShadowMapMVP[i].viewProjection;
        shadowVar["ShadowVPs"]["gStaggeredDirVP"] = mStaggeredDirectionalLightMVP.viewProjection;
        shadowVar["gImportanceMap"][i] = mpImportanceMapHelper->getImportanceMap(i);
    }
    const auto accelDataSize = lights.size();
    for (uint i = 0; i < accelDataSize; i++)
    {
        shadowVar["gSampleDistribution"][i] = mpImportanceMapHelper->getSampleDistribution(i);
        shadowVar["gLinkedListData"][i] = mLinkedListData[i];
    }

    shadowVar["gLinearSampler"] = mpLinearSampler;
    shadowVar["gPointSampler"] = mpPointSampler;
}

void IDSMLinkedList::setShadowMask(const ShaderVar& var, ref<Texture> maskTex, std::vector<ref<Buffer>>& maskISM, bool enable)
{
    mUseMask = enable;
    if (mUseMask)
    {
        auto shadowVar = var["gIDSMLinkedList"];

        shadowVar["gShadowMask"] = maskTex;
        for (uint i = 0; i < maskISM.size(); i++)
            shadowVar["gMaskShadowMap"][i] = maskISM[i];
    }
}

bool IDSMLinkedList::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
    if (auto group = widget.group("IDSM-LL Settings"))
    {
        mResolutionChanged |= group.var("Node Buffer size (Resolution x this)", mApproxNumElementsPerPixel, 1u, 32u, 1u);
        group.text("Note: IDSM tries to fill the buffer, so this affects quality and runtime");
        std::string bufferSize = "Buffer Elements: " + std::to_string(mResolution.x * mResolution.y * mApproxNumElementsPerPixel);
        group.text(bufferSize);
        group.var("Ray Budget Multiplier (Resolution x this)", mSampleOverestimate, 1.0f, 4.f);
        group.tooltip(
            "Controls the maximum dispatch size of the IDSM generation shader. Defines the upper limit for the budget distribution. \n"
            "Max Dispatch Size: [SMRes.x * Overestimate , SMRes.y * Overestimate]"
        );

       mpImportanceMapHelper->renderUI(group);

       if (auto statsGroup = group.group("Stats"))
       {
           if (mpScene)
           {
               if (auto group2 = statsGroup.group("Buffer Size Infos"))
               {
                   const auto loopSize = mpScene->getLightCount();
                   for (uint i = 0; i < loopSize; i++)
                   {
                       if (i > 0)
                           group2.separator();
                       uint dataBufferSize = mUseColoredTransparency ? 12u : 4u;
                       group2.text(mpScene->getLight(i)->getName());
                       group2.text("Buffer Size:        " + std::to_string(mLinkedListNodeBufferSize));
                       float dataMem = (mLinkedListNodeBufferSize * (sizeof(float) + dataBufferSize)) / 1e6f;
                       std::string dataMemStr = std::to_string(dataMem);
                       group2.text("Data Memory:     " + dataMemStr.substr(0, dataMemStr.find(".") + 3) + " MB");

                       group2.text("Used Elements:    " + std::to_string(uint(mUIElementCounter[i])));
                       std::string neededMem = std::to_string((mUIElementCounter[i] * (sizeof(float) + mLinkedListDataFormatSize)) / 1e6f);
                       std::string fillRate = std::to_string(((mUIElementCounter[i]) / float(mLinkedListNodeBufferSize)) * 100.f);
                       group2.text(
                           "Used Element Buffer Memory:   " + neededMem.substr(0, neededMem.find(".") + 3) + " MB (" +
                           fillRate.substr(0, fillRate.find(".") + 2) + "%)"
                       );
                   }
                   group2.separator();
               }
           }
       }

       if (auto group2 = group.group("Debug / Experimental"))
       {
           group2.var("Generate only every X Frame", mSkipGenerationFrameCount, 1u, UINT_MAX);
           group2.tooltip(
               "Number of generated frames is 1/X. Currently poorly optimized (No load distribution, every SM is generated in the same "
               "Frame)"
           );

           group2.checkbox("Debug Show Importance", mDebugEnableShowImportance);
           if (mDebugEnableShowImportance)
           {
               group2.text("Press the \"Show in Debug Window\" on top and select the output");
               group2.text("\"IDSMRenderer.outDebug\" in the Debug Window. Brightness for IM and SD can be adjusted below.");
               group2.text("Top Left: Importance Map. For all MipMaps to work, check \"Reduce Use MipMap version\"");
               group2.text("Top Right: Sample Distribution.");
               group2.text("Bottom Left: Non-Opaque Object Mask.");
               if (mpScene)
               {
                   uint lightCount = mpScene->getLightCount();
                   if (lightCount > 1)
                       group2.slider("Selected Light", mDebugSelectedLight, 0u, lightCount - 1u);
                   else
                       mDebugSelectedLight = 0;
               }

               uint mipCount = mpImportanceMapHelper->getImportanceMap(0)->getMipCount();
               if (mipCount > 1)
                   group2.slider("Selected Mipmap", mDebugSelectedMipLevel, 0u, mipCount - 1u);
               else
                   mDebugSelectedMipLevel = 0;

               group2.var("Scale Factor IM", mDebugScaleFactorIM, 0.f, FLT_MAX, 1.f);
               group2.var("Scale Factor SD", mDebugScaleFactorSD, 0.f, FLT_MAX, 1.f);
           }
       }

    }
    return dirty;
}

void IDSMLinkedList::debugPass(
    RenderContext* pRenderContext,
    const RenderData& renderData,
    ref<Texture> debugOut,
    ref<Texture> mask
)
{
    // Early out
    if (!mDebugEnableShowImportance)
        return;

    FALCOR_PROFILE(pRenderContext, "ShowSampleDistribution");

    if (!mpDebugShowImportancePass)
    {
        Program::Desc desc;
        //desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderShowImportanceMap).csEntry("main").setShaderModel("6_6");
        //desc.addTypeConformances(mpScene->getTypeConformances());

        DefineList defines;

        mpDebugShowImportancePass = ComputePass::create(mpDevice, desc, defines, true);
    }
    FALCOR_ASSERT(mpDebugShowImportancePass);

    // Dispatch Dims
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    auto var = mpDebugShowImportancePass->getRootVar();
    auto& pImportanceMap = mpImportanceMapHelper->getImportanceMap(mDebugSelectedLight);
    auto& pSampleDistribution = mpImportanceMapHelper->getSampleDistribution(mDebugSelectedLight);

    var["CB"]["gDispatchSize"] = targetDim;
    var["CB"]["gSMRes"] = uint2(pImportanceMap->getWidth(), pImportanceMap->getHeight()) / (1u << mDebugSelectedMipLevel);
    var["CB"]["gScaleFactorIM"] = mDebugScaleFactorIM;
    var["CB"]["gScaleFactorSD"] = mDebugScaleFactorSD;

    var["gDebug"] = debugOut;
    var["gSampleDistribution"].setSrv(pSampleDistribution->getSRV(mDebugSelectedMipLevel, 1));
    var["gImportanceMap"].setSrv(pImportanceMap->getSRV(mDebugSelectedMipLevel, 1));
    var["gMask"] = mask;

    mpDebugShowImportancePass->execute(pRenderContext, uint3(targetDim, 1));
}
