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
#include "IDSMAccelerationStructure.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/SampleGenerators/HaltonSamplePattern.h"

namespace
{
    //Shader Paths
    const std::string kShaderFolder = "RenderPasses/IDSMRenderer/IDSMAccelerationStructure/";
    const std::string kGenShader = kShaderFolder + "GenerateIDSMAccelerationStructure.rt.slang";
    const std::string kShaderDebugShowShadowAccelRaster = kShaderFolder + "DebugShowShadowAccel.3d.slang";

    //UI
    const Gui::DropdownList kAccelDebugVisModes = {{0, "Transparency (Heatmap)"}, {1, "AABB index"}, {2, "Pixel"}, {3, "DepthPoints"}};
    const Gui::DropdownList kAccelDataFormat = {{1, "Uint"}, {2, "Uint2"}, {4, "Uint4"}};

}; // namespace

IDSMAccelerationStructure::IDSMAccelerationStructure(ref<Device> pDevice, ref<Scene> pScene) : DeepShadowMapMethod(pDevice, pScene)
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

    mpImportanceMapHelper = std::make_unique<ImportanceMapHelper>(mpDevice, mpScene->getLightCount(), mResolution);
}

void IDSMAccelerationStructure::prepareResources(RenderContext* pRenderContext)
{

    //This is triggered if either the resolution or number of lights changed
    if (mResolutionChanged)
    {
        mAccelShadowAABB.clear();
        mpShadowAccelerationStrucure.reset();
        //The following buffers need to be cleared when light count changes
        mAccelShadowCounter.clear();
        mAccelShadowCounterCPU.clear();
        mAccelFenceWaitValues.clear();
        mAccelShadowNumPoints.clear();
        mpImportanceMapHelper->updateResolution(mResolution);
    }

    if ((mTransparencyBufferUsesColor != mUseColoredTransparency) || mResolutionChanged)
    {
        mAccelShadowData.clear();
        mTransparencyBufferUsesColor = mUseColoredTransparency;
    }

    mResolutionChanged = false;
    updateSMMatrices();

    // Create AVSM trace program
    if (!mGenAccelShadowPip.pProgram)
    {
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kGenShader);
        desc.setMaxPayloadSize(32u); 
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(1u);

        mGenAccelShadowPip.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mGenAccelShadowPip.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("miss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("closestHit", "anyHit"));
        }

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");

        mGenAccelShadowPip.pProgram = RtProgram::create(mpDevice, desc, defines);
    }

    auto& lights = mpScene->getLights();

    // Create / Destroy resources
    {
        const uint numBuffers = lights.size();
        const uint numAccelBuffers = mUseOneAABBForAllLights ? 1 : lights.size();

        if (mAccelShadowAABB.empty())
        {
            mAccelShadowAABB.resize(numAccelBuffers);
            mAccelShadowMaxNumPoints = mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel;
            for (uint i = 0; i < numAccelBuffers; i++)
            {
                mAccelShadowAABB[i] = Buffer::createStructured(
                    mpDevice, sizeof(AABB), mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false
                );
                mAccelShadowAABB[i]->setName("AccelShadowAABB_" + std::to_string(i));
            }
        }
        // Counter
        if (mAccelShadowCounter.empty())
        {
            mAccelShadowCounter.resize(kFramesInFlight);
            mAccelShadowCounterCPU.resize(kFramesInFlight);
            mAccelFenceWaitValues.resize(kFramesInFlight);
            mAccelShadowNumPoints.resize(numAccelBuffers);

            std::vector<uint> initData(numAccelBuffers, 0);
            for (uint i = 0; i < kFramesInFlight; i++)
            {
                mAccelShadowCounter[i] = Buffer::createStructured(
                    mpDevice, sizeof(uint), numAccelBuffers, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    Buffer::CpuAccess::None, initData.data(), false
                );
                mAccelShadowCounter[i]->setName("AccelShadowAABBCounter_" + std::to_string(i));

                mAccelShadowCounterCPU[i] = Buffer::createStructured(
                    mpDevice, sizeof(uint), numAccelBuffers, ResourceBindFlags::None, Buffer::CpuAccess::Read, initData.data(), false
                );
                mAccelShadowCounterCPU[i]->setName("AccelShadowAABBCounterCPU_" + std::to_string(i));

                mAccelFenceWaitValues[i] = 0;
            }

            for (uint i = 0; i < numAccelBuffers; i++)
                mAccelShadowNumPoints[i] = mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel;
        }
        if (mAccelShadowData.empty())
        {
            mAccelShadowData.resize(numAccelBuffers);
            uint dataSize = mUseColoredTransparency ? 3 : 1;
            dataSize *= sizeof(float);
            for (uint i = 0; i < numAccelBuffers; i++)
            {
                mAccelShadowData[i] = Buffer::createStructured(
                    mpDevice, dataSize , mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false
                );       
                mAccelShadowData[i]->setName("AccelShadowData" + std::to_string(i));
            }
        }

        if (!mpShadowAccelerationStrucure)
        {
            std::vector<uint64_t> aabbCount;
            std::vector<uint64_t> aabbGPUAddress;
            for (uint i = 0; i < numAccelBuffers; i++)
            {
                aabbCount.push_back(mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel);
                aabbGPUAddress.push_back(mAccelShadowAABB[i]->getGpuAddress());
            }
            //Note: TLAS update is slightly faster (~0.02 ms) even though rebuild is usally recommended. Tracing times do not change between toggeling the update mode
            mpShadowAccelerationStrucure = std::make_unique<CustomAccelerationStructure>(
                mpDevice, aabbCount, aabbGPUAddress, CustomAccelerationStructure::BuildMode::None,
                CustomAccelerationStructure::UpdateMode::TLASOnly
            );
            mpShadowAccelerationStrucure->setMinBLASUpdateCount(kMinAABBUpdateCount);
        }

        //For stats
        if (!mpStatsRaysDistributedBuffer)
        {
            mStatsDistributedRayCountsPerLight.resize(numBuffers);
            for (auto& statsBuf : mStatsDistributedRayCountsPerLight)
                statsBuf = 0;
            mpStatsRaysDistributedBuffer = Buffer::create(
                mpDevice, sizeof(uint) * numBuffers, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                Buffer::CpuAccess::None, mStatsDistributedRayCountsPerLight.data()
            );
            mpStatsRaysDistributedBuffer->setName("StatsRaySampleCount");
            mpStatsRaysDistributedBufferCPU = Buffer::create(
                mpDevice, sizeof(uint) * numBuffers, ResourceBindFlags::None, Buffer::CpuAccess::Read,
                mStatsDistributedRayCountsPerLight.data()
            );
            mpStatsRaysDistributedBufferCPU->setName("StatsRaySampleCountCPURead");
        }
    }
}

std::array<float4, 4> IDSMAccelerationStructure::getCameraFrustumPlanes()
{
    // TODO add motion prediction
    const CameraData& data = mpScene->getCamera()->getData();
    const float fovY = focalLengthToFovY(data.focalLength, data.frameHeight);
    const float3 camU = normalize(data.cameraU);
    const float3 camV = normalize(data.cameraV);
    const float3 camW = normalize(data.cameraW);

    const float halfVSide = data.farZ * math::tan(fovY * 0.5f);
    const float halfHSide = halfVSide * data.aspectRatio;
    const float3 frontTimesFar = camW * data.farZ;

    // Frustum Planes. Data struct xyz = N ; w = distance
    std::array<float4, 4> frustumPlanes;
    // Top
    float3 N = math::normalize(math::cross(camU, frontTimesFar - camV * halfVSide));
    frustumPlanes[0] = float4(N, math::dot(N, data.posW));
    // Bottom
    N = math::normalize(math::cross(frontTimesFar + camV * halfVSide, camU));
    frustumPlanes[1] = float4(N, math::dot(N, data.posW));
    // Left
    N = math::normalize(math::cross(camV, frontTimesFar + camU * halfHSide));
    frustumPlanes[2] = float4(N, math::dot(N, data.posW));
    // Right
    N = math::normalize(math::cross(frontTimesFar - camU * halfHSide, camV));
    frustumPlanes[3] = float4(N, math::dot(N, data.posW));

    return frustumPlanes;
}

void IDSMAccelerationStructure::dummyProfileGeneration(RenderContext* pRenderContext)
{

    mpImportanceMapHelper->dummyRenderPassProfile(pRenderContext);

    {
        FALCOR_PROFILE(pRenderContext, "ClearAccelAABBBuffers");
    }
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
    {
        FALCOR_PROFILE(pRenderContext, "buildCustomBlas");
    }
    {
        FALCOR_PROFILE(pRenderContext, "buildCustomTlas");
    }
}

void IDSMAccelerationStructure::generate(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "Generate_IDSM_AS");

    // Abort early if disabled
    bool skipGeneration = (mSkipFrameCount % mSkipGenerationFrameCount) != 0;
    mSkipFrameCount++;
    if ((mAccelDebugShowAS.enable && mAccelDebugShowAS.stopGeneration) || skipGeneration)
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
    int frameInFlight = mStagingCount; // For sync if optimization is used

    int lastFrameInFlight = 0;
    lastFrameInFlight = frameInFlight - 1;
    lastFrameInFlight = lastFrameInFlight < 0 ? kFramesInFlight - 1 : lastFrameInFlight;
    uint maxRayBudget = mResolution.x * mResolution.y * mSampleOverestimate * mSampleOverestimate;
    uint maxNodeSize = mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel;                 

    //Get the sample distribution ready
    mpImportanceMapHelper->generateSampleDistribution(
        pRenderContext, mAccelShadowCounter[lastFrameInFlight], maxNodeSize, maxRayBudget, mAccelShadowCounter[frameInFlight],
        mUseOneAABBForAllLights
    );

    // Clear Counter
    uint clearSize = mUseOneAABBForAllLights ? 1 : lights.size();
    pRenderContext->clearUAV(mAccelShadowCounter[frameInFlight]->getUAV(0u, clearSize).get(), uint4(0));

    // Defines
    mGenAccelShadowPip.pProgram->addDefine("MAX_IDX", std::to_string(mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel));
    mGenAccelShadowPip.pProgram->addDefine("MIDPOINT_PERCENTAGE", std::to_string(mMidpointPercentage));
    mGenAccelShadowPip.pProgram->addDefine("MIDPOINT_DEPTH_BIAS", std::to_string(mMidpointDepthBias));
    mGenAccelShadowPip.pProgram->addDefine("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");
    mGenAccelShadowPip.pProgram->addDefine("ACCEL_USE_FRUSTUM_CULLING", mAccelUseFrustumCulling ? "1" : "0");
    mGenAccelShadowPip.pProgram->addDefine("SAMPLE_DIST_MIPS", std::to_string(mpImportanceMapHelper->getSampleDistribution(0)->getMipCount()));
    mGenAccelShadowPip.pProgram->addDefine("USE_ONE_AABB_BUFFER_FOR_ALL_LIGHTS", mUseOneAABBForAllLights ? "1" : "0");
    mGenAccelShadowPip.pProgram->addDefine("USE_HALTON_SAMPLE_PATTERN", mpHaltonBuffer ? "1" : "0");
    mGenAccelShadowPip.pProgram->addDefine("NUM_HALTON_SAMPLES", std::to_string(mJitterSampleCount));
    mGenAccelShadowPip.pProgram->addDefine("USE_RANDOM_RANDOM_SOFT_SHADOWS", mEnableRandomSoftShadows ? "1" : "0");
    mGenAccelShadowPip.pProgram->addDefine("RANDOM_SOFT_SHADOWS_POS_RADIUS", std::to_string(mRandomSoftShadowsPositionRadius));
    mGenAccelShadowPip.pProgram->addDefine("RANDOM_SOFT_SHADOWS_DIR_SPREAD", std::to_string(mRandomSoftShadowsDirSpread));
    mGenAccelShadowPip.pProgram->addDefine("TRACE_NON_OPAQUE_ONLY", mUseMask ? "1" : "0"); //Trace non-opaque only if mask is used
    mGenAccelShadowPip.pProgram->addDefine("INCLUDE_CAST_SHADOW_INSTANCE_MASK_BIT", mEnableBlacklistWithShadowMaterialFlag ? "0" : "1"); // Determines if the castShadow instance mask bit is used
    mGenAccelShadowPip.pProgram->addDefine("STATS_WRITE_TOTAL_SAMPLE_COUNT", mEnableStats ? "1" : "0");


    // Create Program Vars
    if (!mGenAccelShadowPip.pVars)
    {
        mGenAccelShadowPip.pProgram->addDefines(mpSampleGenerator->getDefines());
        mGenAccelShadowPip.pProgram->setTypeConformances(mpScene->getTypeConformances());
        mGenAccelShadowPip.pVars = RtProgramVars::create(mpDevice, mGenAccelShadowPip.pProgram, mGenAccelShadowPip.pBindingTable);
        mpSampleGenerator->setShaderData(mGenAccelShadowPip.pVars->getRootVar());
    }

    FALCOR_ASSERT(mGenAccelShadowPip.pVars);
    auto var = mGenAccelShadowPip.pVars->getRootVar();

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

        var["gAABB"] = mUseOneAABBForAllLights ? mAccelShadowAABB[0] : mAccelShadowAABB[i];
        var["gCounter"] = mAccelShadowCounter[frameInFlight];
        var["gData"] = mUseOneAABBForAllLights ? mAccelShadowData[0] : mAccelShadowData[i];
        var["gImportanceMap"] = mpImportanceMapHelper->getImportanceMap(i);
        var["gSampleDistribution"] = mpImportanceMapHelper->getSampleDistribution(i);
        var["gHaltonSamples"] = mpHaltonBuffer;
        var["gStatsTotalSampleCountBuffer"] = mpStatsRaysDistributedBuffer;

        // Get dimensions of ray dispatch.
        uint2 targetDim = uint2(float2(mResolution) * mSampleOverestimate);
                    
        FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

        // Spawn the rays.
        mpScene->raytrace(pRenderContext, mGenAccelShadowPip.pProgram.get(), mGenAccelShadowPip.pVars, uint3(targetDim, 1));

        mFrameCount++;
    }

    const uint numAABBs = mUseOneAABBForAllLights ? 1 : lights.size();

    //Copy Counter from GPU to CPU
    {
        // Copy to CPU
        pRenderContext->copyBufferRegion(
            mAccelShadowCounterCPU[mStagingCount].get(), 0, mAccelShadowCounter[mStagingCount].get(), 0, sizeof(uint32_t) * numAABBs
        );

        // Frame in flight for the counter
        mAccelFenceWaitValues[mStagingCount] = mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        mStagingCount = (mStagingCount + 1) % kFramesInFlight;

        uint64_t& fenceWaitVal = mAccelFenceWaitValues[mStagingCount];
        // Wait for the GPU to finish the frame
        mpFence->syncCpu(fenceWaitVal);

        void* data = mAccelShadowCounterCPU[mStagingCount]->map(Buffer::MapType::Read);
        std::memcpy(mAccelShadowNumPoints.data(), data, sizeof(uint) * numAABBs);
        mAccelShadowCounterCPU[mStagingCount]->unmap();
    }

    // Clear unused AABBs
    mpShadowAccelerationStrucure->clearAABBBuffers(pRenderContext, mAccelShadowAABB, true, mAccelShadowCounter[frameInFlight]);

    // Build the Acceleration structure
    std::vector<uint64_t> aabbCount;
    
    for (uint i = 0; i < numAABBs; i++)
    {
        aabbCount.push_back(mAccelShadowMaxNumPoints);
    }
    mpShadowAccelerationStrucure->update(pRenderContext, aabbCount);

    // Copy Stat Counter
    {
        // Copy to CPU
        pRenderContext->copyBufferRegion(mpStatsRaysDistributedBufferCPU.get(), 0, mpStatsRaysDistributedBuffer.get(), 0,
                                         sizeof(uint32_t) * lights.size()
        );

        void* data = mpStatsRaysDistributedBufferCPU->map(Buffer::MapType::Read);
        std::memcpy(mStatsDistributedRayCountsPerLight.data(), data, sizeof(uint) * lights.size());
        mpStatsRaysDistributedBufferCPU->unmap();
    }
}

DefineList IDSMAccelerationStructure::getDefines()
{
    DefineList defines = {};
    defines.add(DeepShadowMapMethod::getDefines());
    defines.add("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");
    defines.add("ACCEL_USE_RAY_INLINE", mAccelUseRayTracingInline ? "1" : "0");
    defines.add("ACCEL_USE_ONE_AABB_FOR_ALL_LIGHTS", mUseOneAABBForAllLights ? "1" : "0");
    return defines;
}

void IDSMAccelerationStructure::setShaderData(const ShaderVar& var)
{
    auto shadowVar = var["gIDSMAccelerationStructure"];

    shadowVar["SMCB"]["gSMSize"] = mResolution;
    shadowVar["SMCB"]["gNear"] = mNearFar.x;
    shadowVar["SMCB"]["gFar"] = mNearFar.y;
    shadowVar["SMCB"]["gMipCount"] = mpImportanceMapHelper->getSampleDistribution(0)->getMipCount();
    uint2 opaqueSMMaxDispatch = getShaderDispatchSize();
    shadowVar["SMCB"]["gMaxBufferSize"] = opaqueSMMaxDispatch.x * opaqueSMMaxDispatch.y;

    auto& lights = mpScene->getLights();
    for (uint i = 0; i < lights.size(); i++)
    {
        shadowVar["ShadowVPs"]["gShadowMapVP"][i] = mShadowMapMVP[i].viewProjection;
        shadowVar["ShadowVPs"]["gStaggeredDirVP"] = mStaggeredDirectionalLightMVP.viewProjection;
        shadowVar["gImportanceMap"][i] = mpImportanceMapHelper->getImportanceMap(i);
        shadowVar["gSampleDistribution"][i] = mpImportanceMapHelper->getSampleDistribution(i);
    }
    const auto accelDataSize = mUseOneAABBForAllLights ? 1 : lights.size();
    for (uint i = 0; i < accelDataSize; i++)
    {
        shadowVar["gAccelShadowData"][i] = mAccelShadowData[i];
        shadowVar["gShadowAABBs"][i] = mAccelShadowAABB[i];
    }

    shadowVar["gPointSampler"] = mpPointSampler;
    shadowVar["gLinearSampler"] = mpLinearSampler;

    mpShadowAccelerationStrucure->bindTlas(shadowVar, "gShadowAS");
}

void IDSMAccelerationStructure::setShadowMask(
    const ShaderVar& var,
    ref<Texture> maskTex,
    std::vector<ref<Buffer>>& maskISM,
    bool enable
)
{
    mUseMask = enable;
    if (mUseMask)
    {
        auto shadowVar = var["gIDSMAccelerationStructure"];

        shadowVar["gShadowMask"] = maskTex;
        for (uint i = 0; i < maskISM.size(); i++)
            shadowVar["gMaskShadowMap"][i] = maskISM[i];
    }
}

bool IDSMAccelerationStructure::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
    if (auto group = widget.group("IDSM-AS Settings"))
    {
        mResolutionChanged |= group.var("Node Buffer size (Resolution x this)", mAccelApproxNumElementsPerPixel, 1u, 32u, 1u);
        group.text("Note: IDSM tries to fill the buffer, so this affects quality and runtime");
        std::string bufferSize = "Buffer Elements: " + std::to_string(mResolution.x * mResolution.y * mAccelApproxNumElementsPerPixel);
        group.text(bufferSize);

        group.var("Ray Budget Multiplier (Resolution x this)", mSampleOverestimate, 1.0f, 4.f);
        group.tooltip("Controls the maximum dispatch size of the IDSM generation shader. Defines the upper limit for the budget distribution. \n"
            "Max Dispatch Size: [SMRes.x * Overestimate , SMRes.y * Overestimate]");

        mpImportanceMapHelper->renderUI(group);

        mResolutionChanged |= group.checkbox("Use one Acceleration Structure for all IDSMs", mUseOneAABBForAllLights);
        group.tooltip("Uses one AABB Buffer (and therefore BLAS) for all lights. Light coordinates are put side by side on the x axis");

        if (auto statsGroup = group.group("Stats"))
        {
            if (mpScene)
            {
                if (auto group2 = statsGroup.group("Buffer Size Infos"))
                {
                    const auto loopSize = mUseOneAABBForAllLights ? 1 : mpScene->getLightCount();
                    for (uint i = 0; i < loopSize; i++)
                    {
                        if (i > 0)
                            group2.separator();
                        uint dataBufferSize = mUseColoredTransparency ? 12u : 4u;
                        group2.text(mUseOneAABBForAllLights ? "Total" : mpScene->getLight(i)->getName());
                        group2.text("Buffer Size:        " + std::to_string(mAccelShadowMaxNumPoints));
                        float accelMem = (mAccelShadowMaxNumPoints * sizeof(AABB)) / 1e6f;
                        float dataMem = (mAccelShadowMaxNumPoints * dataBufferSize) / 1e6f;
                        std::string accelMemStr = std::to_string(accelMem);
                        std::string dataMemStr = std::to_string(dataMem);
                        std::string totalMemStr = std::to_string((accelMem + dataMem));
                        group2.text("AABB Memory:     " + accelMemStr.substr(0, accelMemStr.find(".") + 3) + " MB");
                        group2.text("Data Memory:     " + dataMemStr.substr(0, dataMemStr.find(".") + 3) + " MB");
                        group2.text("Total Memory:    " + totalMemStr.substr(0, totalMemStr.find(".") + 3) + " MB");

                        group2.text("Used Elements:    " + std::to_string(uint(mAccelShadowNumPoints[i])));
                        accelMem = (mAccelShadowNumPoints[i] * sizeof(AABB)) / 1e6f;
                        dataMem = (mAccelShadowNumPoints[i] * dataBufferSize) / 1e6f;
                        std::string neededAABBMem = std::to_string(accelMem);
                        std::string neededDataMem = std::to_string(dataMem);
                        std::string neededTotalMem = std::to_string(accelMem + dataMem);
                        std::string fillRate = std::to_string(((mAccelShadowNumPoints[i]) / float(mAccelShadowMaxNumPoints)) * 100.f);
                        group2.text("Used AABB Memory:   " + neededAABBMem.substr(0, neededAABBMem.find(".") + 3) + " MB");
                        group2.text("Used Data Memory:   " + neededDataMem.substr(0, neededDataMem.find(".") + 3) + " MB");
                        group2.text(
                            "Used Total Memory:   " + neededTotalMem.substr(0, neededTotalMem.find(".") + 3) + " MB (" +
                            fillRate.substr(0, fillRate.find(".") + 2) + "%)"
                        );
                    }
                    group2.separator();
                }

                if (auto group2 = statsGroup.group("Ray Sample Count Info"))
                {
                    mEnableStats = true;
                    uint total = 0;
                    for (const auto& count : mStatsDistributedRayCountsPerLight)
                        total += count;
                    group2.text("Total Count: " + std::to_string(total));
                    if (mpScene->getLightCount() > 1)
                    {
                        for (uint i = 0; i < mpScene->getLightCount(); i++)
                        {
                            group2.text(mpScene->getLight(i)->getName() + ": " + std::to_string(mStatsDistributedRayCountsPerLight[i]));
                        }
                    }
                }
                else
                {
                    mEnableStats = false;
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

           group2.checkbox("Enable Accelerations Structure Visualization", mAccelDebugShowAS.enable);
           if (mAccelDebugShowAS.enable && mpScene)
           {
               if (auto group3 = group2.group("AS Visualization Settings", true))
               {
                   group3.text("Enable \"List All Outputs\" and switch \"Output\" to ");
                   group3.text("\"IDSMRenderer.outDebug\". For better visibility stop");
                   group3.text("the time (Spacebar) and check the \"Stop Generation\" checkbox below.");
                   if (mpScene->getLightCount() > 1 && !mUseOneAABBForAllLights)
                       group3.slider("Selected Light", mAccelDebugShowAS.selectedLight, 0u, mpScene->getLightCount() - 1);
                   group3.var("Clip X", mAccelDebugShowAS.clipX, 0.f, float(mResolution.x), 0.1f);
                   group3.var("Clip Y", mAccelDebugShowAS.clipY, 0.f, float(mResolution.y), 0.1f);
                   group3.var("Clip Z", mAccelDebugShowAS.clipZ, -FLT_MAX, FLT_MAX, 0.1f);

                   group3.var("Blend with Output", mAccelDebugShowAS.blendT, 0.f, 1.f, 0.001f);
                   if (group3.dropdown("Mode", kAccelDebugVisModes, mAccelDebugShowAS.visMode))
                       mAccelDebugShowAS.stopGeneration = mAccelDebugShowAS.visMode == 1 ? true : mAccelDebugShowAS.stopGeneration;
                   group3.checkbox("Stop Generation", mAccelDebugShowAS.stopGeneration);
               }
           }
        }

    }
   
    return dirty;
}

void IDSMAccelerationStructure::debugPass(
    RenderContext* pRenderContext,
    const RenderData& renderData,
    ref<Texture> debugOut,
    ref<Texture> colorOut
)
{
    // Early return if disabled
    if (!mAccelDebugShowAS.enable)
        return;

    FALCOR_PROFILE(pRenderContext, "ShowAccel");

    const uint2 dims = renderData.getDefaultTextureDims();
    if (!mpDebugDepth || math::any(uint2(mpDebugDepth->getWidth(), mpDebugDepth->getHeight()) != dims))
    {
        mpDebugDepth =
            Texture::create2D(mpDevice, dims.x, dims.y, ResourceFormat::D32Float, 1u, 1u, nullptr, ResourceBindFlags::DepthStencil);
        mpDebugDepth->setName("DebugRasterDepth");
    }

    // Init Program
    if (!mRasterShowAccelPass.pProgram)
    {
        // Init program
        Program::Desc desc;
        desc.addShaderLibrary(kShaderDebugShowShadowAccelRaster).vsEntry("vsMain").psEntry("psMain").gsEntry("gsMain");
        desc.setShaderModel("6_6");

        auto defines = mpScene->getSceneDefines();
        defines.add("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");
        defines.add("USE_ONE_AABB_FOR_ALL", mUseOneAABBForAllLights ? "1" : "0");
        defines.add("COUNT_LIGHTS", std::to_string(mpScene->getLightCount()));
        // Create Program and state
        mRasterShowAccelPass.pProgram = GraphicsProgram::create(mpDevice, desc, defines);
        mRasterShowAccelPass.pState = GraphicsState::create(mpDevice);

        // Set state
        mRasterShowAccelPass.pState->setProgram(mRasterShowAccelPass.pProgram);
        mRasterShowAccelPass.pState->setVao(Vao::create(Vao::Topology::PointList));

        // Set raster state
        RasterizerState::Desc rsDesc;
        rsDesc.setCullMode(RasterizerState::CullMode::None);
        rsDesc.setFillMode(RasterizerState::FillMode::Solid);
        mRasterShowAccelPass.pState->setRasterizerState(RasterizerState::create(rsDesc));

        mRasterShowAccelPass.pFBO = Fbo::create(mpDevice);
    }

    // Set draw target
    mRasterShowAccelPass.pFBO->attachColorTarget(debugOut, 0);
    mRasterShowAccelPass.pFBO->attachDepthStencilTarget(mpDebugDepth);
    pRenderContext->clearFbo(mRasterShowAccelPass.pFBO.get(), float4(0, 0, 0, 1), 1.0, 0);
    mRasterShowAccelPass.pState->setFbo(mRasterShowAccelPass.pFBO);

    // Runtime Defines
    mRasterShowAccelPass.pProgram->addDefine("USE_COLOR_TRANSPARENCY", mUseColoredTransparency ? "1" : "0");
    mRasterShowAccelPass.pProgram->addDefine("USE_ONE_AABB_FOR_ALL", mUseOneAABBForAllLights ? "1" : "0");

    // Vars
    if (!mRasterShowAccelPass.pVars)
        mRasterShowAccelPass.pVars = GraphicsVars::create(mpDevice, mRasterShowAccelPass.pProgram.get());

    uint frameInFlight = 0;
    // Staging count was increased at the end of the generation code, so take one less
    frameInFlight = mStagingCount == 0 ? kFramesInFlight - 1 : mStagingCount - 1;

    //Get index of directional light if the scene contains it
    uint directionalIndex = UINT_MAX;
    auto& lights = mpScene->getLights();
    for (uint i = 0; i <lights.size(); i++)
    {
        if (lights[i]->getType() == LightType::Directional)
        {
            directionalIndex = i;
            break;
        }
    }

    auto var = mRasterShowAccelPass.pVars->getRootVar();

    var["gScene"] = mpScene->getParameterBlock();
    var["CB"]["gSMSize"] = mResolution;
    var["CB"]["gNear"] = mNearFar.x;
    var["CB"]["gFar"] = mNearFar.y;
    var["CB"]["gSelectedLight"] = mUseOneAABBForAllLights ? 0 : mAccelDebugShowAS.selectedLight;
    var["CB"]["gCullMin"] = float3(mAccelDebugShowAS.clipX.x, mAccelDebugShowAS.clipY.x, mAccelDebugShowAS.clipZ.x);
    var["CB"]["gCullMax"] = float3(mAccelDebugShowAS.clipX.y, mAccelDebugShowAS.clipY.y, mAccelDebugShowAS.clipZ.y);
    var["CB"]["gBlendT"] = mAccelDebugShowAS.blendT;
    var["CB"]["gVisMode"] = mAccelDebugShowAS.visMode;
    var["CB"]["gDirectionalIdx"] = directionalIndex;

    //Set the viewProj matrices
    for (uint i = 0; i < mpScene->getLightCount(); i++)
    {
        var["LightMatrices"]["gInvView"][i] = mShadowMapMVP[i].invView;
        var["LightMatrices"]["gInvProj"][i] = mShadowMapMVP[i].invProjection;
        var["InvViewProjections"]["gInvViewProj"][i] = mShadowMapMVP[i].invViewProjection;
    }

    var["gShadowAABB"] = mAccelShadowAABB[mAccelDebugShowAS.selectedLight];
    var["gShadowCounter"] = mAccelShadowCounter[frameInFlight];
    var["gShadowData"] = mAccelShadowData[mAccelDebugShowAS.selectedLight];
    var["gOutputColor"] = colorOut; // For blending

    pRenderContext->draw(mRasterShowAccelPass.pState.get(), mRasterShowAccelPass.pVars.get(), mAccelShadowMaxNumPoints, 0);
}
