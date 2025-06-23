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
#pragma once
#include "../DeepShadowMapMethod.h"
#include "Rendering/AccelerationStructure/CustomAccelerationStructure.h"

class DSMAccelerationStructure : public DeepShadowMapMethod
{
public:
    virtual ~DSMAccelerationStructure() = default;

    DSMAccelerationStructure(ref<Device> pDevice, ref<Scene> pScene);

    /** Generate resources needed to evaluate the Shadow Method
     */
    virtual void generate(RenderContext* pRenderContext, const RenderData& renderData) override;

    /** Returns defines needed for the method
     */
    virtual DefineList getDefines() override;

    /** Set the needed shader data for the method (textures,buffer, etc)
     */
    virtual void setShaderData(const ShaderVar& var) override;

    /** Additional mask to reject the backprojectio
     */
    virtual void setShadowMask(const ShaderVar& var, ref<Texture> maskTex, ref<Texture> maskSM, bool enable = true) override;

    /** Render UI for the method
     */
    virtual bool renderUI(Gui::Widgets& widget) override;

    /** Optional Debug pass
     */
    virtual void debugPass(RenderContext* pRenderContext, const RenderData& renderData,  ref<Texture> debugOut = nullptr, ref<Texture> colorOut = nullptr) override;

private:
    void prepareResources(RenderContext* pRenderContext);
    std::array<float4, 4> DSMAccelerationStructure::getCameraFrustumPlanes();

    //Runtime
    uint mFrameCount = 0;
    bool mUseOpaqueSM = false;  //Use opaque shadow map

    //Sync Resources
    static const uint kFramesInFlight = 3; ///< Number of frames in flight for GPU/CPU sync
    ref<GpuFence> mpFence;                 ///< Fence for CPU/GPU syncs
    uint mStagingCount = 0;

    // Accel shadow settings
    bool mUseOneAABBForAllLights = false;        //Use one data and AABB buffer
    uint mAccelApproxNumElementsPerPixel = 4u;
    std::vector<uint> mAccelShadowNumPoints;
    std::vector<uint64_t> mAccelFenceWaitValues; // Fence values forCounter sync
    uint mAccelShadowMaxNumPoints = 0;
    bool mAccelShadowUseCPUCounterOptimization = true;
    float mAccelShadowOverestimation = 1.1f;
    uint mAccelDataFormatSize = 4; // Size of the data struct for the accel data
    bool mRebuildAccelDataBuffer = true;
    bool mAccelUsePCF = false;
    RayFlags mAccelRayFlags = RayFlags::None;

    struct
    {
        bool enable = false;
        uint selectedLight = 0;
        float2 clipX = float2(0, 4096);
        float2 clipY = float2(0, 4096);
        float2 clipZ = float2(-10000.f, 10000.f);
        float blendT = 0.2f;
        uint visMode = 0;
        bool stopGeneration = false;
    } mAccelDebugShowAS;

    std::vector<ref<Buffer>> mAccelShadowAABB;                                 // For Accel AABB points
    std::vector<ref<Buffer>> mAccelShadowCounter;                              // Counter for inserting points
    std::vector<ref<Buffer>> mAccelShadowCounterCPU;                           // Counter for inserting points
    std::vector<ref<Buffer>> mAccelShadowData;                                 // Transparency Data
    std::unique_ptr<CustomAccelerationStructure> mpShadowAccelerationStrucure; // AS
    ref<Texture> mpDebugDepth;                                                 // Depth for the debug pass
    ref<Sampler> mpPointSampler;                                               // Point sampler

    RayTracingPipeline mGenAccelShadowPip; //RayTracingPipeline
    RasterPipeline mRasterShowAccelPass;
};
