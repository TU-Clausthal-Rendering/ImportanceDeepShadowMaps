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
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "ParticleDataTypes.slang"

using namespace Falcor;

class ParticlePass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ParticlePass, "ParticlePass", "Animates the Particle Systems for a scene");
        
    struct ParticleSettings
    {
        float lifetime = 500.f;
        float randomLifetime = 30.f;
        float radius = 1.f;
        float randomRadiusOffset = 0.5f;
        float3 spawnPosition = float3(0, 0, 0);
        float3 initialVelocityDirection = float3(0, 1, 0);
        float initialVelocityStrength = 1.0;
        float velocityRandom = 0.5;
        float gravity = 0.1f;
        float gravityRandom = 0.01f;
        float spawnRadius = 0.1f;
        float spreadAngle = 1.2f;
        float3 windDirection = float3(0,1,0);
        float windStrength = 0.f;
        float shirnkAtEnd = -1.f;
    };

    static ref<ParticlePass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ParticlePass>(pDevice, props); }

    ParticlePass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    
    void storeCurrentConfiguration();
    void refreshFileList();
    bool loadConfigurationFile(std::string filename);

    /** Dispatches the particle pass
    */
    void dispatchParticlePass(RenderContext* pRenderContext, float deltaT);

    /** Fills the initial data buffer. Vector has the size of the global particle buffer
      * If index is >=0 only data of the responding particle system is filled
    */
    std::vector<ParticleAnimateData> getInitialData(int index = -1);

    double lastFrameTime = 0.0; //For clock
    uint mFrameCounter = 0;             //Frame Counter for the sample generator
    bool mReinitializeBuffer = false;   //Resets the buffer data
    bool mReset = false;                //Resets all Particle simimulations
    bool mPaused = false;                 //Pause simulation
    bool mDisablePause = false;         //Disable Pause (for video recording)

    bool mEnableSimulateOnStartup = true;     //Simumlates the max lifetime on startup
    bool mUseSimulation = true;         //Simulate the max lifetime
    uint mSimulationSteps = 100;        //Number of iterations for the simulation (deltaT is maxLifetime/steps)

    ref<Scene> mpScene; //reference to the scene
    ref<SampleGenerator> mpSampleGenerator; ///< GPU sample generator.
    std::vector<ParticleSettings> mParticleSettings;
    std::vector<uint> mParticleBufferOffsets;   //Offset in the global particle buffer
    uint mTotalBufferSize = 0;

    std::string mConfigurationName = "ParticleSettings";
    std::filesystem::path mScenePath;
    Gui::DropdownList mFileList;    //List of all files in the scene
    uint mSelectedFile = 0;


    ref<Buffer> mpParticleAnimateDataBuffer; //Extra date needed to animate the Particle Points
    ref<ComputePass> mpUpdateParticlePointsPass; //Compute pass to update the particle points
};
