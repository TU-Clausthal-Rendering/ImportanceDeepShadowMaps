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
#include "ParticlePass.h"
#include "Utils/Timing/Clock.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <random>

using json = nlohmann::json;

//Float3 json defines
namespace Falcor::math
{
    void to_json(json& j, const float3& v)
    {
        j = {v.x, v.y, v.z};
    }

    void from_json(const json& j, float3& v)
    {
        j[0].get_to(v.x);
        j[1].get_to(v.y);
        j[2].get_to(v.z);
    }
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ParticlePass>();
}

namespace
{
    const std::string kShaderUpdateParticles = "RenderPasses/ParticlePass/UpdateParticlePoints.cs.slang";
    const std::string kShaderModel = "6_6";

    //JSON Keys
    const std::string kJSONKeyLifetime = "lifetime";
    const std::string kJSONKeyLifetimeRandom = "lifetimeRandom";
    const std::string kJSONKeyRadius = "radius";
    const std::string kJSONKeyRandomRadius = "rndRadius";
    const std::string kJSONKeySpawnPosition = "spawnPosition";
    const std::string kJSONKeyInitialVelocity = "initialVelocity";
    const std::string kJSONKeyRandomVelocity = "randomVelocity";
    const std::string kJSONKeyGravity = "gravity";
    const std::string kJSONKeyGravityRandom = "gravityRandom";
    const std::string kJSONKeySpawnRadius = "spawnRadius";
    const std::string kJSONKeySpreadAngle = "spreadAngle";
    const std::string kJSONKeyWind = "wind";
    const std::string kJSONKeyShrinkAtEnd = "shrinkAtEnd";
}

//Json defines for settings type
void from_json(const json& j, ParticlePass::ParticleSettings& settings) {
    if (j.contains(kJSONKeyLifetime)) j[kJSONKeyLifetime].get_to(settings.lifetime);
    if (j.contains(kJSONKeyLifetimeRandom)) j[kJSONKeyLifetimeRandom].get_to(settings.randomLifetime);
    if (j.contains(kJSONKeyRadius)) j[kJSONKeyRadius].get_to(settings.radius);
    if (j.contains(kJSONKeyRandomRadius)) j[kJSONKeyRandomRadius].get_to(settings.randomRadiusOffset);
    if (j.contains(kJSONKeySpawnPosition)) j[kJSONKeySpawnPosition].get_to(settings.spawnPosition);
    if (j.contains(kJSONKeyInitialVelocity))
    {
        j[kJSONKeyInitialVelocity].get_to(settings.initialVelocityDirection);
        settings.initialVelocityStrength = math::length(settings.initialVelocityDirection);
        settings.initialVelocityDirection =
            settings.initialVelocityStrength > 0.f ? settings.initialVelocityDirection / settings.initialVelocityStrength : float3(0, 1, 0);
    }
    if (j.contains(kJSONKeyRandomVelocity)) j[kJSONKeyRandomVelocity].get_to(settings.velocityRandom);
    if (j.contains(kJSONKeyGravity)) j[kJSONKeyGravity].get_to(settings.gravity);
    if (j.contains(kJSONKeyGravityRandom)) j[kJSONKeyGravityRandom].get_to(settings.gravityRandom);
    if (j.contains(kJSONKeySpawnRadius)) j[kJSONKeySpawnRadius].get_to(settings.spawnRadius);
    if (j.contains(kJSONKeySpreadAngle))j[kJSONKeySpreadAngle].get_to(settings.spreadAngle);
    if (j.contains(kJSONKeyWind)) {
        j[kJSONKeyWind].get_to(settings.windDirection);
        settings.windStrength = math::length(settings.windDirection);
        settings.windDirection =
            settings.windStrength > 0.f ? settings.windDirection / settings.windStrength : float3(0, 1, 0);
    }
    if (j.contains(kJSONKeyShrinkAtEnd))j[kJSONKeyShrinkAtEnd].get_to(settings.shirnkAtEnd);
 }

 void to_json(json& j, const ParticlePass::ParticleSettings& settings)
{
    j[kJSONKeyLifetime] = settings.lifetime;
    j[kJSONKeyLifetimeRandom] = settings.randomLifetime;
    j[kJSONKeyRadius] = settings.radius;
    j[kJSONKeyRandomRadius] = settings.randomRadiusOffset;
    j[kJSONKeySpawnPosition] = settings.spawnPosition;
    j[kJSONKeyInitialVelocity] = settings.initialVelocityDirection * settings.initialVelocityStrength;
    j[kJSONKeyRandomVelocity] = settings.velocityRandom;
    j[kJSONKeyGravity] = settings.gravity;
    j[kJSONKeyGravityRandom] = settings.gravityRandom;
    j[kJSONKeySpawnRadius] = settings.spawnRadius;
    j[kJSONKeySpreadAngle] = settings.spreadAngle;
    j[kJSONKeyWind] = settings.windDirection * settings.windStrength;
    j[kJSONKeyShrinkAtEnd] = settings.shirnkAtEnd;
 }

ParticlePass::ParticlePass(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

Properties ParticlePass::getProperties() const
{
    return {};
}

RenderPassReflection ParticlePass::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    return reflector;
}

void ParticlePass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) {
    if (pScene)
    {
        mReinitializeBuffer = true;
        mpScene = pScene;
        mScenePath = mpScene->getPath().parent_path();
        refreshFileList(); //Get possible configuration

        // Reset old buffers
        mParticleSettings.clear();
        mParticleBufferOffsets.clear();
        mpParticleAnimateDataBuffer.reset();

        //Set all particles in the scene to active and initialize default settings
        auto& particleSystems = mpScene->getParticleSystem();
        mTotalBufferSize = 0;
        for (auto& ps : particleSystems)
        {
            ps.active = true;
            ParticleSettings particleSetting{};
            particleSetting.spawnPosition = ps.spawnPosition;
            particleSetting.radius = ps.intitialRadius;
            mParticleSettings.push_back(particleSetting);
            mParticleBufferOffsets.push_back(mTotalBufferSize);
            mTotalBufferSize += ps.numberParticles;
        }

        //Load Settings from file if exist
        if (mTotalBufferSize > 0 && !mFileList.empty())
        {
            loadConfigurationFile(mFileList[0].label);
        }
            
        if (mTotalBufferSize > 0)
        {
            //Set initial data for the buffer
            std::vector<ParticleAnimateData> initialData = getInitialData();

            //Create buffer
            mpParticleAnimateDataBuffer = Buffer::createStructured(
                mpDevice, sizeof(ParticleAnimateData), mTotalBufferSize, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                Buffer::CpuAccess::None, initialData.data(), false
            );
            mpParticleAnimateDataBuffer->setName("ParticlePass::ParticleAnimateData");
        }
    }
        
}

void ParticlePass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    //if scene is not set do nothing
    if (!mpScene)
        return;

    auto& particleSystems = mpScene->getParticleSystem();

    //Check if particle system is set, else return
    if (particleSystems.empty())
        return;

    if (mReinitializeBuffer)
    {
        std::vector<ParticleAnimateData> initialData = getInitialData();
        mpParticleAnimateDataBuffer->setBlob(initialData.data(), 0, mTotalBufferSize * sizeof(ParticleAnimateData));
        pRenderContext->flush(true); //Ensures that the data is uploaded
        mReset = !mEnableSimulateOnStartup; //Normal reset is not needed with simulation
        mReinitializeBuffer = false;
        mUseSimulation = mEnableSimulateOnStartup;
    }

    //Update Point pass
    if (!mpUpdateParticlePointsPass)
    {
        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderUpdateParticles).csEntry("main").setShaderModel(kShaderModel);

        DefineList defines;
        defines.add(mpScene->getSceneDefines());
        defines.add(mpSampleGenerator->getDefines());

        mpUpdateParticlePointsPass = ComputePass::create(mpDevice, desc, defines, true);   
    }

    //Get deltaT
    auto& renderDict = renderData.getDictionary();
    auto pGlobalClock = static_cast<Clock*>(renderDict[kRenderGlobalClock]);
    double currentTime = pGlobalClock->getTime();

    float deltaT = static_cast<float>(math::clamp(currentTime - lastFrameTime,0.0000001, 1.0)); //Cap deltaT at 1000ms
    lastFrameTime = currentTime;

    //Handle Pause
    const auto& camera = mpScene->getCamera();
    auto cameraChanges = camera->getChanges();
    auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
    bool cameraMoved = (cameraChanges & ~excluded) != Camera::Changes::None;

    bool isPaused = pGlobalClock->isPaused() || mPaused;
    isPaused &= !mDisablePause;
    for (auto& ps : particleSystems)
        ps.paused = isPaused && !cameraMoved;

    if (isPaused)
        return;
        
    //Get Simulated deltaT instead
    if (mUseSimulation)
    {
        //Get max lifetime of all particle systems
        float maxLifetime = 0;
        for (const auto& pSett : mParticleSettings)
            maxLifetime = math::max(maxLifetime, pSett.lifetime + pSett.randomLifetime);
        deltaT = (maxLifetime * 1.1f) / float(mSimulationSteps);
    }

    //Dispatch either once of multiple times
    uint dispatchCount = mUseSimulation ? mSimulationSteps : 1;
    for (uint i = 0; i < dispatchCount; i++)
        dispatchParticlePass(pRenderContext, deltaT);
        
    mUseSimulation = false;
}

void ParticlePass::dispatchParticlePass(RenderContext* pRenderContext, float deltaT) {
    FALCOR_PROFILE(pRenderContext, "UpdateParticlePoints");

    FALCOR_ASSERT(mpUpdateParticlePointsPass);
    // Get Resources
    auto& pParticlePointsBuffer = mpScene->getParticlePointsBuffer();
    const auto& particleSystems = mpScene->getParticleSystem();

    auto var = mpUpdateParticlePointsPass->getRootVar();
    // Set Shader data valid for all particle spawners
    mpSampleGenerator->setShaderData(var);
    var["gParticlePointDesc"] = pParticlePointsBuffer;
    var["gParticleAnimateData"] = mpParticleAnimateDataBuffer;

     // One dispatch per particle system
    for (uint i = 0; i < particleSystems.size(); i++)
    {
        auto& ps = particleSystems[i];
        auto& pSett = mParticleSettings[i];

        if (!ps.active)
            continue;
        FALCOR_PROFILE(pRenderContext, ps.name);
        // Set constant buffer
        var["CB"]["gReset"] = mReset;
        var["CB"]["gFrameCount"] = mFrameCounter;
        var["CB"]["gParticleBufferOffset"] = ps.particleBufferOffset;
        var["CB"]["gNumParticles"] = ps.numberParticles;
        var["CB"]["gRestPosition"] = ps.spawnPosition;
        var["CB"]["gBaseRadius"] = pSett.radius;
        var["CB"]["gRadiusOffset"] = pSett.randomRadiusOffset;

        var["CB"]["gDeltaT"] = deltaT;
        var["CB"]["gInitialPosition"] = pSett.spawnPosition;
        var["CB"]["gInitialVelocity"] = pSett.initialVelocityDirection * pSett.initialVelocityStrength;
        var["CB"]["gVelocityRandom"] = pSett.velocityRandom;
        var["CB"]["gMaxLifetime"] = pSett.lifetime;
        var["CB"]["gRandomLifetime"] = pSett.randomLifetime;
        var["CB"]["gGravity"] = pSett.gravity;
        var["CB"]["gGravityRandom"] = pSett.gravityRandom;
        var["CB"]["gSpawnRadius"] = pSett.spawnRadius;
        var["CB"]["gSpreadAngle"] = pSett.spreadAngle;
        var["CB"]["gWind"] = pSett.windDirection * pSett.windStrength;
        var["CB"]["gShrinkAtEnd"] = pSett.shirnkAtEnd;

        mpUpdateParticlePointsPass->execute(pRenderContext, uint3(ps.numberParticles, 1, 1));
        mFrameCounter++;
    }

    pRenderContext->uavBarrier(pParticlePointsBuffer.get());
    pRenderContext->uavBarrier(mpParticleAnimateDataBuffer.get());
    mReset = false;
}

void ParticlePass::renderUI(Gui::Widgets& widget)
{
    if (!mpScene)
    {
        widget.text("Please load in a scene for options to appear");
        return;
    }
        
    auto& particleSystems = mpScene->getParticleSystem();
    if (particleSystems.empty())
    {
        widget.text(
            "There is no particle system in the current scene \n Please add one or more in the .pyscene with "
            "\"sceneBuilder.addParticleSystem(name, material, numberOfParticles, restPosition)\""
        );
        return;
    }

    auto labelText = [](std::string text, uint idx) {
        text = text + "##" + std::to_string(idx);
        return text;
    };

    widget.text("Particle Systems:");
    for (uint i=0; i<particleSystems.size(); i++)
    {
        auto& ps = particleSystems[i];
        auto& pSett = mParticleSettings[i];
        if (auto group = widget.group(ps.name))
        {
            group.checkbox(labelText("Enable", i).c_str(), ps.active);
            group.text("Max Number of Particles: " + std::to_string(ps.numberParticles));
            group.var(labelText("BaseRadius", i).c_str(), pSett.radius, 1e-7f, FLT_MAX, 0.001f);
            group.var(labelText("Random Radius Offset", i).c_str(), pSett.randomRadiusOffset, 0.f, FLT_MAX, 0.001f);
            group.tooltip("Random radius offset applied when spawing the particle. BaseRadius + [-offset, offset]");
            group.var(labelText("RestPosition", i).c_str(), ps.spawnPosition);
            group.tooltip("Position where all inactive particles are moved");

            group.var(labelText("Lifetime", i).c_str(), pSett.lifetime, 0.0f, FLT_MAX, 0.1f);
            group.tooltip("Time a particle lives");
            group.var(labelText("Lifetime Random", i).c_str(), pSett.randomLifetime, 0.f, FLT_MAX, 0.1f);
            group.tooltip("Random Offset for the lifetime. Lifetime + [-rnd, rnd]");
            group.var(labelText("SpawnPosition", i).c_str(), pSett.spawnPosition);
            group.tooltip("Position where a active particle spawns. A particles spawns if their current lifetime exeeds the Lifetime");
            if(group.var(labelText("InitialVelocityDirection", i).c_str(), pSett.initialVelocityDirection))
                pSett.initialVelocityDirection = math::normalize(pSett.initialVelocityDirection);
            group.var(labelText("InitialVelocityStrength", i).c_str(), pSett.initialVelocityStrength, 0.f);
            group.var(labelText("VelocityRandomOffset", i).c_str(), pSett.velocityRandom, 0.f, FLT_MAX, 0.001f);
            group.tooltip("Random Factor for Velocity applied when spawning the particle. Random Velocity between Velocity*[-Offset, Offset] is added to the initial velocity. Set to 0 to disable");
            group.var(labelText("Gravity", i).c_str(), pSett.gravity);
            group.var(labelText("Gravity Random", i).c_str(), pSett.gravityRandom, 0.f, FLT_MAX, 0.001f);
            group.tooltip("Absolute Random offset applied to gravity. Gravity + [-random,random]"); //TODO change to mass?
            group.var(labelText("SpawnRadius", i).c_str(), pSett.spawnRadius, 0.f);
            group.var(labelText("SpreadAngle", i).c_str(), pSett.spreadAngle, 0.f, static_cast<float>(M_PI) * 2.f, 0.001f);
            if (group.var(labelText("Wind Direction", i).c_str(), pSett.windDirection))
                pSett.windDirection = math::normalize(pSett.windDirection);
            group.var(labelText("Wind Stength", i).c_str(), pSett.windStrength, 0.f, FLT_MAX, 0.001f);
            group.var(labelText("ShrinkAtEnd", i).c_str(), pSett.shirnkAtEnd, -1.f, pSett.lifetime, 0.001f);
            group.tooltip("Shrinks the particle radius lineary at the end if lifetime of the particle is smaller than this value.");
        }
    }
    widget.separator();

    mReinitializeBuffer |= widget.button("Reset All");

    widget.checkbox("Enable Simulation on Reset", mEnableSimulateOnStartup);
    if (mEnableSimulateOnStartup)
    {
        widget.var("Simulation Steps", mSimulationSteps, 1u, UINT_MAX, 1u);
        widget.tooltip(
            "Defines how many steps the simulation has. DeltaT for Simulation is determined by the max of all ParticleSystem Lifetimes / "
            "steps."
        );
    }

    if (!mFileList.empty())
    {
        widget.dropdown("Particle Configs", mFileList, mSelectedFile);
        if (widget.button("Load Config File"))
            mReinitializeBuffer |= loadConfigurationFile(mFileList[mSelectedFile].label);
    }
        

    widget.textbox("Particle Config Name", mConfigurationName);
    bool storeConfig = widget.button("Store Current Configuration");
    if (storeConfig)
        storeCurrentConfiguration();

    widget.checkbox("Disable Pause", mDisablePause);
    if (!mDisablePause)
        widget.checkbox("Pause Simulation", mPaused);
}

void ParticlePass::storeCurrentConfiguration() {
    //Do nothing if scene is not set
    if (!mpScene)
        return;

    auto& particleSystems = mpScene->getParticleSystem();
    if (particleSystems.empty())
        return;

    //Check if name is empty and set a default name
    std::string fileName = mConfigurationName;
    if (fileName.empty())
        fileName = "ParticleSettings";
    fileName += ".prtsett";

    auto pathToFile = mScenePath;
    pathToFile.append(fileName);

    std::ofstream ofs(pathToFile);
    if (!ofs.good())
    {
        logWarning("Failed to open particle settings file '{}' for writing.", pathToFile);
        return;
    }

    json j;
    for (uint i = 0; i < mParticleSettings.size(); i++)
        j[particleSystems[i].name] = mParticleSettings[i];

    ofs << j.dump(4);
    ofs.close();

    logInfo("Successfully stored Particle Configuration at: '{}'", pathToFile);
    refreshFileList();
}

bool checkIfValidConfigurationFile(ref<Scene> pScene, const std::filesystem::path& pathToFile) {
    std::ifstream ifs(pathToFile);
    if (!ifs.good())
    {
        logWarning("Failed to open Particle Settings file '{}' for reading.", pathToFile);
        return false;
    }

    bool valid = true;
    auto& particleSystems = pScene->getParticleSystem();
    try
    {
        json j = json::parse(ifs);
        for (uint i = 0; i < particleSystems.size(); i++)
        {
            // Check if there is a entry with the particle system name
            if (!j.contains(particleSystems[i].name))
                valid = false;
        }
    }
    catch (const std::exception& e)
    {
        logWarning("Error when deserializing Particle Setting from '{}': {}", pathToFile, e.what());
        return false;
    }

    return valid;
}

void ParticlePass::refreshFileList()
{
    if (!mpScene)
        return;

    mFileList.clear();
    Gui::DropdownValue v;
    v.value = 0;

    std::vector <std::string> validFiles;

    for (const auto& entry : std::filesystem::directory_iterator(mScenePath))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".prtsett")
        {
            if (checkIfValidConfigurationFile(mpScene, entry.path()))
            {
                v.label = entry.path().filename().replace_extension().string();
                mFileList.push_back(v);
                ++v.value;
            }
        }
    }
    mSelectedFile = 0;
}

bool ParticlePass::loadConfigurationFile(std::string filename) {
    if (!mpScene)
        return false;

    auto& particleSystems = mpScene->getParticleSystem();
    FALCOR_ASSERT(particleSystems.size() == mParticleSettings.size()); //We assume that both the particle system and settings are initialized

    //Create the path from the filename
    FALCOR_ASSERT(!filename.empty());
    filename += ".prtsett";
    auto pathToFile = mScenePath;
    pathToFile.append(filename);

    std::ifstream ifs(pathToFile);
    if (!ifs.good())
    {
        logWarning("Failed to open Particle Settings file '{}' for reading.", pathToFile);
        return false;
    }

    //Parse json
    try
    {
        json j = json::parse(ifs);
        for (uint i = 0; i < particleSystems.size(); i++)
        {
            //Check if there is a entry with the particle system name
            if (j.contains(particleSystems[i].name))
            {
                mParticleSettings[i] = j[particleSystems[i].name];
            }
        }
    }
    catch (const std::exception& e)
    {
        logWarning("Error when deserializing Particle Setting from '{}': {}", pathToFile, e.what());
        return false;
    }

    return true;
}

std::vector<ParticleAnimateData> ParticlePass::getInitialData(int index)
{
    std::vector<ParticleAnimateData> initialData(mTotalBufferSize);
    auto& particleSystems = mpScene->getParticleSystem();

    uint offset = index >= 0 ? mParticleBufferOffsets[index] : 0;
    uint i = index >= 0 ? index : 0;
    uint size = index >= 0 ? index + 1 : particleSystems.size();

    //Init random number generator
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(-1.0, std::nextafter(1.0,DBL_MAX)); //Random Value in [-1,1]

    for (i; i < size; i++)
    {
        auto& ps = particleSystems[i];
        auto& pSett = mParticleSettings[i];
        float combinedLifetime = pSett.lifetime + pSett.randomLifetime;
        for (uint j = 0; j < ps.numberParticles; j++)
        {
            ParticleAnimateData data{};
            data.lifetime = combinedLifetime * static_cast<float>(dist(mt));
            data.velocity = float3(0);
            data.isValid = false;
            initialData[j + offset] = data;
        }
        offset += ps.numberParticles;
    }

    return initialData;
}
