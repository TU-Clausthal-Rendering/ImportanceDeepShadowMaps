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
#include "Scene.h"
#include "SceneDefines.slangh"
#include "SceneBuilder.h"
#include "Importer.h"
#include "Curves/CurveConfig.h"
#include "SDFs/SDFGrid.h"
#include "SDFs/NormalizedDenseSDFGrid/NDSDFGrid.h"
#include "SDFs/SparseBrickSet/SDFSBS.h"
#include "SDFs/SparseVoxelOctree/SDFSVO.h"
#include "SDFs/SparseVoxelSet/SDFSVS.h"
#include "Core/API/Device.h"
#include "Core/API/RenderContext.h"
#include "Core/API/IndirectCommands.h"
#include "Utils/StringUtils.h"
#include "Utils/ObjectIDPython.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/MathHelpers.h"
#include "Utils/Math/Vector.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/UI/InputTypes.h"
#include "Utils/Scripting/ScriptWriter.h"

#include <fstream>
#include <numeric>
#include <sstream>

namespace Falcor
{
    static_assert(sizeof(MeshDesc) % 16 == 0, "MeshDesc size should be a multiple of 16");
    static_assert(sizeof(GeometryInstanceData) == 32, "GeometryInstanceData size should be 32");
    static_assert(sizeof(PackedStaticVertexData) % 16 == 0, "PackedStaticVertexData size should be a multiple of 16");

    namespace
    {
        // Large scenes are split into multiple BLAS groups in order to reduce build memory usage.
        // The target is max 0.5GB intermediate memory per BLAS group. Note that this is not a strict limit.
        const size_t kMaxBLASBuildMemory = 1ull << 29;

        const std::string kParameterBlockName = "gScene";
        const std::string kGeometryInstanceBufferName = "geometryInstances";
        const std::string kMeshBufferName = "meshes";
        const std::string kIndexBufferName = "indexData";
        const std::string kVertexBufferName = "vertices";
        const std::string kPrevVertexBufferName = "prevVertices";
        const std::string kProceduralPrimAABBBufferName = "proceduralPrimitiveAABBs";
        const std::string kCurveBufferName = "curves";
        const std::string kCurveIndexBufferName = "curveIndices";
        const std::string kCurveVertexBufferName = "curveVertices";
        const std::string kPrevCurveVertexBufferName = "prevCurveVertices";
        const std::string kSDFGridsArrayName = "sdfGrids";
        const std::string kCustomPrimitiveBufferName = "customPrimitives";
        const std::string kMaterialsBlockName = "materials";
        const std::string kLightsBufferName = "lights";
        const std::string kGridVolumesBufferName = "gridVolumes";

        const std::string kStats = "stats";
        const std::string kBounds = "bounds";
        const std::string kAnimations = "animations";
        const std::string kLoopAnimations = "loopAnimations";
        const std::string kCamera = "camera";
        const std::string kCameras = "cameras";
        const std::string kCameraSpeed = "cameraSpeed";
        const std::string kSetCameraBounds = "setCameraBounds";
        const std::string kLights = "lights";
        const std::string kLightProfile = "lightProfile";
        const std::string kAnimated = "animated";
        const std::string kRenderSettings = "renderSettings";
        const std::string kUpdateCallback = "updateCallback";
        const std::string kEnvMap = "envMap";
        const std::string kMaterials = "materials";
        const std::string kGridVolumes = "gridVolumes";
        const std::string kGetLight = "getLight";
        const std::string kGetMaterial = "getMaterial";
        const std::string kGetGridVolume = "getGridVolume";
        const std::string kSetEnvMap = "setEnvMap";
        const std::string kAddViewpoint = "addViewpoint";
        const std::string kRemoveViewpoint = "removeViewpoint";
        const std::string kSelectViewpoint = "selectViewpoint";

        const Gui::DropdownList kUpDirectionList =
        {
            { (uint32_t)Scene::UpDirection::XPos, "X+" },
            { (uint32_t)Scene::UpDirection::XNeg, "X-" },
            { (uint32_t)Scene::UpDirection::YPos, "Y+" },
            { (uint32_t)Scene::UpDirection::YNeg, "Y-" },
            { (uint32_t)Scene::UpDirection::ZPos, "Z+" },
            { (uint32_t)Scene::UpDirection::ZNeg, "Z-" },
        };

        const Gui::DropdownList kCameraControllerTypeList =
        {
            { (uint32_t)Scene::CameraControllerType::FirstPerson, "First Person" },
            { (uint32_t)Scene::CameraControllerType::Orbiter, "Orbiter" },
            { (uint32_t)Scene::CameraControllerType::SixDOF, "6-DOF" },
        };

        // Checks if the transform flips the coordinate system handedness (its determinant is negative).
        bool doesTransformFlip(const float4x4& m)
        {
            return determinant(float3x3(m)) < 0.f;
        }
    }

    const FileDialogFilterVec& Scene::getFileExtensionFilters()
    {
        // This assumes that all importer plugins are loaded when this is first called.
        static FileDialogFilterVec sFilters = [] () {
            FileDialogFilterVec filters;
            auto extensions = Importer::getSupportedExtensions();
            filters.reserve(extensions.size());
            std::transform(extensions.begin(), extensions.end(), std::back_inserter(filters), [](const auto& ext) {
                return FileDialogFilter(ext);
            });
            return filters;
        }();
        return sFilters;
    }

    Scene::Scene(ref<Device> pDevice, SceneData&& sceneData)
        : mpDevice(pDevice)
    {
        // Copy/move scene data to member variables.
        mPath = sceneData.path;
        mRenderSettings = sceneData.renderSettings;
        mCameras = std::move(sceneData.cameras);
        mSelectedCamera = sceneData.selectedCamera;
        mCameraSpeed = sceneData.cameraSpeed;
        mLights = std::move(sceneData.lights);

        mpMaterials = std::move(sceneData.pMaterials);
        mGridVolumes = std::move(sceneData.gridVolumes);
        mGrids = std::move(sceneData.grids);
        mpEnvMap = sceneData.pEnvMap;
        mpLightProfile = sceneData.pLightProfile;
        mSceneGraph = std::move(sceneData.sceneGraph);
        mMetadata = std::move(sceneData.metadata);

        // Merge all geometry instance lists into one.
        mGeometryInstanceData.reserve(sceneData.meshInstanceData.size() + sceneData.curveInstanceData.size() + sceneData.sdfGridInstances.size());
        mGeometryInstanceData.insert(std::end(mGeometryInstanceData), std::begin(sceneData.meshInstanceData), std::end(sceneData.meshInstanceData));
        mGeometryInstanceData.insert(std::end(mGeometryInstanceData), std::begin(sceneData.curveInstanceData), std::end(sceneData.curveInstanceData));
        mGeometryInstanceData.insert(std::end(mGeometryInstanceData), std::begin(sceneData.sdfGridInstances), std::end(sceneData.sdfGridInstances));

        mMeshDesc = std::move(sceneData.meshDesc);
        mMeshNames = std::move(sceneData.meshNames);
        mMeshBBs = std::move(sceneData.meshBBs);
        mMeshIdToInstanceIds = std::move(sceneData.meshIdToInstanceIds);
        mMeshGroups = std::move(sceneData.meshGroups);

        mUseCompressedHitInfo = sceneData.useCompressedHitInfo;
        mHas16BitIndices = sceneData.has16BitIndices;
        mHas32BitIndices = sceneData.has32BitIndices;

        mCurveDesc = std::move(sceneData.curveDesc);
        mCurveBBs = std::move(sceneData.curveBBs);
        mCurveIndexData = std::move(sceneData.curveIndexData);
        mCurveStaticData = std::move(sceneData.curveStaticData);

        mSDFGrids = std::move(sceneData.sdfGrids);
        mSDFGridDesc = std::move(sceneData.sdfGridDesc);
        mSDFGridMaxLODCount = std::move(sceneData.sdfGridMaxLODCount);

        mCustomPrimitiveDesc = std::move(sceneData.customPrimitiveDesc);
        mCustomPrimitiveAABBs = std::move(sceneData.customPrimitiveAABBs);

        mParticleSystems = std::move(sceneData.particleSystems);

        // Setup additional resources.
        mFrontClockwiseRS[RasterizerState::CullMode::None] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(false).setCullMode(RasterizerState::CullMode::None));
        mFrontClockwiseRS[RasterizerState::CullMode::Back] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(false).setCullMode(RasterizerState::CullMode::Back));
        mFrontClockwiseRS[RasterizerState::CullMode::Front] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(false).setCullMode(RasterizerState::CullMode::Front));
        mFrontCounterClockwiseRS[RasterizerState::CullMode::None] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(true).setCullMode(RasterizerState::CullMode::None));
        mFrontCounterClockwiseRS[RasterizerState::CullMode::Back] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(true).setCullMode(RasterizerState::CullMode::Back));
        mFrontCounterClockwiseRS[RasterizerState::CullMode::Front] = RasterizerState::create(RasterizerState::Desc().setFrontCounterCW(true).setCullMode(RasterizerState::CullMode::Front));

        // Setup volume grid -> id map.
        for (size_t i = 0; i < mGrids.size(); ++i) mGridIDs.emplace(mGrids[i], (uint32_t)i);

        // Set default SDF grid config.
        setSDFGridConfig();

        // Create vertex array objects for meshes and curves.
        createMeshVao(sceneData.meshDrawCount, sceneData.meshIndexData, sceneData.meshStaticData, sceneData.meshSkinningData);
        createCurveVao(mCurveIndexData, mCurveStaticData);
        createMeshUVTiles(mMeshDesc, sceneData.meshIndexData, sceneData.meshStaticData);

        // Create animation controller.
        mpAnimationController = std::make_unique<AnimationController>(mpDevice, this, sceneData.meshStaticData, sceneData.meshSkinningData, sceneData.prevVertexCount, sceneData.animations);

        // Some runtime mesh data validation. These are essentially asserts, but large scenes are mostly opened in Release
        for (const auto& mesh : mMeshDesc)
        {
            if (mesh.isDynamic())
            {
                if (mesh.prevVbOffset + mesh.vertexCount > sceneData.prevVertexCount) throw RuntimeError("Cached Mesh Animation: Invalid prevVbOffset");
            }
        }
        for (const auto &mesh : sceneData.cachedMeshes)
        {
            if (!mMeshDesc[mesh.meshID.get()].isAnimated()) throw RuntimeError("Cached Mesh Animation: Referenced mesh ID is not dynamic");
            if (mesh.timeSamples.size() != mesh.vertexData.size()) throw RuntimeError("Cached Mesh Animation: Time sample count mismatch.");
            for (const auto &vertices : mesh.vertexData)
            {
                if (vertices.size() != mMeshDesc[mesh.meshID.get()].vertexCount) throw RuntimeError("Cached Mesh Animation: Vertex count mismatch.");
            }
        }
        for (const auto& cache : sceneData.cachedCurves)
        {
            if (cache.tessellationMode != CurveTessellationMode::LinearSweptSphere)
            {
                if (!mMeshDesc[cache.geometryID.get()].isAnimated()) throw RuntimeError("Cached Curve Animation: Referenced mesh ID is not dynamic");
            }
        }

        // Must be placed after curve data/AABB creation.
        mpAnimationController->addAnimatedVertexCaches(std::move(sceneData.cachedCurves), std::move(sceneData.cachedMeshes), sceneData.meshStaticData);

        // Finalize scene.
        finalize();
    }

    ref<Scene> Scene::create(ref<Device> pDevice, const std::filesystem::path& path, const Settings& settings)
    {
        return SceneBuilder(pDevice, path, settings).getScene();
    }

    ref<Scene> Scene::create(ref<Device> pDevice, SceneData&& sceneData)
    {
        return ref<Scene>(new Scene(pDevice, std::move(sceneData)));
    }

    void Scene::updateSceneDefines()
    {
        DefineList defines;

        // The following defines are currently static and do not change at runtime.
        defines.add("SCENE_GRID_COUNT", std::to_string(mGrids.size()));
        defines.add("SCENE_HAS_INDEXED_VERTICES", hasIndexBuffer() ? "1" : "0");
        defines.add("SCENE_HAS_16BIT_INDICES", mHas16BitIndices ? "1" : "0");
        defines.add("SCENE_HAS_32BIT_INDICES", mHas32BitIndices ? "1" : "0");
        defines.add("SCENE_USE_LIGHT_PROFILE", mpLightProfile != nullptr ? "1" : "0");

        defines.add(mHitInfo.getDefines());
        defines.add(getSceneSDFGridDefines());

        // The following defines may change at runtime.
        defines.add("SCENE_DIFFUSE_ALBEDO_MULTIPLIER", std::to_string(mRenderSettings.diffuseAlbedoMultiplier));
        defines.add("SCENE_GEOMETRY_TYPES", std::to_string((uint32_t)mGeometryTypes));

        defines.add(mpMaterials->getDefines());

        mSceneDefines = defines;
    }

    DefineList Scene::getSceneDefines() const
    {
        return mSceneDefines;
    }

    DefineList Scene::getSceneSDFGridDefines() const
    {
        DefineList defines;

        // Setup static defines for enum values.
        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_NDSDF", std::to_string((uint32_t)SDFGrid::Type::NormalizedDenseGrid));
        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SVS", std::to_string((uint32_t)SDFGrid::Type::SparseVoxelSet));
        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SBS", std::to_string((uint32_t)SDFGrid::Type::SparseBrickSet));
        defines.add("SCENE_SDF_GRID_IMPLEMENTATION_SVO", std::to_string((uint32_t)SDFGrid::Type::SparseVoxelOctree));

        defines.add("SCENE_SDF_NO_INTERSECTION_METHOD", std::to_string((uint32_t)SDFGridIntersectionMethod::None));
        defines.add("SCENE_SDF_NO_VOXEL_SOLVER", std::to_string((uint32_t)SDFGridIntersectionMethod::GridSphereTracing));
        defines.add("SCENE_SDF_VOXEL_SPHERE_TRACING", std::to_string((uint32_t)SDFGridIntersectionMethod::VoxelSphereTracing));

        defines.add("SCENE_SDF_NO_GRADIENT_EVALUATION_METHOD", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::None));
        defines.add("SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::NumericDiscontinuous));
        defines.add("SCENE_SDF_GRADIENT_NUMERIC_CONTINUOUS", std::to_string((uint32_t)SDFGridGradientEvaluationMethod::NumericContinuous));

        // Setup dynamic defines based on current configuration.
        defines.add("SCENE_SDF_GRID_COUNT", std::to_string(mSDFGrids.size()));
        defines.add("SCENE_SDF_GRID_MAX_LOD_COUNT", std::to_string(mSDFGridMaxLODCount));

        defines.add("SCENE_SDF_GRID_IMPLEMENTATION", std::to_string((uint32_t)mSDFGridConfig.implementation));
        defines.add("SCENE_SDF_VOXEL_INTERSECTION_METHOD", std::to_string((uint32_t)mSDFGridConfig.intersectionMethod));
        defines.add("SCENE_SDF_GRADIENT_EVALUATION_METHOD", std::to_string((uint32_t)mSDFGridConfig.gradientEvaluationMethod));
        defines.add("SCENE_SDF_SOLVER_MAX_ITERATION_COUNT", std::to_string(mSDFGridConfig.solverMaxIterations));
        defines.add("SCENE_SDF_OPTIMIZE_VISIBILITY_RAYS", mSDFGridConfig.optimizeVisibilityRays ? "1" : "0");

        return defines;
    }

    Program::TypeConformanceList Scene::getTypeConformances() const
    {
        return mTypeConformances;
    }

    Program::ShaderModuleList Scene::getShaderModules() const
    {
        return mpMaterials->getShaderModules();
    }

    const ref<LightCollection>& Scene::getLightCollection(RenderContext* pRenderContext)
    {
        if (!mpLightCollection)
        {
            checkInvariant(mFinalized, "getLightCollection() called before scene is ready for use");

            mpLightCollection = LightCollection::create(mpDevice, pRenderContext, this);
            mpLightCollection->setShaderData(mpSceneBlock->getRootVar()["lightCollection"]);

            mSceneStats.emissiveMemoryInBytes = mpLightCollection->getMemoryUsageInBytes();
        }
        return mpLightCollection;
    }

    void Scene::rasterize(
        RenderContext* pRenderContext,
        GraphicsState* pState,
        GraphicsVars* pVars,
        RasterizerState::CullMode cullMode,
        RasterizerState::MeshRenderMode meshRenderMode,
        bool drawShadowCastable
    )
    {
        rasterize(
            pRenderContext, pState, pVars, mFrontClockwiseRS[cullMode], mFrontCounterClockwiseRS[cullMode],
            mFrontCounterClockwiseRS[RasterizerState::CullMode::None], meshRenderMode, drawShadowCastable
        );
    }

    void Scene::rasterize(
        RenderContext* pRenderContext,
        GraphicsState* pState,
        GraphicsVars* pVars,
        const ref<RasterizerState>& pRasterizerStateCW,
        const ref<RasterizerState>& pRasterizerStateCCW,
        const ref<RasterizerState>& pRasterizerStateDS,
        RasterizerState::MeshRenderMode meshRenderMode,
        bool drawShadowCastable
    )
    {
        FALCOR_PROFILE(pRenderContext, "rasterizeScene");

        pVars->setParameterBlock(kParameterBlockName, mpSceneBlock);

        auto pCurrentRS = pState->getRasterizerState();
        bool isIndexed = hasIndexBuffer();
        // Lamda for checking particles
        auto checkSkipParticle = [&](const DrawArgs& draw) {
            bool skip = false;
            if (draw.particleOrientationMode != ParticleOrientationMode::None)
            {
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleCamera) &&
                     draw.particleOrientationMode == ParticleOrientationMode::Camera);
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleXY) &&
                     draw.particleOrientationMode == ParticleOrientationMode::XY_Plane);
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleYZ) &&
                     draw.particleOrientationMode == ParticleOrientationMode::YZ_Plane);
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleXZ) &&
                     draw.particleOrientationMode == ParticleOrientationMode::XZ_Plane);
            }

            return skip;
        };

        for (const auto& draw : mDrawArgs)
        {
            FALCOR_ASSERT(draw.count > 0);

            // Skip meshes that should not cast a shadow
            if (!draw.isCastShadow && !drawShadowCastable)
                continue;

            //Skip particles
            if (checkSkipParticle(draw))
                continue;

            //Skip opaque meshes
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipOpaque) && draw.isOpaque)
                continue;

            //Skip Non-opaque meshes
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipNonOpaque) && !draw.isOpaque)
                continue;

            //Skip static meshes if desired
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipStatic) && !draw.isDynamic)
                continue;

            // Skip dynamic meshes if desired
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipDynamic) && draw.isDynamic)
                continue;

            //Skip non double sided if desired
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipNonDoubleSided) && !draw.ignoreWinding)
                continue;

            // Set state.
            pState->setVao(draw.ibFormat == ResourceFormat::R16Uint ? mpMeshVao16Bit : mpMeshVao);

            if (draw.ignoreWinding)
                pState->setRasterizerState(pRasterizerStateDS);
            else if (draw.ccw)
                pState->setRasterizerState(pRasterizerStateCCW);
            else
                pState->setRasterizerState(pRasterizerStateCW);

            // Draw the primitives.
            if (isIndexed)
            {
                pRenderContext->drawIndexedIndirect(pState, pVars, draw.count, draw.pBuffer.get(), 0, nullptr, 0);
            }
            else
            {
                pRenderContext->drawIndirect(pState, pVars, draw.count, draw.pBuffer.get(), 0, nullptr, 0);
            }
        }

        pState->setRasterizerState(pCurrentRS);
    }

     void Scene::rasterizeFrustumCulling(
        RenderContext* pRenderContext,
        GraphicsState* pState,
        GraphicsVars* pVars,
        RasterizerState::CullMode cullMode,
        RasterizerState::MeshRenderMode meshRenderMode,
        bool drawShadowCastable,
        ref<FrustumCulling> pFrustumCulling
    )
    {
        rasterizeFrustumCulling(
            pRenderContext, pState, pVars, mFrontClockwiseRS[cullMode], mFrontCounterClockwiseRS[cullMode],
            mFrontCounterClockwiseRS[RasterizerState::CullMode::None], meshRenderMode,drawShadowCastable, pFrustumCulling
        );
    }

    void Scene::rasterizeFrustumCulling(
        RenderContext* pRenderContext,
        GraphicsState* pState,
        GraphicsVars* pVars,
        const ref<RasterizerState>& pRasterizerStateCW,
        const ref<RasterizerState>& pRasterizerStateCCW,
        const ref<RasterizerState>& pRasterizerStateDS,
        RasterizerState::MeshRenderMode meshRenderMode,
        bool drawShadowCastable,
        ref<FrustumCulling> pFrustumCulling
    )
    {
        FALCOR_PROFILE(pRenderContext, "rasterizeScene");

        pVars->setParameterBlock(kParameterBlockName, mpSceneBlock);

        auto pCurrentRS = pState->getRasterizerState();
        bool isIndexed = hasIndexBuffer();

        bool updateDynamicGeomFrustum = true;

        //If there was no called culling, use the camera one
        if (!pFrustumCulling)
        {
            bool forceUpdate = mSelectedCamera != mFrustumCullingSelectedCamera;
            if (forceUpdate)
                mFrustumCullingSelectedCamera = mSelectedCamera;

            const auto& camera = mCameras[mSelectedCamera];
            if (!mpCameraCulling)
            {
                mpCameraCulling = make_ref<FrustumCulling>(camera);
            }

            pFrustumCulling = mpCameraCulling;

            //Update the frustum only for the first rasterize pass
            if (!mFrustumCullingUpdated || forceUpdate)
            {
                auto cameraChanges = camera->getChanges();
                auto excluded = Camera::Changes::Jitter | Camera::Changes::History;
                if (((cameraChanges & ~excluded) != Camera::Changes::None) || forceUpdate)
                {
                    pFrustumCulling->updateFrustum(camera);
                }
                updateDynamicGeomFrustum = true;
                mFrustumCullingUpdated = true;
            }
        }
        else //Custom culling is used
        {
            // TODO add handling for custom frustum culling
            // For now we assume that every custom culling is called once
            updateDynamicGeomFrustum = true;
        }

        //Initialize the draw buffers, with the mDrawArgs buffer as template
        if (mDrawArgs.size() != pFrustumCulling->getDrawBufferSize())
        {
            std::vector<ref<Buffer>> drawBuffers;
            std::vector<bool> hasDynamicGeometry;
            for (const auto& draw : mDrawArgs)
            {
                drawBuffers.push_back(draw.pBuffer);
                hasDynamicGeometry.push_back(draw.isDynamic);
            }
                
            pFrustumCulling->createDrawBuffer(mpDevice, mpFence, pRenderContext, drawBuffers, hasDynamicGeometry);
        }

        // Create an custom draw argument buffer for this frame
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();
        auto& pDrawBuffers = pFrustumCulling->getDrawBuffers();
        auto& pDrawBufferCounts = pFrustumCulling->getDrawCounts();

        //Check if any buffer needs an update
        bool needUpdate = false;
        for (uint i = 0; i < mDrawArgs.size(); i++)
            needUpdate |= !pFrustumCulling->isBufferValid(i);

        if (needUpdate || (updateDynamicGeomFrustum && pFrustumCulling->hasDynamic()))
            pFrustumCulling->startUpdate(mFenceSyncLastFrame);

        //Lamda for checking particles
        auto checkSkipParticle = [&](const DrawArgs& draw)
        {
            bool skip = false;
            if (draw.particleOrientationMode != ParticleOrientationMode::None)
            {
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleCamera) &&
                     draw.particleOrientationMode == ParticleOrientationMode::Camera);
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleXY) &&
                     draw.particleOrientationMode == ParticleOrientationMode::XY_Plane);
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleYZ) &&
                     draw.particleOrientationMode == ParticleOrientationMode::YZ_Plane);
                skip |=
                    (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipParticleXZ) &&
                     draw.particleOrientationMode == ParticleOrientationMode::XZ_Plane);
            }

            return skip;
        };

        for (uint i=0; i<mDrawArgs.size(); i++)
        {
            const auto& draw = mDrawArgs[i];
            FALCOR_ASSERT(draw.count > 0);

            //Skip meshes that should not cast a shadow
            if (!draw.isCastShadow && !drawShadowCastable)
                continue;

            // Skip particles
            if (checkSkipParticle(draw))
                continue;

            // Skip opaque meshes
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipOpaque) && draw.isOpaque)
                continue;

            // Skip Non-opaque meshes
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipNonOpaque) && !draw.isOpaque)
                continue;

            // Skip static meshes if desired
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipStatic) && !draw.isDynamic)
                continue;

            // Skip dynamic meshes if desired
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipDynamic) && draw.isDynamic)
                continue;

            // Skip non double sided if desired
            if (is_set(meshRenderMode, RasterizerState::MeshRenderMode::SkipNonDoubleSided) && !draw.ignoreWinding)
                continue;

            bool bufferValid = pFrustumCulling->isBufferValid(i);
            bool isParticle = draw.particleOrientationMode != ParticleOrientationMode::None;

            //Skip Frustum culling for all particles
            if (isParticle)
            {
                pDrawBufferCounts[i] = draw.count;
                pDrawBuffers[i] = draw.pBuffer;
            }
            else if (isIndexed && (!bufferValid || draw.isDynamic))
            {
                std::vector<DrawIndexedArguments> drawArguments;
                std::vector<uint> passedDrawInstances;        //Draw instances used for dynamic geometry
                for (auto& instanceID : mDrawArgsInstanceIDs[i])
                {
                    const auto& instance = mGeometryInstanceData[instanceID];
                    const auto& worldMat = globalMatrices[instance.globalMatrixID];
                    const AABB& meshBB = mMeshBBs[instance.geometryID];
                    const auto& mesh = mMeshDesc[instance.geometryID];

                    //If the mesh passes the culling test, add to draw buffer
                    // TODO: Add a better/functioning precalculated BB for skinned meshes
                    if (pFrustumCulling->isInFrustum(meshBB.transform(worldMat)) || mesh.isSkinned())
                    {
                        DrawIndexedArguments drawArg;
                        drawArg.IndexCountPerInstance = mesh.indexCount;
                        drawArg.InstanceCount = 1;
                        drawArg.StartIndexLocation = mesh.ibOffset * (mesh.use16BitIndices() ? 2 : 1);
                        drawArg.BaseVertexLocation = mesh.vbOffset;
                        drawArg.StartInstanceLocation = instanceID;

                        drawArguments.push_back(drawArg);
                        passedDrawInstances.push_back(instanceID);
                    }
                }
                //For dynamic check if we need to update the draw buffer
                bool updateDrawBuffer = true;
                if (draw.isDynamic)
                    updateDrawBuffer = pFrustumCulling->checkDynamicInstances(i, passedDrawInstances);
                    
                if (updateDrawBuffer)
                    pFrustumCulling->updateDrawBuffer(pRenderContext,i, drawArguments);
            }
            else if ((!bufferValid || draw.isDynamic))
            {
                std::vector<DrawArguments> drawArguments;
                std::vector<uint> passedDrawInstances; // Draw instances used for dynamic geometry
                for (auto& instanceID : mDrawArgsInstanceIDs[i])
                {
                    const auto& instance = mGeometryInstanceData[instanceID];
                    const auto& worldMat = globalMatrices[instance.globalMatrixID];
                    const AABB& meshBB = mMeshBBs[instance.geometryID];
                    const auto& mesh = mMeshDesc[instance.geometryID];
                    // If the mesh passes the culling test, add to draw buffer
                    // TODO: Add a better/functioning precalculated BB for skinned meshes
                    if (pFrustumCulling->isInFrustum(meshBB.transform(worldMat)) || mesh.isSkinned())
                    {
                        
                        DrawArguments drawArg;
                        drawArg.VertexCountPerInstance = mesh.vertexCount;
                        drawArg.InstanceCount = 1;
                        drawArg.StartVertexLocation = mesh.vbOffset;
                        drawArg.StartInstanceLocation = instanceID;

                        drawArguments.push_back(drawArg);
                        passedDrawInstances.push_back(instanceID);
                    }
                }
                // For dynamic check if we need to update the draw buffer
                bool updateDrawBuffer = true;
                if (draw.isDynamic)
                    updateDrawBuffer = pFrustumCulling->checkDynamicInstances(i, passedDrawInstances);

                if (updateDrawBuffer)
                    pFrustumCulling->updateDrawBuffer(pRenderContext, i, drawArguments);
            }

            //Check if everything was culled
            if (pDrawBufferCounts[i] == 0)
                continue;

            // Set state.
            pState->setVao(draw.ibFormat == ResourceFormat::R16Uint ? mpMeshVao16Bit : mpMeshVao);

            if (draw.ignoreWinding)
                pState->setRasterizerState(pRasterizerStateDS);
            else if (draw.ccw)
                pState->setRasterizerState(pRasterizerStateCCW);
            else
                pState->setRasterizerState(pRasterizerStateCW);

            // Draw the primitives.
            if (isIndexed)
            {
                pRenderContext->drawIndexedIndirect(pState, pVars, pDrawBufferCounts[i], pDrawBuffers[i].get(), 0, nullptr, 0);
            }
            else
            {
                pRenderContext->drawIndirect(pState, pVars, pDrawBufferCounts[i], pDrawBuffers[i].get(), 0, nullptr, 0);
            }
        }

        
        pState->setRasterizerState(pCurrentRS);
    }

    uint32_t Scene::getRaytracingMaxAttributeSize() const
    {
        bool hasDisplacedMesh = hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh);
        if (hasDisplacedMesh) return 12;


        return 8;
    }

    void Scene::raytrace(RenderContext* pRenderContext, RtProgram* pProgram, const ref<RtProgramVars>& pVars, uint3 dispatchDims)
    {
        FALCOR_PROFILE(pRenderContext, "raytraceScene");

        FALCOR_ASSERT(pRenderContext && pProgram && pVars);
        // Check for valid number of geometries.
        // We either expect a single geometry (used for "dummy shared binding tables") or matching the number of geometries in the scene.
        if (pVars->getRayTypeCount() > 0 && pVars->getGeometryCount() != 1 && pVars->getGeometryCount() != getGeometryCount())
        {
            logWarning("RtProgramVars geometry count mismatch");
        }

        uint32_t rayTypeCount = pVars->getRayTypeCount();
        setRaytracingShaderData(pRenderContext, pVars->getRootVar(), rayTypeCount);

        // Set ray type constant.
        pVars->getRootVar()["DxrPerFrame"]["rayTypeCount"] = rayTypeCount;

        pRenderContext->raytrace(pProgram, pVars.get(), dispatchDims.x, dispatchDims.y, dispatchDims.z);
    }

    void Scene::createMeshVao(uint32_t drawCount, const std::vector<uint32_t>& indexData, const std::vector<PackedStaticVertexData>& staticData, const std::vector<SkinningVertexData>& skinningData)
    {
        if (drawCount == 0) return;

        // Create the index buffer.
        size_t ibSize = sizeof(uint32_t) * indexData.size();
        if (ibSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Index buffer size exceeds 4GB");
        }

        ref<Buffer> pIB;
        if (ibSize > 0)
        {
            ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
            pIB = Buffer::create(mpDevice, ibSize, ibBindFlags, Buffer::CpuAccess::None, indexData.data());
        }

        // Create the vertex data structured buffer.
        const size_t vertexCount = (uint32_t)staticData.size();
        size_t staticVbSize = sizeof(PackedStaticVertexData) * vertexCount;
        if (staticVbSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Vertex buffer size exceeds 4GB");
        }

        ref<Buffer> pStaticBuffer;
        if (vertexCount > 0)
        {
            ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
            pStaticBuffer = Buffer::createStructured(mpDevice, sizeof(PackedStaticVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, nullptr, false);
            pStaticBuffer->setName("Scene::StaticVerticesBuffer");
        }

        Vao::BufferVec pVBs(kVertexBufferCount);
        pVBs[kStaticDataBufferIndex] = pStaticBuffer;

        // Create the draw ID buffer.
        // This is only needed when rasterizing meshes in the scene.
        ResourceFormat drawIDFormat = drawCount <= (1 << 16) ? ResourceFormat::R16Uint : ResourceFormat::R32Uint;

        ref<Buffer> pDrawIDBuffer;
        if (drawIDFormat == ResourceFormat::R16Uint)
        {
            FALCOR_ASSERT(drawCount <= (1 << 16));
            std::vector<uint16_t> drawIDs(drawCount);
            for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
            pDrawIDBuffer = Buffer::create(mpDevice, drawCount * sizeof(uint16_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());
            pDrawIDBuffer->setName("Scene::VertexDrawIDBuffer");
        }
        else if (drawIDFormat == ResourceFormat::R32Uint)
        {
            std::vector<uint32_t> drawIDs(drawCount);
            for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
            pDrawIDBuffer = Buffer::create(mpDevice, drawCount * sizeof(uint32_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());
            pDrawIDBuffer->setName("Scene::VertexDrawIDBuffer");
        }
        else FALCOR_UNREACHABLE();

        FALCOR_ASSERT(pDrawIDBuffer);
        pVBs[kDrawIdBufferIndex] = pDrawIDBuffer;

        // Create vertex layout.
        // The layout only initializes the vertex data and draw ID layout. The skinning data doesn't get passed into the vertex shader.
        ref<VertexLayout> pLayout = VertexLayout::create();

        // Add the packed static vertex data layout.
        ref<VertexBufferLayout> pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(VERTEX_POSITION_NAME, offsetof(PackedStaticVertexData, position), ResourceFormat::RGB32Float, 1, VERTEX_POSITION_LOC);
        pStaticLayout->addElement(VERTEX_PACKED_NORMAL_TANGENT_CURVE_RADIUS_NAME, offsetof(PackedStaticVertexData, packedNormalTangentCurveRadius), ResourceFormat::RGB32Float, 1, VERTEX_PACKED_NORMAL_TANGENT_CURVE_RADIUS_LOC);
        pStaticLayout->addElement(VERTEX_TEXCOORD_NAME, offsetof(PackedStaticVertexData, texCrd), ResourceFormat::RG32Float, 1, VERTEX_TEXCOORD_LOC);
        pLayout->addBufferLayout(kStaticDataBufferIndex, pStaticLayout);

        // Add the draw ID layout.
        ref<VertexBufferLayout> pInstLayout = VertexBufferLayout::create();
        pInstLayout->addElement(INSTANCE_DRAW_ID_NAME, 0, drawIDFormat, 1, INSTANCE_DRAW_ID_LOC);
        pInstLayout->setInputClass(VertexBufferLayout::InputClass::PerInstanceData, 1);
        pLayout->addBufferLayout(kDrawIdBufferIndex, pInstLayout);

        // Create the VAO objects.
        // Note that the global index buffer can be mixed 16/32-bit format.
        // For drawing the meshes we need separate VAOs for these cases.
        mpMeshVao = Vao::create(Vao::Topology::TriangleList, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
        mpMeshVao16Bit = Vao::create(Vao::Topology::TriangleList, pLayout, pVBs, pIB, ResourceFormat::R16Uint);
    }

    void Scene::createCurveVao(const std::vector<uint32_t>& indexData, const std::vector<StaticCurveVertexData>& staticData)
    {
        if (indexData.empty() || staticData.empty()) return;

        // Create the index buffer.
        size_t ibSize = sizeof(uint32_t) * indexData.size();
        if (ibSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Curve index buffer size exceeds 4GB");
        }

        ref<Buffer> pIB = nullptr;
        if (ibSize > 0)
        {
            ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
            pIB = Buffer::create(mpDevice, ibSize, ibBindFlags, Buffer::CpuAccess::None, indexData.data());
        }

        // Create the vertex data as structured buffers.
        const size_t vertexCount = (uint32_t)staticData.size();
        size_t staticVbSize = sizeof(StaticCurveVertexData) * vertexCount;
        if (staticVbSize > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Curve vertex buffer exceeds 4GB");
        }

        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
        // Also upload the curve vertex data.
        ref<Buffer> pStaticBuffer = Buffer::createStructured(mpDevice, sizeof(StaticCurveVertexData), (uint32_t)vertexCount, vbBindFlags, Buffer::CpuAccess::None, staticData.data(), false);

        // Curves do not need DrawIDBuffer.
        Vao::BufferVec pVBs(kVertexBufferCount - 1);
        pVBs[kStaticDataBufferIndex] = pStaticBuffer;

        // Create vertex layout.
        // The layout only initializes the vertex data layout. The skinning data doesn't get passed into the vertex shader.
        ref<VertexLayout> pLayout = VertexLayout::create();

        // Add the packed static vertex data layout.
        ref<VertexBufferLayout> pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(CURVE_VERTEX_POSITION_NAME, offsetof(StaticCurveVertexData, position), ResourceFormat::RGB32Float, 1, CURVE_VERTEX_POSITION_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_RADIUS_NAME, offsetof(StaticCurveVertexData, radius), ResourceFormat::R32Float, 1, CURVE_VERTEX_RADIUS_LOC);
        pStaticLayout->addElement(CURVE_VERTEX_TEXCOORD_NAME, offsetof(StaticCurveVertexData, texCrd), ResourceFormat::RG32Float, 1, CURVE_VERTEX_TEXCOORD_LOC);
        pLayout->addBufferLayout(kStaticDataBufferIndex, pStaticLayout);

        // Create the VAO objects.
        mpCurveVao = Vao::create(Vao::Topology::LineStrip, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
    }

    void Scene::createMeshUVTiles(const std::vector<MeshDesc>& meshDescs, const std::vector<uint32_t>& indexData, const std::vector<PackedStaticVertexData>& staticData)
    {
        const uint8_t* indexData8 = reinterpret_cast<const uint8_t*>(indexData.data());

        for (const MeshDesc& desc : meshDescs)
        {
            // This tile captures any triangles that span more than one unit square, e.g., for tiled textures
            Rectangle largeTriangleTile;
            std::map<int2, Rectangle> tiles;

            const uint tcount = desc.getTriangleCount();
            for (uint tidx = 0; tidx < tcount; ++tidx)
            {
                // Compute local vertex indices within the mesh.
                uint32_t vidx[3] = {};
                if (desc.useVertexIndices())
                {
                    FALCOR_ASSERT(indexData8 != nullptr);
                    uint baseIndex = desc.ibOffset * 4;
                    if (desc.use16BitIndices())
                    {
                        baseIndex += tidx * 3 * sizeof(uint16_t);
                        vidx[0] = reinterpret_cast<const uint16_t*>(indexData8 + baseIndex)[0];
                        vidx[1] = reinterpret_cast<const uint16_t*>(indexData8 + baseIndex)[1];
                        vidx[2] = reinterpret_cast<const uint16_t*>(indexData8 + baseIndex)[2];
                    }
                    else
                    {
                        baseIndex += tidx * 3 * sizeof(uint32_t);
                        vidx[0] = reinterpret_cast<const uint32_t*>(indexData8 + baseIndex)[0];
                        vidx[1] = reinterpret_cast<const uint32_t*>(indexData8 + baseIndex)[1];
                        vidx[2] = reinterpret_cast<const uint32_t*>(indexData8 + baseIndex)[2];
                    }
                }
                else
                {
                    uint baseIndex = tidx * 3;
                    vidx[0] = baseIndex + 0;
                    vidx[1] = baseIndex + 1;
                    vidx[2] = baseIndex + 2;
                }
                FALCOR_ASSERT(vidx[0] < desc.vertexCount);
                FALCOR_ASSERT(vidx[1] < desc.vertexCount);
                FALCOR_ASSERT(vidx[2] < desc.vertexCount);

                // Load vertices from global vertex buffer.
                // Note that the mesh local vbOffset is added to address into the global vertex buffer.
                FALCOR_ASSERT((size_t)desc.vbOffset + desc.vertexCount <= staticData.size());
                StaticVertexData vertices[3];
                vertices[0] = staticData[(size_t)desc.vbOffset + vidx[0]].unpack();
                vertices[1] = staticData[(size_t)desc.vbOffset + vidx[1]].unpack();
                vertices[2] = staticData[(size_t)desc.vbOffset + vidx[2]].unpack();

                int2 v0 = int2(std::floor(vertices[0].texCrd[0]), std::floor(vertices[0].texCrd[1]));
                int2 v1 = int2(std::floor(vertices[1].texCrd[0]), std::floor(vertices[1].texCrd[1]));
                int2 v2 = int2(std::floor(vertices[2].texCrd[0]), std::floor(vertices[2].texCrd[1]));

                Rectangle* tile;
                if (all(v0 == v1 && v0 == v2))
                    tile = &tiles[v0];
                else
                    tile = &largeTriangleTile;

                tile->include(vertices[0].texCrd);
                tile->include(vertices[1].texCrd);
                tile->include(vertices[2].texCrd);
            }

            mMeshUVTiles.push_back(std::vector<Rectangle>());
            for (auto& tile : tiles)
            {
                if (largeTriangleTile.contains(tile.second))
                    continue;
                mMeshUVTiles.back().push_back(tile.second);
            }

            if (largeTriangleTile.valid())
                mMeshUVTiles.back().push_back(largeTriangleTile);
        }
    }

    void Scene::setSDFGridConfig()
    {
        if (mSDFGrids.empty()) return;

        for (const ref<SDFGrid>& pSDFGrid : mSDFGrids)
        {
            if (mSDFGridConfig.implementation == SDFGrid::Type::None)
            {
                mSDFGridConfig.implementation = pSDFGrid->getType();
            }
            else if (mSDFGridConfig.implementation != pSDFGrid->getType())
            {
                throw RuntimeError("All SDF grids in the same scene must currently be of the same type.");
            }
        }

        // Set default SDF grid config and compute allowed SDF grid UI settings list.

        switch (mSDFGridConfig.implementation)
        {
        case SDFGrid::Type::NormalizedDenseGrid:
        {
            mSDFGridConfig.intersectionMethod = SDFGridIntersectionMethod::VoxelSphereTracing;
            mSDFGridConfig.gradientEvaluationMethod = SDFGridGradientEvaluationMethod::NumericDiscontinuous;
            mSDFGridConfig.solverMaxIterations = 256;
            mSDFGridConfig.optimizeVisibilityRays = true;

            mSDFGridConfig.intersectionMethodList =
            {
                { uint32_t(SDFGridIntersectionMethod::GridSphereTracing), "Grid Sphere Tracing" },
                { uint32_t(SDFGridIntersectionMethod::VoxelSphereTracing), "Voxel Sphere Tracing" },
            };

            mSDFGridConfig.gradientEvaluationMethodList =
            {
                { uint32_t(SDFGridGradientEvaluationMethod::NumericDiscontinuous), "Numeric Discontinuous" },
                { uint32_t(SDFGridGradientEvaluationMethod::NumericContinuous), "Numeric Continuous" },
            };

            break;
        }
        case SDFGrid::Type::SparseVoxelSet:
        case SDFGrid::Type::SparseBrickSet:
        {
            mSDFGridConfig.intersectionMethod = SDFGridIntersectionMethod::VoxelSphereTracing;
            mSDFGridConfig.gradientEvaluationMethod = SDFGridGradientEvaluationMethod::NumericDiscontinuous;
            mSDFGridConfig.solverMaxIterations = 256;
            mSDFGridConfig.optimizeVisibilityRays = true;

            mSDFGridConfig.intersectionMethodList =
            {
                { uint32_t(SDFGridIntersectionMethod::VoxelSphereTracing), "Voxel Sphere Tracing" },
            };

            mSDFGridConfig.gradientEvaluationMethodList =
            {
                { uint32_t(SDFGridGradientEvaluationMethod::NumericDiscontinuous), "Numeric Discontinuous" },
            };

            break;
        case SDFGrid::Type::SparseVoxelOctree:
            mSDFGridConfig.intersectionMethod = SDFGridIntersectionMethod::VoxelSphereTracing;
            mSDFGridConfig.gradientEvaluationMethod = SDFGridGradientEvaluationMethod::NumericDiscontinuous;
            mSDFGridConfig.solverMaxIterations = 256;
            mSDFGridConfig.optimizeVisibilityRays = true;

            mSDFGridConfig.intersectionMethodList =
            {
                { uint32_t(SDFGridIntersectionMethod::VoxelSphereTracing), "Voxel Sphere Tracing" },
            };

            mSDFGridConfig.gradientEvaluationMethodList =
            {
                { uint32_t(SDFGridGradientEvaluationMethod::NumericDiscontinuous), "Numeric Discontinuous" },
            };
            break;

        }
        }
    }

    void Scene::initSDFGrids()
    {
        if (mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
        {
            mSDFGridConfig.implementationData.SBS.virtualBrickCoordsBitCount = 0;
            mSDFGridConfig.implementationData.SBS.brickLocalVoxelCoordsBitCount = 0;
        }
        else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
        {
            mSDFGridConfig.implementationData.SVO.svoIndexBitCount = 0;
        }

        for (const ref<SDFGrid>& pSDFGrid : mSDFGrids)
        {
            pSDFGrid->createResources(mpDevice->getRenderContext());

            if (mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
            {
                const SDFSBS* pSBS = reinterpret_cast<const SDFSBS*>(pSDFGrid.get());
                mSDFGridConfig.implementationData.SBS.virtualBrickCoordsBitCount = std::max(mSDFGridConfig.implementationData.SBS.virtualBrickCoordsBitCount, pSBS->getVirtualBrickCoordsBitCount());
                mSDFGridConfig.implementationData.SBS.brickLocalVoxelCoordsBitCount = std::max(mSDFGridConfig.implementationData.SBS.brickLocalVoxelCoordsBitCount, pSBS->getBrickLocalVoxelCoordsBrickCount());
            }
            else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
            {
                const SDFSVO* pSVO = reinterpret_cast<const SDFSVO*>(pSDFGrid.get());
                mSDFGridConfig.implementationData.SVO.svoIndexBitCount = std::max(mSDFGridConfig.implementationData.SVO.svoIndexBitCount, pSVO->getSVOIndexBitCount());
            }
        }
    }

    void Scene::createParameterBlock()
    {
        // Create parameter block.
        ref<ComputeProgram> pProgram = ComputeProgram::createFromFile(mpDevice, "Scene/SceneBlock.slang", "main", getSceneDefines());
        ref<const ParameterBlockReflection> pReflection = pProgram->getReflector()->getParameterBlock(kParameterBlockName);
        FALCOR_ASSERT(pReflection);

        mpSceneBlock = ParameterBlock::create(mpDevice, pReflection);
        auto var = mpSceneBlock->getRootVar();

        // Create GPU buffers.
        if (!mGeometryInstanceData.empty() &&
            (!mpGeometryInstancesBuffer || mpGeometryInstancesBuffer->getElementCount() < mGeometryInstanceData.size()))
        {
            mpGeometryInstancesBuffer = Buffer::createStructured(mpDevice, var[kGeometryInstanceBufferName], (uint32_t)mGeometryInstanceData.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpGeometryInstancesBuffer->setName("Scene::mpGeometryInstancesBuffer");
        }

        if (!mMeshDesc.empty() &&
            (!mpMeshesBuffer || mpMeshesBuffer->getElementCount() < mMeshDesc.size()))
        {
            mpMeshesBuffer = Buffer::createStructured(mpDevice, var[kMeshBufferName], (uint32_t)mMeshDesc.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpMeshesBuffer->setName("Scene::mpMeshesBuffer");
        }

        if (!mCurveDesc.empty() &&
            (!mpCurvesBuffer || mpCurvesBuffer->getElementCount() < mCurveDesc.size()))
        {
            mpCurvesBuffer = Buffer::createStructured(mpDevice, var[kCurveBufferName], (uint32_t)mCurveDesc.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpCurvesBuffer->setName("Scene::mpCurvesBuffer");
        }

        if (!mLights.empty() &&
            (!mpLightsBuffer || mpLightsBuffer->getElementCount() < mLights.size()))
        {
            mpLightsBuffer = Buffer::createStructured(mpDevice, var[kLightsBufferName], (uint32_t)mLights.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpLightsBuffer->setName("Scene::mpLightsBuffer");
        }

        if (!mGridVolumes.empty() &&
            (!mpGridVolumesBuffer || mpGridVolumesBuffer->getElementCount() < mGridVolumes.size()))
        {
            mpGridVolumesBuffer = Buffer::createStructured(mpDevice, var[kGridVolumesBufferName], (uint32_t)mGridVolumes.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpGridVolumesBuffer->setName("Scene::mpGridVolumesBuffer");
        }

        // Bind resources to parameter block.
        var[kGeometryInstanceBufferName] = mpGeometryInstancesBuffer;
        var[kMeshBufferName] = mpMeshesBuffer;
        var[kCurveBufferName] = mpCurvesBuffer;
        var[kLightsBufferName] = mpLightsBuffer;
        var[kGridVolumesBufferName] = mpGridVolumesBuffer;

        // Bind materials parameter block.
        var[kMaterialsBlockName].setParameterBlock(mpMaterials->getParameterBlock());
    }

    void Scene::uploadResources()
    {
        FALCOR_ASSERT(mpAnimationController);

        auto var = mpSceneBlock->getRootVar();

        // Upload geometry.
        if (!mMeshDesc.empty()) mpMeshesBuffer->setBlob(mMeshDesc.data(), 0, sizeof(MeshDesc) * mMeshDesc.size());
        if (!mCurveDesc.empty()) mpCurvesBuffer->setBlob(mCurveDesc.data(), 0, sizeof(CurveDesc) * mCurveDesc.size());

        auto sdfGridsVar = var[kSDFGridsArrayName];

        for (uint32_t i = 0; i < mSDFGrids.size(); i++)
        {
            const ref<SDFGrid>& pGrid = mSDFGrids[i];
            pGrid->setShaderData(sdfGridsVar[i]);
        }

        if (mpMeshVao != nullptr)
        {
            if (hasIndexBuffer()) var[kIndexBufferName] = mpMeshVao->getIndexBuffer();
            var[kVertexBufferName] = mpMeshVao->getVertexBuffer(Scene::kStaticDataBufferIndex);
            var[kPrevVertexBufferName] = mpAnimationController->getPrevVertexData(); // Can be nullptr
        }

        if (mpCurveVao != nullptr)
        {
            var[kCurveIndexBufferName] = mpCurveVao->getIndexBuffer();
            var[kCurveVertexBufferName] = mpCurveVao->getVertexBuffer(Scene::kStaticDataBufferIndex);
            var[kPrevCurveVertexBufferName] = mpAnimationController->getPrevCurveVertexData();
        }
    }

    void Scene::uploadSelectedCamera()
    {
        getCamera()->setShaderData(mpSceneBlock->getRootVar()[kCamera]);
    }

    void Scene::updateBounds()
    {
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();

        mSceneBB = AABB();

        for (const auto& inst : mGeometryInstanceData)
        {
            const float4x4& transform = globalMatrices[inst.globalMatrixID];
            switch (inst.getType())
            {
            case GeometryType::TriangleMesh:
            case GeometryType::DisplacedTriangleMesh:
            {
                const AABB& meshBB = mMeshBBs[inst.geometryID];
                mSceneBB |= meshBB.transform(transform);
                break;
            }
            case GeometryType::Curve:
            {
                const AABB& curveBB = mCurveBBs[inst.geometryID];
                mSceneBB |= curveBB.transform(transform);
                break;
            }
            case GeometryType::SDFGrid:
            {
                float3x3 transform3x3 = float3x3(transform);
                transform3x3[0] = abs(transform3x3[0]);
                transform3x3[1] = abs(transform3x3[1]);
                transform3x3[2] = abs(transform3x3[2]);
                float3 center = transform.getCol(3).xyz();
                float3 halfExtent = transformVector(transform3x3, float3(0.5f));
                mSceneBB |= AABB(center - halfExtent, center + halfExtent);
                break;
            }
            }
        }

        for (const auto& aabb : mCustomPrimitiveAABBs)
        {
            mSceneBB |= aabb;
        }

        for (const auto& pGridVolume : mGridVolumes)
        {
            mSceneBB |= pGridVolume->getBounds();
        }
    }

    void Scene::updateGeometryInstances(bool forceUpdate)
    {
        if (mGeometryInstanceData.empty()) return;

        bool dataChanged = false;
        const auto& globalMatrices = mpAnimationController->getGlobalMatrices();

        for (auto& inst : mGeometryInstanceData)
        {
            if (inst.getType() == GeometryType::TriangleMesh || inst.getType() == GeometryType::DisplacedTriangleMesh)
            {
                uint32_t prevFlags = inst.flags;

                FALCOR_ASSERT(inst.globalMatrixID < globalMatrices.size());
                const float4x4& transform = globalMatrices[inst.globalMatrixID];
                bool isTransformFlipped = doesTransformFlip(transform);
                bool isObjectFrontFaceCW = getMesh(MeshID::fromSlang(inst.geometryID)).isFrontFaceCW();
                bool isWorldFrontFaceCW = isObjectFrontFaceCW ^ isTransformFlipped;

                if (isTransformFlipped) inst.flags |= (uint32_t)GeometryInstanceFlags::TransformFlipped;
                else inst.flags &= ~(uint32_t)GeometryInstanceFlags::TransformFlipped;

                if (isObjectFrontFaceCW) inst.flags |= (uint32_t)GeometryInstanceFlags::IsObjectFrontFaceCW;
                else inst.flags &= ~(uint32_t)GeometryInstanceFlags::IsObjectFrontFaceCW;

                if (isWorldFrontFaceCW) inst.flags |= (uint32_t)GeometryInstanceFlags::IsWorldFrontFaceCW;
                else inst.flags &= ~(uint32_t)GeometryInstanceFlags::IsWorldFrontFaceCW;

                dataChanged |= (inst.flags != prevFlags);
            }
        }

        if (forceUpdate || dataChanged)
        {
            uint32_t byteSize = (uint32_t)(mGeometryInstanceData.size() * sizeof(GeometryInstanceData));
            mpGeometryInstancesBuffer->setBlob(mGeometryInstanceData.data(), 0, byteSize);
        }
    }

    Scene::UpdateFlags Scene::updateRaytracingAABBData(bool forceUpdate)
    {
        // This function updates the global list of AABBs for all procedural primitives.
        // TODO: Move this code to the GPU. Then the CPU copies of some buffers won't be needed anymore.
        Scene::UpdateFlags flags = Scene::UpdateFlags::None;

        size_t curveAABBCount = 0;
        for (const auto& curve : mCurveDesc) curveAABBCount += curve.indexCount;

        size_t customAABBCount = mCustomPrimitiveAABBs.size();
        size_t totalAABBCount = curveAABBCount + customAABBCount;

        if (totalAABBCount > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Procedural primitive count exceeds the maximum");
        }

        // If there are no procedural primitives, clear the CPU buffer and return.
        // We'll leave the GPU buffer to be lazily re-allocated when needed.
        if (totalAABBCount == 0)
        {
            mRtAABBRaw.clear();
            return flags;
        }

        mRtAABBRaw.resize(totalAABBCount);
        uint32_t index = 0;

        size_t firstUpdated = std::numeric_limits<size_t>::max();
        size_t lastUpdated = 0;

        if (forceUpdate)
        {
            // Compute AABBs of curve segments.
            for (const auto& curve : mCurveDesc)
            {
                // Track range of updated AABBs.
                // TODO: Per-curve flag to indicate changes. For now assume all curves need updating.
                firstUpdated = std::min(firstUpdated, (size_t)index);
                lastUpdated = std::max(lastUpdated, (size_t)index + curve.indexCount);

                const auto* indexData = &mCurveIndexData[curve.ibOffset];
                const auto* staticData = &mCurveStaticData[curve.vbOffset];

                for (uint32_t j = 0; j < curve.indexCount; j++)
                {
                    AABB curveSegBB;
                    uint32_t v = indexData[j];

                    for (uint32_t k = 0; k <= curve.degree; k++)
                    {
                        curveSegBB.include(staticData[v + k].position - float3(staticData[v + k].radius));
                        curveSegBB.include(staticData[v + k].position + float3(staticData[v + k].radius));
                    }

                    mRtAABBRaw[index++] = static_cast<RtAABB>(curveSegBB);
                }
                flags |= Scene::UpdateFlags::CurvesMoved;
            }
            FALCOR_ASSERT(index == curveAABBCount);
        }
        index = (uint32_t)curveAABBCount;

        if (forceUpdate || mCustomPrimitivesChanged || mCustomPrimitivesMoved)
        {
            mCustomPrimitiveAABBOffset = index;

            // Track range of updated AABBs.
            firstUpdated = std::min(firstUpdated, (size_t)index);
            lastUpdated = std::max(lastUpdated, (size_t)index + customAABBCount);

            for (auto& aabb : mCustomPrimitiveAABBs)
            {
                mRtAABBRaw[index++] = static_cast<RtAABB>(aabb);
            }
            FALCOR_ASSERT(index == totalAABBCount);
            flags |= Scene::UpdateFlags::CustomPrimitivesMoved;
        }

        // Create/update GPU buffer. This is used in BLAS creation and also bound to the scene for lookup in shaders.
        // Requires unordered access and will be in Non-Pixel Shader Resource state.
        if (mpRtAABBBuffer == nullptr || mpRtAABBBuffer->getElementCount() < (uint32_t)mRtAABBRaw.size())
        {
            mpRtAABBBuffer = Buffer::createStructured(mpDevice, sizeof(RtAABB), (uint32_t)mRtAABBRaw.size(), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, mRtAABBRaw.data(), false);
            mpRtAABBBuffer->setName("Scene::mpRtAABBBuffer");

            // Bind the new buffer to the scene.
            FALCOR_ASSERT(mpSceneBlock);
            mpSceneBlock->setBuffer(kProceduralPrimAABBBufferName, mpRtAABBBuffer);
        }
        else if (firstUpdated < lastUpdated)
        {
            size_t bytes = sizeof(RtAABB) * mRtAABBRaw.size();
            FALCOR_ASSERT(mpRtAABBBuffer && mpRtAABBBuffer->getSize() >= bytes);

            // Update the modified range of the GPU buffer.
            size_t offset = firstUpdated * sizeof(RtAABB);
            bytes = (lastUpdated - firstUpdated) * sizeof(RtAABB);
            mpRtAABBBuffer->setBlob(mRtAABBRaw.data() + firstUpdated, offset, bytes);
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateDisplacement(RenderContext* pRenderContext, bool forceUpdate)
    {
        if (!hasGeometryType(GeometryType::DisplacedTriangleMesh)) return UpdateFlags::None;

        // For now we assume that displaced meshes are static.
        // Create AABB and AABB update task buffers.
        if (!mDisplacement.pAABBBuffer)
        {
            mDisplacement.meshData.resize(mMeshDesc.size());
            mDisplacement.updateTasks.clear();

            uint32_t AABBOffset = 0;

            for (uint32_t meshID = 0; meshID < mMeshDesc.size(); ++meshID)
            {
                const auto& mesh = mMeshDesc[meshID];

                if (!mesh.isDisplaced())
                {
                    mDisplacement.meshData[meshID] = {};
                    continue;
                }

                uint32_t AABBCount = mesh.getTriangleCount();
                mDisplacement.meshData[meshID] = { AABBOffset, AABBCount };
                AABBOffset += AABBCount;

                DisplacementUpdateTask task;
                task.meshID = meshID;
                task.triangleIndex = 0;
                task.AABBIndex = mDisplacement.meshData[meshID].AABBOffset;
                task.count = mDisplacement.meshData[meshID].AABBCount;
                mDisplacement.updateTasks.push_back(task);
            }

            mDisplacement.pAABBBuffer = Buffer::createStructured(mpDevice, sizeof(RtAABB), AABBOffset, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);

            FALCOR_ASSERT(mDisplacement.updateTasks.size() < std::numeric_limits<uint32_t>::max());
            mDisplacement.pUpdateTasksBuffer = Buffer::createStructured(mpDevice, (uint32_t)sizeof(DisplacementUpdateTask), (uint32_t)mDisplacement.updateTasks.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, mDisplacement.updateTasks.data());
        }

        FALCOR_ASSERT(!mDisplacement.updateTasks.empty());

        // We cannot access the scene parameter block until its finalized.
        if (!mFinalized) return UpdateFlags::None;

        // Update the AABB data.
        if (!mDisplacement.pUpdatePass)
        {
            mDisplacement.pUpdatePass = ComputePass::create(mpDevice, "Scene/Displacement/DisplacementUpdate.cs.slang", "main", getSceneDefines());
            mDisplacement.needsUpdate = true;
        }

        if (mDisplacement.needsUpdate)
        {
            // TODO: Only update objects with modified materials.

            FALCOR_PROFILE(pRenderContext, "updateDisplacement");

            mDisplacement.pUpdatePass->getVars()->setParameterBlock(kParameterBlockName, mpSceneBlock);

            auto var = mDisplacement.pUpdatePass->getRootVar()["CB"];
            var["gTaskCount"] = (uint32_t)mDisplacement.updateTasks.size();
            var["gTasks"] = mDisplacement.pUpdateTasksBuffer;
            var["gAABBs"] = mDisplacement.pAABBBuffer;

            mDisplacement.pUpdatePass->execute(mpDevice->getRenderContext(), uint3(DisplacementUpdateTask::kThreadCount, (uint32_t)mDisplacement.updateTasks.size(), 1));

            mCustomPrimitivesChanged = true; // Trigger a BVH update.
            mDisplacement.needsUpdate = false;
            return UpdateFlags::DisplacementChanged;
        }

        return UpdateFlags::None;
    }

    Scene::UpdateFlags Scene::updateSDFGrids(RenderContext* pRenderContext)
    {
        UpdateFlags updateFlags = UpdateFlags::None;
        if (!is_set(mGeometryTypes, GeometryTypeFlags::SDFGrid)) return updateFlags;

        auto sdfGridsVar = mpSceneBlock->getRootVar()[kSDFGridsArrayName];

        for (uint32_t sdfGridID = 0; sdfGridID < mSDFGrids.size(); ++sdfGridID)
        {
            ref<SDFGrid>& pSDFGrid = mSDFGrids[sdfGridID];
            if (pSDFGrid->mpDevice != mpDevice)
                throw RuntimeError("SDFGrid '{}' was created with a different device than the Scene", pSDFGrid->getName());
            SDFGrid::UpdateFlags sdfGridUpdateFlags = pSDFGrid->update(pRenderContext);

            if (is_set(sdfGridUpdateFlags, SDFGrid::UpdateFlags::AABBsChanged))
            {
                updateGeometryStats();

                // Clear any previous BLAS data. This will trigger a full BLAS/TLAS rebuild.
                // TODO: Support partial rebuild of just the procedural primitives.
                mBlasDataValid = false;
                updateFlags |= Scene::UpdateFlags::SDFGeometryChanged;
            }

            if (is_set(sdfGridUpdateFlags, SDFGrid::UpdateFlags::BuffersReallocated))
            {
                updateGeometryStats();
                pSDFGrid->setShaderData(sdfGridsVar[sdfGridID]);
                updateFlags |= Scene::UpdateFlags::SDFGeometryChanged;
            }
        }

        return updateFlags;
    }

    Scene::UpdateFlags Scene::updateProceduralPrimitives(bool forceUpdate)
    {
        // Update the AABB buffer.
        // The bounds are updated if any primitive has moved or been added/removed.
        Scene::UpdateFlags flags = updateRaytracingAABBData(forceUpdate);

        // Update the procedural primitives metadata.
        if (forceUpdate || mCustomPrimitivesChanged)
        {
            auto var = mpSceneBlock->getRootVar();

            // Update the custom primitives buffer.
            if (!mCustomPrimitiveDesc.empty())
            {
                if (mpCustomPrimitivesBuffer == nullptr || mpCustomPrimitivesBuffer->getElementCount() < (uint32_t)mCustomPrimitiveDesc.size())
                {
                    mpCustomPrimitivesBuffer = Buffer::createStructured(mpDevice, var[kCustomPrimitiveBufferName], (uint32_t)mCustomPrimitiveDesc.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, mCustomPrimitiveDesc.data(), false);
                    mpCustomPrimitivesBuffer->setName("Scene::mpCustomPrimitivesBuffer");

                    // Bind the buffer to the scene.
                    FALCOR_ASSERT(mpSceneBlock);
                    mpSceneBlock->setBuffer(kCustomPrimitiveBufferName, mpCustomPrimitivesBuffer);
                }
                else
                {
                    size_t bytes = sizeof(CustomPrimitiveDesc) * mCustomPrimitiveDesc.size();
                    FALCOR_ASSERT(mpCustomPrimitivesBuffer && mpCustomPrimitivesBuffer->getSize() >= bytes);
                    mpCustomPrimitivesBuffer->setBlob(mCustomPrimitiveDesc.data(), 0, bytes);
                }
            }

            // Update scene constants.
            uint32_t customPrimitiveInstanceOffset = getGeometryInstanceCount();
            uint32_t customPrimitiveInstanceCount = getCustomPrimitiveCount();

            var["customPrimitiveInstanceOffset"] = customPrimitiveInstanceOffset;
            var["customPrimitiveInstanceCount"] = customPrimitiveInstanceCount;
            var["customPrimitiveAABBOffset"] = mCustomPrimitiveAABBOffset;

            flags |= Scene::UpdateFlags::GeometryChanged;
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateParticles(RenderContext* pRenderContext, bool forceUpdate) {
        Scene::UpdateFlags flags = Scene::UpdateFlags::None;

        //Return early if there is no particle system
        if (mParticleSystems.empty())
            return flags;

        //Create/Recreate particle system data
        if (forceUpdate || !mpParticlePointBuffer)
        {
            //Count total number of particles
            uint totalParticles = 0;
            std::vector<ParticlePointDesc> initialData;
            for (auto& ps : mParticleSystems)
            {
                ps.particleBufferOffset = totalParticles;
                totalParticles += ps.numberParticles;
                //Fill initial data
                for (uint i = 0; i < ps.numberParticles; i++)
                {
                    ParticlePointDesc pointDesc;
                    pointDesc.position = ps.spawnPosition;
                    pointDesc.radius = ps.intitialRadius;
                    initialData.push_back(pointDesc);
                }
            }
                
            FALCOR_ASSERT(totalParticles > 0);

            //Create Buffer
            mpParticlePointBuffer = Buffer::createStructured(
                mpDevice, sizeof(ParticlePointDesc), totalParticles, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                Buffer::CpuAccess::None, initialData.data(), false
            );
            mpParticlePointBuffer->setName("Scene::ParticlePointBuffer");        
        }

        // Check if one particle system is active
        bool oneActive = false;
        bool paused = true;
        for (const auto& ps : mParticleSystems)
        {
            oneActive |= ps.active;
            paused &= ps.paused;
        }
            

        if (!oneActive || paused)
            return flags;

        // Update the particle systems
        FALCOR_PROFILE(pRenderContext, "UpdateParticles");
        flags |= UpdateFlags::MeshesChanged;

        //Create compute shader if it does not exist
        if (!mpUpdateParticlesPass)
        {
            Program::Desc desc;
            std::string shaderFile = "Scene/Particles/UpdateParticles.cs.slang";
            desc.addShaderLibrary(shaderFile).csEntry("main").setShaderModel("6_6");

            DefineList defines;
            defines.add(getSceneDefines());

            mpUpdateParticlesPass = ComputePass::create(mpDevice, desc, defines, true);
        }

        //Bind buffer valid for all pass execution
        auto var = mpUpdateParticlesPass->getRootVar();
        

        var["gParticlePoints"] = mpParticlePointBuffer;
        var["gVertexBuffer"] = mpMeshVao->getVertexBuffer(kStaticDataBufferIndex);
        var["gPrevVertexBuffer"] = mpAnimationController->getPrevVertexData();
        bool validAnimationData = mpAnimationController->getPrevVertexData() != nullptr;
        //var["gIndexBuffer"] =  mpMeshVao->getIndexBuffer();

        // Get Camera Data and upload to constant buffer
        const auto& camera = getCamera();
        const float3 cameraU = math::normalize(camera->getData().cameraU);
        const float3 cameraV = math::normalize(camera->getData().cameraV);

        var["ConstantCameraBuffer"]["gCamU"] = cameraU; 
        var["ConstantCameraBuffer"]["gCamV"] = cameraV; 

        uint indexOffset = 0;
        for (auto& ps : mParticleSystems)
        {
            //Get Mesh information
            uint4 vbOffsets;
            uint4 prevVBOffset;
            for (uint i = 0; i < 4; i++)
            {
                auto& mesh = getMesh(ps.meshIDs[i]);
                vbOffsets[i] = mesh.vbOffset;
                prevVBOffset[i] = mesh.prevVbOffset;
            }
                
            var["CB"]["gIndexOffset"] = indexOffset;
            var["CB"]["gNumParticles"] = ps.numberParticles;
            var["CB"]["gCopyPrev"] = validAnimationData;
            var["CB"]["gVBOffsets"] = vbOffsets;
            var["CB"]["gPrevVBOffset"] = prevVBOffset;

            mpUpdateParticlesPass->execute(pRenderContext, float3(ps.numberParticles, 1, 1));
            indexOffset += ps.numberParticles;
            pRenderContext->uavBarrier(mpMeshVao->getVertexBuffer(kStaticDataBufferIndex).get());
            if (validAnimationData)
                pRenderContext->uavBarrier(mpAnimationController->getPrevVertexData().get());
        }

        return flags;
    }

    void Scene::updateGeometryTypes()
    {
        mGeometryTypes = GeometryTypeFlags(0);
        if (getMeshCount() > 0) mGeometryTypes |= GeometryTypeFlags::TriangleMesh;
        auto hasDisplaced = std::any_of(mMeshDesc.begin(), mMeshDesc.end(), [](const auto& mesh) { return mesh.isDisplaced(); });
        if (hasDisplaced) mGeometryTypes |= GeometryTypeFlags::DisplacedTriangleMesh;
        if (getCurveCount() > 0) mGeometryTypes |= GeometryTypeFlags::Curve;
        if (getSDFGridCount() > 0) mGeometryTypes |= GeometryTypeFlags::SDFGrid;
        if (getCustomPrimitiveCount() > 0) mGeometryTypes |= GeometryTypeFlags::Custom;
    }

    void Scene::finalize()
    {
        RenderContext* pRenderContext = mpDevice->getRenderContext();

        // Perform setup that affects the scene defines.
        initSDFGrids();
        mHitInfo.init(*this, mUseCompressedHitInfo);
        updateGeometryTypes();

        // Prepare the materials.
        // This sets up defines and materials parameter block, which are needed for creating the scene parameter block.
        updateMaterials(true);

        // Prepare scene defines.
        // These are currently assumed not to change beyond this point.
        // The defines are needed by the functions below for setting up the scene parameter block.
        updateSceneDefines();
        mPrevSceneDefines = mSceneDefines;

        // Prepare and upload resources.
        // The order of these calls is important as there are dependencies between them.
        createParameterBlock(); // Requires scene defines

        mpAnimationController->animate(pRenderContext, 0); // Requires Scene block to exist
        updateGeometry(pRenderContext, true); // Requires scene defines
        updateGeometryInstances(true);

        // DEMO21: Setup light profile.
        if (mpLightProfile)
        {
            mpLightProfile->bake(mpDevice->getRenderContext());
            mpLightProfile->setShaderData(mpSceneBlock->getRootVar()[kLightProfile]);
        }

        updateBounds();
        createDrawList();
        if (mCameras.size() == 0)
        {
            // Create a new camera to use in the event of a scene with no cameras
            mCameras.push_back(Camera::create());
            resetCamera();
        }
        setCameraController(mCamCtrlType);
        initializeCameras();
        uploadSelectedCamera();
        addViewpoint();
        updateLights(true);
        updateGridVolumes(true);
        updateEnvMap(true);
        uploadResources(); // Upload data after initialization is complete

        // Update stats and UI data.
        updateGeometryStats();
        updateMaterialStats();
        updateLightStats();
        updateGridVolumeStats();
        prepareUI();

        // Validate assumption that scene defines didn't change.
        updateSceneDefines();
        checkInvariant(mSceneDefines == mPrevSceneDefines, "Scene defines changed unexpectedly");

        mFinalized = true;
    }

    void Scene::initializeCameras()
    {
        for (auto& camera : mCameras)
        {
            updateAnimatable(*camera, *mpAnimationController, true);
            camera->beginFrame();
        }
    }

    void Scene::prepareUI()
    {
        for (uint32_t camId = 0; camId < (uint32_t)mCameras.size(); camId++)
        {
            mCameraList.push_back({ camId, mCameras[camId]->getName() });
        }
    }

    void Scene::updateGeometryStats()
    {
        auto& s = mSceneStats;

        s.meshCount = getMeshCount();
        s.meshInstanceCount = 0;
        s.meshInstanceOpaqueCount = 0;
        s.transformCount = getAnimationController()->getGlobalMatrices().size();
        s.uniqueVertexCount = 0;
        s.uniqueTriangleCount = 0;
        s.instancedVertexCount = 0;
        s.instancedTriangleCount = 0;
        s.curveCount = getCurveCount();
        s.curveInstanceCount = 0;
        s.uniqueCurvePointCount = 0;
        s.uniqueCurveSegmentCount = 0;
        s.instancedCurvePointCount = 0;
        s.instancedCurveSegmentCount = 0;
        s.sdfGridCount = getSDFGridCount();
        s.sdfGridDescriptorCount = getSDFGridDescCount();
        s.sdfGridInstancesCount = 0;

        s.customPrimitiveCount = getCustomPrimitiveCount();

        for (uint32_t instanceID = 0; instanceID < getGeometryInstanceCount(); instanceID++)
        {
            const auto& instance = getGeometryInstance(instanceID);
            switch (instance.getType())
            {
            case GeometryType::TriangleMesh:
            case GeometryType::DisplacedTriangleMesh:
            {
                s.meshInstanceCount++;
                const auto& mesh = getMesh(MeshID::fromSlang(instance.geometryID));
                s.instancedVertexCount += mesh.vertexCount;
                s.instancedTriangleCount += mesh.getTriangleCount();

                auto pMaterial = getMaterial(MaterialID::fromSlang(instance.materialID));
                if (pMaterial->isOpaque()) s.meshInstanceOpaqueCount++;
                break;
            }
            case GeometryType::Curve:
            {
                s.curveInstanceCount++;
                const auto& curve = getCurve(CurveID::fromSlang(instance.geometryID));
                s.instancedCurvePointCount += curve.vertexCount;
                s.instancedCurveSegmentCount += curve.getSegmentCount();
                break;
            }
            case GeometryType::SDFGrid:
            {
                s.sdfGridInstancesCount++;
                break;
            }
            }
        }

        for (MeshID meshID{ 0 }; meshID.get() < getMeshCount(); ++meshID)
        {
            const auto& mesh = getMesh(meshID);
            s.uniqueVertexCount += mesh.vertexCount;
            s.uniqueTriangleCount += mesh.getTriangleCount();
        }

        for (CurveID curveID{ 0 }; curveID.get() < getCurveCount(); ++curveID)
        {
            const auto& curve = getCurve(curveID);
            s.uniqueCurvePointCount += curve.vertexCount;
            s.uniqueCurveSegmentCount += curve.getSegmentCount();
        }

        // Calculate memory usage.
        s.indexMemoryInBytes = 0;
        s.vertexMemoryInBytes = 0;
        s.geometryMemoryInBytes = 0;
        s.animationMemoryInBytes = 0;

        if (mpMeshVao)
        {
            const auto& pIB = mpMeshVao->getIndexBuffer();
            const auto& pVB = mpMeshVao->getVertexBuffer(kStaticDataBufferIndex);
            const auto& pDrawID = mpMeshVao->getVertexBuffer(kDrawIdBufferIndex);

            s.indexMemoryInBytes += pIB ? pIB->getSize() : 0;
            s.vertexMemoryInBytes += pVB ? pVB->getSize() : 0;
            s.geometryMemoryInBytes += pDrawID ? pDrawID->getSize() : 0;
        }

        s.curveIndexMemoryInBytes = 0;
        s.curveVertexMemoryInBytes = 0;

        if (mpCurveVao != nullptr)
        {
            const auto& pCurveIB = mpCurveVao->getIndexBuffer();
            const auto& pCurveVB = mpCurveVao->getVertexBuffer(kStaticDataBufferIndex);

            s.curveIndexMemoryInBytes += pCurveIB ? pCurveIB->getSize() : 0;
            s.curveVertexMemoryInBytes += pCurveVB ? pCurveVB->getSize() : 0;
        }

        s.sdfGridMemoryInBytes = 0;

        for (const ref<SDFGrid>& pSDFGrid : mSDFGrids)
        {
            s.sdfGridMemoryInBytes += pSDFGrid->getSize();
        }

        s.geometryMemoryInBytes += mpGeometryInstancesBuffer ? mpGeometryInstancesBuffer->getSize() : 0;
        s.geometryMemoryInBytes += mpMeshesBuffer ? mpMeshesBuffer->getSize() : 0;
        s.geometryMemoryInBytes += mpCurvesBuffer ? mpCurvesBuffer->getSize() : 0;
        s.geometryMemoryInBytes += mpCustomPrimitivesBuffer ? mpCustomPrimitivesBuffer->getSize() : 0;
        s.geometryMemoryInBytes += mpRtAABBBuffer ? mpRtAABBBuffer->getSize() : 0;

        for (const auto& draw : mDrawArgs)
        {
            FALCOR_ASSERT(draw.pBuffer);
            s.geometryMemoryInBytes += draw.pBuffer->getSize();
        }

        s.animationMemoryInBytes += getAnimationController()->getMemoryUsageInBytes();
    }

    void Scene::updateMaterialStats()
    {
        mSceneStats.materials = mpMaterials->getStats();
    }

    void Scene::updateRaytracingBLASStats()
    {
        auto& s = mSceneStats;

        s.blasGroupCount = mBlasGroups.size();
        s.blasCount = mBlasData.size();
        s.blasCompactedCount = 0;
        s.blasOpaqueCount = 0;
        s.blasGeometryCount = 0;
        s.blasOpaqueGeometryCount = 0;
        s.blasMemoryInBytes = 0;
        s.blasScratchMemoryInBytes = 0;

        for (const auto& blas : mBlasData)
        {
            if (blas.useCompaction) s.blasCompactedCount++;
            s.blasMemoryInBytes += blas.blasByteSize;

            // Count number of opaque geometries in BLAS.
            uint64_t opaque = 0;
            for (const auto& desc : blas.geomDescs)
            {
                if (is_set(desc.flags, RtGeometryFlags::Opaque)) opaque++;
            }

            if (opaque == blas.geomDescs.size()) s.blasOpaqueCount++;
            s.blasGeometryCount += blas.geomDescs.size();
            s.blasOpaqueGeometryCount += opaque;
        }

        if (mpBlasScratch) s.blasScratchMemoryInBytes += mpBlasScratch->getSize();
        if (mpBlasStaticWorldMatrices) s.blasScratchMemoryInBytes += mpBlasStaticWorldMatrices->getSize();
    }

    void Scene::updateRaytracingTLASStats()
    {
        auto& s = mSceneStats;

        s.tlasCount = 0;
        s.tlasMemoryInBytes = 0;
        s.tlasScratchMemoryInBytes = 0;

        for (const auto& [i, tlas] : mTlasCache)
        {
            if (tlas.pTlasBuffer)
            {
                s.tlasMemoryInBytes += tlas.pTlasBuffer->getSize();
                s.tlasCount++;
            }
            if (tlas.pInstanceDescs) s.tlasScratchMemoryInBytes += tlas.pInstanceDescs->getSize();
        }
        if (mpTlasScratch) s.tlasScratchMemoryInBytes += mpTlasScratch->getSize();
    }

    void Scene::updateLightStats()
    {
        auto& s = mSceneStats;

        s.activeLightCount = mActiveLights.size();;
        s.totalLightCount = mLights.size();
        s.pointLightCount = 0;
        s.directionalLightCount = 0;
        s.rectLightCount = 0;
        s.discLightCount = 0;
        s.sphereLightCount = 0;
        s.distantLightCount = 0;

        for (const auto& light : mLights)
        {
            switch (light->getType())
            {
            case LightType::Point:
                s.pointLightCount++;
                break;
            case LightType::Directional:
                s.directionalLightCount++;
                break;
            case LightType::Rect:
                s.rectLightCount++;
                break;
            case LightType::Disc:
                s.discLightCount++;
                break;
            case LightType::Sphere:
                s.sphereLightCount++;
                break;
            case LightType::Distant:
                s.distantLightCount++;
                break;
            }
        }

        s.lightsMemoryInBytes = mpLightsBuffer ? mpLightsBuffer->getSize() : 0;
    }

    void Scene::updateGridVolumeStats()
    {
        auto& s = mSceneStats;

        s.gridVolumeCount = mGridVolumes.size();
        s.gridVolumeMemoryInBytes = mpGridVolumesBuffer ? mpGridVolumesBuffer->getSize() : 0;

        s.gridCount = mGrids.size();
        s.gridVoxelCount = 0;
        s.gridMemoryInBytes = 0;

        for (const auto& pGrid : mGrids)
        {
            s.gridVoxelCount += pGrid->getVoxelCount();
            s.gridMemoryInBytes += pGrid->getGridSizeInBytes();
        }
    }

    bool Scene::updateAnimatable(Animatable& animatable, const AnimationController& controller, bool force)
    {
        NodeID nodeID = animatable.getNodeID();

        // It is possible for this to be called on an object with no associated node in the scene graph (kInvalidNode),
        // e.g. non-animated lights. This check ensures that we return immediately instead of trying to check
        // matrices for a non-existent node.
        if (nodeID == NodeID::Invalid()) return false;

        if (force || (animatable.hasAnimation() && animatable.isAnimated()))
        {
            if (!controller.isMatrixChanged(nodeID) && !force) return false;

            float4x4 transform = controller.getGlobalMatrices()[nodeID.get()];
            animatable.updateFromAnimation(transform);
            return true;
        }
        return false;
    }

    void Scene::signalFence(RenderContext* pRenderContext)
    {
        //Create the Fence
        if (!mpFence)
        {
            mpFence = GpuFence::create(mpDevice);
            mpFence->breakStrongReferenceToDevice();
        }

        mFenceSyncLastFrame = mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
    }

    Scene::UpdateFlags Scene::updateSelectedCamera(bool forceUpdate)
    {
        auto camera = mCameras[mSelectedCamera];

        if (forceUpdate || (camera->hasAnimation() && camera->isAnimated()))
        {
            updateAnimatable(*camera, *mpAnimationController, forceUpdate);
        }
        else
        {
            mpCamCtrl->update();
        }

        UpdateFlags flags = UpdateFlags::None;
        auto cameraChanges = camera->beginFrame();
        if (mCameraSwitched || cameraChanges != Camera::Changes::None)
        {
            uploadSelectedCamera();
            if (is_set(cameraChanges, Camera::Changes::Movement)) flags |= UpdateFlags::CameraMoved;
            if ((cameraChanges & (~Camera::Changes::Movement)) != Camera::Changes::None) flags |= UpdateFlags::CameraPropertiesChanged;
            if (mCameraSwitched) flags |= UpdateFlags::CameraSwitched;
        }
        mCameraSwitched = false;
        return flags;
    }

    Scene::UpdateFlags Scene::updateLights(bool forceUpdate)
    {
        Light::Changes combinedChanges = Light::Changes::None;

        // Animate lights and get list of changes.
        for (const auto& light : mLights)
        {
            if (light->isActive() || forceUpdate)
            {
                updateAnimatable(*light, *mpAnimationController, forceUpdate);
            }

            auto changes = light->beginFrame();
            combinedChanges |= changes;
        }

        // Update changed lights.
        uint32_t activeLightIndex = 0;
        mActiveLights.clear();

        for (const auto& light : mLights)
        {
            if (!light->isActive()) continue;

            mActiveLights.push_back(light);

            auto changes = light->getChanges();
            if (changes != Light::Changes::None || is_set(combinedChanges, Light::Changes::Active) || forceUpdate)
            {
                // TODO: This is slow since the buffer is not CPU writable. Copy into CPU buffer and upload once instead.
                mpLightsBuffer->setElement(activeLightIndex, light->getData());
            }

            activeLightIndex++;
        }

        if (combinedChanges != Light::Changes::None || forceUpdate)
        {
            mpSceneBlock->getRootVar()["lightCount"] = (uint32_t)mActiveLights.size();
            updateLightStats();
        }

        // Compute update flags.
        UpdateFlags flags = UpdateFlags::None;
        if (is_set(combinedChanges, Light::Changes::Intensity)) flags |= UpdateFlags::LightIntensityChanged;
        if (is_set(combinedChanges, Light::Changes::Position)) flags |= UpdateFlags::LightsMoved;
        if (is_set(combinedChanges, Light::Changes::Direction)) flags |= UpdateFlags::LightsMoved;
        if (is_set(combinedChanges, Light::Changes::Active)) flags |= UpdateFlags::LightCountChanged;
        const Light::Changes otherChanges = ~(Light::Changes::Intensity | Light::Changes::Position | Light::Changes::Direction | Light::Changes::Active);
        if ((combinedChanges & otherChanges) != Light::Changes::None) flags |= UpdateFlags::LightPropertiesChanged;

        return flags;
    }

    Scene::UpdateFlags Scene::updateGridVolumes(bool forceUpdate)
    {
        GridVolume::UpdateFlags combinedUpdates = GridVolume::UpdateFlags::None;

        // Update animations and get combined updates.
        for (const auto& pGridVolume : mGridVolumes)
        {
            if (pGridVolume->mpDevice != mpDevice)
                throw RuntimeError("GridVolume '{}' was created with a different device than the Scene.", pGridVolume->getName());
            updateAnimatable(*pGridVolume, *mpAnimationController, forceUpdate);
            combinedUpdates |= pGridVolume->getUpdates();
        }

        // Early out if no volumes have changed.
        if (!forceUpdate && combinedUpdates == GridVolume::UpdateFlags::None) return UpdateFlags::None;

        // Upload grids.
        if (forceUpdate)
        {
            auto var = mpSceneBlock->getRootVar()["grids"];
            for (size_t i = 0; i < mGrids.size(); ++i)
            {
                mGrids[i]->setShaderData(var[i]);
            }
        }

        // Upload volumes and clear updates.
        uint32_t volumeIndex = 0;
        for (const auto& pGridVolume : mGridVolumes)
        {
            if (forceUpdate || pGridVolume->getUpdates() != GridVolume::UpdateFlags::None)
            {
                // Fetch copy of volume data.
                auto data = pGridVolume->getData();
                data.densityGrid = (pGridVolume->getDensityGrid() ? mGridIDs.at(pGridVolume->getDensityGrid()) : SdfGridID::Invalid()).getSlang();
                data.emissionGrid = (pGridVolume->getEmissionGrid() ? mGridIDs.at(pGridVolume->getEmissionGrid()) : SdfGridID::Invalid()).getSlang();
                // Merge grid and volume transforms.
                const auto& densityGrid = pGridVolume->getDensityGrid();
                if (densityGrid)
                {
                    data.transform = mul(data.transform, densityGrid->getTransform());
                    data.invTransform = mul(densityGrid->getInvTransform(), data.invTransform);
                }
                mpGridVolumesBuffer->setElement(volumeIndex, data);
            }
            pGridVolume->clearUpdates();
            volumeIndex++;
        }

        mpSceneBlock->getRootVar()["gridVolumeCount"] = (uint32_t)mGridVolumes.size();

        UpdateFlags flags = UpdateFlags::None;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::TransformChanged)) flags |= UpdateFlags::GridVolumesMoved;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::PropertiesChanged)) flags |= UpdateFlags::GridVolumePropertiesChanged;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::GridsChanged)) flags |= UpdateFlags::GridVolumeGridsChanged;
        if (is_set(combinedUpdates, GridVolume::UpdateFlags::BoundsChanged)) flags |= UpdateFlags::GridVolumeBoundsChanged;

        return flags;
    }

    Scene::UpdateFlags Scene::updateEnvMap(bool forceUpdate)
    {
        UpdateFlags flags = UpdateFlags::None;

        if (mpEnvMap)
        {
            if (mpEnvMap->mpDevice != mpDevice)
                throw RuntimeError("EnvMap was created with a different device than the Scene.");
            auto envMapChanges = mpEnvMap->beginFrame();
            if (envMapChanges != EnvMap::Changes::None || mEnvMapChanged || forceUpdate)
            {
                if (envMapChanges != EnvMap::Changes::None) flags |= UpdateFlags::EnvMapPropertiesChanged;
                mpEnvMap->setShaderData(mpSceneBlock->getRootVar()[kEnvMap]);
            }
        }
        mSceneStats.envMapMemoryInBytes = mpEnvMap ? mpEnvMap->getMemoryUsageInBytes() : 0;

        if (mEnvMapChanged)
        {
            flags |= UpdateFlags::EnvMapChanged;
            mEnvMapChanged = false;
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateMaterials(bool forceUpdate)
    {
        // Update material system.
        FALCOR_ASSERT(mpMaterials);
        Material::UpdateFlags materialUpdates = mpMaterials->update(forceUpdate);

        UpdateFlags flags = UpdateFlags::None;
        if (forceUpdate || materialUpdates != Material::UpdateFlags::None)
        {
            flags |= UpdateFlags::MaterialsChanged;

            // Bind materials parameter block to scene.
            if (mpSceneBlock)
            {
                mpSceneBlock->setParameterBlock(kMaterialsBlockName, mpMaterials->getParameterBlock());
            }

            // If displacement parameters have changed, we need to trigger displacement update.
            if (is_set(materialUpdates, Material::UpdateFlags::DisplacementChanged))
            {
                mDisplacement.needsUpdate = true;
            }

            // Update type conformances.
            auto prevTypeConformances = mTypeConformances;
            mTypeConformances = mpMaterials->getTypeConformances();
            if (mTypeConformances != prevTypeConformances)
            {
                flags |= UpdateFlags::TypeConformancesChanged;
            }

            // Pass on update flag indicating shader code changes.
            if (is_set(materialUpdates, Material::UpdateFlags::CodeChanged))
            {
                flags |= UpdateFlags::ShaderCodeChanged;
            }

            updateMaterialStats();
        }

        return flags;
    }

    Scene::UpdateFlags Scene::updateGeometry(RenderContext* pRenderContext, bool forceUpdate)
    {
        UpdateFlags flags = updateProceduralPrimitives(forceUpdate);
        flags |= updateDisplacement(pRenderContext, forceUpdate);
        flags |= updateParticles(pRenderContext, forceUpdate);

        if (forceUpdate || mCustomPrimitivesChanged)
        {
            updateGeometryStats();

            // Mark previous BLAS data as invalid. This will trigger a full BLAS/TLAS rebuild.
            // TODO: Support partial rebuild of just the procedural primitives.
            mBlasDataValid = false;
        }

        mCustomPrimitivesMoved = false;
        mCustomPrimitivesChanged = false;
        return flags;
    }

    Scene::UpdateFlags Scene::update(RenderContext* pRenderContext, double currentTime)
    {
        // Run scene update callback.
        if (mUpdateCallback) mUpdateCallback(ref<Scene>(this), currentTime);

        mUpdates = UpdateFlags::None;

        // Perform updates that may affect the scene defines.
        updateGeometryTypes();
        mUpdates |= updateMaterials(false);

        // Update scene defines.
        // These are currently assumed not to change beyond this point.
        updateSceneDefines();
        if (mSceneDefines != mPrevSceneDefines)
        {
            mUpdates |= UpdateFlags::SceneDefinesChanged;
            mPrevSceneDefines = mSceneDefines;
        }

        // TODO: If scene defined changed we should re-create the scene parameter block
        // here as the defines may affect resource declarations.
        // The assumption is currently that resources do not change after scene creation.
        // When changing this behavior we should make sure that all accesses to the (new)
        // scene block are placed below this point.
        checkInvariant(!is_set(mUpdates, UpdateFlags::SceneDefinesChanged), "Scene doesn't yet support modifications that change the scene defines.");

        if (mpAnimationController->animate(pRenderContext, currentTime))
        {
            mUpdates |= UpdateFlags::SceneGraphChanged;
            if (mpAnimationController->hasSkinnedMeshes()) mUpdates |= UpdateFlags::MeshesChanged;

            for (const auto& inst : mGeometryInstanceData)
            {
                if (mpAnimationController->isMatrixChanged(NodeID{ inst.globalMatrixID }))
                {
                    mUpdates |= UpdateFlags::GeometryMoved;
                }
            }

            // We might end up setting the flag even if curves haven't changed (if looping is disabled for example).
            if (mpAnimationController->hasAnimatedCurveCaches()) mUpdates |= UpdateFlags::CurvesMoved;
            if (mpAnimationController->hasAnimatedMeshCaches()) mUpdates |= UpdateFlags::MeshesChanged;
        }

        for (const auto& pGridVolume : mGridVolumes)
        {
            pGridVolume->updatePlayback(currentTime);
        }

        mUpdates |= updateSelectedCamera(false);
        mUpdates |= updateLights(false);
        mUpdates |= updateGridVolumes(false);
        mUpdates |= updateEnvMap(false);
        mUpdates |= updateGeometry(pRenderContext, false);
        mUpdates |= updateSDFGrids(pRenderContext);
        pRenderContext->flush();

        if (is_set(mUpdates, UpdateFlags::GeometryMoved))
        {
            invalidateTlasCache();
            updateGeometryInstances(false);
        }

        //Signal Fence for this frame
        signalFence(pRenderContext);

        // Update existing BLASes if skinned animation and/or procedural primitives moved.
        bool updateProcedural = is_set(mUpdates, UpdateFlags::CurvesMoved) || is_set(mUpdates, UpdateFlags::CustomPrimitivesMoved);
        bool blasUpdateRequired = is_set(mUpdates, UpdateFlags::MeshesChanged) || updateProcedural;

        if (mBlasDataValid && blasUpdateRequired)
        {
            invalidateTlasCache();
            buildBlas(pRenderContext);
        }

        // Update light collection
        if (mpLightCollection && mpLightCollection->update(pRenderContext))
        {
            mUpdates |= UpdateFlags::LightCollectionChanged;
            mSceneStats.emissiveMemoryInBytes = mpLightCollection->getMemoryUsageInBytes();
        }
        else if (!mpLightCollection)
        {
            mSceneStats.emissiveMemoryInBytes = 0;
        }

        if (mRenderSettings != mPrevRenderSettings)
        {
            mUpdates |= UpdateFlags::RenderSettingsChanged;
            mPrevRenderSettings = mRenderSettings;
        }

        if (mSDFGridConfig != mPrevSDFGridConfig)
        {
            mUpdates |= UpdateFlags::SDFGridConfigChanged;
            mPrevSDFGridConfig = mSDFGridConfig;
        }

        // Validate assumption that scene defines didn't change.
        updateSceneDefines();
        checkInvariant(mSceneDefines == mPrevSceneDefines, "Scene defines changed unexpectedly");

        mFrustumCullingUpdated = false;

        return mUpdates;
    }

    void Scene::renderUI(Gui::Widgets& widget)
    {
        if (mpAnimationController->hasAnimations())
        {
            bool isEnabled = mpAnimationController->isEnabled();
            if (widget.checkbox("Animate Scene", isEnabled)) mpAnimationController->setEnabled(isEnabled);

            if (auto animGroup = widget.group("Animations"))
            {
                mpAnimationController->renderUI(animGroup);
            }
        }

        auto camera = mCameras[mSelectedCamera];
        if (camera->hasAnimation())
        {
            bool isAnimated = camera->isAnimated();
            if (widget.checkbox("Animate Camera", isAnimated)) camera->setIsAnimated(isAnimated);
        }

        auto upDirection = getUpDirection();
        if (widget.dropdown("Up Direction", kUpDirectionList, reinterpret_cast<uint32_t&>(upDirection)))
        {
            setUpDirection(upDirection);
        }

        auto cameraControllerType = getCameraControllerType();
        if (widget.dropdown("Camera Controller", kCameraControllerTypeList, reinterpret_cast<uint32_t&>(cameraControllerType)))
        {
            setCameraController(cameraControllerType);
        }

        if (widget.var("Camera Speed", mCameraSpeed, 0.f, std::numeric_limits<float>::max(), 0.01f))
        {
            mpCamCtrl->setCameraSpeed(mCameraSpeed);
        }

        if (mCameraList.size() > 1)
        {
            uint32_t camIndex = mSelectedCamera;
            if (widget.dropdown("Selected Camera", mCameraList, camIndex)) selectCamera(camIndex);
        }

        if (widget.button("Add Viewpoint")) addViewpoint();

        if (mViewpoints.size() > 1)
        {
            if (widget.button("Remove Viewpoint", true)) removeViewpoint();

            static uint32_t animationLength = 30;
            widget.var("Animation Length", animationLength, 1u, 120u);

            if (widget.button("Save Viewpoints"))
            {
                static const FileDialogFilterVec kFileExtensionFilters = { { "txt", "Text Files"} };
                std::filesystem::path path = "cameraPath.txt";
                if (saveFileDialog(kFileExtensionFilters, path))
                {
                    std::ofstream file(path, std::ios::out);
                    if (file.is_open())
                    {
                        for (uint32_t i = 0; i < mViewpoints.size(); i++)
                        {
                            const Viewpoint& vp = mViewpoints[i];
                            float timePoint = animationLength * float(i) / mViewpoints.size();

                            file << timePoint << ", Transform(";
                            file << "position = float3(" << vp.position.x << ", " << vp.position.y << ", " << vp.position.z << "), ";
                            file << "target = float3(" << vp.target.x << ", " << vp.target.y << ", " << vp.target.z << "), ";
                            file << "up = float3(" << vp.up.x << ", " << vp.up.y << ", " << vp.up.z << "))" << std::endl;
                        }

                        const Viewpoint& vp = mViewpoints[0];
                        file << animationLength << ", Transform(";
                        file << "position = float3(" << vp.position.x << ", " << vp.position.y << ", " << vp.position.z << "), ";
                        file << "target = float3(" << vp.target.x << ", " << vp.target.y << ", " << vp.target.z << "), ";
                        file << "up = float3(" << vp.up.x << ", " << vp.up.y << ", " << vp.up.z << "))" << std::endl;

                        file.close();
                    }
                }
            }

            Gui::DropdownList viewpoints;
            viewpoints.push_back({ 0, "Default Viewpoint" });
            for (uint32_t viewId = 1; viewId < (uint32_t)mViewpoints.size(); viewId++)
            {
                viewpoints.push_back({ viewId, "Viewpoint " + std::to_string(viewId) });
            }
            uint32_t viewIndex = mCurrentViewpoint;
            if (widget.dropdown("Viewpoints", viewpoints, viewIndex)) selectViewpoint(viewIndex);
        }

        if (auto cameraGroup = widget.group("Camera"))
        {
            camera->renderUI(cameraGroup);
        }

        if (auto renderSettingsGroup = widget.group("Render Settings"))
        {
            renderSettingsGroup.checkbox("Use environment light", mRenderSettings.useEnvLight);
            renderSettingsGroup.tooltip("This enables using the environment map as a distant light source.", true);

            renderSettingsGroup.checkbox("Use analytic lights", mRenderSettings.useAnalyticLights);
            renderSettingsGroup.tooltip("This enables using analytic lights.", true);

            renderSettingsGroup.checkbox("Use emissive", mRenderSettings.useEmissiveLights);
            renderSettingsGroup.tooltip("This enables using emissive triangles as lights.", true);

            renderSettingsGroup.checkbox("Use grid volumes", mRenderSettings.useGridVolumes);
            renderSettingsGroup.tooltip("This enables rendering of grid volumes.", true);

            renderSettingsGroup.slider("Diffuse albedo multiplier", mRenderSettings.diffuseAlbedoMultiplier);
        }

        if (mSDFGridConfig.implementation != SDFGrid::Type::None)
        {
            if (auto sdfGridConfigGroup = widget.group("SDF Grid Settings"))
            {
                sdfGridConfigGroup.dropdown("Intersection Method", mSDFGridConfig.intersectionMethodList, reinterpret_cast<uint32_t&>(mSDFGridConfig.intersectionMethod));
                sdfGridConfigGroup.dropdown("Gradient Evaluation Method", mSDFGridConfig.gradientEvaluationMethodList, reinterpret_cast<uint32_t&>(mSDFGridConfig.gradientEvaluationMethod));
                sdfGridConfigGroup.var("Solver Max Iteration Count", mSDFGridConfig.solverMaxIterations, 0u, 512u, 1u);
                sdfGridConfigGroup.text(fmt::format("Data structure: {}", to_string(mSDFGridConfig.implementation)).c_str());
                sdfGridConfigGroup.checkbox("Optimize Visibility Rays", mSDFGridConfig.optimizeVisibilityRays);
            }
        }

        if (auto envMapGroup = widget.group("EnvMap"))
        {
            if (envMapGroup.button("Load"))
            {
                std::filesystem::path path;
                if (openFileDialog(Bitmap::getFileDialogFilters(ResourceFormat::RGBA32Float), path))
                {
                    if (!loadEnvMap(path))
                    {
                        msgBox("Error", fmt::format("Failed to load environment map from '{}'.", path), MsgBoxType::Ok, MsgBoxIcon::Warning);
                    }
                }
            }

            if (mpEnvMap && envMapGroup.button("Clear", true)) setEnvMap(nullptr);

            if (mpEnvMap) mpEnvMap->renderUI(envMapGroup);
        }

        if (auto lightsGroup = widget.group("Lights"))
        {
            uint32_t lightID = 0;
            for (auto& light : mLights)
            {
                auto name = std::to_string(lightID) + ": " + light->getName();
                if (auto lightGroup = lightsGroup.group(name))
                {
                    light->renderUI(lightGroup);
                }
                lightID++;
            }
        }

        if (mpLightProfile)
        {
            if (auto lightProfileGroup = widget.group("Light Profile"))
            {
                mpLightProfile->renderUI(lightProfileGroup);
            }
        }

        if (auto materialsGroup = widget.group("Materials"))
        {
            mpMaterials->renderUI(materialsGroup);
        }

        if (auto volumesGroup = widget.group("Grid volumes"))
        {
            uint32_t volumeID = 0;
            for (auto& pGridVolume : mGridVolumes)
            {
                auto name = std::to_string(volumeID) + ": " + pGridVolume->getName();
                if (auto volumeGroup = volumesGroup.group(name))
                {
                    pGridVolume->renderUI(volumeGroup);
                }
                volumeID++;
            }
        }

        if (auto statsGroup = widget.group("Statistics"))
        {
            const auto& s = mSceneStats;
            const double bytesPerTexel = s.materials.textureTexelCount > 0 ? (double)s.materials.textureMemoryInBytes / s.materials.textureTexelCount : 0.0;
            const double channelsPerTexel = (double)s.materials.textureTexelChannelCount / s.materials.textureTexelCount;

            std::ostringstream oss;
            oss << "Path: " << mPath << std::endl;
            oss << "Bounds: (" << mSceneBB.minPoint.x << "," << mSceneBB.minPoint.y << "," << mSceneBB.minPoint.z << ")-(" << mSceneBB.maxPoint.x << "," << mSceneBB.maxPoint.y << "," << mSceneBB.maxPoint.z << ")" << std::endl;
            oss << "Total scene memory: " << formatByteSize(s.getTotalMemory()) << std::endl;

            // Geometry stats.
            oss << "Geometry stats:" << std::endl
                << "  Mesh count: " << s.meshCount << std::endl
                << "  Mesh instance count (total): " << s.meshInstanceCount << std::endl
                << "  Mesh instance count (opaque): " << s.meshInstanceOpaqueCount << std::endl
                << "  Mesh instance count (non-opaque): " << (s.meshInstanceCount - s.meshInstanceOpaqueCount) << std::endl
                << "  Transform matrix count: " << s.transformCount << std::endl
                << "  Unique triangle count: " << s.uniqueTriangleCount << std::endl
                << "  Unique vertex count: " << s.uniqueVertexCount << std::endl
                << "  Instanced triangle count: " << s.instancedTriangleCount << std::endl
                << "  Instanced vertex count: " << s.instancedVertexCount << std::endl
                << "  Index  buffer memory: " << formatByteSize(s.indexMemoryInBytes) << std::endl
                << "  Vertex buffer memory: " << formatByteSize(s.vertexMemoryInBytes) << std::endl
                << "  Geometry data memory: " << formatByteSize(s.geometryMemoryInBytes) << std::endl
                << "  Animation data memory: " << formatByteSize(s.animationMemoryInBytes) << std::endl
                << "  Curve count: " << s.curveCount << std::endl
                << "  Curve instance count: " << s.curveInstanceCount << std::endl
                << "  Unique curve segment count: " << s.uniqueCurveSegmentCount << std::endl
                << "  Unique curve point count: " << s.uniqueCurvePointCount << std::endl
                << "  Instanced curve segment count: " << s.instancedCurveSegmentCount << std::endl
                << "  Instanced curve point count: " << s.instancedCurvePointCount << std::endl
                << "  Curve index buffer memory: " << formatByteSize(s.curveIndexMemoryInBytes) << std::endl
                << "  Curve vertex buffer memory: " << formatByteSize(s.curveVertexMemoryInBytes) << std::endl
                << "  SDF grid count: " << s.sdfGridCount << std::endl
                << "  SDF grid descriptor count: " << s.sdfGridDescriptorCount << std::endl
                << "  SDF grid instances count: " << s.sdfGridInstancesCount << std::endl
                << "  SDF grid memory: " << formatByteSize(s.sdfGridMemoryInBytes) << std::endl
                << "  Custom primitive count: " << s.customPrimitiveCount << std::endl
                << std::endl;

            // Raytracing stats.
            oss << "Raytracing stats:" << std::endl
                << "  BLAS groups: " << s.blasGroupCount << std::endl
                << "  BLAS count (total): " << s.blasCount << std::endl
                << "  BLAS count (compacted): " << s.blasCompactedCount << std::endl
                << "  BLAS count (opaque): " << s.blasOpaqueCount << std::endl
                << "  BLAS count (non-opaque): " << (s.blasCount - s.blasOpaqueCount) << std::endl
                << "  BLAS geometries (total): " << s.blasGeometryCount << std::endl
                << "  BLAS geometries (opaque): " << s.blasOpaqueGeometryCount << std::endl
                << "  BLAS geometries (non-opaque): " << (s.blasGeometryCount - s.blasOpaqueGeometryCount) << std::endl
                << "  BLAS memory (final): " << formatByteSize(s.blasMemoryInBytes) << std::endl
                << "  BLAS memory (scratch): " << formatByteSize(s.blasScratchMemoryInBytes) << std::endl
                << "  TLAS count: " << s.tlasCount << std::endl
                << "  TLAS memory (final): " << formatByteSize(s.tlasMemoryInBytes) << std::endl
                << "  TLAS memory (scratch): " << formatByteSize(s.tlasScratchMemoryInBytes) << std::endl
                << std::endl;

            // Material stats.
            oss << "Materials stats:" << std::endl
                << "  Material types: " << s.materials.materialTypeCount << std::endl
                << "  Material count (total): " << s.materials.materialCount << std::endl
                << "  Material count (opaque): " << s.materials.materialOpaqueCount << std::endl
                << "  Material count (non-opaque): " << (s.materials.materialCount - s.materials.materialOpaqueCount) << std::endl
                << "  Material memory: " << formatByteSize(s.materials.materialMemoryInBytes) << std::endl
                << "  Texture count (total): " << s.materials.textureCount << std::endl
                << "  Texture count (compressed): " << s.materials.textureCompressedCount << std::endl
                << "  Texture texel count: " << s.materials.textureTexelCount << std::endl
                << "  Texture memory: " << formatByteSize(s.materials.textureMemoryInBytes) << std::endl
                << "  Bytes/texel (average): " << std::fixed << std::setprecision(2) << bytesPerTexel << std::endl
                << "  Channels/texel (average): " << std::fixed << std::setprecision(2) << channelsPerTexel << std::endl
                << std::endl;

            // Analytic light stats.
            oss << "Analytic light stats:" << std::endl
                << "  Active light count: " << s.activeLightCount << std::endl
                << "  Total light count: " << s.totalLightCount << std::endl
                << "  Point light count: " << s.pointLightCount << std::endl
                << "  Directional light count: " << s.directionalLightCount << std::endl
                << "  Rect light count: " << s.rectLightCount << std::endl
                << "  Disc light count: " << s.discLightCount << std::endl
                << "  Sphere light count: " << s.sphereLightCount << std::endl
                << "  Distant light count: " << s.distantLightCount << std::endl
                << "  Analytic lights memory: " << formatByteSize(s.lightsMemoryInBytes) << std::endl
                << std::endl;

            // Emissive light stats.
            oss << "Emissive light stats:" << std::endl;
            if (mpLightCollection)
            {
                const auto& stats = mpLightCollection->getStats(mpDevice->getRenderContext());
                oss << "  Active triangle count: " << stats.trianglesActive << std::endl
                    << "  Active uniform triangle count: " << stats.trianglesActiveUniform << std::endl
                    << "  Active textured triangle count: " << stats.trianglesActiveTextured << std::endl
                    << "  Details:" << std::endl
                    << "    Total mesh count: " << stats.meshLightCount << std::endl
                    << "    Textured mesh count: " << stats.meshesTextured << std::endl
                    << "    Total triangle count: " << stats.triangleCount << std::endl
                    << "    Texture triangle count: " << stats.trianglesTextured << std::endl
                    << "    Culled triangle count: " << stats.trianglesCulled << std::endl
                    << "  Emissive lights memory: " << formatByteSize(s.emissiveMemoryInBytes) << std::endl;
            }
            else
            {
                oss << "  N/A" << std::endl;
            }
            oss << std::endl;

            // Environment map stats.
            oss << "Environment map:" << std::endl;
            if (mpEnvMap)
            {
                oss << "  Filename: " << mpEnvMap->getPath().string() << std::endl
                    << "  Resolution: " << mpEnvMap->getEnvMap()->getWidth() << "x" << mpEnvMap->getEnvMap()->getHeight() << std::endl
                    << "  Texture memory: " << formatByteSize(s.envMapMemoryInBytes) << std::endl;
            }
            else
            {
                oss << "  N/A" << std::endl;
            }
            oss << std::endl;

            // Grid volume stats.
            oss << "Grid volume stats:" << std::endl
                << "  Grid volume count: " << s.gridVolumeCount << std::endl
                << "  Grid volume memory: " << formatByteSize(s.gridVolumeMemoryInBytes) << std::endl
                << std::endl;

            // Grid stats.
            oss << "Grid stats:" << std::endl
                << "  Grid count: " << s.gridCount << std::endl
                << "  Grid voxel count: " << s.gridVoxelCount << std::endl
                << "  Grid memory: " << formatByteSize(s.gridMemoryInBytes) << std::endl
                << std::endl;

            if (statsGroup.button("Print to log")) logInfo("\n" + oss.str());

            statsGroup.text(oss.str());
        }

        // Filtering mode
        // Camera controller
    }

    bool Scene::useEnvBackground() const
    {
        return mpEnvMap != nullptr;
    }

    bool Scene::useEnvLight() const
    {
        return mRenderSettings.useEnvLight && mpEnvMap != nullptr && mpEnvMap->getIntensity() > 0.f;
    }

    bool Scene::useAnalyticLights() const
    {
        return mRenderSettings.useAnalyticLights && mActiveLights.empty() == false;
    }

    bool Scene::useEmissiveLights() const
    {
        return mRenderSettings.useEmissiveLights && mpLightCollection != nullptr && mpLightCollection->getActiveLightCount(mpDevice->getRenderContext()) > 0;
    }

    bool Scene::useGridVolumes() const
    {
        return mRenderSettings.useGridVolumes && mGridVolumes.empty() == false;
    }

    void Scene::setCamera(const ref<Camera>& pCamera)
    {
        auto it = std::find(mCameras.begin(), mCameras.end(), pCamera);
        if (it != mCameras.end())
        {
            selectCamera((uint32_t)std::distance(mCameras.begin(), it));
        }
        else if (pCamera)
        {
            logWarning("Selected camera '{}' does not exist.", pCamera->getName());
        }
    }

    void Scene::selectCamera(uint32_t index)
    {
        if (index == mSelectedCamera) return;
        if (index >= mCameras.size())
        {
            logWarning("Selected camera index {} is invalid.", index);
            return;
        }

        mSelectedCamera = index;
        mCameraSwitched = true;
        setCameraController(mCamCtrlType);
    }

    void Scene::setCameraControlsEnabled(bool value)
    {
        mCameraControlsEnabled = value;

        // Reset the stored input state of the camera controller.
        if (!value) mpCamCtrl->resetInputState();
    }

    void Scene::resetCamera(bool resetDepthRange)
    {
        auto camera = getCamera();
        float radius = mSceneBB.radius();
        camera->setPosition(mSceneBB.center());
        camera->setTarget(mSceneBB.center() + float3(0, 0, -1));
        camera->setUpVector(float3(0, 1, 0));

        if (resetDepthRange)
        {
            float nearZ = std::max(0.1f, radius / 750.0f);
            float farZ = radius * 50;
            camera->setDepthRange(nearZ, farZ);
        }
    }

    void Scene::setCameraSpeed(float speed)
    {
        mCameraSpeed = std::clamp(speed, 0.f, std::numeric_limits<float>::max());
        mpCamCtrl->setCameraSpeed(speed);
    }

    void Scene::setCameraBounds(const AABB& aabb)
    {
        mCameraBounds = aabb;
        mpCamCtrl->setCameraBounds(aabb);
    }

    void Scene::addViewpoint()
    {
        auto camera = getCamera();
        addViewpoint(camera->getPosition(), camera->getTarget(), camera->getUpVector(), mSelectedCamera);
    }

    void Scene::addViewpoint(const float3& position, const float3& target, const float3& up, uint32_t cameraIndex)
    {
        Viewpoint viewpoint = { cameraIndex, position, target, up };
        mViewpoints.push_back(viewpoint);
        mCurrentViewpoint = (uint32_t)mViewpoints.size() - 1;
    }

    void Scene::removeViewpoint()
    {
        if (mCurrentViewpoint == 0)
        {
            logWarning("Cannot remove default viewpoint.");
            return;
        }
        mViewpoints.erase(mViewpoints.begin() + mCurrentViewpoint);
        mCurrentViewpoint = std::min(mCurrentViewpoint, (uint32_t)mViewpoints.size() - 1);
    }

    void Scene::selectViewpoint(uint32_t index)
    {
        if (index >= mViewpoints.size())
        {
            logWarning("Viewpoint does not exist.");
            return;
        }

        auto& viewpoint = mViewpoints[index];
        selectCamera(viewpoint.index);
        auto camera = getCamera();
        camera->setPosition(viewpoint.position);
        camera->setTarget(viewpoint.target);
        camera->setUpVector(viewpoint.up);
        mCurrentViewpoint = index;
    }

    uint32_t Scene::getGeometryCount() const
    {
        // The BLASes currently hold the geometries in the order: meshes, curves, SDF grids, custom primitives.
        // We calculate the total number of geometries as the sum of the respective kind.

        size_t totalGeometries = mMeshDesc.size() + mCurveDesc.size() + mCustomPrimitiveDesc.size() + getSDFGridGeometryCount();
        FALCOR_ASSERT_LT(totalGeometries, std::numeric_limits<uint32_t>::max());
        return (uint32_t)totalGeometries;
    }

    std::vector<GlobalGeometryID> Scene::getGeometryIDs(GeometryType geometryType) const
    {
        if (!hasGeometryType(geometryType)) return {};

        std::vector<GlobalGeometryID> geometryIDs;
        uint32_t geometryCount = getGeometryCount();
        for (GlobalGeometryID geometryID{ 0 }; geometryID.get() < geometryCount; ++geometryID)
        {
            if (getGeometryType(geometryID) == geometryType) geometryIDs.push_back(geometryID);
        }
        return geometryIDs;
    }

    std::vector<GlobalGeometryID> Scene::getGeometryIDs(GeometryType geometryType, MaterialType materialType) const
    {
        if (!hasGeometryType(geometryType)) return {};

        std::vector<GlobalGeometryID> geometryIDs;
        uint32_t geometryCount = getGeometryCount();
        for (GlobalGeometryID geometryID{ 0 }; geometryID.get() < geometryCount; ++geometryID)
        {
            auto pMaterial = getGeometryMaterial(geometryID);
            if (getGeometryType(geometryID) == geometryType && pMaterial && pMaterial->getType() == materialType)
            {
                geometryIDs.push_back(geometryID);
            }
        }
        return geometryIDs;
    }

    std::vector<GlobalGeometryID> Scene::getGeometryIDs(const Material* material) const
    {
        std::vector<GlobalGeometryID> geometryIDs;
        uint32_t geometryCount = getGeometryCount();
        for (GlobalGeometryID geometryID{ 0 }; geometryID.get() < geometryCount; ++geometryID)
        {
            if(getGeometryMaterial(geometryID).get() == material)
                geometryIDs.push_back(geometryID);
        }
        return geometryIDs;
    }

    std::vector<Rectangle> Scene::getGeometryUVTiles(GlobalGeometryID geometryID) const
    {
        GlobalGeometryID::IntType geometryIdx = geometryID.get();

        std::vector<Rectangle> result;
        if (geometryIdx >= mMeshUVTiles.size())
            return result;
        checkInvariant(getGeometryType(geometryID) == GeometryType::TriangleMesh ||
                       getGeometryType(geometryID) == GeometryType::DisplacedTriangleMesh
        , "Can get UV tiles only from triangle-based meshes only");
        return mMeshUVTiles[geometryIdx];
    }

    Scene::GeometryType Scene::getGeometryType(GlobalGeometryID geometryID) const
    {
        // Map global geometry ID to which type of geometry it represents.
        if (geometryID.get() < mMeshDesc.size())
        {
            if (mMeshDesc[geometryID.get()].isDisplaced()) return GeometryType::DisplacedTriangleMesh;
            else return GeometryType::TriangleMesh;
        }
        else if (geometryID.get() < mMeshDesc.size() + mCurveDesc.size()) return GeometryType::Curve;
        else if (geometryID.get() < mMeshDesc.size() + mCurveDesc.size() + mSDFGridDesc.size()) return GeometryType::SDFGrid;
        else if (geometryID.get() < mMeshDesc.size() + mCurveDesc.size() + mSDFGridDesc.size() + mCustomPrimitiveDesc.size()) return GeometryType::Custom;
        else throw ArgumentError("'geometryID' is invalid.");
    }

    uint32_t Scene::getSDFGridGeometryCount() const
    {
        switch (mSDFGridConfig.implementation)
        {
        case SDFGrid::Type::None:
            return 0;
        case SDFGrid::Type::NormalizedDenseGrid:
        case SDFGrid::Type::SparseVoxelOctree:
            return mSDFGrids.empty() ? 0 : 1;
        case SDFGrid::Type::SparseVoxelSet:
        case SDFGrid::Type::SparseBrickSet:
            return (uint32_t)mSDFGrids.size();
        default:
            FALCOR_UNREACHABLE();
            return 0;
        }
    }

    SdfGridID Scene::findSDFGridIDFromGeometryInstanceID(uint32_t geometryInstanceID) const
    {
        NodeID nodeID{ getGeometryInstance(geometryInstanceID).globalMatrixID };

        for (const auto& sdf : mSDFGridDesc)
        {
            auto instanceIt = std::find(sdf.instances.begin(), sdf.instances.end(), nodeID);
            if (instanceIt != sdf.instances.end())
            {
                return sdf.sdfGridID;
            }
        }
        return SdfGridID::Invalid();
    }

    std::vector<uint32_t> Scene::getGeometryInstanceIDsByType(GeometryType type) const
    {
        std::vector<uint32_t> instanceIDs;
        for (uint32_t i = 0; i < getGeometryInstanceCount(); ++i)
        {
            const GeometryInstanceData& instanceData = mGeometryInstanceData[i];
            if (instanceData.getType() == type) instanceIDs.push_back(i);
        }
        return instanceIDs;
    }

    ref<Material> Scene::getGeometryMaterial(GlobalGeometryID geometryID) const
    {
        GlobalGeometryID::IntType geometryIdx = geometryID.get();
        if (geometryIdx < mMeshDesc.size())
        {
            return mpMaterials->getMaterial(MaterialID::fromSlang(mMeshDesc[geometryIdx].materialID));
        }
        geometryIdx -= (uint32_t)mMeshDesc.size();

        if (geometryIdx < mCurveDesc.size())
        {
            return mpMaterials->getMaterial(MaterialID::fromSlang(mCurveDesc[geometryIdx].materialID));
        }
        geometryIdx -= (uint32_t)mCurveDesc.size();

        if (geometryIdx < mSDFGridDesc.size())
        {
            return mpMaterials->getMaterial(mSDFGridDesc[geometryIdx].materialID);
        }
        geometryIdx -= (uint32_t)mSDFGridDesc.size();

        if (geometryIdx < mCustomPrimitiveDesc.size())
        {
            return nullptr;
        }
        geometryIdx -= (uint32_t)mCustomPrimitiveDesc.size();

        throw ArgumentError("'geometryID' is invalid.");
    }

    uint32_t Scene::getCustomPrimitiveIndex(GlobalGeometryID geometryID) const
    {
        if (getGeometryType(geometryID) != GeometryType::Custom)
        {
            throw ArgumentError("'geometryID' ({}) does not refer to a custom primitive.", geometryID);
        }

        size_t customPrimitiveOffset = mMeshDesc.size() + mCurveDesc.size() + mSDFGridDesc.size();
        FALCOR_ASSERT(geometryID.get() >= (uint32_t)customPrimitiveOffset && geometryID.get() < getGeometryCount());
        return geometryID.get() - (uint32_t)customPrimitiveOffset;
    }

    const CustomPrimitiveDesc& Scene::getCustomPrimitive(uint32_t index) const
    {
        if (index >= getCustomPrimitiveCount())
        {
            throw ArgumentError("'index' ({}) is out of range.", index);
        }
        return mCustomPrimitiveDesc[index];
    }

    const AABB& Scene::getCustomPrimitiveAABB(uint32_t index) const
    {
        if (index >= getCustomPrimitiveCount())
        {
            throw ArgumentError("'index' ({}) is out of range.", index);
        }
        return mCustomPrimitiveAABBs[index];
    }

    uint32_t Scene::addCustomPrimitive(uint32_t userID, const AABB& aabb)
    {
        // Currently each custom primitive has exactly one AABB. This may change in the future.
        FALCOR_ASSERT(mCustomPrimitiveDesc.size() == mCustomPrimitiveAABBs.size());
        if (mCustomPrimitiveAABBs.size() > std::numeric_limits<uint32_t>::max())
        {
            throw RuntimeError("Custom primitive count exceeds the maximum");
        }

        const uint32_t index = (uint32_t)mCustomPrimitiveDesc.size();

        CustomPrimitiveDesc desc = {};
        desc.userID = userID;
        desc.aabbOffset = (uint32_t)mCustomPrimitiveAABBs.size();

        mCustomPrimitiveDesc.push_back(desc);
        mCustomPrimitiveAABBs.push_back(aabb);
        mCustomPrimitivesChanged = true;

        return index;
    }

    void Scene::removeCustomPrimitives(uint32_t first, uint32_t last)
    {
        if (first > last || last > getCustomPrimitiveCount())
        {
            throw ArgumentError("'first' ({}) and 'last' ({}) is not a valid range of custom primitives.", first, last);
        }

        if (first == last) return;

        mCustomPrimitiveDesc.erase(mCustomPrimitiveDesc.begin() + first, mCustomPrimitiveDesc.begin() + last);
        mCustomPrimitiveAABBs.erase(mCustomPrimitiveAABBs.begin() + first, mCustomPrimitiveAABBs.begin() + last);

        // Update AABB offsets for all subsequent primitives.
        // The offset is currently redundant since there is one AABB per primitive. This may change in the future.
        for (uint32_t i = first; i < mCustomPrimitiveDesc.size(); i++)
        {
            mCustomPrimitiveDesc[i].aabbOffset = i;
        }

        mCustomPrimitivesChanged = true;
    }

    void Scene::updateCustomPrimitive(uint32_t index, const AABB& aabb)
    {
        if (index >= getCustomPrimitiveCount())
        {
            throw ArgumentError("'index' ({}) is out of range.", index);
        }

        if (mCustomPrimitiveAABBs[index] != aabb)
        {
            mCustomPrimitiveAABBs[index] = aabb;
            mCustomPrimitivesMoved = true;
        }
    }

    ref<GridVolume> Scene::getGridVolumeByName(const std::string& name) const
    {
        for (const auto& v : mGridVolumes)
        {
            if (v->getName() == name) return v;
        }

        return nullptr;
    }

    ref<Light> Scene::getLightByName(const std::string& name) const
    {
        for (const auto& l : mLights)
        {
            if (l->getName() == name) return l;
        }

        return nullptr;
    }

    void Scene::toggleAnimations(bool animate)
    {
        for (auto& light : mLights) light->setIsAnimated(animate);
        for (auto& camera : mCameras) camera->setIsAnimated(animate);
        mpAnimationController->setEnabled(animate);
    }

    bool Scene::hasDynamicGeometry() {
        bool hasDynamic = false;
        //Take the drawArgs to quickly check for dynamic geometry
        for (const auto& drawArgs : mDrawArgs)
            hasDynamic |= drawArgs.isDynamic;
        return hasDynamic;
    }

    void Scene::setBlasUpdateMode(UpdateMode mode)
    {
        if (mode != mBlasUpdateMode) mRebuildBlas = true;
        mBlasUpdateMode = mode;
    }

    void Scene::createDrawList()
    {
        // This function creates argument buffers for draw indirect calls to rasterize the scene.
        // The updateGeometryInstances() function must have been called before so that the flags are accurate.
        //
        // Note that we create four draw buffers to handle all combinations of:
        // 1) mesh is using 16- or 32-bit indices,
        // 2) mesh triangle winding is CW or CCW after transformation.
        //
        // TODO: Update the draw args if a mesh undergoes animation that flips the winding.

        //mGeometryInstanceData[desc.instanceID].globalMatrixID;

        mDrawArgs.clear();
        mDrawArgsInstanceIDs.clear();

        //Correctly mark animated geometry

        //Create a set containing all nodes with animation data
        std::set<uint32_t> animatedNodes;
        for (const auto& animation : mpAnimationController->getAnimations())
        {
            uint nodeID = animation->getNodeID().get();
            if (nodeID != NodeID::kInvalidID) {
                animatedNodes.insert(nodeID);
            } 
        }
        //Add all changed matrixes to the node map
        for (uint node = 0; node < mpAnimationController->getGlobalMatrices().size(); node++)
        {
            NodeID nodeID = NodeID{node};
            if (mpAnimationController->isMatrixChanged(nodeID))
            {
                if (node != NodeID::kInvalidID)
                {
                    animatedNodes.insert(node);
                } 
            }
        }

        for (const auto& instance : mGeometryInstanceData)
        {
            //Skip non triangles
            if (instance.getType() != GeometryType::TriangleMesh)
                continue;

            const uint32_t instNodeID = instance.globalMatrixID;
            auto& mesh = mMeshDesc[instance.geometryID];
            //If node was found mark mesh as isAnimated
            if (auto search = animatedNodes.find(instNodeID); search != animatedNodes.end())
            {
                //auto& mesh = mMeshDesc[instance.geometryID];
                if(!mesh.isAnimated())
                    mesh.flags |= (uint32_t)MeshFlags::IsAnimated;
            }
        }

        // Helper to create the draw-indirect buffer. This is for opaque clockwise or counterclockwise meshes that throw shadows 
        auto createDrawBuffer = [this](
                                    const auto& drawMeshes, bool ccw, bool isDynamic, const std::vector<uint>& instanceIDs,
                                    ResourceFormat ibFormat = ResourceFormat::Unknown
                                )
        {
            if (drawMeshes.size() > 0)
            {
                DrawArgs draw;
                draw.pBuffer = Buffer::create(mpDevice, sizeof(drawMeshes[0]) * drawMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None, drawMeshes.data());
                draw.pBuffer->setName("Scene draw buffer");
                FALCOR_ASSERT(drawMeshes.size() <= std::numeric_limits<uint32_t>::max());
                draw.count = (uint32_t)drawMeshes.size();
                draw.ccw = ccw;
                draw.ignoreWinding = false;
                draw.isDynamic = isDynamic;
                draw.isCastShadow = true;
                draw.isOpaque = true;
                draw.ibFormat = ibFormat;

                mDrawArgs.push_back(draw);
                mDrawArgsInstanceIDs.push_back(instanceIDs);
            }
        };

        // Helper to create the draw-indirect buffer for double sided meshes. Non Shadow throwable, as well as non opaque are always flagged as double sided
        auto createDoubleSidedDrawBuffer = [this](const auto& drawMeshes, bool isDynamic, bool isCastShadow, bool isOpaque,
                                    const std::vector<uint>& instanceIDs, ResourceFormat ibFormat = ResourceFormat::Unknown, ParticleOrientationMode particleOrientationMode = ParticleOrientationMode::None)
        {
            if (drawMeshes.size() > 0)
            {
                DrawArgs draw;
                draw.pBuffer = Buffer::create(
                    mpDevice, sizeof(drawMeshes[0]) * drawMeshes.size(), Resource::BindFlags::IndirectArg, Buffer::CpuAccess::None,
                    drawMeshes.data()
                );
                draw.pBuffer->setName("Scene draw buffer");
                FALCOR_ASSERT(drawMeshes.size() <= std::numeric_limits<uint32_t>::max());
                draw.count = (uint32_t)drawMeshes.size();
                draw.ccw = true; // ignored
                draw.ignoreWinding = true; //Implicid
                draw.isDynamic = isDynamic;
                draw.isCastShadow = isCastShadow;
                draw.isOpaque = isOpaque;
                draw.ibFormat = ibFormat;
                draw.particleOrientationMode = particleOrientationMode;

                mDrawArgs.push_back(draw);
                mDrawArgsInstanceIDs.push_back(instanceIDs);
            }
        };

        if (hasIndexBuffer())
        {
            //Objects with shadow ignore are always drawn without face culling
            std::vector<DrawIndexedArguments> drawClockwiseMeshes[2], drawCounterClockwiseMeshes[2], drawDoubleSidedMeshes[8]; //32Bit indices; 0 static ; 1 dynamic; 2,3 +non shadow casting , 4-7 +non-opaque
            std::vector<DrawIndexedArguments> drawClockwiseMeshes16B[2], drawCounterClockwiseMeshes16B[4], drawDoubleSidedMeshes16B[8]; //16Bit indices; 0 static ; 1 dynamic; 2,3 +non shadow casting , 4-7 +non-opaque
            std::vector<uint> drawClockwiseMeshesIDs[2], drawCounterClockwiseMeshesIDs[2], drawDoubleSidedMeshesIDs[8];       //32Bit indices; 0 static ; 1 dynamic; 2,3 +non shadow casting , 4-7 +non-opaque
            std::vector<uint> drawClockwiseMeshesIDs16B[2], drawCounterClockwiseMeshesIDs16B[2], drawDoubleSidedMeshesIDs16B[8];       //16Bit indices; 0 static ; 1 dynamic; 2,3 +non shadow casting , 4-7 +non-opaque
            std::vector<DrawIndexedArguments> drawParticles[4]; //Mesh groups for particles. They only exists in a specific configuration
            std::vector<uint> drawParticlesMeshIDs[4];

            uint32_t instanceID = 0;
            //Create a map to identify particles
            std::map<uint32_t, uint32_t> particleMap;
            
            for (uint i = 0; i < mParticleSystems.size(); i++)
            {
                auto& ps = mParticleSystems[i];
                /*
                particleMap.insert((uint32_t)ps.meshIDs[0].get(), (uint32_t)ParticleOrientationMode::Camera);
                particleMap.insert((uint32_t)ps.meshIDs[1].get(), (uint32_t)ParticleOrientationMode::XY_Plane);
                particleMap.insert((uint32_t)ps.meshIDs[2].get(), (uint32_t)ParticleOrientationMode::YZ_Plane);
                particleMap.insert((uint32_t)ps.meshIDs[3].get(), (uint32_t)ParticleOrientationMode::XZ_Plane);
                */
                particleMap[(uint32_t)ps.meshIDs[0].get()] = (uint32_t)ParticleOrientationMode::Camera;
                particleMap[(uint32_t)ps.meshIDs[1].get()] = (uint32_t)ParticleOrientationMode::XY_Plane;
                particleMap[(uint32_t)ps.meshIDs[2].get()] = (uint32_t)ParticleOrientationMode::YZ_Plane;
                particleMap[(uint32_t)ps.meshIDs[3].get()] = (uint32_t)ParticleOrientationMode::XZ_Plane;
            }
            
                
            for (const auto& instance : mGeometryInstanceData)
            {
                if (instance.getType() != GeometryType::TriangleMesh) continue;

                const auto& mesh = mMeshDesc[instance.geometryID];
                bool use16Bit = mesh.use16BitIndices();
                bool isDynamic = mesh.isAnimated() || mesh.isDynamic(); 
                const auto mat = getMaterial(MaterialID::fromSlang(mesh.materialID));
                bool isCastShadow = mat->isCastShadow();
                bool isParticle = particleMap.count(instance.geometryID) > 0;

                DrawIndexedArguments draw;
                draw.IndexCountPerInstance = mesh.indexCount;
                draw.InstanceCount = 1;
                draw.StartIndexLocation = mesh.ibOffset * (use16Bit ? 2 : 1);
                draw.BaseVertexLocation = mesh.vbOffset;
                draw.StartInstanceLocation = instanceID;

                int i = isDynamic ? 1 : 0;
                if (isParticle)
                {
                    uint orientationID = particleMap[instance.geometryID] - 1u; //Orientation enum starts at 1 if the particle is set
                    
                    drawParticles[orientationID].push_back(draw);
                    drawParticlesMeshIDs[orientationID].push_back(instance.geometryID);
                }
                else if (mat->isDoubleSided() || !mat->isOpaque() || !isCastShadow)
                {
                    //Offset the non shadowed materials
                    if (!isCastShadow)
                        i += 2;
                    if (!mat->isOpaque())
                        i += 4;
                    if (use16Bit){
                        drawDoubleSidedMeshes16B[i].push_back(draw);
                        drawDoubleSidedMeshesIDs16B[i].push_back(instanceID);
                    }
                    else{
                        drawDoubleSidedMeshes[i].push_back(draw);
                        drawDoubleSidedMeshesIDs[i].push_back(instanceID);
                    }
                    
                }
                else if (instance.isWorldFrontFaceCW())
                {
                    if (use16Bit){
                        drawClockwiseMeshes16B[i].push_back(draw);
                        drawClockwiseMeshesIDs16B[i].push_back(instanceID);
                    }else{
                        drawClockwiseMeshes[i].push_back(draw);
                        drawClockwiseMeshesIDs[i].push_back(instanceID);
                    }
                    
                }
                else
                {
                    if (use16Bit)
                    {
                        drawCounterClockwiseMeshes16B[i].push_back(draw);
                        drawCounterClockwiseMeshesIDs16B[i].push_back(instanceID);
                    }else{
                        drawCounterClockwiseMeshes[i].push_back(draw);
                        drawCounterClockwiseMeshesIDs[i].push_back(instanceID);
                    }
                    
                }

                instanceID++;
            }

            //Clockwise
            createDrawBuffer(drawClockwiseMeshes16B[0], false, false, drawClockwiseMeshesIDs16B[0], ResourceFormat::R16Uint);
            createDrawBuffer(drawClockwiseMeshes16B[1], false, true, drawClockwiseMeshesIDs16B[1], ResourceFormat::R16Uint);
            createDrawBuffer(drawClockwiseMeshes[0], false, false, drawClockwiseMeshesIDs[0], ResourceFormat::R32Uint);
            createDrawBuffer(drawClockwiseMeshes[1], false, true, drawClockwiseMeshesIDs[1], ResourceFormat::R32Uint);

            // Counter Clockwise
            createDrawBuffer(drawCounterClockwiseMeshes16B[0], true, false, drawCounterClockwiseMeshesIDs16B[0], ResourceFormat::R16Uint);
            createDrawBuffer(drawCounterClockwiseMeshes16B[1], true, true, drawCounterClockwiseMeshesIDs16B[1], ResourceFormat::R16Uint);
            createDrawBuffer(drawCounterClockwiseMeshes[0], true, false, drawCounterClockwiseMeshesIDs[0], ResourceFormat::R32Uint);
            createDrawBuffer(drawCounterClockwiseMeshes[1], true, true, drawCounterClockwiseMeshesIDs[1], ResourceFormat::R32Uint);

            //Double Sided 16Bit indices            
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[0], false, true, true, drawDoubleSidedMeshesIDs16B[0], ResourceFormat::R16Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[1], true, true, true, drawDoubleSidedMeshesIDs16B[1], ResourceFormat::R16Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[2], false, false, true, drawDoubleSidedMeshesIDs16B[2], ResourceFormat::R16Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[3], true, false, true, drawDoubleSidedMeshesIDs16B[3], ResourceFormat::R16Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[4], false, true, false, drawDoubleSidedMeshesIDs16B[4], ResourceFormat::R16Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[5], true, true, false, drawDoubleSidedMeshesIDs16B[5], ResourceFormat::R16Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[6], false, false, false, drawDoubleSidedMeshesIDs16B[6], ResourceFormat::R16Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes16B[7], true, false, false, drawDoubleSidedMeshesIDs16B[7], ResourceFormat::R16Uint);

            //Double Sided 32Bit indices   
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[0], false, true, true, drawDoubleSidedMeshesIDs[0], ResourceFormat::R32Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[1], true, true, true, drawDoubleSidedMeshesIDs[1], ResourceFormat::R32Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[2], false, false, true, drawDoubleSidedMeshesIDs[2], ResourceFormat::R32Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[3], true, false, true, drawDoubleSidedMeshesIDs[3], ResourceFormat::R32Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[4], false, true, false, drawDoubleSidedMeshesIDs[4], ResourceFormat::R32Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[5], true, true, false, drawDoubleSidedMeshesIDs[5], ResourceFormat::R32Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[6], false, false, false, drawDoubleSidedMeshesIDs[6], ResourceFormat::R32Uint);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[7], true, false, false, drawDoubleSidedMeshesIDs[7], ResourceFormat::R32Uint);

            //Particles
            createDoubleSidedDrawBuffer(drawParticles[0], true, true, false, drawParticlesMeshIDs[0], ResourceFormat::R32Uint, ParticleOrientationMode::Camera);
            createDoubleSidedDrawBuffer(drawParticles[1], true, true, false, drawParticlesMeshIDs[1], ResourceFormat::R32Uint, ParticleOrientationMode::XY_Plane);
            createDoubleSidedDrawBuffer(drawParticles[2], true, true, false, drawParticlesMeshIDs[2], ResourceFormat::R32Uint, ParticleOrientationMode::YZ_Plane);
            createDoubleSidedDrawBuffer(drawParticles[3], true, true, false, drawParticlesMeshIDs[3], ResourceFormat::R32Uint, ParticleOrientationMode::XZ_Plane);
        }
        else
        {
            std::vector<DrawArguments> drawClockwiseMeshes[2], drawCounterClockwiseMeshes[2], drawDoubleSidedMeshes[8]; //0 static ; 1 dynamic; 2,3 +non shadow casting , 4-7 +non-opaque
            std::vector<uint> drawClockwiseMeshesIDs[2], drawCounterClockwiseMeshesIDs[2], drawDoubleSidedMeshesIDs[8]; //0 static ; 1 dynamic; 2,3 +non shadow casting , 4-7 +non-opaque

            uint32_t instanceID = 0;
            for (const auto& instance : mGeometryInstanceData)
            {
                if (instance.getType() != GeometryType::TriangleMesh) continue;

                const auto& mesh = mMeshDesc[instance.geometryID];
                FALCOR_ASSERT(mesh.indexCount == 0);
                const auto mat = getMaterial(MaterialID::fromSlang(mesh.materialID));
                bool isDynamic = mesh.isAnimated() || mesh.isDynamic(); 
                bool isCastShadow = mat->isCastShadow();

                DrawArguments draw;
                draw.VertexCountPerInstance = mesh.vertexCount;
                draw.InstanceCount = 1;
                draw.StartVertexLocation = mesh.vbOffset;
                draw.StartInstanceLocation = instanceID;
                uint i = isDynamic ? 1 : 0;
                
                if (mat->isDoubleSided() || !mat->isOpaque() || !isCastShadow)
                {
                    if (!isCastShadow)
                        i += 2;
                    if (!mat->isOpaque())
                        i += 4;
                    drawDoubleSidedMeshes[i].push_back(draw);
                    drawDoubleSidedMeshesIDs[i].push_back(instanceID);
                }
                else if (instance.isWorldFrontFaceCW())
                {
                    drawClockwiseMeshes[i].push_back(draw);
                    drawClockwiseMeshesIDs[i].push_back(instanceID);
                }
                else{
                    drawCounterClockwiseMeshes[i].push_back(draw);
                    drawCounterClockwiseMeshesIDs[i].push_back(instanceID);
                }

                instanceID++;   
            }

            //Clockwise
            createDrawBuffer(drawClockwiseMeshes[0], false, false, drawClockwiseMeshesIDs[0]);
            createDrawBuffer(drawClockwiseMeshes[1], false, true, drawClockwiseMeshesIDs[1]);
            //Counter-Clockwise
            createDrawBuffer(drawCounterClockwiseMeshes[0], true, false, drawCounterClockwiseMeshesIDs[0]);
            createDrawBuffer(drawCounterClockwiseMeshes[1], true, true, drawCounterClockwiseMeshesIDs[1]);
            //DoubleSided or shadow throwable or non-opaque
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[0], false, true, true, drawDoubleSidedMeshesIDs[0]);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[1], true, true, true, drawDoubleSidedMeshesIDs[1]);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[2], false, false, true, drawDoubleSidedMeshesIDs[2]);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[3], true, false, true, drawDoubleSidedMeshesIDs[3]);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[4], false, true, false, drawDoubleSidedMeshesIDs[4]);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[5], true, true, false, drawDoubleSidedMeshesIDs[5]);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[6], false, false, false, drawDoubleSidedMeshesIDs[6]);
            createDoubleSidedDrawBuffer(drawDoubleSidedMeshes[7], true, false, false, drawDoubleSidedMeshesIDs[7]);
        }
    }

    void Scene::initGeomDesc(RenderContext* pRenderContext)
    {
        // This function initializes all geometry descs to prepare for BLAS build.
        // If the scene has no geometries the 'mBlasData' array will be left empty.

        // First compute total number of BLASes to build:
        // - Triangle meshes have been grouped beforehand and we build one BLAS per mesh group.
        // - Curves and procedural primitives are currently placed in a single BLAS each, if they exist.
        // - SDF grids are placed in individual BLASes.
        const uint32_t totalBlasCount = (uint32_t)mMeshGroups.size() + (mCurveDesc.empty() ? 0 : 1) + getSDFGridGeometryCount() + (mCustomPrimitiveDesc.empty() ? 0 : 1);

        mBlasData.clear();
        mBlasData.resize(totalBlasCount);
        mRebuildBlas = true;

        if (!mMeshGroups.empty())
        {
            FALCOR_ASSERT(mpMeshVao);
            const ref<VertexBufferLayout>& pVbLayout = mpMeshVao->getVertexLayout()->getBufferLayout(kStaticDataBufferIndex);
            const ref<Buffer>& pVb = mpMeshVao->getVertexBuffer(kStaticDataBufferIndex);
            const ref<Buffer>& pIb = mpMeshVao->getIndexBuffer();
            const auto& globalMatrices = mpAnimationController->getGlobalMatrices();

            // Normally static geometry is already pre-transformed to world space by the SceneBuilder,
            // but if that isn't the case, we let DXR transform static geometry as part of the BLAS build.
            // For this we need the GPU address of the transform matrix of each mesh in row-major format.
            // Since glm uses column-major format we lazily create a buffer with the transposed matrices.
            // Note that this is sufficient to do once only as the transforms for static meshes can't change.
            // TODO: Use AnimationController's matrix buffer directly when we've switched to a row-major matrix library.
            auto getStaticMatricesBuffer = [&]()
            {
                if (!mpBlasStaticWorldMatrices)
                {
                    std::vector<float4x4> transposedMatrices;
                    transposedMatrices.reserve(globalMatrices.size());
                    for (const auto& m : globalMatrices) transposedMatrices.push_back(transpose(m));

                    uint32_t float4Count = (uint32_t)transposedMatrices.size() * 4;
                    mpBlasStaticWorldMatrices = Buffer::createStructured(mpDevice, sizeof(float4), float4Count, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, transposedMatrices.data(), false);
                    mpBlasStaticWorldMatrices->setName("Scene::mpBlasStaticWorldMatrices");

                    // Transition the resource to non-pixel shader state as expected by DXR.
                    pRenderContext->resourceBarrier(mpBlasStaticWorldMatrices.get(), Resource::State::NonPixelShader);
                }
                return mpBlasStaticWorldMatrices;
            };

            // Iterate over the mesh groups. One BLAS will be created for each group.
            // Each BLAS may contain multiple geometries.
            for (size_t i = 0; i < mMeshGroups.size(); i++)
            {
                const auto& meshList = mMeshGroups[i].meshList;
                const bool isStatic = mMeshGroups[i].isStatic;
                const bool isDisplaced = mMeshGroups[i].isDisplaced;
                const bool isParticle = mMeshGroups[i].isParticle(); // Particles have some special properties

                auto& blas = mBlasData[i];
                auto& geomDescs = blas.geomDescs;
                geomDescs.resize(meshList.size());
                blas.hasProceduralPrimitives = false;
                blas.isParticle = isParticle;

                // Track what types of triangle winding exist in the final BLAS.
                // The SceneBuilder should have ensured winding is consistent, but keeping the check here as a safeguard.
                uint32_t triangleWindings = 0; // bit 0 indicates CW, bit 1 CCW.

                for (size_t j = 0; j < meshList.size(); j++)
                {
                    const MeshID meshID = meshList[j];
                    const MeshDesc& mesh = mMeshDesc[meshID.get()];
                    bool frontFaceCW = mesh.isFrontFaceCW();
                    blas.hasDynamicMesh |= mesh.isDynamic() | isParticle;

                    RtGeometryDesc& desc = geomDescs[j];

                    if (!isDisplaced)
                    {
                        desc.type = RtGeometryType::Triangles;
                        desc.content.triangles.transform3x4 = 0; // The default is no transform

                        if (isStatic)
                        {
                            // Static meshes will be pre-transformed when building the BLAS.
                            // Lookup the matrix ID here. If it is an identity matrix, no action is needed.
                            FALCOR_ASSERT(mMeshIdToInstanceIds[meshID.get()].size() == 1);
                            uint32_t instanceID = mMeshIdToInstanceIds[meshID.get()][0];
                            FALCOR_ASSERT(instanceID < mGeometryInstanceData.size());
                            uint32_t matrixID = mGeometryInstanceData[instanceID].globalMatrixID;

                            FALCOR_ASSERT(matrixID < globalMatrices.size());
                            if (globalMatrices[matrixID] != float4x4::identity())
                            {
                                // Get the GPU address of the transform in row-major format.
                                desc.content.triangles.transform3x4 = getStaticMatricesBuffer()->getGpuAddress() + matrixID * 64ull;

                                if (determinant(globalMatrices[matrixID]) < 0.f) frontFaceCW = !frontFaceCW;
                            }
                        }
                        triangleWindings |= frontFaceCW ? 1 : 2;

                        // If this is an opaque mesh, set the opaque flag
                        auto pMaterial = mpMaterials->getMaterial(MaterialID::fromSlang(mesh.materialID));
                        desc.flags = pMaterial->isOpaque() && !isParticle ? RtGeometryFlags::Opaque : RtGeometryFlags::None;
                        desc.flags |= mAdditionalASGeometryFlags;

                        // Set the position data
                        desc.content.triangles.vertexData = pVb->getGpuAddress() + (mesh.vbOffset * pVbLayout->getStride());
                        desc.content.triangles.vertexStride = pVbLayout->getStride();
                        desc.content.triangles.vertexCount = mesh.vertexCount;
                        desc.content.triangles.vertexFormat = pVbLayout->getElementFormat(0);

                        // Set index data
                        if (pIb)
                        {
                            // The global index data is stored in a dword array.
                            // Each mesh specifies whether its indices are in 16-bit or 32-bit format.
                            ResourceFormat ibFormat = mesh.use16BitIndices() ? ResourceFormat::R16Uint : ResourceFormat::R32Uint;
                            desc.content.triangles.indexData = pIb->getGpuAddress() + mesh.ibOffset * sizeof(uint32_t);
                            desc.content.triangles.indexCount = mesh.indexCount;
                            desc.content.triangles.indexFormat = ibFormat;
                        }
                        else
                        {
                            FALCOR_ASSERT(mesh.indexCount == 0);
                            desc.content.triangles.indexData = 0;
                            desc.content.triangles.indexCount = 0;
                            desc.content.triangles.indexFormat = ResourceFormat::Unknown;
                        }
                    }
                    else
                    {
                        // Displaced triangle mesh, requires custom intersection.
                        desc.type = RtGeometryType::ProcedurePrimitives;
                        desc.flags = RtGeometryFlags::Opaque;

                        desc.content.proceduralAABBs.count = mDisplacement.meshData[meshID.get()].AABBCount;
                        uint64_t bbStartOffset = mDisplacement.meshData[meshID.get()].AABBOffset * sizeof(RtAABB);
                        desc.content.proceduralAABBs.data = mDisplacement.pAABBBuffer->getGpuAddress() + bbStartOffset;
                        desc.content.proceduralAABBs.stride = sizeof(RtAABB);
                    }
                }

                FALCOR_ASSERT(!(isStatic && blas.hasDynamicMesh && !isParticle));

                if (triangleWindings == 0x3)
                {
                    logWarning("Mesh group {} has mixed triangle winding. Back/front face culling won't work correctly.", i);
                }
            }
        }

        // Procedural primitives other than displaced triangle meshes and SDF grids are placed in two BLASes at the end.
        // The geometries in these BLASes are using the following layout:
        //
        //  +----------+----------+-----+----------+
        //  |          |          |     |          |
        //  |  Curve0  |  Curve1  | ... |  CurveM  |
        //  |          |          |     |          |
        //  |          |          |     |          |
        //  +----------+----------+-----+----------+
        // SDF grids either create a shared BLAS or one BLAS per SDF grid:
        //  +----------+          +----------+ +----------+     +----------+
        //  |          |          |          | |          |     |          |
        //  | SDFGrid  |          | SDFGrid0 | | SDFGrid1 | ... | SDFGridN |
        //  |  Shared  |    or    |          | |          |     |          |
        //  | Geometry |          |          | |          |     |          |
        //  +----------+          +----------+ +----------+     +----------+
        //
        //  +----------+----------+-----+----------+
        //  |          |          |     |          |
        //  |  Custom  |  Custom  | ... |  Custom  |
        //  |  Prim0   |  Prim1   |     |  PrimN   |
        //  |          |          |     |          |
        //  +----------+----------+-----+----------+
        //
        // Each procedural primitive indexes a range of AABBs in a global AABB buffer.
        //
        size_t blasDataIndex = mMeshGroups.size();
        uint64_t bbAddressOffset = 0;
        if (!mCurveDesc.empty())
        {
            FALCOR_ASSERT(mpRtAABBBuffer && mpRtAABBBuffer->getElementCount() >= mRtAABBRaw.size());

            auto& blas = mBlasData[blasDataIndex++];
            blas.geomDescs.resize(mCurveDesc.size());
            blas.hasProceduralPrimitives = true;
            blas.hasDynamicCurve |= mpAnimationController->hasAnimatedCurveCaches();

            uint32_t geomIndexOffset = 0;

            for (const auto& curve : mCurveDesc)
            {
                // One geometry desc per curve.
                RtGeometryDesc& desc = blas.geomDescs[geomIndexOffset++];

                desc.type = RtGeometryType::ProcedurePrimitives;
                desc.flags = RtGeometryFlags::Opaque;
                desc.content.proceduralAABBs.count = curve.indexCount;
                desc.content.proceduralAABBs.data = mpRtAABBBuffer->getGpuAddress() + bbAddressOffset;
                desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                bbAddressOffset += sizeof(RtAABB) * curve.indexCount;
            }
        }

        if (!mSDFGrids.empty())
        {
            if (mSDFGridConfig.implementation == SDFGrid::Type::NormalizedDenseGrid ||
                mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
            {
                // All ND SDF Grid instances share the same BLAS and AABB buffer.
                const ref<SDFGrid>& pSDFGrid = mSDFGrids.back();

                auto& blas = mBlasData[blasDataIndex];
                blas.hasProceduralPrimitives = true;
                blas.geomDescs.resize(1);

                RtGeometryDesc& desc = blas.geomDescs.back();
                desc.type = RtGeometryType::ProcedurePrimitives;
                desc.flags = RtGeometryFlags::Opaque;
                desc.content.proceduralAABBs.count = pSDFGrid->getAABBCount();
                desc.content.proceduralAABBs.data = pSDFGrid->getAABBBuffer()->getGpuAddress();
                desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                blasDataIndex++;
            }
            else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelSet ||
                     mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
            {
                for (uint32_t s = 0; s < mSDFGrids.size(); s++)
                {
                    const ref<SDFGrid>& pSDFGrid = mSDFGrids[s];

                    auto& blas = mBlasData[blasDataIndex + s];
                    blas.hasProceduralPrimitives = true;
                    blas.geomDescs.resize(1);

                    RtGeometryDesc& desc = blas.geomDescs.back();
                    desc.type = RtGeometryType::ProcedurePrimitives;
                    desc.flags = RtGeometryFlags::Opaque;
                    desc.content.proceduralAABBs.count = pSDFGrid->getAABBCount();
                    desc.content.proceduralAABBs.data = pSDFGrid->getAABBBuffer()->getGpuAddress();
                    desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                    FALCOR_ASSERT(desc.content.proceduralAABBs.count > 0);
                }

                blasDataIndex += mSDFGrids.size();
            }
        }

        if (!mCustomPrimitiveDesc.empty())
        {
            FALCOR_ASSERT(mpRtAABBBuffer && mpRtAABBBuffer->getElementCount() >= mRtAABBRaw.size());

            auto& blas = mBlasData.back();
            blas.geomDescs.resize(mCustomPrimitiveDesc.size());
            blas.hasProceduralPrimitives = true;

            uint32_t geomIndexOffset = 0;

            for (const auto& customPrim : mCustomPrimitiveDesc)
            {
                RtGeometryDesc& desc = blas.geomDescs[geomIndexOffset++];
                desc.type = RtGeometryType::ProcedurePrimitives;
                desc.flags = RtGeometryFlags::None;

                desc.content.proceduralAABBs.count = 1; // Currently only one AABB per user-defined prim supported
                desc.content.proceduralAABBs.data = mpRtAABBBuffer->getGpuAddress() + bbAddressOffset;
                desc.content.proceduralAABBs.stride = sizeof(RtAABB);

                bbAddressOffset += sizeof(RtAABB);
            }
        }

        // Verify that the total geometry count matches the expectation.
        size_t totalGeometries = 0;
        for (const auto& blas : mBlasData) totalGeometries += blas.geomDescs.size();
        if (totalGeometries != getGeometryCount()) throw RuntimeError("Total geometry count mismatch");

        mBlasDataValid = true;
    }

    void Scene::preparePrebuildInfo(RenderContext* pRenderContext)
    {
        for (auto& blas : mBlasData)
        {
            // Determine how BLAS build/update should be done.
            // The default choice is to compact all static BLASes and those that don't need to be rebuilt every frame.
            // For all other BLASes, compaction just adds overhead.
            // TODO: Add compaction on/off switch for profiling.
            // TODO: Disable compaction for skinned meshes if update performance becomes a problem.
            blas.updateMode = blas.isParticle ? mBlasParticleUpdateMode : mBlasUpdateMode;
            blas.useCompaction = (!blas.hasDynamicGeometry()) || blas.updateMode != UpdateMode::Rebuild;

            // Setup build parameters.
            RtAccelerationStructureBuildInputs& inputs = blas.buildInputs;
            inputs.kind = RtAccelerationStructureKind::BottomLevel;
            inputs.descCount = (uint32_t)blas.geomDescs.size();
            inputs.geometryDescs = blas.geomDescs.data();
            inputs.flags = RtAccelerationStructureBuildFlags::None;

            // Add necessary flags depending on settings.
            if (blas.useCompaction)
            {
                inputs.flags |= RtAccelerationStructureBuildFlags::AllowCompaction;
            }
            if ((blas.hasDynamicGeometry() || blas.hasProceduralPrimitives) && blas.updateMode == UpdateMode::Refit)
            {
                inputs.flags |= RtAccelerationStructureBuildFlags::AllowUpdate;
            }
            // Set optional performance hints.
            // TODO: Set FAST_BUILD for skinned meshes if update/rebuild performance becomes a problem.
            // TODO: Add FAST_TRACE on/off switch for profiling. It is disabled by default as it is scene-dependent.
            //if (!blas.hasSkinnedMesh)
            //{
            //    inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            //}

            if (blas.hasDynamicGeometry())
            {
                inputs.flags |= RtAccelerationStructureBuildFlags::PreferFastBuild;
            }

            // Get prebuild info.
            blas.prebuildInfo = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), inputs);

            // Figure out the padded allocation sizes to have proper alignment.
            FALCOR_ASSERT(blas.prebuildInfo.resultDataMaxSize > 0);
            blas.resultByteSize = align_to(kAccelerationStructureByteAlignment, blas.prebuildInfo.resultDataMaxSize);

            uint64_t scratchByteSize = std::max(blas.prebuildInfo.scratchDataSize, blas.prebuildInfo.updateScratchDataSize);
            blas.scratchByteSize = align_to(kAccelerationStructureByteAlignment, scratchByteSize);
        }
    }

    void Scene::computeBlasGroups()
    {
        mBlasGroups.clear();
        uint64_t groupSize = 0;

        for (uint32_t blasId = 0; blasId < mBlasData.size(); blasId++)
        {
            auto& blas = mBlasData[blasId];
            size_t blasSize = blas.resultByteSize + blas.scratchByteSize;

            // Start new BLAS group on first iteration or if group size would exceed the target.
            if (groupSize == 0 || groupSize + blasSize > kMaxBLASBuildMemory)
            {
                mBlasGroups.push_back({});
                groupSize = 0;
            }

            // Add BLAS to current group.
            FALCOR_ASSERT(mBlasGroups.size() > 0);
            auto& group = mBlasGroups.back();
            group.blasIndices.push_back(blasId);
            blas.blasGroupIndex = (uint32_t)mBlasGroups.size() - 1;

            // Update data offsets and sizes.
            blas.resultByteOffset = group.resultByteSize;
            blas.scratchByteOffset = group.scratchByteSize;
            group.resultByteSize += blas.resultByteSize;
            group.scratchByteSize += blas.scratchByteSize;

            groupSize += blasSize;
        }

        // Validation that all offsets and sizes are correct.
        uint64_t totalResultSize = 0;
        uint64_t totalScratchSize = 0;
        std::set<uint32_t> blasIDs;

        for (size_t blasGroupIndex = 0; blasGroupIndex < mBlasGroups.size(); blasGroupIndex++)
        {
            uint64_t resultSize = 0;
            uint64_t scratchSize = 0;

            const auto& group = mBlasGroups[blasGroupIndex];
            FALCOR_ASSERT(!group.blasIndices.empty());

            for (auto blasId : group.blasIndices)
            {
                FALCOR_ASSERT(blasId < mBlasData.size());
                const auto& blas = mBlasData[blasId];

                FALCOR_ASSERT(blasIDs.insert(blasId).second);
                FALCOR_ASSERT(blas.blasGroupIndex == blasGroupIndex);

                FALCOR_ASSERT(blas.resultByteSize > 0);
                FALCOR_ASSERT(blas.resultByteOffset == resultSize);
                resultSize += blas.resultByteSize;

                FALCOR_ASSERT(blas.scratchByteSize > 0);
                FALCOR_ASSERT(blas.scratchByteOffset == scratchSize);
                scratchSize += blas.scratchByteSize;

                FALCOR_ASSERT(blas.blasByteOffset == 0);
                FALCOR_ASSERT(blas.blasByteSize == 0);
            }

            FALCOR_ASSERT(resultSize == group.resultByteSize);
            FALCOR_ASSERT(scratchSize == group.scratchByteSize);
        }
        FALCOR_ASSERT(blasIDs.size() == mBlasData.size());
    }

    void Scene::buildBlas(RenderContext* pRenderContext)
    {
        FALCOR_PROFILE(pRenderContext, "buildBlas");

        if (!mBlasDataValid) throw RuntimeError("buildBlas() BLAS data is invalid");
        if (!pRenderContext->getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing))
        {
            throw RuntimeError("Raytracing is not supported by the current device");
        }

        // Add barriers for the VB and IB which will be accessed by the build.
        if (mpMeshVao)
        {
            const ref<Buffer>& pVb = mpMeshVao->getVertexBuffer(kStaticDataBufferIndex);
            const ref<Buffer>& pIb = mpMeshVao->getIndexBuffer();
            pRenderContext->resourceBarrier(pVb.get(), Resource::State::NonPixelShader);
            if (pIb) pRenderContext->resourceBarrier(pIb.get(), Resource::State::NonPixelShader);
        }

        if (mpCurveVao)
        {
            const ref<Buffer>& pCurveVb = mpCurveVao->getVertexBuffer(kStaticDataBufferIndex);
            const ref<Buffer>& pCurveIb = mpCurveVao->getIndexBuffer();
            pRenderContext->resourceBarrier(pCurveVb.get(), Resource::State::NonPixelShader);
            pRenderContext->resourceBarrier(pCurveIb.get(), Resource::State::NonPixelShader);
        }

        if (!mSDFGrids.empty())
        {
            if (mSDFGridConfig.implementation == SDFGrid::Type::NormalizedDenseGrid ||
                mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelOctree)
            {
                pRenderContext->resourceBarrier(mSDFGrids.back()->getAABBBuffer().get(), Resource::State::NonPixelShader);
            }
            else if (mSDFGridConfig.implementation == SDFGrid::Type::SparseVoxelSet ||
                     mSDFGridConfig.implementation == SDFGrid::Type::SparseBrickSet)
            {
                for (const ref<SDFGrid>& pSDFGrid : mSDFGrids)
                {
                    pRenderContext->resourceBarrier(pSDFGrid->getAABBBuffer().get(), Resource::State::NonPixelShader);
                }
            }
        }

        if (mpRtAABBBuffer)
        {
            pRenderContext->resourceBarrier(mpRtAABBBuffer.get(), Resource::State::NonPixelShader);
        }

        // On the first time, or if a full rebuild is necessary we will:
        // - Update all build inputs and prebuild info
        // - Compute BLAS groups
        // - Calculate total intermediate buffer sizes
        // - Build all BLASes into an intermediate buffer
        // - Calculate total compacted buffer size
        // - Compact/clone all BLASes to their final location

        if (mRebuildBlas)
        {
            // Invalidate any previous TLASes as they won't be valid anymore.
            invalidateTlasCache();

            if (mBlasData.empty())
            {
                logInfo("Skipping BLAS build due to no geometries");

                mBlasGroups.clear();
                mBlasObjects.clear();
            }
            else
            {
                logInfo("Initiating BLAS build for {} mesh groups", mBlasData.size());

                // Compute pre-build info per BLAS and organize the BLASes into groups
                // in order to limit GPU memory usage during BLAS build.
                preparePrebuildInfo(pRenderContext);
                computeBlasGroups();

                logInfo("BLAS build split into {} groups", mBlasGroups.size());

                // Compute the required maximum size of the result and scratch buffers.
                uint64_t resultByteSize = 0;
                uint64_t scratchByteSize = 0;
                size_t maxBlasCount = 0;

                for (const auto& group : mBlasGroups)
                {
                    resultByteSize = std::max(resultByteSize, group.resultByteSize);
                    scratchByteSize = std::max(scratchByteSize, group.scratchByteSize);
                    maxBlasCount = std::max(maxBlasCount, group.blasIndices.size());
                }
                FALCOR_ASSERT(resultByteSize > 0 && scratchByteSize > 0);

                logInfo("BLAS build result buffer size: {}", formatByteSize(resultByteSize));
                logInfo("BLAS build scratch buffer size: {}", formatByteSize(scratchByteSize));

                // Allocate result and scratch buffers.
                // The scratch buffer we'll retain because it's needed for subsequent rebuilds and updates.
                // TODO: Save memory by reducing the scratch buffer to the minimum required for the dynamic objects.
                if (mpBlasScratch == nullptr || mpBlasScratch->getSize() < scratchByteSize)
                {
                    mpBlasScratch = Buffer::create(mpDevice, scratchByteSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
                    mpBlasScratch->setName("Scene::mpBlasScratch");
                }

                ref<Buffer> pResultBuffer = Buffer::create(mpDevice, resultByteSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                FALCOR_ASSERT(pResultBuffer && mpBlasScratch);

                // Create post-build info pool for readback.
                RtAccelerationStructurePostBuildInfoPool::Desc compactedSizeInfoPoolDesc;
                compactedSizeInfoPoolDesc.queryType = RtAccelerationStructurePostBuildInfoQueryType::CompactedSize;
                compactedSizeInfoPoolDesc.elementCount = (uint32_t)maxBlasCount;
                ref<RtAccelerationStructurePostBuildInfoPool> compactedSizeInfoPool = RtAccelerationStructurePostBuildInfoPool::create(mpDevice.get(), compactedSizeInfoPoolDesc);

                RtAccelerationStructurePostBuildInfoPool::Desc currentSizeInfoPoolDesc;
                currentSizeInfoPoolDesc.queryType = RtAccelerationStructurePostBuildInfoQueryType::CurrentSize;
                currentSizeInfoPoolDesc.elementCount = (uint32_t)maxBlasCount;
                ref<RtAccelerationStructurePostBuildInfoPool> currentSizeInfoPool = RtAccelerationStructurePostBuildInfoPool::create(mpDevice.get(), currentSizeInfoPoolDesc);

                bool hasDynamicGeometry = false;
                bool hasProceduralPrimitives = false;

                mBlasObjects.resize(mBlasData.size());

                // Iterate over BLAS groups. For each group build and compact all BLASes.
                for (size_t blasGroupIndex = 0; blasGroupIndex < mBlasGroups.size(); blasGroupIndex++)
                {
                    auto& group = mBlasGroups[blasGroupIndex];

                    // Allocate array to hold intermediate blases for the group.
                    std::vector<ref<RtAccelerationStructure>> intermediateBlases(group.blasIndices.size());

                    // Insert barriers. The buffers are now ready to be written.
                    pRenderContext->uavBarrier(pResultBuffer.get());
                    pRenderContext->uavBarrier(mpBlasScratch.get());

                    // Reset the post-build info pools to receive new info.
                    compactedSizeInfoPool->reset(pRenderContext);
                    currentSizeInfoPool->reset(pRenderContext);

                    // Build the BLASes into the intermediate result buffer.
                    // We output post-build info in order to find out the final size requirements.
                    for (size_t i = 0; i < group.blasIndices.size(); ++i)
                    {
                        const uint32_t blasId = group.blasIndices[i];
                        const auto& blas = mBlasData[blasId];

                        hasDynamicGeometry |= blas.hasDynamicGeometry();
                        hasProceduralPrimitives |= blas.hasProceduralPrimitives;

                        RtAccelerationStructure::Desc createDesc = {};
                        createDesc.setBuffer(pResultBuffer, blas.resultByteOffset, blas.resultByteSize);
                        createDesc.setKind(RtAccelerationStructureKind::BottomLevel);
                        auto blasObject = RtAccelerationStructure::create(mpDevice, createDesc);
                        intermediateBlases[i] = blasObject;

                        RtAccelerationStructure::BuildDesc asDesc = {};
                        asDesc.inputs = blas.buildInputs;
                        asDesc.scratchData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
                        asDesc.dest = blasObject.get();

                        // Need to find out the post-build compacted BLAS size to know the final allocation size.
                        RtAccelerationStructurePostBuildInfoDesc postbuildInfoDesc = {};
                        if (blas.useCompaction)
                        {
                            postbuildInfoDesc.type = RtAccelerationStructurePostBuildInfoQueryType::CompactedSize;
                            postbuildInfoDesc.index = (uint32_t)i;
                            postbuildInfoDesc.pool = compactedSizeInfoPool.get();
                        }
                        else
                        {
                            postbuildInfoDesc.type = RtAccelerationStructurePostBuildInfoQueryType::CurrentSize;
                            postbuildInfoDesc.index = (uint32_t)i;
                            postbuildInfoDesc.pool = currentSizeInfoPool.get();
                        }

                        pRenderContext->buildAccelerationStructure(asDesc, 1, &postbuildInfoDesc);
                    }

                    // Read back the calculated final size requirements for each BLAS.

                    group.finalByteSize = 0;
                    for (size_t i = 0; i < group.blasIndices.size(); i++)
                    {
                        const uint32_t blasId = group.blasIndices[i];
                        auto& blas = mBlasData[blasId];

                        // Check the size. Upon failure a zero size may be reported.
                        uint64_t byteSize = 0;
                        if (blas.useCompaction)
                        {
                            byteSize = compactedSizeInfoPool->getElement(pRenderContext, (uint32_t)i);
                        }
                        else
                        {
                            byteSize = currentSizeInfoPool->getElement(pRenderContext, (uint32_t)i);
                            // For platforms that does not support current size query, use prebuild size.
                            if (byteSize == 0)
                            {
                                byteSize = blas.prebuildInfo.resultDataMaxSize;
                            }
                        }
                        FALCOR_ASSERT(byteSize <= blas.prebuildInfo.resultDataMaxSize);
                        if (byteSize == 0) throw RuntimeError("Acceleration structure build failed for BLAS index {}", blasId);

                        blas.blasByteSize = align_to(kAccelerationStructureByteAlignment, byteSize);
                        blas.blasByteOffset = group.finalByteSize;
                        group.finalByteSize += blas.blasByteSize;
                    }
                    FALCOR_ASSERT(group.finalByteSize > 0);

                    logInfo("BLAS group " + std::to_string(blasGroupIndex) + " final size: " + formatByteSize(group.finalByteSize));

                    // Allocate final BLAS buffer.
                    auto& pBlas = group.pBlas;
                    if (pBlas == nullptr || pBlas->getSize() < group.finalByteSize)
                    {
                        pBlas = Buffer::create(mpDevice, group.finalByteSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                        pBlas->setName("Scene::mBlasGroups[" + std::to_string(blasGroupIndex) + "].pBlas");
                    }
                    else
                    {
                        // If we didn't need to reallocate, just insert a barrier so it's safe to use.
                        pRenderContext->uavBarrier(pBlas.get());
                    }

                    // Insert barrier. The result buffer is now ready to be consumed.
                    // TOOD: This is probably not necessary since we flushed above, but it's not going to hurt.
                    pRenderContext->uavBarrier(pResultBuffer.get());

                    // Compact/clone all BLASes to their final location.
                    for (size_t i = 0; i < group.blasIndices.size(); ++i)
                    {
                        const uint32_t blasId = group.blasIndices[i];
                        auto& blas = mBlasData[blasId];

                        RtAccelerationStructure::Desc blasDesc = {};
                        blasDesc.setBuffer(pBlas, blas.blasByteOffset, blas.blasByteSize);
                        blasDesc.setKind(RtAccelerationStructureKind::BottomLevel);
                        mBlasObjects[blasId] = RtAccelerationStructure::create(mpDevice, blasDesc);

                        pRenderContext->copyAccelerationStructure(
                            mBlasObjects[blasId].get(),
                            intermediateBlases[i].get(),
                            blas.useCompaction ? RenderContext::RtAccelerationStructureCopyMode::Compact : RenderContext::RtAccelerationStructureCopyMode::Clone);
                    }

                    // Insert barrier. The BLAS buffer is now ready for use.
                    pRenderContext->uavBarrier(pBlas.get());
                }

                // Release scratch buffer if there is no animated content. We will not need it.
                if (!hasDynamicGeometry && !hasProceduralPrimitives) mpBlasScratch.reset();
            }

            updateRaytracingBLASStats();
            mRebuildBlas = false;
            return;
        }

        // If we get here, all BLASes have previously been built and compacted. We will:
        // - Skip the ones that have no animated geometries.
        // - Update or rebuild in-place the ones that are animated.

        FALCOR_ASSERT(!mRebuildBlas);
        bool updateProcedural = is_set(mUpdates, UpdateFlags::CurvesMoved) || is_set(mUpdates, UpdateFlags::CustomPrimitivesMoved);

        for (const auto& group : mBlasGroups)
        {
            // Determine if any BLAS in the group needs to be updated.
            bool needsUpdate = false;
            for (uint32_t blasId : group.blasIndices)
            {
                const auto& blas = mBlasData[blasId];
                if (blas.hasProceduralPrimitives && updateProcedural) needsUpdate = true;
                if (!blas.hasProceduralPrimitives && blas.hasDynamicGeometry()) needsUpdate = true;
            }

            if (!needsUpdate) continue;

            // At least one BLAS in the group needs to be updated.
            // Insert barriers. The buffers are now ready to be written.
            auto& pBlas = group.pBlas;
            FALCOR_ASSERT(pBlas && mpBlasScratch);
            pRenderContext->uavBarrier(pBlas.get());
            pRenderContext->uavBarrier(mpBlasScratch.get());

            // Iterate over all BLASes in group.
            for (uint32_t blasId : group.blasIndices)
            {
                const auto& blas = mBlasData[blasId];

                // Skip BLASes that do not need to be updated.
                if (blas.hasProceduralPrimitives && !updateProcedural) continue;
                if (!blas.hasProceduralPrimitives && !blas.hasDynamicGeometry()) continue;

                // Rebuild/update BLAS.
                RtAccelerationStructure::BuildDesc asDesc = {};
                asDesc.inputs = blas.buildInputs;
                asDesc.scratchData = mpBlasScratch->getGpuAddress() + blas.scratchByteOffset;
                asDesc.dest = mBlasObjects[blasId].get();

                if (blas.updateMode == UpdateMode::Refit)
                {
                    // Set source address to destination address to update in place.
                    asDesc.source = asDesc.dest;
                    asDesc.inputs.flags |= RtAccelerationStructureBuildFlags::PerformUpdate;
                }
                else
                {
                    // We'll rebuild in place. The BLAS should not be compacted, check that size matches prebuild info.
                    FALCOR_ASSERT(blas.blasByteSize == align_to(kAccelerationStructureByteAlignment, blas.prebuildInfo.resultDataMaxSize));
                }
                pRenderContext->buildAccelerationStructure(asDesc, 0, nullptr);
            }

            // Insert barrier. The BLAS buffer is now ready for use.
            pRenderContext->uavBarrier(pBlas.get());
        }
    }

    void Scene::fillInstanceDesc(std::vector<RtInstanceDesc>& instanceDescs, uint32_t rayTypeCount, bool perMeshHitEntry) const
    {
        instanceDescs.clear();
        uint32_t instanceContributionToHitGroupIndex = 0;
        uint32_t instanceID = 0;

        for (size_t i = 0; i < mMeshGroups.size(); i++)
        {
            const auto& meshList = mMeshGroups[i].meshList;
            const bool isStatic = mMeshGroups[i].isStatic;
            const bool isNotCastShadow = !mMeshGroups[i].isCastShadow;
            const bool isParticle = mMeshGroups[i].isParticle();

            FALCOR_ASSERT(mBlasData[i].blasGroupIndex < mBlasGroups.size());
            const auto& pBlas = mBlasGroups[mBlasData[i].blasGroupIndex].pBlas;
            FALCOR_ASSERT(pBlas);

            RtInstanceDesc desc = {};
            desc.accelerationStructure = pBlas->getGpuAddress() + mBlasData[i].blasByteOffset;
            //Set instance mask. Only last 4 bits are used. They are set as follows: PtcUni | PtcCam | Shadow | Normal
            if (isNotCastShadow || isParticle)
            {
                uint bitShiftParticle = (uint)mMeshGroups[i].particleOrientation + 1; //Only valid if isPartice is true
                desc.instanceMask = 0;
                desc.instanceMask |= isNotCastShadow && !isParticle ? 1 << 1 : 0; //Particles ignore the castShadow Material flag
                desc.instanceMask |= isParticle ? 1 << bitShiftParticle : 0;
            }
            else
            {
                desc.instanceMask = 1; // Normal Geometry 
            }
            
            desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

            instanceContributionToHitGroupIndex += rayTypeCount * (uint32_t)meshList.size();

            // We expect all meshes in a group to have identical triangle winding. Verify that assumption here.
            FALCOR_ASSERT(!meshList.empty());
            const bool frontFaceCW = mMeshDesc[meshList[0].get()].isFrontFaceCW();
            for (size_t j = 1; j < meshList.size(); j++)
            {
                FALCOR_ASSERT(mMeshDesc[meshList[j].get()].isFrontFaceCW() == frontFaceCW);
            }

            // Set the triangle winding for the instance if it differs from the default.
            // The default in DXR is that a triangle is front facing if its vertices appear clockwise
            // from the ray origin, in object space in a left-handed coordinate system.
            // Note that Falcor uses a right-handed coordinate system, so we have to invert the flag.
            // Since these winding direction rules are defined in object space, they are unaffected by instance transforms.
            if (frontFaceCW) desc.flags = desc.flags | RtGeometryInstanceFlags::TriangleFrontCounterClockwise;

            // From the scene builder we can expect the following:
            //
            // If BLAS is marked as static:
            // - The meshes are pre-transformed to world-space.
            // - The meshes are guaranteed to be non-instanced, so only one INSTANCE_DESC with an identity transform is needed.
            //
            // If BLAS is not marked as static:
            // - The meshes are guaranteed to be non-instanced or be identically instanced, one INSTANCE_DESC per TLAS instance is needed.
            // - The global matrices are the same for all meshes in an instance.
            //
            FALCOR_ASSERT(!meshList.empty());
            size_t instanceCount = mMeshIdToInstanceIds[meshList[0].get()].size();

            FALCOR_ASSERT(instanceCount > 0);
            for (size_t instanceIdx = 0; instanceIdx < instanceCount; instanceIdx++)
            {
                // Validate that the ordering is matching our expectations:
                // InstanceID() + GeometryIndex() should look up the correct mesh instance.
                for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)meshList.size(); geometryIndex++)
                {
                    const auto& instances = mMeshIdToInstanceIds[meshList[geometryIndex].get()];
                    FALCOR_ASSERT(instances.size() == instanceCount);
                    FALCOR_ASSERT(instances[instanceIdx] == instanceID + geometryIndex);
                }

                desc.instanceID = instanceID;
                instanceID += (uint32_t)meshList.size();

                float4x4 transform4x4 = float4x4::identity();
                if (!isStatic)
                {
                    // For non-static meshes, the matrices for all meshes in an instance are guaranteed to be the same.
                    // Just pick the matrix from the first mesh.
                    const uint32_t matrixId = mGeometryInstanceData[desc.instanceID].globalMatrixID;
                    transform4x4 = mpAnimationController->getGlobalMatrices()[matrixId];

                    // Verify that all meshes have matching tranforms.
                    for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)meshList.size(); geometryIndex++)
                    {
                        FALCOR_ASSERT(matrixId == mGeometryInstanceData[desc.instanceID + geometryIndex].globalMatrixID);
                    }
                }
                std::memcpy(desc.transform, &transform4x4, sizeof(desc.transform));

                // Verify that instance data has the correct instanceIndex and geometryIndex.
                for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)meshList.size(); geometryIndex++)
                {
                    FALCOR_ASSERT((uint32_t)instanceDescs.size() == mGeometryInstanceData[desc.instanceID + geometryIndex].instanceIndex);
                    FALCOR_ASSERT(geometryIndex == mGeometryInstanceData[desc.instanceID + geometryIndex].geometryIndex);
                }

                instanceDescs.push_back(desc);
            }
        }

        uint32_t totalBlasCount = (uint32_t)mMeshGroups.size() + (mCurveDesc.empty() ? 0 : 1) + getSDFGridGeometryCount() + (mCustomPrimitiveDesc.empty() ? 0 : 1);
        FALCOR_ASSERT((uint32_t)mBlasData.size() == totalBlasCount);

        size_t blasDataIndex = mMeshGroups.size();
        // One instance for curves.
        if (!mCurveDesc.empty())
        {
            const auto& blasData = mBlasData[blasDataIndex++];
            FALCOR_ASSERT(blasData.blasGroupIndex < mBlasGroups.size());
            const auto& pBlas = mBlasGroups[blasData.blasGroupIndex].pBlas;
            FALCOR_ASSERT(pBlas);

            RtInstanceDesc desc = {};
            desc.accelerationStructure = pBlas->getGpuAddress() + blasData.blasByteOffset;
            desc.instanceMask = 0xFF;
            desc.instanceID = instanceID;
            instanceID += (uint32_t)mCurveDesc.size();

            // Start procedural primitive hit group after the triangle hit groups.
            desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

            instanceContributionToHitGroupIndex += rayTypeCount * (uint32_t)mCurveDesc.size();

            // For cached curves, the matrices for all curves in an instance are guaranteed to be the same.
            // Just pick the matrix from the first curve.
            auto it = std::find_if(mGeometryInstanceData.begin(), mGeometryInstanceData.end(), [](const auto& inst) { return inst.getType() == GeometryType::Curve; });
            FALCOR_ASSERT(it != mGeometryInstanceData.end());
            const uint32_t matrixId = it->globalMatrixID;
            desc.setTransform(mpAnimationController->getGlobalMatrices()[matrixId]);

            // Verify that instance data has the correct instanceIndex and geometryIndex.
            for (uint32_t geometryIndex = 0; geometryIndex < (uint32_t)mCurveDesc.size(); geometryIndex++)
            {
                FALCOR_ASSERT((uint32_t)instanceDescs.size() == mGeometryInstanceData[desc.instanceID + geometryIndex].instanceIndex);
                FALCOR_ASSERT(geometryIndex == mGeometryInstanceData[desc.instanceID + geometryIndex].geometryIndex);
            }

            instanceDescs.push_back(desc);
        }

        // One instance per SDF grid instance.
        if (!mSDFGrids.empty())
        {
            bool sdfGridInstancesHaveUniqueBLASes = true;
            switch (mSDFGridConfig.implementation)
            {
            case SDFGrid::Type::NormalizedDenseGrid:
            case SDFGrid::Type::SparseVoxelOctree:
                sdfGridInstancesHaveUniqueBLASes = false;
                break;
            case SDFGrid::Type::SparseVoxelSet:
            case SDFGrid::Type::SparseBrickSet:
                sdfGridInstancesHaveUniqueBLASes = true;
                break;
            default:
                FALCOR_UNREACHABLE();
            }

            for (const GeometryInstanceData& instance : mGeometryInstanceData)
            {
                if (instance.getType() != GeometryType::SDFGrid) continue;

                const BlasData& blasData = mBlasData[blasDataIndex + (sdfGridInstancesHaveUniqueBLASes ? instance.geometryID : 0)];
                const auto& pBlas = mBlasGroups[blasData.blasGroupIndex].pBlas;

                RtInstanceDesc desc = {};
                desc.accelerationStructure = pBlas->getGpuAddress() + blasData.blasByteOffset;
                desc.instanceMask = 0xFF;
                desc.instanceID = instanceID;
                instanceID++;

                // Start SDF grid hit group after the curve hit groups.
                desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

                desc.setTransform(mpAnimationController->getGlobalMatrices()[instance.globalMatrixID]);

                // Verify that instance data has the correct instanceIndex and geometryIndex.
                FALCOR_ASSERT((uint32_t)instanceDescs.size() == instance.instanceIndex);
                FALCOR_ASSERT(0 == instance.geometryIndex);

                instanceDescs.push_back(desc);
            }

            blasDataIndex += (sdfGridInstancesHaveUniqueBLASes ? mSDFGrids.size() : 1);
            instanceContributionToHitGroupIndex += rayTypeCount * (uint32_t)mSDFGridDesc.size();
        }

        // One instance with identity transform for custom primitives.
        if (!mCustomPrimitiveDesc.empty())
        {
            FALCOR_ASSERT(mBlasData.back().blasGroupIndex < mBlasGroups.size());
            const auto& pBlas = mBlasGroups[mBlasData.back().blasGroupIndex].pBlas;
            FALCOR_ASSERT(pBlas);

            RtInstanceDesc desc = {};
            desc.accelerationStructure = pBlas->getGpuAddress() + mBlasData.back().blasByteOffset;
            desc.instanceMask = 0xFF;
            desc.instanceID = instanceID;
            instanceID += (uint32_t)mCustomPrimitiveDesc.size();

            // Start procedural primitive hit group after the curve hit group.
            desc.instanceContributionToHitGroupIndex = perMeshHitEntry ? instanceContributionToHitGroupIndex : 0;

            instanceContributionToHitGroupIndex += rayTypeCount * (uint32_t)mCustomPrimitiveDesc.size();

            float4x4 identityMat = float4x4::identity();
            std::memcpy(desc.transform, &identityMat, sizeof(desc.transform));
            instanceDescs.push_back(desc);
        }
    }

    void Scene::invalidateTlasCache()
    {
        for (auto& tlas : mTlasCache)
        {
            tlas.second.pTlasObject = nullptr;
        }
    }

    void Scene::buildTlas(RenderContext* pRenderContext, uint32_t rayTypeCount, bool perMeshHitEntry)
    {
        FALCOR_PROFILE(pRenderContext, "buildTlas");

        TlasData tlas;
        auto it = mTlasCache.find(rayTypeCount);
        if (it != mTlasCache.end()) tlas = it->second;

        // Prepare instance descs.
        // Note if there are no instances, we'll build an empty TLAS.
        fillInstanceDesc(mInstanceDescs, rayTypeCount, perMeshHitEntry);

        RtAccelerationStructureBuildInputs inputs = {};
        inputs.kind = RtAccelerationStructureKind::TopLevel;
        inputs.descCount = (uint32_t)mInstanceDescs.size();
        inputs.flags = RtAccelerationStructureBuildFlags::None;

        // Add build flags for dynamic scenes if TLAS should be updating instead of rebuilt
        if ((mpAnimationController->hasAnimations() || mpAnimationController->hasAnimatedVertexCaches()) && mTlasUpdateMode == UpdateMode::Refit)
        {
            inputs.flags |= RtAccelerationStructureBuildFlags::AllowUpdate;

            // If TLAS has been built already and it was built with ALLOW_UPDATE
            if (tlas.pTlasObject != nullptr && tlas.updateMode == UpdateMode::Refit) inputs.flags |= RtAccelerationStructureBuildFlags::PerformUpdate;
        }

        tlas.updateMode = mTlasUpdateMode;

        // On first build for the scene, create scratch buffer and cache prebuild info. As long as INSTANCE_DESC count doesn't change, we can reuse these
        if (mpTlasScratch == nullptr)
        {
            // Prebuild
            mTlasPrebuildInfo = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), inputs);
            mpTlasScratch = Buffer::create(mpDevice, mTlasPrebuildInfo.scratchDataSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            mpTlasScratch->setName("Scene::mpTlasScratch");

            // #SCENE This isn't guaranteed according to the spec, and the scratch buffer being stored should be sized differently depending on update mode
            FALCOR_ASSERT(mTlasPrebuildInfo.updateScratchDataSize <= mTlasPrebuildInfo.scratchDataSize);
        }

        // Setup GPU buffers
        RtAccelerationStructure::BuildDesc asDesc = {};
        asDesc.inputs = inputs;

        // If first time building this TLAS
        if (tlas.pTlasObject == nullptr)
        {
            {
                // Allocate a new buffer for the TLAS only if the existing buffer isn't big enough.
                if (!tlas.pTlasBuffer || tlas.pTlasBuffer->getSize() < mTlasPrebuildInfo.resultDataMaxSize)
                {
                    tlas.pTlasBuffer = Buffer::create(mpDevice, mTlasPrebuildInfo.resultDataMaxSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
                    tlas.pTlasBuffer->setName("Scene TLAS buffer");
                }
            }
            if (!mInstanceDescs.empty())
            {
                // Allocate a new buffer for the TLAS instance desc input only if the existing buffer isn't big enough.
                if (!tlas.pInstanceDescs || tlas.pInstanceDescs->getSize() < mInstanceDescs.size() * sizeof(RtInstanceDesc))
                {
                    tlas.pInstanceDescs = Buffer::create(mpDevice, (uint32_t)mInstanceDescs.size() * sizeof(RtInstanceDesc), Buffer::BindFlags::None, Buffer::CpuAccess::Write, mInstanceDescs.data());
                    tlas.pInstanceDescs->setName("Scene instance descs buffer");
                }
                else
                {
                    tlas.pInstanceDescs->setBlob(mInstanceDescs.data(), 0, mInstanceDescs.size() * sizeof(RtInstanceDesc));
                }
            }

            RtAccelerationStructure::Desc asCreateDesc = {};
            asCreateDesc.setKind(RtAccelerationStructureKind::TopLevel);
            asCreateDesc.setBuffer(tlas.pTlasBuffer, 0, mTlasPrebuildInfo.resultDataMaxSize);
            tlas.pTlasObject = RtAccelerationStructure::create(mpDevice, asCreateDesc);
        }
        // Else update instance descs and barrier TLAS buffers
        else
        {
            FALCOR_ASSERT(mpAnimationController->hasAnimations() || mpAnimationController->hasAnimatedVertexCaches());
            pRenderContext->uavBarrier(tlas.pTlasBuffer.get());
            pRenderContext->uavBarrier(mpTlasScratch.get());
            if (tlas.pInstanceDescs)
            {
                FALCOR_ASSERT(!mInstanceDescs.empty());
                tlas.pInstanceDescs->setBlob(mInstanceDescs.data(), 0, inputs.descCount * sizeof(RtInstanceDesc));
            }
            asDesc.source = tlas.pTlasObject.get(); // Perform the update in-place
        }

        FALCOR_ASSERT(tlas.pTlasBuffer && tlas.pTlasBuffer->getGfxResource() && mpTlasScratch->getGfxResource());
        FALCOR_ASSERT(inputs.descCount == 0 || (tlas.pInstanceDescs && tlas.pInstanceDescs->getGfxResource()));

        asDesc.inputs.instanceDescs = tlas.pInstanceDescs ? tlas.pInstanceDescs->getGpuAddress() : 0;
        asDesc.scratchData = mpTlasScratch->getGpuAddress();
        asDesc.dest = tlas.pTlasObject.get();

        // Set the source buffer to update in place if this is an update
        if ((inputs.flags & RtAccelerationStructureBuildFlags::PerformUpdate) != RtAccelerationStructureBuildFlags::None)
        {
            asDesc.source = asDesc.dest;
        }

        // Create TLAS
        if (tlas.pInstanceDescs)
        {
            pRenderContext->resourceBarrier(tlas.pInstanceDescs.get(), Resource::State::NonPixelShader);
        }
        pRenderContext->buildAccelerationStructure(asDesc, 0, nullptr);
        pRenderContext->uavBarrier(tlas.pTlasBuffer.get());

        mTlasCache[rayTypeCount] = tlas;
        updateRaytracingTLASStats();
    }

    void Scene::setRaytracingShaderData(RenderContext* pRenderContext, const ShaderVar& var, uint32_t rayTypeCount)
    {
        // On first execution or if BLASes need to be rebuilt, create BLASes for all geometries.
        if (!mBlasDataValid)
        {
            initGeomDesc(pRenderContext);
            buildBlas(pRenderContext);
        }

        // On first execution, when meshes have moved, when there's a new ray type count, or when a BLAS has changed, create/update the TLAS
        //
        // The raytracing shader table has one hit record per ray type and geometry. We need to know the ray type count in order to setup the indexing properly.
        // Note that for DXR 1.1 ray queries, the shader table is not used and the ray type count doesn't matter and can be set to zero.
        //
        auto tlasIt = mTlasCache.find(rayTypeCount);
        if (tlasIt == mTlasCache.end() || !tlasIt->second.pTlasObject)
        {
            // We need a hit entry per mesh right now to pass GeometryIndex()
            buildTlas(pRenderContext, rayTypeCount, true);

            // If new TLAS was just created, get it so the iterator is valid
            if (tlasIt == mTlasCache.end()) tlasIt = mTlasCache.find(rayTypeCount);
        }
        FALCOR_ASSERT(mpSceneBlock);

        // Bind TLAS.
        FALCOR_ASSERT(tlasIt != mTlasCache.end() && tlasIt->second.pTlasObject)
        mpSceneBlock->getRootVar()["rtAccel"].setAccelerationStructure(tlasIt->second.pTlasObject);

        // Bind Scene parameter block.
        getCamera()->setShaderData(mpSceneBlock->getRootVar()[kCamera]); // TODO REMOVE: Shouldn't be needed anymore?
        var[kParameterBlockName] = mpSceneBlock;
    }

    std::vector<uint32_t> Scene::getMeshBlasIDs() const
    {
        const uint32_t invalidID = uint32_t(-1);
        std::vector<uint32_t> blasIDs(mMeshDesc.size(), invalidID);

        for (uint32_t blasID = 0; blasID < (uint32_t)mMeshGroups.size(); blasID++)
        {
            for (auto meshID : mMeshGroups[blasID].meshList)
            {
                FALCOR_ASSERT_LT(meshID.get(), blasIDs.size());
                blasIDs[meshID.get()] = blasID;
            }
        }

        for (auto blasID : blasIDs) FALCOR_ASSERT(blasID != invalidID);
        return blasIDs;
    }

    NodeID Scene::getParentNodeID(NodeID nodeID) const
    {
        if (nodeID.get() >= mSceneGraph.size()) throw ArgumentError("'nodeID' ({}) is out of range", nodeID);
        return mSceneGraph[nodeID.get()].parent;
    }

    void Scene::nullTracePass(RenderContext* pRenderContext, const uint2& dim)
    {
        auto pDevice = pRenderContext->getDevice();

        if (!pDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        {
            logWarning("Raytracing Tier 1.1 is not supported by the current device.");
            return;
        }

        RtAccelerationStructureBuildInputs inputs = {};
        inputs.kind = RtAccelerationStructureKind::TopLevel;
        inputs.descCount = 0;
        inputs.flags = RtAccelerationStructureBuildFlags::None;

        RtAccelerationStructurePrebuildInfo prebuildInfo = RtAccelerationStructure::getPrebuildInfo(pDevice.get(), inputs);

        auto pScratch = Buffer::create(pDevice, prebuildInfo.scratchDataSize, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
        auto pTlasBuffer = Buffer::create(pDevice, prebuildInfo.resultDataMaxSize, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);

        RtAccelerationStructure::Desc createDesc = {};
        createDesc.setKind(RtAccelerationStructureKind::TopLevel);
        createDesc.setBuffer(pTlasBuffer, 0, prebuildInfo.resultDataMaxSize);
        ref<RtAccelerationStructure> tlasObject = RtAccelerationStructure::create(pDevice, createDesc);

        RtAccelerationStructure::BuildDesc asDesc = {};
        asDesc.inputs = inputs;
        asDesc.scratchData = pScratch->getGpuAddress();
        asDesc.dest = tlasObject.get();

        pRenderContext->buildAccelerationStructure(asDesc, 0, nullptr);
        pRenderContext->uavBarrier(pTlasBuffer.get());

        Program::Desc desc;
        desc.addShaderLibrary("Scene/NullTrace.cs.slang").csEntry("main").setShaderModel("6_5");
        auto pass = ComputePass::create(pDevice, desc);
        auto var = pass->getRootVar();
        var["gOutput"] = Texture::create2D(pDevice, dim.x, dim.y, ResourceFormat::R8Uint, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess);
        var["gTlas"].setAccelerationStructure(tlasObject);

        for (size_t i = 0; i < 100; i++)
        {
            pass->execute(pRenderContext, uint3(dim, 1));
        }
    }

    void Scene::setEnvMap(ref<EnvMap> pEnvMap)
    {
        if (mpEnvMap == pEnvMap) return;
        mpEnvMap = pEnvMap;
        mEnvMapChanged = true;
    }

    bool Scene::loadEnvMap(const std::filesystem::path& path)
    {
        auto pEnvMap = EnvMap::createFromFile(mpDevice, path);
        if (!pEnvMap)
        {
            logWarning("Failed to load environment map from '{}'.", path);
            return false;
        }
        setEnvMap(pEnvMap);
        return true;
    }

    void Scene::setCameraAspectRatio(float ratio)
    {
        getCamera()->setAspectRatio(ratio);
    }

    void Scene::setUpDirection(UpDirection upDirection)
    {
        mUpDirection = upDirection;
        mpCamCtrl->setUpDirection((CameraController::UpDirection)upDirection);
    }

    void Scene::setCameraController(CameraControllerType type)
    {
        if (!mCameraSwitched && mCamCtrlType == type && mpCamCtrl) return;

        auto camera = getCamera();
        switch (type)
        {
        case CameraControllerType::FirstPerson:
            mpCamCtrl = std::make_unique<FirstPersonCameraController>(camera);
            break;
        case CameraControllerType::Orbiter:
            mpCamCtrl = std::make_unique<OrbiterCameraController>(camera);
            ((OrbiterCameraController*)mpCamCtrl.get())->setModelParams(mSceneBB.center(), mSceneBB.radius(), 3.5f);
            break;
        case CameraControllerType::SixDOF:
            mpCamCtrl = std::make_unique<SixDoFCameraController>(camera);
            break;
        default:
            FALCOR_UNREACHABLE();
        }
        mpCamCtrl->setUpDirection((CameraController::UpDirection)mUpDirection);
        mpCamCtrl->setCameraSpeed(mCameraSpeed);
        mpCamCtrl->setCameraBounds(mCameraBounds);
        mCamCtrlType = type;
    }

    bool Scene::onMouseEvent(const MouseEvent& mouseEvent)
    {
        if (mCameraControlsEnabled)
        {
            // DEMO21, but I think it makes sense, if the camera did anything, stop the animation for it.
            if (mpCamCtrl->onMouseEvent(mouseEvent))
            {
                auto& camera = mCameras[mSelectedCamera];
                camera->setIsAnimated(false);
                return true;
            }
        }

        return false;
    }

    bool Scene::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
        {
            if (keyEvent.mods == Input::ModifierFlags::None)
            {
                if (keyEvent.key == Input::Key::F3)
                {
                    addViewpoint();
                    return true;
                }
            }
            // DEMO21, but I think it makes sense, to have these controls
            else if (keyEvent.key == Input::Key::C || keyEvent.key == Input::Key::F7)
            {
                // Force camera animation on.
                auto camera = mCameras[mSelectedCamera];
                camera->setIsAnimated(true);
                return true;
            }
            else if (keyEvent.key == Input::Key::F8)
            {
                auto camera = mCameras[mSelectedCamera];
            }
        }
        if (mCameraControlsEnabled)
        {
            // DEMO21, but I think it makes sense, if the camera did anything, stop the animation for it.
            if (mpCamCtrl->onKeyEvent(keyEvent))
            {
                auto& camera = mCameras[mSelectedCamera];
                camera->setIsAnimated(false);
                return true;
            }
        }

        return false;
    }

    bool Scene::onGamepadEvent(const GamepadEvent& gamepadEvent)
    {
        return false;
    }

    bool Scene::onGamepadState(const GamepadState& gamepadState)
    {
        if (mCameraControlsEnabled)
        {
            return mpCamCtrl->onGamepadState(gamepadState);
        }

        return false;
    }

    std::string Scene::getScript(const std::string& sceneVar)
    {
        std::string c;

        // Render settings.
        c += ScriptWriter::makeSetProperty(sceneVar, kRenderSettings, mRenderSettings);

        // Animations.
        if (hasAnimation() && !isAnimated())
        {
            c += ScriptWriter::makeSetProperty(sceneVar, kAnimated, false);
        }
        for (size_t i = 0; i < mLights.size(); ++i)
        {
            const auto& light = mLights[i];
            if (light->hasAnimation() && !light->isAnimated())
            {
                c += ScriptWriter::makeSetProperty(sceneVar + "." + kGetLight + "(" + std::to_string(i) + ").", kAnimated, false);
            }
        }

        // Camera.
        if (mSelectedCamera != 0)
        {
            c += sceneVar + "." + kCamera + " = " + sceneVar + "." + kCameras + "[" + std::to_string(mSelectedCamera) + "]\n";
        }
        c += getCamera()->getScript(sceneVar + "." + kCamera);

        // Camera speed.
        c += ScriptWriter::makeSetProperty(sceneVar, kCameraSpeed, mCameraSpeed);

        // Viewpoints.
        if (hasSavedViewpoints())
        {
            for (size_t i = 1; i < mViewpoints.size(); i++)
            {
                auto v = mViewpoints[i];
                c += ScriptWriter::makeMemberFunc(sceneVar, kAddViewpoint, v.position, v.target, v.up, v.index);
            }
        }

        return c;
    }

    void Scene::updateNodeTransform(uint32_t nodeID, const float4x4& transform)
    {
        FALCOR_ASSERT(nodeID < mSceneGraph.size());

        Node& node = mSceneGraph[nodeID];
        node.transform = validateTransformMatrix(transform);
        mpAnimationController->setNodeEdited(nodeID);
    }

    inline pybind11::dict toPython(const Scene::SceneStats& stats)
    {
        pybind11::dict d;

        // Geometry stats
        d["meshCount"] = stats.meshCount;
        d["meshInstanceCount"] = stats.meshInstanceCount;
        d["meshInstanceOpaqueCount"] = stats.meshInstanceOpaqueCount;
        d["transformCount"] = stats.transformCount;
        d["uniqueTriangleCount"] = stats.uniqueTriangleCount;
        d["uniqueVertexCount"] = stats.uniqueVertexCount;
        d["instancedTriangleCount"] = stats.instancedTriangleCount;
        d["instancedVertexCount"] = stats.instancedVertexCount;
        d["indexMemoryInBytes"] = stats.indexMemoryInBytes;
        d["vertexMemoryInBytes"] = stats.vertexMemoryInBytes;
        d["geometryMemoryInBytes"] = stats.geometryMemoryInBytes;
        d["animationMemoryInBytes"] = stats.animationMemoryInBytes;

        // Curve stats
        d["curveCount"] = stats.curveCount;
        d["curveInstanceCount"] = stats.curveInstanceCount;
        d["uniqueCurveSegmentCount"] = stats.uniqueCurveSegmentCount;
        d["uniqueCurvePointCount"] = stats.uniqueCurvePointCount;
        d["instancedCurveSegmentCount"] = stats.instancedCurveSegmentCount;
        d["instancedCurvePointCount"] = stats.instancedCurvePointCount;
        d["curveIndexMemoryInBytes"] = stats.curveIndexMemoryInBytes;
        d["curveVertexMemoryInBytes"] = stats.curveVertexMemoryInBytes;

        // SDF grid stats
        d["sdfGridCount"] = stats.sdfGridCount;
        d["sdfGridDescriptorCount"] = stats.sdfGridDescriptorCount;
        d["sdfGridInstancesCount"] = stats.sdfGridInstancesCount;
        d["sdfGridMemoryInBytes"] = stats.sdfGridMemoryInBytes;

        // Custom primitive stats
        d["customPrimitiveCount"] = stats.customPrimitiveCount;

        // Material stats
        d["materialCount"] = stats.materials.materialCount;
        d["materialOpaqueCount"] = stats.materials.materialOpaqueCount;
        d["materialMemoryInBytes"] = stats.materials.materialMemoryInBytes;
        d["textureCount"] = stats.materials.textureCount;
        d["textureCompressedCount"] = stats.materials.textureCompressedCount;
        d["textureTexelCount"] = stats.materials.textureTexelCount;
        d["textureTexelChannelCount"] = stats.materials.textureTexelChannelCount;
        d["textureMemoryInBytes"] = stats.materials.textureMemoryInBytes;

        // Raytracing stats
        d["blasGroupCount"] = stats.blasGroupCount;
        d["blasCount"] = stats.blasCount;
        d["blasCompactedCount"] = stats.blasCompactedCount;
        d["blasOpaqueCount"] = stats.blasOpaqueCount;
        d["blasGeometryCount"] = stats.blasGeometryCount;
        d["blasOpaqueGeometryCount"] = stats.blasOpaqueGeometryCount;
        d["blasMemoryInBytes"] = stats.blasMemoryInBytes;
        d["blasScratchMemoryInBytes"] = stats.blasScratchMemoryInBytes;
        d["tlasCount"] = stats.tlasCount;
        d["tlasMemoryInBytes"] = stats.tlasMemoryInBytes;
        d["tlasScratchMemoryInBytes"] = stats.tlasScratchMemoryInBytes;

        // Light stats
        d["activeLightCount"] = stats.activeLightCount;
        d["totalLightCount"] = stats.totalLightCount;
        d["pointLightCount"] = stats.pointLightCount;
        d["directionalLightCount"] = stats.directionalLightCount;
        d["rectLightCount"] = stats.rectLightCount;
        d["discLightCount"] = stats.discLightCount;
        d["sphereLightCount"] = stats.sphereLightCount;
        d["distantLightCount"] = stats.distantLightCount;
        d["lightsMemoryInBytes"] = stats.lightsMemoryInBytes;
        d["envMapMemoryInBytes"] = stats.envMapMemoryInBytes;
        d["emissiveMemoryInBytes"] = stats.emissiveMemoryInBytes;

        // Volume stats
        d["gridVolumeCount"] = stats.gridVolumeCount;
        d["gridVolumeMemoryInBytes"] = stats.gridVolumeMemoryInBytes;

        // Grid stats
        d["gridCount"] = stats.gridCount;
        d["gridVoxelCount"] = stats.gridVoxelCount;
        d["gridMemoryInBytes"] = stats.gridMemoryInBytes;

        return d;
    }

    FALCOR_SCRIPT_BINDING(Scene)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Rectangle)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Light)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(GridVolume)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Animation)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(AABB)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(Camera)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(EnvMap)
        FALCOR_SCRIPT_BINDING_DEPENDENCY(SDFGrid)

        // RenderSettings
        pybind11::class_<Scene::RenderSettings> renderSettings(m, "SceneRenderSettings");
        renderSettings.def_readwrite("useEnvLight", &Scene::RenderSettings::useEnvLight);
        renderSettings.def_readwrite("useAnalyticLights", &Scene::RenderSettings::useAnalyticLights);
        renderSettings.def_readwrite("useEmissiveLights", &Scene::RenderSettings::useEmissiveLights);
        renderSettings.def_readwrite("useGridVolumes", &Scene::RenderSettings::useGridVolumes);
        renderSettings.def_readwrite("diffuseAlbedoMultiplier", &Scene::RenderSettings::diffuseAlbedoMultiplier);
        renderSettings.def(pybind11::init<>([](bool useEnvLight, bool useAnalyticLights, bool useEmissiveLights, bool useGridVolumes, float diffuseAlbedoMultiplier) {
            Scene::RenderSettings settings;
            settings.useEnvLight = useEnvLight;
            settings.useAnalyticLights = useAnalyticLights;
            settings.useEmissiveLights = useEmissiveLights;
            settings.useGridVolumes = useGridVolumes;
            settings.diffuseAlbedoMultiplier = diffuseAlbedoMultiplier;
            return settings;
        }), "useEnvLight"_a = true, "useAnalyticLights"_a = true, "useEmissiveLights"_a = true, "useGridVolumes"_a = true, "diffuseAlbedoMultiplier"_a = 1.f);
        renderSettings.def("__repr__", [](const Scene::RenderSettings& self) {
            return fmt::format(
                "SceneRenderSettings(useEnvLight={}, useAnalyticLights={}, useEmissiveLights={}, useGridVolumes={}, diffuseAlbedoMultiplier={})",
                self.useEnvLight ? "True" : "False",
                self.useAnalyticLights ? "True" : "False",
                self.useEmissiveLights ? "True" : "False",
                self.useGridVolumes ? "True" : "False",
                self.diffuseAlbedoMultiplier
            );
        });

        // Scene
        pybind11::class_<Scene, ref<Scene>> scene(m, "Scene");

        scene.def_property_readonly(kStats.c_str(), [](const Scene* pScene) { return toPython(pScene->getSceneStats()); });
        scene.def_property_readonly(kBounds.c_str(), &Scene::getSceneBounds, pybind11::return_value_policy::copy);
        scene.def_property(kCamera.c_str(), &Scene::getCamera, &Scene::setCamera);
        scene.def_property(kEnvMap.c_str(), &Scene::getEnvMap, &Scene::setEnvMap);
        scene.def_property_readonly(kAnimations.c_str(), &Scene::getAnimations);
        scene.def_property_readonly(kCameras.c_str(), &Scene::getCameras);
        scene.def_property_readonly(kLights.c_str(), &Scene::getLights);
        scene.def_property_readonly(kGridVolumes.c_str(), &Scene::getGridVolumes);
        scene.def_property_readonly("volumes", &Scene::getGridVolumes); // PYTHONDEPRECATED
        scene.def_property(kCameraSpeed.c_str(), &Scene::getCameraSpeed, &Scene::setCameraSpeed);
        scene.def_property(kAnimated.c_str(), &Scene::isAnimated, &Scene::setIsAnimated);
        scene.def_property(kLoopAnimations.c_str(), &Scene::isLooped, &Scene::setIsLooped);
        scene.def_property(kRenderSettings.c_str(), pybind11::overload_cast<>(&Scene::getRenderSettings, pybind11::const_), &Scene::setRenderSettings);
        scene.def_property(kUpdateCallback.c_str(), &Scene::getUpdateCallback, &Scene::setUpdateCallback);

        scene.def(kSetEnvMap.c_str(), &Scene::loadEnvMap, "path"_a);
        scene.def(kGetLight.c_str(), &Scene::getLight, "index"_a);
        scene.def(kGetLight.c_str(), &Scene::getLightByName, "name"_a);
        scene.def(kGetGridVolume.c_str(), &Scene::getGridVolume, "index"_a);
        scene.def(kGetGridVolume.c_str(), &Scene::getGridVolumeByName, "name"_a);
        scene.def("getVolume", &Scene::getGridVolume, "index"_a); // PYTHONDEPRECATED
        scene.def("getVolume", &Scene::getGridVolumeByName, "name"_a); // PYTHONDEPRECATED
        scene.def(kSetCameraBounds.c_str(), [](Scene* pScene, const float3& minPoint, const float3& maxPoint) {
            pScene->setCameraBounds(AABB(minPoint, maxPoint));
            }, "minPoint"_a, "maxPoint"_a);
        scene.def("getGeometryUVTiles", &Scene::getGeometryUVTiles, "geometryID"_a);

        // Materials
        scene.def_property_readonly(kMaterials.c_str(), &Scene::getMaterials);
        scene.def(kGetMaterial.c_str(), &Scene::getMaterial, "index"_a);
        scene.def(kGetMaterial.c_str(), &Scene::getMaterialByName, "name"_a);
        scene.def("addMaterial", &Scene::addMaterial, "material"_a);
        scene.def("getGeometryIDsForMaterial", [](const Scene* scene, const ref<Material>& pMaterial)
        {
            return scene->getGeometryIDs(pMaterial.get());
        }, "material"_a);


        // Viewpoints
        scene.def(kAddViewpoint.c_str(), pybind11::overload_cast<>(&Scene::addViewpoint)); // add current camera as viewpoint
        scene.def(kAddViewpoint.c_str(), pybind11::overload_cast<const float3&, const float3&, const float3&, uint32_t>(&Scene::addViewpoint), "position"_a, "target"_a, "up"_a, "cameraIndex"_a = 0); // add specified viewpoint
        scene.def(kRemoveViewpoint.c_str(), &Scene::removeViewpoint); // remove the selected viewpoint
        scene.def(kSelectViewpoint.c_str(), &Scene::selectViewpoint, "index"_a); // select a viewpoint by index
    }
}
