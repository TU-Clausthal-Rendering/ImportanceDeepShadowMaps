add_plugin(IDSMRenderer)

target_sources(IDSMRenderer PRIVATE
	DSMAccelerationStructure/DSMAccelerationStructure.h
	DSMAccelerationStructure/DSMAccelerationStructure.cpp
	DSMAccelerationStructure/GenerateDSMAccelerationStructure.rt.slang
	DSMAccelerationStructure/DSMAccelerationStructure.slang
	DSMAccelerationStructure/DebugShowShadowAccel.3d.slang

    DSMLinkedList/DSMLinkedList.h
	DSMLinkedList/DSMLinkedList.cpp
	DSMLinkedList/DSMLinkedList.slang
	DSMLinkedList/DSMLinkedListData.slang
	DSMLinkedList/GenerateDSMLinkedList.rt.slang
	DSMLinkedList/GenerateLinkedListNeighbors.cs.slang

	IDSMAccelerationStructure/IDSMAccelerationStructure.h
	IDSMAccelerationStructure/IDSMAccelerationStructure.cpp
	IDSMAccelerationStructure/GenerateIDSMAccelerationStructure.rt.slang
	IDSMAccelerationStructure/IDSMAccelerationStructure.slang
	IDSMAccelerationStructure/DebugShowShadowAccel.3d.slang

    IDSMLinkedList/IDSMLinkedList.h
	IDSMLinkedList/IDSMLinkedList.cpp
	IDSMLinkedList/IDSMLinkedList.slang
	IDSMLinkedList/GenerateIDSMLinkedList.rt.slang
	IDSMLinkedList/IDSMLinkedListData.slang
	IDSMLinkedList/DebugShowImportance.cs.slang

    ImportanceMapHelpers/ImportanceMapHelper.h
    ImportanceMapHelpers/ImportanceMapHelper.cpp
    ImportanceMapHelpers/ReduceMipMaps.cs.slang
    ImportanceMapHelpers/Reduce.cs.slang
    ImportanceMapHelpers/DistributeBudget.cs.slang
    ImportanceMapHelpers/GenSampleDistribution.cs.slang
    ImportanceMapHelpers/RaySampleFromSampleDistribution.slang
    ImportanceMapHelpers/HeadIndexFromSampleDistribution.slang

    IDSMMaskAndOpaqueShadowMap/IDSMMaskAndOpaqueShadowMap.cpp
    IDSMMaskAndOpaqueShadowMap/IDSMMaskAndOpaqueShadowMap.h
    IDSMMaskAndOpaqueShadowMap/GenerateIDSMMask.3d.slang
    IDSMMaskAndOpaqueShadowMap/AccumulateMask.cs.slang
    IDSMMaskAndOpaqueShadowMap/GenMaskShadowMap.rt.slang
    IDSMMaskAndOpaqueShadowMap/GenMaskImportanceShadowMap.rt.slang

	DeepShadowMapMethod.h
	DeepShadowMapMethod.cpp

	IDSMRenderer.cpp
	IDSMRenderer.h
	
	EvaluateAnalyticLight.slang
	EvalTransparenciesDirect.rt.slang
    RayReflections.rt.slang
	PathTracer.rt.slang
    SharedShadowDefines.slang
)

target_copy_shaders(IDSMRenderer RenderPasses/IDSMRenderer)
target_source_group(IDSMRenderer "RenderPasses")
