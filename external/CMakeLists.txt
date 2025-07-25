# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------

# d3d12
add_library(d3d12 INTERFACE)
target_link_libraries(d3d12 INTERFACE dxgi.lib d3d12.lib)

# On Linux, we need GTK3.
if(FALCOR_LINUX)
    find_package(GTK3 REQUIRED)
    add_library(gtk3 INTERFACE)
    target_include_directories(gtk3 INTERFACE ${GTK3_INCLUDE_DIRS})
    target_link_libraries(gtk3 INTERFACE ${GTK3_LIBRARIES})
endif()

# -----------------------------------------------------------------------------
# Dependencies in git submodules
# -----------------------------------------------------------------------------

# args
add_library(args INTERFACE)
target_include_directories(args INTERFACE args)

# fmt
message(STATUS "Configure fmt")
add_subdirectory(fmt)
set_target_properties(fmt PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "Libraries"
)

# glfw
message(STATUS "Configure glfw")
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
add_subdirectory(glfw)

# imgui
add_library(imgui INTERFACE)
target_include_directories(imgui INTERFACE imgui)

# pybind11
message(STATUS "Configure pybind11")
add_subdirectory(pybind11)

# vulkan-headers
add_subdirectory(vulkan-headers)


# -----------------------------------------------------------------------------
# Dependencies in local files
# -----------------------------------------------------------------------------

# mikktspace
add_library(mikktspace STATIC mikktspace/mikktspace.c)
target_include_directories(mikktspace PUBLIC mikktspace)
set_target_properties(mikktspace PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    FOLDER "Libraries"
)

# imgui_addons
add_library(imgui_addons INTERFACE)
target_include_directories(imgui_addons INTERFACE imgui_addons)

# external_includes
add_library(external_includes INTERFACE)
target_include_directories(external_includes INTERFACE include)

# -----------------------------------------------------------------------------
# Dependencies in packman
# -----------------------------------------------------------------------------

set(PACKMAN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packman)

# -----------------------------------------------------------------------------
# Pre-built dependencies in falcor_deps packman package (built using vcpkg)
# -----------------------------------------------------------------------------

# Falcor uses a set of pre-built libraries provided by the falcor_deps package
# that are generated by vcpkg. Building in vcpkg allows to make sure to have
# compatible inter-dependencies. The main libraries provided are:
#
# OpenEXR, OpenVDB, assimp, FreeImage, pugixml, hdf5, lz4, zlib
#
# vcpkg does not currently have the ability to generate a set of imported
# targets for use in CMake (it offers to use FindXXX scripts, but this
# currently has issues). Instead we generate our own set of imported targets
# here.

set(FALCOR_DEPS_DIR ${PACKMAN_DIR}/deps)

# OpenEXR
# Note: Using an INTERFACE target to simplify linking against all the various libraries in OpenEXR
if(FALCOR_WINDOWS)
    add_library(OpenEXR INTERFACE)
    target_include_directories(OpenEXR INTERFACE ${FALCOR_DEPS_DIR}/include)
    target_link_directories(OpenEXR INTERFACE
        $<$<CONFIG:Release>:${FALCOR_DEPS_DIR}/lib>
        $<$<CONFIG:Debug>:${FALCOR_DEPS_DIR}/debug/lib>
    )
    target_link_libraries(OpenEXR INTERFACE
        $<$<CONFIG:Release>:OpenEXR-3_1.lib Imath-3_1.lib IlmThread-3_1.lib Iex-3_1.lib>
        $<$<CONFIG:Debug>:OpenEXR-3_1_d.lib Imath-3_1_d.lib IlmThread-3_1_d.lib Iex-3_1_d.lib>
    )
elseif(FALCOR_LINUX)
    add_library(OpenEXR INTERFACE)
    target_include_directories(OpenEXR INTERFACE ${FALCOR_DEPS_DIR}/include)
    target_link_directories(OpenEXR INTERFACE
        $<$<CONFIG:Release>:${FALCOR_DEPS_DIR}/lib>
        $<$<CONFIG:Debug>:${FALCOR_DEPS_DIR}/debug/lib>
    )
    target_link_libraries(OpenEXR INTERFACE
        $<$<CONFIG:Release>:OpenEXR-3_1.so Imath-3_1.so IlmThread-3_1.so Iex-3_1.so>
        $<$<CONFIG:Debug>:OpenEXR-3_1_d.so Imath-3_1_d.so IlmThread-3_1_d.so Iex-3_1_d.so>
    )
endif()

# tbb
if(FALCOR_WINDOWS)
    add_library(tbb SHARED IMPORTED GLOBAL)
    set_target_properties(tbb PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_IMPLIB_RELEASE ${FALCOR_DEPS_DIR}/lib/tbb.lib
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/bin/tbb.dll
        IMPORTED_IMPLIB_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/tbb_debug.lib
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/bin/tbb_debug.dll
    )
    target_compile_definitions(tbb INTERFACE
        $<$<CONFIG:Release>:TBB_USE_DEBUG=0;__TBB_NO_IMPLICIT_LINKAGE=1>
        $<$<CONFIG:Debug>:TBB_USE_DEBUG=1;__TBB_NO_IMPLICIT_LINKAGE=1>
    )
elseif(FALCOR_LINUX)
    add_library(tbb SHARED IMPORTED GLOBAL)
    set_target_properties(tbb PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/libtbb.so
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/libtbb_debug.so
    )
    target_compile_definitions(tbb INTERFACE
        $<$<CONFIG:Release>:TBB_USE_DEBUG=0>
        $<$<CONFIG:Debug>:TBB_USE_DEBUG=1>
    )
endif()

# opensubdiv
# Note: only static libs are supported under windows. Use static for linux as well for symmetry.
if(FALCOR_WINDOWS)
    add_library(opensubdiv STATIC IMPORTED GLOBAL)
    set_target_properties(opensubdiv PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/osdCPU.lib
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/osdCPU.lib
    )
elseif(FALCOR_LINUX)
    add_library(opensubdiv STATIC IMPORTED GLOBAL)
    set_target_properties(opensubdiv PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/libosdCPU.a
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/libosdCPU.a
    )
endif()

# OpenVDB
if(FALCOR_WINDOWS)
    add_library(OpenVDB SHARED IMPORTED GLOBAL)
    set_target_properties(OpenVDB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_IMPLIB_RELEASE ${FALCOR_DEPS_DIR}/lib/openvdb.lib
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/bin/openvdb.dll
        IMPORTED_IMPLIB_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/openvdb.lib
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/bin/openvdb.dll
        INTERFACE_LINK_LIBRARIES "tbb;blosc"
    )
elseif(FALCOR_LINUX)
    add_library(OpenVDB SHARED IMPORTED GLOBAL)
    set_target_properties(OpenVDB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/libopenvdb.so
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/libopenvdb.so
        INTERFACE_LINK_LIBRARIES "tbb;blosc"
    )
endif()

# assimp
# Note: For better performance in Debug builds we always use the release version.
if(FALCOR_WINDOWS)
    add_library(assimp SHARED IMPORTED GLOBAL)
    set_target_properties(assimp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_IMPLIB ${FALCOR_DEPS_DIR}/lib/assimp-vc143-mt.lib
        IMPORTED_LOCATION ${FALCOR_DEPS_DIR}/bin/assimp-vc143-mt.dll
    )
elseif(FALCOR_LINUX)
    add_library(assimp SHARED IMPORTED GLOBAL)
    set_target_properties(assimp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/libassimp.so
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/libassimpd.so
    )
endif()

# FreeImage
if(FALCOR_WINDOWS)
    add_library(FreeImage SHARED IMPORTED GLOBAL)
    set_target_properties(FreeImage PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_IMPLIB_RELEASE ${FALCOR_DEPS_DIR}/lib/FreeImage.lib
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/bin/FreeImage.dll
        IMPORTED_IMPLIB_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/FreeImaged.lib
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/bin/FreeImaged.dll
    )
elseif(FALCOR_LINUX)
    add_library(FreeImage SHARED IMPORTED GLOBAL)
    set_target_properties(FreeImage PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/libFreeImage.so
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/libFreeImaged.so
    )
endif()

# hdf5
if(FALCOR_WINDOWS)
    add_library(hdf5 SHARED IMPORTED GLOBAL)
    set_target_properties(hdf5 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_IMPLIB_RELEASE ${FALCOR_DEPS_DIR}/lib/hdf5.lib
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/bin/hdf5.dll
        IMPORTED_IMPLIB_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/hdf5_D.lib
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/bin/hdf5_D.dll
        INTERFACE_COMPILE_DEFINITIONS H5_BUILT_AS_DYNAMIC_LIB
    )
elseif(FALCOR_LINUX)
    add_library(hdf5 SHARED IMPORTED GLOBAL)
    set_target_properties(hdf5 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/libhdf5.so
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/libhdf5_debug.so
        INTERFACE_COMPILE_DEFINITIONS H5_BUILT_AS_DYNAMIC_LIB
    )
endif()

# lz4
# Note: For better performance in Debug builds we always use the release version.
if(FALCOR_WINDOWS)
    add_library(lz4 STATIC IMPORTED GLOBAL)
    set_target_properties(lz4 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION ${FALCOR_DEPS_DIR}/lib/lz4.lib
    )
elseif(FALCOR_LINUX)
    add_library(lz4 STATIC IMPORTED GLOBAL)
    set_target_properties(lz4 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION ${FALCOR_DEPS_DIR}/lib/liblz4.a
    )
endif()

# zlib
# Note: For better performance in Debug builds we always use the release version.
if(FALCOR_WINDOWS)
    add_library(zlib STATIC IMPORTED GLOBAL)
    set_target_properties(zlib PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION ${FALCOR_DEPS_DIR}/lib/zlib.lib
    )
elseif(FALCOR_LINUX)
    add_library(zlib STATIC IMPORTED GLOBAL)
    set_target_properties(zlib PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION ${FALCOR_DEPS_DIR}/lib/libz.a
    )
endif()

# pugixml
if(FALCOR_WINDOWS)
    add_library(pugixml STATIC IMPORTED GLOBAL)
    set_target_properties(pugixml PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/pugixml.lib
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/pugixml.lib
    )
elseif(FALCOR_LINUX)
    add_library(pugixml STATIC IMPORTED GLOBAL)
    set_target_properties(pugixml PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${FALCOR_DEPS_DIR}/include
        IMPORTED_LOCATION_RELEASE ${FALCOR_DEPS_DIR}/lib/libpugixml.a
        IMPORTED_LOCATION_DEBUG ${FALCOR_DEPS_DIR}/debug/lib/libpugixml.a
    )
endif()

# -----------------------------------------------------------------------------
# Pre-built dependencies in packman
# -----------------------------------------------------------------------------

# aftermath
set(AFTERMATH_DIR ${PACKMAN_DIR}/aftermath)
if(EXISTS ${AFTERMATH_DIR}/include/GFSDK_Aftermath.h)
    if(FALCOR_WINDOWS)
        add_library(aftermath SHARED IMPORTED GLOBAL)
        set_target_properties(aftermath PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${AFTERMATH_DIR}/include
            IMPORTED_IMPLIB ${AFTERMATH_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.lib
            IMPORTED_LOCATION ${AFTERMATH_DIR}/lib/x64/GFSDK_Aftermath_Lib.x64.dll
        )
    elseif(FALCOR_LINUX)
        add_library(aftermath SHARED IMPORTED GLOBAL)
        set_target_properties(aftermath PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${AFTERMATH_DIR}/include
            IMPORTED_LOCATION ${AFTERMATH_DIR}/lib/x64/libGFSDK_Aftermath_Lib.x64.so
    )
    endif()
    set(FALCOR_HAS_AFTERMATH ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_AFTERMATH OFF PARENT_SCOPE)
endif()

# nvapi
set(NVAPI_DIR ${PACKMAN_DIR}/nvapi)
if(EXISTS ${NVAPI_DIR}/nvapi.h)
    add_library(nvapi STATIC IMPORTED GLOBAL)
    set_target_properties(nvapi PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NVAPI_DIR}
        IMPORTED_LOCATION ${NVAPI_DIR}/amd64/nvapi64.lib
    )
    set(FALCOR_HAS_NVAPI ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_NVAPI OFF PARENT_SCOPE)
endif()

# slang
set(SLANG_DIR ${PACKMAN_DIR}/slang)
set(FALCOR_SLANG_CONFIG release CACHE STRING "Slang config to use.")
set_property(CACHE FALCOR_SLANG_CONFIG PROPERTY STRINGS release debug)
if(FALCOR_WINDOWS)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/slang.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/slang.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/${FALCOR_SLANG_CONFIG}/libslang.so
    )
endif()

# slang-gfx
if(FALCOR_WINDOWS)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_IMPLIB ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/gfx.lib
        IMPORTED_LOCATION ${SLANG_DIR}/bin/windows-x64/${FALCOR_SLANG_CONFIG}/gfx.dll
    )
elseif(FALCOR_LINUX)
    add_library(slang-gfx SHARED IMPORTED GLOBAL)
    set_target_properties(slang-gfx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${SLANG_DIR}
        IMPORTED_LOCATION ${SLANG_DIR}/bin/linux-x64/${FALCOR_SLANG_CONFIG}/libgfx.so
    )
endif()

# pix
set(PIX_DIR ${PACKMAN_DIR}/pix)
if(EXISTS ${PIX_DIR}/Include/WinPixEventRuntime/pix3.h)
    add_library(pix SHARED IMPORTED GLOBAL)
    set_target_properties(pix PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${PIX_DIR}/Include
        IMPORTED_IMPLIB ${PIX_DIR}/bin/x64/WinPixEventRuntime.lib
        IMPORTED_LOCATION ${PIX_DIR}/bin/x64/WinPixEventRuntime.dll
    )
    set(FALCOR_HAS_PIX ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_PIX OFF PARENT_SCOPE)
endif()

# nanovdb
set(NANOVDB_DIR ${PACKMAN_DIR}/nanovdb)
add_library(nanovdb INTERFACE)
target_include_directories(nanovdb INTERFACE ${NANOVDB_DIR}/include)

# nv-usd
set(NV_USD_RELEASE_DIR ${PACKMAN_DIR}/nv-usd-release)
set(NV_USD_DEBUG_DIR ${PACKMAN_DIR}/nv-usd-debug)
if((EXISTS ${NV_USD_RELEASE_DIR}) AND (EXISTS ${NV_USD_DEBUG_DIR}))
    if(FALCOR_WINDOWS)
        add_library(nv-usd INTERFACE)
        target_include_directories(nv-usd INTERFACE
            $<$<CONFIG:Release>:${NV_USD_RELEASE_DIR}/include ${NV_USD_RELEASE_DIR}/include/boost-1_76>
            $<$<CONFIG:Debug>:${NV_USD_DEBUG_DIR}/include ${NV_USD_DEBUG_DIR}/include/boost-1_76>
        )
        target_link_directories(nv-usd INTERFACE
            $<$<CONFIG:Release>:${NV_USD_RELEASE_DIR}/lib ${NV_USD_RELEASE_DIR}/plugin/usd>
            $<$<CONFIG:Debug>:${NV_USD_DEBUG_DIR}/lib ${NV_USD_DEBUG_DIR}/plugin/usd>
        )
        target_link_libraries(nv-usd INTERFACE
            usd_ms.lib
        )
        target_compile_definitions(nv-usd INTERFACE
            _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
            $<$<CONFIG:Release>:TBB_USE_DEBUG=0>
            $<$<CONFIG:Debug>:TBB_USE_DEBUG=1>
        )
    elseif(FALCOR_LINUX)
        add_library(nv-usd SHARED IMPORTED GLOBAL)
        set_target_properties(nv-usd PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NV_USD_RELEASE_DIR}/include;${NV_USD_RELEASE_DIR}/include/boost"
            IMPORTED_LOCATION_RELEASE ${NV_USD_RELEASE_DIR}/lib/libusd_ms.so
            IMPORTED_LOCATION_DEBUG ${NV_USD_DEBUG_DIR}/lib/libusd_ms.so
            INTERFACE_LINK_LIBRARIES "tbb"
        )
    endif()
    set(FALCOR_HAS_NV_USD ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_NV_USD OFF PARENT_SCOPE)
endif()

# mdl-sdk
set(MDL_SDK_DIR ${PACKMAN_DIR}/mdl-sdk)
if(EXISTS ${MDL_SDK_DIR})
    add_library(mdl-sdk INTERFACE)
    target_include_directories(mdl-sdk INTERFACE ${MDL_SDK_DIR}/include)
    set(FALCOR_HAS_MDL_SDK ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_MDL_SDK OFF PARENT_SCOPE)
endif()

# dxcompiler
set(DXCOMPILER_DIR ${PACKMAN_DIR}/dxcompiler)
add_library(dxcompiler SHARED IMPORTED GLOBAL)
set_target_properties(dxcompiler PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${DXCOMPILER_DIR}/inc
    IMPORTED_IMPLIB ${DXCOMPILER_DIR}/lib/x64/dxcompiler.lib
    IMPORTED_LOCATION ${DXCOMPILER_DIR}/bin/x64/dxcompiler.dll
)

# nrd
set(NRD_DIR ${PACKMAN_DIR}/nrd)
if(EXISTS ${NRD_DIR}/Include/NRD.h)
    add_library(nrd SHARED IMPORTED GLOBAL)
    set_target_properties(nrd PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NRD_DIR}/Include
        IMPORTED_IMPLIB_RELEASE ${NRD_DIR}/Lib/Release/NRD.lib
        IMPORTED_LOCATION_RELEASE ${NRD_DIR}/Lib/Release/NRD.dll
        IMPORTED_IMPLIB_DEBUG ${NRD_DIR}/Lib/Debug/NRD.lib
        IMPORTED_LOCATION_DEBUG ${NRD_DIR}/Lib/Debug/NRD.dll
    )
    set(FALCOR_HAS_NRD ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_NRD OFF PARENT_SCOPE)
endif()

# dlss

#The official .zip from the github always contains the folder with the version name.
#The following code handles that and sets the correct directory
file(GLOB DLSS_DIR_LIST "${PACKMAN_DIR}/dlss/dlss-*")
list(LENGTH DLSS_DIR_LIST DLSS_COUNT)
if(NOT DLSS_COUNT EQUAL 1)
    message(FATAL_ERROR "Expected exactly one DLSS directory in the packman repo (dlss), found ${DLSS_COUNT}")
endif()
# Extract the first (and only) result from the list
list(GET DLSS_DIR_LIST 0 DLSS_DIR)
set(FALCOR_DLSS_DIR ${DLSS_DIR} PARENT_SCOPE)
if(EXISTS ${DLSS_DIR}/include/nvsdk_ngx.h)
    add_library(dlss STATIC IMPORTED GLOBAL)
    set_target_properties(dlss PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${DLSS_DIR}/include
		if(EXISTS ${DLSS_DIR}/lib/Windows_x86_64/x86_64/nvsdk_ngx_d.lib)
			IMPORTED_LOCATION_RELEASE ${DLSS_DIR}/lib/Windows_x86_64/x86_64/nvsdk_ngx_d.lib
			IMPORTED_LOCATION_DEBUG ${DLSS_DIR}/lib/Windows_x86_64/x86_64/nvsdk_ngx_d_dbg.lib
		else()
			IMPORTED_LOCATION_RELEASE ${DLSS_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib
			IMPORTED_LOCATION_DEBUG ${DLSS_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib
        INTERFACE_COMPILE_DEFINITIONS NV_WINDOWS
    )
    set(FALCOR_HAS_DLSS ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_DLSS OFF PARENT_SCOPE)
endif()

# FFXAPI
set(FFX_DIR ${PACKMAN_DIR}/fsr/ffx-api)
set(FFX_LIB_DIR ${PACKMAN_DIR}/fsr/PrebuiltSignedDLL)
set(FFX_API_BACKEND DX12_X64)
add_library(ffxapi SHARED IMPORTED GLOBAL)
set_target_properties(ffxapi PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${FFX_DIR}/include
    IMPORTED_LOCATION_RELEASE ${FFX_LIB_DIR}/amd_fidelityfx_dx12.dll
    IMPORTED_LOCATION_DEBUG ${FFX_LIB_DIR}/amd_fidelityfx_dx12.dll
    IMPORTED_IMPLIB_RELEASE ${FFX_LIB_DIR}/amd_fidelityfx_dx12.lib
    IMPORTED_IMPLIB_DEBUG ${FFX_LIB_DIR}/amd_fidelityfx_dx12.lib
)
set(FALCOR_HAS_FFXAPI ON PARENT_SCOPE)

# optix
set(OPTIX_DIR ${PACKMAN_DIR}/optix)
if(EXISTS ${OPTIX_DIR}/include/optix.h)
    add_library(optix INTERFACE)
    target_include_directories(optix INTERFACE ${OPTIX_DIR}/include)
    set(FALCOR_HAS_OPTIX ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_OPTIX OFF PARENT_SCOPE)
endif()

# nvtt
set(NVTT_DIR ${PACKMAN_DIR}/nvtt)
if(FALCOR_WINDOWS)
    add_library(nvtt SHARED IMPORTED GLOBAL)
    set_target_properties(nvtt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NVTT_DIR}/include
        IMPORTED_IMPLIB ${NVTT_DIR}/lib/x64-v142/nvtt30106.lib
        IMPORTED_LOCATION ${NVTT_DIR}/nvtt30106.dll
    )
elseif(FALCOR_LINUX)
    add_library(nvtt SHARED IMPORTED GLOBAL)
    set_target_properties(nvtt PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES ${NVTT_DIR}/include
        IMPORTED_LOCATION ${NVTT_DIR}/libnvtt.so
        IMPORTED_NO_SONAME TRUE
    )
endif()

# agility-sdk
set(D3D12_AGILITY_SDK_DIR ${PACKMAN_DIR}/agility-sdk)
if(EXISTS ${D3D12_AGILITY_SDK_DIR}/build/native/include/d3d12.h)
    add_library(agility-sdk INTERFACE)
    target_include_directories(agility-sdk INTERFACE ${D3D12_AGILITY_SDK_DIR}/build/native/include)
    set(FALCOR_HAS_D3D12_AGILITY_SDK ON PARENT_SCOPE)
else()
    set(FALCOR_HAS_D3D12_AGILITY_SDK OFF PARENT_SCOPE)
endif()

# rtxdi
set(RTXDI_DIR ${PACKMAN_DIR}/rtxdi)
add_library(rtxdi INTERFACE)
target_include_directories(rtxdi INTERFACE ${RTXDI_DIR}/rtxdi-sdk/include)

