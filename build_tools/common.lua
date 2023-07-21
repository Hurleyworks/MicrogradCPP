
local ROOT = "../"

	language "C++"

	defines{
		"JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED", "OIIO_STATIC_DEFINE", "__TBB_NO_IMPLICIT_LINKAGE",
		"_D_CORE_DLL","_D_TINY_DLL","_D_NEWTON_DLL","_D_COLLISION_DLL"
	}
	flags { "MultiProcessorCompile", "NoMinimalRebuild" }
	
	local VCPKG_DIR = "../../OpenSource/vcpkg/installed/x64-windows-static/"
	local PRECOMPILED_DIR = "../../precompiled/"
	local CORE_DIR = ROOT .. "core/source/"
	local JAHLEY_DIR = ROOT .. "core/source/jahley/"
	local THIRD_PARTY_DIR = "../thirdparty/"
	local MODULE_DIR = "../modules/"
	
	local CUDA_INCLUDE_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
	local CUDA_EXTRA_DIR = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/extras/cupti/include"
	local CUDA_LIB_DIR =  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64"
	
	local OPTIX_ROOT = "C:/ProgramData/NVIDIA Corporation"
	local OPTIX7_INCLUDE_DIR = OPTIX_ROOT .. "/OptiX SDK 7.7.0/include"
	
	local NEWTON_SDK_ROOT = "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/sdk/"
	local NEWTON_THIRD_PARTY_DIR = "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/thirdParty/"
	
	includedirs
	{
		CORE_DIR,
		JAHLEY_DIR,
		MODULE_DIR,
		
		CUDA_INCLUDE_DIR,
		CUDA_EXTRA_DIR,
		OPTIX7_INCLUDE_DIR,
		
		VCPKG_DIR,
		VCPKG_DIR .. "include",
		THIRD_PARTY_DIR,
		THIRD_PARTY_DIR .. "g3log/src",
		THIRD_PARTY_DIR .. "optiXUtil/src",
		THIRD_PARTY_DIR .. "reproc++",
		THIRD_PARTY_DIR .. "benchmark/include",
		PRECOMPILED_DIR .. "include",
		PRECOMPILED_DIR .. "include/huse",
		PRECOMPILED_DIR .. "include/tiny-cuda-nn/include",
		PRECOMPILED_DIR .. "include/tiny-cuda-nn/include/tiny-cuda-nn/fmt/include",
		PRECOMPILED_DIR .. "include/tbb",
		PRECOMPILED_DIR .. "include/ozz",
		PRECOMPILED_DIR .. "include/Imath",
		PRECOMPILED_DIR .. "include/drjit_core/include",
		PRECOMPILED_DIR .. "include/drjit/include",
		THIRD_PARTY_DIR .. "glfw/include",
		THIRD_PARTY_DIR .. "nanogui/include",
		THIRD_PARTY_DIR .. "nanogui/ext/glad/include",
		THIRD_PARTY_DIR .. "nanogui/ext/nanovg/src",
		THIRD_PARTY_DIR .. "g3log/src",
		THIRD_PARTY_DIR .. "date/include/date",
		THIRD_PARTY_DIR .. "linalg/eigen34/Eigen",
		THIRD_PARTY_DIR .. "linalg/eigen34",
		THIRD_PARTY_DIR .. "JUCE/modules",
		THIRD_PARTY_DIR .. "rapidobj/",
		THIRD_PARTY_DIR .. "openFBX/src/",
		THIRD_PARTY_DIR .. "cgltfReader",
		THIRD_PARTY_DIR .. "cgltfWriter",
		THIRD_PARTY_DIR .. "fast_obj/source/",
		THIRD_PARTY_DIR .. "meshoptimizer/src/",
		THIRD_PARTY_DIR .. "nanothread/include/",
		THIRD_PARTY_DIR .. "dlib",
		THIRD_PARTY_DIR .. "openvdb/source",
		THIRD_PARTY_DIR .. "openexr/source",
		THIRD_PARTY_DIR .. "instantMeshes/include",
		THIRD_PARTY_DIR .. "instantMeshes",
		THIRD_PARTY_DIR .. "pmp/src",
		THIRD_PARTY_DIR .. "binarytools/src",
		THIRD_PARTY_DIR .. "json",
		THIRD_PARTY_DIR .. "libjpeg/src",
		THIRD_PARTY_DIR .. "stb_image",
		
		NEWTON_SDK_ROOT,
		NEWTON_SDK_ROOT .. "dCollision",
		NEWTON_SDK_ROOT .. "dCore",
		NEWTON_SDK_ROOT .. "dNewton",
		NEWTON_SDK_ROOT .. "dTinyxml",
		NEWTON_SDK_ROOT .. "dNewton/dJoints",
		NEWTON_SDK_ROOT .. "dNewton/dikSolver",
		NEWTON_SDK_ROOT .. "dNewton/dModels",
		NEWTON_SDK_ROOT .. "dNewton/dParticles",
		NEWTON_SDK_ROOT .. "dNewton/dModels/dVehicle",
		NEWTON_SDK_ROOT .. "dNewton/dModels/dCharacter",
		
		--newton Brain
		NEWTON_THIRD_PARTY_DIR .. "brain",	
	}
	
	targetdir (ROOT .. "builds/bin/" .. outputdir .. "/%{prj.name}")
	objdir (ROOT .. "builds/bin-int/" .. outputdir .. "/%{prj.name}")
	
	filter { "system:windows"}
		disablewarnings { 
			"5030", "4244", "4267", "4667", "4018", "4101", "4305", "4316", "4146", "4996", "4554",
		} 
		linkoptions { "-IGNORE:4099" } -- can't find debug file in release folder
		characterset ("MBCS")
		buildoptions { "/Zm250", "/bigobj",}
		
		defines 
		{ 
			"WIN32", "_WINDOWS",
			--https://github.com/KjellKod/g3log/issues/337
			"_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING",
			"CHANGE_G3LOG_DEBUG_TO_DBUG",
		}
		
	filter "configurations:Debug"
	
		postbuildcommands {
			
		}
		links 
		{ 
			"Core",
			"g3log",
			"benchmark",
			"reproc++",
			-- for reproc++
			"ws2_32",
			"opengl32",
			"fmtd",
			"tiny-cuda-nn",
			"tbb",
			"nanothread",
			--"dlib",
			"nanogui",
			"GLFW",
			"JUCE",
			"huse",
			"binarytools",
			
			--"Newton4SDK",
			"ndNewton_d",
			"ndSolverAvx2_d",
			"ndBrain_d", -- in precompiled
			
			--cuda
			"cudart_static",
			"cuda",
			"nvrtc",
			"cublas",
			"curand",
			"cusolver",
			"cudart",
			"cudnn",
			
			"tinyexr",
			"stb_image",
			"meshoptimizer",
			"fast_obj",
			"cgltfReader",
			"drjit-core",
			"drjit-autodiff",
			
			--oiio
			"OpenImageIO_d",
			"OpenImageIO_Util_d",
			"Half-2_5_d",
			"Iex-2_5_d",
			"IexMath-2_5_d",
			"IlmImf-2_5_d",
			"IlmImfUtil-2_5_d",
			"IlmThread-2_5_d",
			"Imath-2_5_d",
			
			"turbojpeg",
			"tiffd",
			"zlibd",
			"lzma",
			"libpng16d",
			"heif",
			"squishd",
			"libde265",
			"boost_thread-vc140-mt-gd",
			"boost_filesystem-vc140-mt-gd",
			"boost_system-vc140-mt-gd",
	
		}
		defines { "DEBUG", "USE_DEBUG_EXCEPTIONS", "EIGEN_NO_DEBUG" }
		symbols "On"
		libdirs { THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  ROOT .. "builds/bin/" .. outputdir .. "/**",
				  PRECOMPILED_DIR .. "bin/" .. outputdir .. "/**",
				  VCPKG_DIR .. "debug/lib",
				  "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/build/sdk/Debug",
				  "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/build/sdk/dNewton/dExtensions/dAvx2/Debug",
				  "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/build/sdk/dNewton/dExtensions/dCuda/Debug",
				  CUDA_LIB_DIR
		}
		
	filter "configurations:Release"
	postbuildcommands {
			
		}
		links 
		{ 
			"Core",
			"g3log",
			"benchmark",
			"reproc++",
			-- for reproc++
			"ws2_32",
			"opengl32",
			"fmt",
			"tiny-cuda-nn",
			"tbb",
			"nanothread",
			"nanogui",
			"GLFW",
			"JUCE",
			"huse",
			"binarytools",
			
			--"Newton4SDK",
			"ndNewton",
			"ndSolverAvx2",
			"ndBrain",
			
			--cuda
			"cudart_static",
			"cuda",
			"nvrtc",
			"cublas",
			"curand",
			"cusolver",
			"cudart",
			"cudnn",
			
			"tinyexr",
			"stb_image",
			"meshoptimizer",
			"fast_obj",
			"cgltfReader",
			"drjit-core",
			"drjit-autodiff",
			
			 --for 0ii0
			"OpenImageIO",
			"OpenImageIO_Util",
			"Half-2_5",
			"Iex-2_5",
			"IexMath-2_5",
			"IlmImf-2_5",
			"IlmImfUtil-2_5",
			"IlmThread-2_5",
			"Imath-2_5",
			"lzma",
			
			"turbojpeg",
			"libpng16",
			"tiff",
			"zlib",
			"heif",
			"squish",
			"libde265",
			"boost_thread-vc140-mt",
			"boost_filesystem-vc140-mt",
			"boost_system-vc140-mt",
			
		}
		defines { "NDEBUG", "EIGEN_NO_DEBUG" }
		optimize "On"
		libdirs { THIRD_PARTY_DIR .. "builds/bin/" .. outputdir .. "/**",
				  ROOT .. "builds/bin/" .. outputdir .. "/**",
				  PRECOMPILED_DIR .. "bin/" .. outputdir .. "/**",
				  VCPKG_DIR .. "lib",
				  "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/build/sdk/Release",
				  "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/build/sdk/dNewton/dExtensions/dAvx2/Release",
				  "../../OpenSource/latestNewton/newton-dynamics/newton-4.00/build/sdk/dNewton/dExtensions/dCuda/Release",
				  CUDA_LIB_DIR
		}
	
	  


	 		

