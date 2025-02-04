add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

includes("./common/xmake.lua")

add_requires("eigen >=3.4.0")
add_requires("cuda", {system = true})

add_requires("vtk >=9.3.1", {configs = {cuda = true}})

set_rundir("$(projectdir)")

target("cirrus")
    set_kind("binary")
    add_headerfiles("src/*.h", "ext/*.h")
    add_files("src/*.cpp", "src/*.cu")
    add_includedirs("src", "ext", {public = true})
    if is_plat("windows") then
        set_values("build.vcxproj.includes", "$(CUDA_PATH)/include")
    end

    add_cugencodes("native")
    add_cuflags("-extended-lambda --std=c++17 -lineinfo")

    add_packages("cuda", {public = true})
    add_packages("eigen", {public = true})
    add_packages("openvdb", {public = true})
    add_packages("vtk", {public = true})
    add_packages("polyscope", {public = true})

    add_deps("common")
