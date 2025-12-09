add_rules("mode.debug", "mode.release", "mode.releasedbg")
set_languages("c++17")

add_requires("fmt =12.1.0")
add_requireconfs("*.fmt", { override = true, version = "12.1.0" })

includes("./common/xmake.lua")

add_requires("eigen >=3.4.0")
--add_requires("cuda")
add_requires("vtk >=9.5.1", {configs = {cuda = true}})
--add_requires("vtk =9.3.1", {configs = {cuda = true}})
--add_requires("stb")
add_requires("polyscope =2.3")
--add_requireconfs("polyscope.imgui", {override = true, version = "1.91.1"})

add_defines("FMT_UNICODE=0")


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

    --add_packages("cuda", {public = true})
    add_packages("eigen", {public = true})
    add_packages("vtk", {public = true})
    --add_packages("stb", {public = true})
    --add_packages("polyscope", {public = true})

    add_deps("common")
