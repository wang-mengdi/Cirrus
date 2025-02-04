set_languages("c++17")

add_requires("fmt =8.1.1")
add_requires("nlohmann_json >=3.10.5")
--add_requires("boost =1.78.0", {configs = {cmake = false}})
--add_requires("boost >=1.86.0")


target("common")
    set_kind("static")
    add_headerfiles("*.h")
    add_files("*.cpp")
    add_includedirs(".",{public=true})
    add_packages("fmt",{public=true})
    add_packages("nlohmann_json",{public=true})
    --add_packages("boost",{public=true})
