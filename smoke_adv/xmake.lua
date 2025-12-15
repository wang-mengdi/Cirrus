set_languages("c++17")

add_requires("tbb")
add_requires("alembic")
add_requires("nlohmann_json >=3.10.5")

target("smoke_adv")
    set_kind("binary")
    add_files("*.cpp")
    add_headerfiles("*.h")
    add_includedirs(".", {public = true})

    add_packages("alembic", {public = true})
    add_packages("fmt",{public=true})
    add_packages("nlohmann_json",{public=true})
    add_packages("tbb", {public = true})