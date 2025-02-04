import os
import sys

if __name__=='__main__':
    proj_name = ""
    #if len(sys.argv)==2:
    #    proj_name=sys.argv[1]
    bin_dir=os.path.join('bin',proj_name)
    build_dir=os.path.join('build',proj_name)
    lua_file=os.path.join('xmake.lua')
    proj_dir=os.path.join('.', proj_name)
    clean_cmd="xmake c -a"
    config_cmd=f"xmake f -o {bin_dir} -p {proj_name} -y -v"
    project_cmd=f"xmake project -k vsxmake -v -a \"x64\" {build_dir}"
    print(clean_cmd)
    os.system(clean_cmd)
    print(config_cmd)
    os.system(config_cmd)
    print(project_cmd)
    os.system(project_cmd)