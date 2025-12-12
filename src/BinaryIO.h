#include "PoissonGrid.h"

namespace BinaryIO {
    void OutputPoissonGridAsJsonAndBin(std::shared_ptr<HAHostTileHolder<Tile>> holder_ptr, const std::vector<std::pair<int, std::string>>& scalar_channels, const std::vector<std::pair<int, std::string>>& vec_channels, const std::filesystem::path& json_path);
}