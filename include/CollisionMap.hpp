#include <map>
#include <vector>

struct CollisionDetector
{

    // Given a current position vector, quantize and fill bins to detect collisions.

    std::map<uint32_t, std::vector<size_t>> position_bins;
};