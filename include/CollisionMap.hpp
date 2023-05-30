#include <map>
#include <vector>
#include <string>

#include <eigen3/Eigen/Eigen>

#include <iostream>

struct CollisionDetector
{
    // Given a current position vector, quantize and fill bins to detect collisions.

    bool run(const Eigen::MatrixXf &positions, std::vector<std::pair<int, int>> &collisions)
    {
        for (int i = 0; i < positions.cols(); i++)
        {
            uint64_t x = int(std::abs(positions(0, i))) / 10;
            uint64_t y = int(std::abs(positions(1, i))) / 10;
            uint64_t z = int(std::abs(positions(2, i))) / 10;

            uint64_t idx = (x << 32) + (y << 16) + z;

            position_bins[idx].push_back(i);
        }

        for (auto bin : position_bins)
        {

            if (bin.second.size() < 2)
                continue;
            // std::cerr << bin.first << " : ";

            for (size_t i = 0; i < bin.second.size(); i++)
            {
                auto p = bin.second[i];
                for (size_t j = i + 1; j < bin.second.size(); j++)
                {
                    auto q = bin.second[j];
                    auto distance = (positions.col(p) - positions.col(q)).norm();
                    if (distance > 5)
                        continue;
                    // std::cerr << "(" << p << "," << q << "," << distance << ") :" << positions.col(p).transpose() << " -> " << positions.col(q).transpose() << std::endl;
                    collisions.push_back({p, q});
                }

                // std::cerr << std::endl;
            }
        }

        return !collisions.empty();
    }

    std::map<uint64_t, std::vector<uint32_t>> position_bins;
};
