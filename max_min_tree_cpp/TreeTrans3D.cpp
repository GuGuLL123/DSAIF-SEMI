#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <vector>
#include <pybind11/embed.h> // 这个文件有助于使用 py::module::attr属性
#include <pybind11/eval.h> // 这个文件有助于使用 py::eval py::exec属性
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <fstream>

namespace py = pybind11;
// py::object np = py::module::import("numpy");

template <typename CountType>
CountType getCol(const CountType &index, const CountType &depth ,const CountType &width)
{
    return (index % (width*depth))/ depth;
}

template <typename CountType>
CountType getDep(const CountType &index, const CountType &depth ,const CountType &width)
{
    return (index % (width*depth))% depth;
}

template <typename CountType>
CountType getRow(const CountType &index, const CountType &depth ,const CountType &width)
{
    return index / (width*depth);
}

template <typename ValueType, typename CountType, typename Iterator, typename ConstIterator>
void getLocation(std::map<ValueType, CountType> &statistics,
                 Iterator iter, ConstIterator end)
{
    CountType location = 0;
    while (iter != end)
    {
        CountType tmp = location + iter->second;
        iter->second = location;
        location = tmp;
        ++iter;
    }
}

template <typename CountType>
CountType findRoot(std::vector<CountType> &parents, CountType pixel)
{
    if (parents[pixel] == pixel)
    {
        return pixel;
    }
    else
    {
        return (parents[pixel] = findRoot(parents, parents[pixel]));
    }
}

// "ascend" or "descend"
template <typename ValueType, typename CountType>
void sortPixels(
    std::vector<CountType> &pixels,
    const ValueType *data,
    const CountType &size,
    const std::string &order = "ascend")
{
    std::map<ValueType, CountType> statistics;

    for (CountType i = 0; i < size; i++){
        statistics.try_emplace(data[i], 0);
        statistics[data[i]] += 1;
    }

    if (order == "ascend")
    {
        getLocation(statistics, statistics.begin(), statistics.cend());
    }
    else if (order == "descend")
    {
        getLocation(statistics, statistics.rbegin(), statistics.crend());
    }


    for (CountType i=0; i < size; i++)
    {
        pixels[statistics[data[i]]++] = i;
    }
}

template <typename ValueType, typename CountType, int ArrayStyle>
py::list MinMaxTree3D(
    const py::array_t<ValueType, ArrayStyle> &intensity,
    const std::string &treeType,
    const uint8_t &&neighbor)
{
    if (intensity.ndim() != 3)
    {
        throw std::runtime_error("numpy.ndarray dims must be 3!");
    }

    const CountType height = intensity.shape(0);
    const CountType width = intensity.shape(1);
    const CountType depth = intensity.shape(2);
    const CountType &size = intensity.size();
    const ValueType *data = intensity.data();

    // 注意下vector删除内存的问题
    std::vector<CountType> pixels;
    pixels.reserve(size);
    if (treeType == "mintree")
    {
        sortPixels(pixels, data, size);
    }
    else if (treeType == "maxtree")
    {
        sortPixels(pixels, data, size, "descend");
    }

    std::vector<std::array<int8_t, 3>> neighborShit;
    neighborShit.reserve(neighbor);
    // 由于数据类型为正整数，因此 附近点+1 防止负数出现
    // 之后验证附近点是否超出图像范围 与 (0, 0)及(h, w)比较
    switch (neighbor)
    {
    case 6:
        neighborShit = {{0, 1,1}, {1, 0,1}, {1, 2,1}, {2, 1,1},{1,1,0},{1,1,2}};
        break;
    case 26:
        neighborShit = {{0, 0 ,1}, {0, 1,1}, {0, 2,1}, {1, 0,1}, {1, 2,1}, {2, 0,1}, {2, 1,1}, {2, 2,1},{0, 0 ,0}, {0, 1,0}, {0, 2,0}, {1, 0,0}, {1, 2,0}, {2, 0,0}, {2, 1,0}, {2, 2,0},{1,1,0},  {0, 0 ,2}, {0, 1,2}, {0, 2,2}, {1, 0,2}, {1, 2,2}, {2, 0,2}, {2, 1,2}, {2, 2,2},{1,1,2}};
        break;
    }

    // 可能需要返回的值用 py::array
    py::array_t<CountType> parMap = py::array_t<CountType>({height, width,depth});
    CountType *parents = parMap.mutable_data();
    // 不需要返回的值用 vector
    std::vector<CountType> area;
    area.reserve(size);



    {
        // union find
        // 临时的值用申明
        std::vector<CountType> tmpPar;
        tmpPar.reserve(size);
        std::vector<bool> visited(size, false);
        CountType p, pX, pY, pZ;

        for (CountType i = 0; i < size; ++i)
        {
            p = pixels[i];
            pX = getRow(p, depth, width);
            pY = getCol(p, depth, width);
            pZ = getDep(p, depth, width);
            parents[p] = p;
            tmpPar[p] = p;
            area[p] = 1;
            for (auto &&[neSX, neSY, neSZ] : neighborShit)
            {
                // 附近点是否超出图像范围
                // 由于数据类型为正整数，因此 +1 防止负数出现
                if (pY + neSY <= 0 or pX + neSX <= 0 or pZ + neSZ <= 0  or pX + neSX > height or pY + neSY > width or pZ + neSZ > depth) continue;
                CountType neP{(pX+ neSX - 1)*(depth * width) + (pY + neSY - 1) * depth + (pZ + neSZ - 1)};
                if (!visited[neP])
                {
                    continue;
                }
                CountType r = findRoot(tmpPar, neP);
                if (r != p)
                {
                    parents[r] = p;
                    tmpPar[r] = p;
                    area[p] += area[r];
                }

            }
            visited[p] = true;
        }

    }

    {
        // canonize tree
        CountType p, tP;
        for (CountType i = size; i >= 1 ; --i)
        {
            p = pixels[i-1];
            tP = parents[p];
            if (data[parents[tP]] == data[tP])
            {
                parents[p] = parents[tP];
            }
        }

    }

    std::map<CountType, std::array<std::vector<CountType>, 4>> tree;
    for (CountType i = 0; i < size; ++i)
    {
        CountType p = pixels[i];    // pixel
        CountType pp = parents[p];  // parent pixel
        
        // init dict key value
        if (data[p] != data[pp] || p == pp)
        {
            tree.try_emplace(p, std::array<std::vector<CountType>, 4>({{}, {}, {}, {area[p]}}));
        }
        tree.try_emplace(pp, std::array<std::vector<CountType>, 4>({{}, {}, {}, {area[pp]}}));

        if (tree.contains(p))
        {
            if (p != pp)
            {
                tree[p][0].emplace_back(pp);
                tree[pp][1].emplace_back(p);
            }
            tree[p][2].emplace_back(p);
        }
        else
        {
            tree[pp][2].emplace_back(p);
        }
    }
    
    // 记录 tree root
    py::list res = py::list();
    res.append(parMap);
    res.append(pixels[size-1]);
    res.append(tree);
    return res;
    // _Float32
}

PYBIND11_MODULE(tree_trans3D, m) // 创建一个 Python 模块，名为 tree_trans ，用变量 m 表示
{
    m.doc() = "pybind11 example module"; // 给模块 m 添加说明文档
    m.def("min_max_tree3D",
          &MinMaxTree3D<uint8_t, uint32_t, py::array::c_style | py::array::forcecast>,
          py::arg("img"),
          py::arg("tree_type") = "mintree",
          py::arg("neighbor") = 6); // 给模块 m 定义一个函数，名为 sum ，绑定到 C++ 代码中的 sum 函数
}
