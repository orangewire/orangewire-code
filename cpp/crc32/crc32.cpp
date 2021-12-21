#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

/*
    CRC32 checksum calculation.
    Based on this Matlab implementation:
    https://www.mathworks.com/matlabcentral/fileexchange/49518-crc-32-computation-algorithm
*/
constexpr uint32_t polynomial = 0xEDB88320;
constexpr uint32_t init_xor = 0xFFFFFFFF;

uint32_t crc32(const std::string &data)
{
    uint32_t crc = init_xor;
    for (char cc : data)
    {
        crc ^= uint32_t(cc);
        for (size_t jj = 0; jj < 8; ++jj)
        {
            uint32_t mask = (crc & 1) ? ~(crc & 1) + 1 : 0;
            crc = (crc >> 1) ^ (polynomial & mask);
        }
    }
    crc = ~crc;
    return crc;
}

// Helper function to change the case of an input character
inline char toggle_case(char input)
{
    if (std::isalpha(input))
        return std::islower(input) ? char(std::toupper(input)) : char(std::tolower(input));
    return input;
}

/*
    Having the diffs and transfer matrix columns stored in the
    same object (AoS) makes it easier to broadcast operations to
    both of them.
*/
struct Column
{
    uint32_t diff;
    uint64_t transfer;
};

/*
    We will compute the effect that singular upper case transformations have
    on the hash for each index in the string.
    Changing the 8th character from lower to upper case for example,
    might flip bits [0, 5, 12, 24, 31] in the hash, so we record this
    set of bits as a diff vector for index= 8.
    The diff vectors are arranged in a diff matrix. We also initialize
    a transfer matrix to identity.
*/
void compute_diffs(const std::string &init_str, std::vector<Column> &columns)
{
    columns.resize(init_str.size());
    uint32_t init_crc = crc32(init_str);

    for (size_t ii = 0; ii < init_str.size(); ++ii)
    {
        // Singular case modification at index ii
        std::string tmp(init_str);
        tmp[ii] = toggle_case(tmp[ii]);

        // Record diff for this transformation
        columns[ii].diff = init_crc ^ crc32(tmp);
        // The transfer matrix is the identity in this basis
        columns[ii].transfer = 1ul << ii;
    }
}

/*
    We want to find the linear combinations of diff vectors that have only
    one non-zero component. These correspond to the combinations of singular
    case transformation that will flip only one bit in the hash.
    Our diff matrix needs to be diagonalized, and the same transformations
    are to be applied to an identity-initialized transfer matrix during
    diagonalization so as to keep track of the change of basis.
*/
void diagonalize(std::vector<Column> &columns)
{
    // Use a GF(2)-Gauss-Jordan elimination to diagonalize the diffs and
    // apply the same operations to the transfer matrix

    for (size_t cur_column = 0; cur_column < columns.size(); ++cur_column)
    {
        // Get the index of the first set bit in this column (pivot)
        uint32_t col_mask = columns[cur_column].diff & ~(columns[cur_column].diff - 1);
        // For all other columns, find 1s in the column where the pivot is found
        for (size_t other_column = 0; other_column < columns.size(); ++other_column)
        {
            if (other_column != cur_column && (columns[other_column].diff & col_mask) > 0)
            {
                // If found, add the pivot column to this column to clear the 1
                columns[other_column].diff ^= columns[cur_column].diff;
                // Same operation on transfer matrix
                columns[other_column].transfer ^= columns[cur_column].transfer;
            }
        }
    }
}

/*
    After diagonalization, we get a diagonal transfer matrix (up to columns reordering),
    and a transfer matrix that enables change of basis from singular case
    transformation basis to singular bit flip basis.
*/
inline void reorder(std::vector<Column> &columns)
{
    // Because we use unsigned integers as binary columns, and because diffs and
    // transfer columns are grouped together, reordering the columns is just a
    // simple sort.
    std::sort(columns.begin(), columns.end(), [](const Column &a, const Column &b) { return a.diff < b.diff; });
}

/*
    After diagonalization, a few columns of the diff matrix are all-zeros. The corresponding
    columns in the transfer matrix represent the case transformations that leave the
    hash unaltered: they are hash collisions!
    These transformations are of no use to us however, so we remove them.
*/
inline void remove_collisions(std::vector<Column> &columns)
{
    // Remove the columns where the diff is null
    columns.erase(std::remove_if(columns.begin(), columns.end(), [](const Column &column) { return column.diff == 0; }),
                  columns.end());
}

/*
    All we need to do now, is to compute a diff between the initial CRC and the target CRC, and
    for all bits that need to be flipped, apply the corresponding case transformation to our
    input string.
*/
std::string control_crc(const std::string &init_str, const std::vector<Column> &columns, uint32_t target)
{
    uint32_t init_crc = crc32(init_str);
    uint32_t need_flip = init_crc ^ target;

    // Let's accumulate all the transformations that need to be applied in
    // a binary mask
    uint32_t mask = 0;
    for (size_t ii = 0; ii < 32; ++ii)
        if (need_flip & (1 << ii))
            mask ^= columns[ii].transfer;

    // Toggle the case at indices where the mask bits are 1
    std::string ctl_str(init_str);
    for (size_t ii = 0; ii < 32; ++ii)
        if (mask & (1 << ii))
            ctl_str[ii] = toggle_case(ctl_str[ii]);

    return ctl_str;
}

void show(std::vector<Column> &columns)
{
    std::cout << "Diffs:" << std::endl;
    for (auto column : columns)
        std::cout << std::bitset<32>(column.diff) << std::endl;

    std::cout << "Transfer matrix:" << std::endl;
    for (auto column : columns)
        std::cout << std::bitset<32>(column.transfer) << std::endl;
}

void apply_elementary_transformation(const std::string &init_str, std::vector<Column> &columns, size_t bit_index)
{
    std::string ctl_str(init_str);
    for (size_t ii = 0; ii < 32; ++ii)
        if (columns[bit_index].transfer & (1 << ii))
            ctl_str[ii] = toggle_case(ctl_str[ii]);

    uint32_t init_crc = crc32(init_str);
    uint32_t ctl_crc = crc32(ctl_str);

    std::cout << "Initial CRC: " << std::bitset<32>(init_crc) << std::endl;
    std::cout << "Final CRC:   " << std::bitset<32>(ctl_crc) << std::endl;
    std::cout << ctl_str << std::endl;
}

std::string random_string()
{
    constexpr std::string_view alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    std::string ret = "";
    std::uniform_int_distribution<size_t> len_dis(32, 128);
    std::uniform_int_distribution<size_t> idx_dis(0, 51);
    std::random_device rng;
    for (size_t ii = 0; ii < len_dis(rng); ++ii)
        ret += alphabet[idx_dis(rng)];

    return ret;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    std::string init_str = random_string();
    std::vector<Column> columns;

    compute_diffs(init_str, columns);
    diagonalize(columns);
    reorder(columns);
    remove_collisions(columns);
    // show(columns);

    // apply_elementary_transformation(init_str, columns, 2);

    auto ctl_str = control_crc(init_str, columns, 0xdeadbeef);

    std::cout << "Initial string: '" << init_str << '\'' << std::endl;
    std::cout << "    -> 0x" << std::hex << crc32(init_str) << std::endl;
    std::cout << "Final string:   '" << ctl_str << '\'' << std::endl;
    std::cout << "    -> 0x" << std::hex << crc32(ctl_str) << std::endl;

    return 0;
}