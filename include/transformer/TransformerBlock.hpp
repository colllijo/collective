#pragma once

#include <vector>

std::vector<std::vector<float>> xavierUniform(int rows, int cols);
std::vector<std::vector<float>> kaimingHe(int rows, int cols);
std::vector<float> zeroBias(int size);

class TransformerBlock
{
public:
	TransformerBlock(int dModel, int dFF);
	~TransformerBlock();

	std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& x) const;

private:
	std::vector<std::vector<float>> Wq, Wk, Wv, W1, W2;
	std::vector<float> bq, bk, bv, b1, b2;
};
