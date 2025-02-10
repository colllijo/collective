#include "transformer/TransformerBlock.hpp"

#include <iostream>
#include <random>

#include "matrix/Matrix.hpp"

TransformerBlock::TransformerBlock(int dModel, int dFF)
{
	Wq = xavierUniform(dModel, dModel);
	Wk = xavierUniform(dModel, dModel);
	Wv = xavierUniform(dModel, dModel);
	W1 = kaimingHe(dModel, dFF);
	W2 = kaimingHe(dFF, dModel);
	bq = zeroBias(dModel);
	bk = zeroBias(dModel);
	bv = zeroBias(dModel);
	b1 = zeroBias(dFF);
	b2 = zeroBias(dModel);
}

TransformerBlock::~TransformerBlock() = default;

std::vector<std::vector<float>> TransformerBlock::forward(const std::vector<std::vector<float>>& x) const
{
	std::vector<std::vector<float>> Q = matmul(x, Wq);
	std::vector<std::vector<float>> K = matmul(x, Wk);
	std::vector<std::vector<float>> V = matmul(x, Wv);

	std::vector<std::vector<float>> attention = selfAttention(Q, K, V, bq, bk, bv);

	for (size_t i = 0; i < x.size(); ++i)
	{
		for (size_t j = 0; j < x.at(i).size(); ++j)
		{
			attention[i][j] += x[i][j];
		}
	}

	std::vector<std::vector<float>> feedForwardOutput(attention.size(), std::vector<float>(W2.at(0).size(), 0.0f));
	for (size_t i = 0; i < attention.size(); ++i)
	{
		feedForwardOutput[i] = feedForward(attention[i], W1, b1, W2, b2);
	}

	return feedForwardOutput;
}

std::random_device rd;
std::mt19937 gen(rd());

// Xavier Uniform Initialization
std::vector<std::vector<float>> xavierUniform(int rows, int cols)
{
	float limit = sqrt(6.0f / (rows + cols));  // Xavier limit
	std::uniform_real_distribution<float> dist(-limit, limit);

	std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			matrix[i][j] = dist(gen);
		}
	}
	return matrix;
}

// Kaiming He Initialization
std::vector<std::vector<float>> kaimingHe(int rows, int cols)
{
	float std_dev = sqrt(2.0f / rows);	// Kaiming He std deviation
	std::normal_distribution<float> dist(0.0f, std_dev);

	std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			matrix[i][j] = dist(gen);
		}
	}
	return matrix;
}

// Bias Initialization (Zero)
std::vector<float> zeroBias(int size)
{
	return std::vector<float>(size, 0.0f);
}
