#include "matrix/Matrix.hpp"

#include <algorithm>
#include <cmath>

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B)
{
	size_t rows = A.size();
	size_t cols = B.at(0).size();
	size_t k = A.at(0).size();

	std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < cols; ++j)
		{
			for (size_t l = 0; l < k; ++l)
			{
				result[i][j] += A[i][l] * B[l][j];
			}
		}
	}

	return result;
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& M)
{
	int rows = M.size();
	int cols = M.at(0).size();
	std::vector<std::vector<float>> result(cols, std::vector<float>(rows, 0.0f));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			result[j][i] = M[i][j];
		}
	}

	return result;
}

std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x)
{
	std::vector<std::vector<float>> result = x;

	for (auto& row : result)
	{
		float sumExp = 0.0f;

		for (float& value : row)
		{
			value = std::exp(value);
			sumExp += value;
		}

		for (float& value : row)
		{
			value /= sumExp;
		}
	}

	return result;
}

std::vector<float> softmax(const std::vector<float>& logits)
{
	std::vector<float> expValues(logits.size());
	float maxLogit = *std::max_element(logits.begin(), logits.end());

	float sumExp = 0.0f;
	for (size_t i = 0; i < logits.size(); ++i)
	{
		expValues[i] = std::exp(logits[i] - maxLogit);
		sumExp += expValues[i];
	}

	for (float& value : expValues)
	{
		value /= sumExp;
	}

	return expValues;
}

std::vector<float> softmax(const std::vector<float>& logits, float temperature)
{
	std::vector<float> scaledLogits = logits;
	float maxLogit = *std::max_element(scaledLogits.begin(), scaledLogits.end());

	float sumExp = 0.0f;
	for (float& logit : scaledLogits)
	{
		logit = std::exp((logit - maxLogit) / temperature);
		sumExp += logit;
	}

	for (float& value : scaledLogits)
	{
		value /= sumExp;
	}

	return scaledLogits;
}

std::vector<std::vector<float>> selfAttention(const std::vector<std::vector<float>>& queries, const std::vector<std::vector<float>>& keys,
											  const std::vector<std::vector<float>>& values, const std::vector<float>& bq, const std::vector<float>& bk,
											  const std::vector<float>& bv)
{
	std::vector<std::vector<float>> Q = queries;
	std::vector<std::vector<float>> K = keys;
	std::vector<std::vector<float>> V = values;

	// Add the biases
	for (size_t i = 0; i < Q.size(); ++i)
	{
		for (size_t j = 0; j < Q.at(i).size(); ++j)
		{
			Q[i][j] += bq[j];
			K[i][j] += bk[j];
			V[i][j] += bv[j];
		}
	}

	std::vector<std::vector<float>> scores = matmul(Q, transpose(K));
	float dK = std::sqrt(K.size());

	for (auto& row : scores)
	{
		for (float& value : row)
		{
			value /= dK;
		}
	}

	return matmul(softmax(scores), V);
}

std::vector<float> feedForward(const std::vector<float>& x, const std::vector<std::vector<float>>& W1, const std::vector<float>& b1, const std::vector<std::vector<float>>& W2,
							   const std::vector<float>& b2)
{
	std::vector<float> hidden(W1.at(0).size(), 0.0f);

	for (size_t i = 0; i < W1.at(0).size(); ++i)
	{
		hidden[i] = b1[i];

		for (size_t j = 0; j < x.size(); ++j)
		{
			hidden[i] += x[j] * W1[j][i];
		}

		hidden[i] = std::max(0.0f, hidden[i]);
	}

	std::vector<float> output(W2.at(0).size(), 0.0f);
	for (size_t i = 0; i < W2.at(0).size(); ++i)
	{
		output[i] = b2[i];

		for (size_t j = 0; j < hidden.size(); ++j)
		{
			output[i] += hidden[j] * W2[j][i];
		}
	}

	return output;
}
