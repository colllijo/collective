#include "embedding/PositionalEncoding.hpp"

#include <cmath>

PositionalEncoding::PositionalEncoding(int maxSequenceLength, int embeddingDimension)
	: maxSequenceLength(maxSequenceLength), embeddingDimension(embeddingDimension), positionalEncodings{}
{
	positionalEncodings.resize(maxSequenceLength, std::vector<float>(embeddingDimension));

	for (int pos = 0; pos < maxSequenceLength; ++pos)
	{
		for (int i = 0; i < embeddingDimension; i += 2)
		{
			float denominator = std::pow(10000.0f, (static_cast<float>(i) / embeddingDimension));

			positionalEncodings[pos][i] = std::sin(pos / denominator);
			if (i + 1 < embeddingDimension)
			{
				positionalEncodings[pos][i + 1] = std::cos(pos / denominator);
			}
		}
	}
}

PositionalEncoding::~PositionalEncoding() = default;

std::vector<float> PositionalEncoding::getEncoding(int position) const
{
	if (position < 0 || position >= maxSequenceLength)
	{
		return std::vector<float>(embeddingDimension, 0.0f);
	}

	return positionalEncodings[position];
}
