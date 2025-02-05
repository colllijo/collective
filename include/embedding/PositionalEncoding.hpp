#pragma once

#include <vector>
class PositionalEncoding
{
public:
	PositionalEncoding(int maxSequenceLength, int embeddingDimension);
	~PositionalEncoding();

	std::vector<float> getEncoding(int position) const;

private:
	int maxSequenceLength;
	int embeddingDimension;
	std::vector<std::vector<float>> positionalEncodings;
};
