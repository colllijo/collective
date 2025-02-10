#pragma once

#include <vector>

#include "embedding/EmbeddingLayer.hpp"
#include "transformer/TransformerBlock.hpp"

class GPT
{
public:
	GPT(int vocabularySize, int maxSequenceLength, int dModel, int numLayers, int dFF);
	~GPT();

	int predictNextToken(const std::vector<int>& tokenIds, float temperature = 1.0f) const;

private:
	int numLayers;
	int maxSequenceLength;
	int dModel;

	EmbeddingLayer embedding;
	PositionalEncoding positionalEncoding;
	std::vector<TransformerBlock> layers;

	std::vector<std::vector<float>> forward(const std::vector<int>& tokenIds) const;
};
