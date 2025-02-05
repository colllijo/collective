#pragma once

#include <vector>
#include "embedding/PositionalEncoding.hpp"

class EmbeddingLayer
{
public:
	EmbeddingLayer(int vocabularySize, int embeddingDimension);
	~EmbeddingLayer();

	std::vector<float> getEmbedding(int tokenId) const;

private:
	int vocabularySize;
	int embeddingDimension;

	std::vector<std::vector<float>> embeddings;
};

std::vector<std::vector<float>> applyPositionalEncoding(const std::vector<int> tokenIds, const EmbeddingLayer& embeddingLayer, const PositionalEncoding& positionalEncoding);
