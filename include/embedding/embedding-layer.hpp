#pragma once

#include <vector>

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
