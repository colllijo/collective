#include "embedding/EmbeddingLayer.hpp"

#include <random>

EmbeddingLayer::EmbeddingLayer(int vocabularySize, int embeddingDimension) : vocabularySize(vocabularySize), embeddingDimension(embeddingDimension), embeddings{}
{
	embeddings.resize(vocabularySize, std::vector<float>(embeddingDimension));

	// Xavier initialization
	std::random_device rd;
	std::mt19937 gen(rd());
	float limit = std::sqrt(6.0f / (embeddingDimension));
	std::uniform_real_distribution<float> dist(-limit, limit);

	for (int i = 0; i < vocabularySize; ++i)
	{
		for (int j = 0; j < embeddingDimension; ++j)
		{
			embeddings[i][j] = dist(gen);
		}
	}
}

EmbeddingLayer::~EmbeddingLayer() = default;

std::vector<float> EmbeddingLayer::getEmbedding(int tokenId) const
{
	if (tokenId < 0 || tokenId >= vocabularySize)
	{
		return std::vector<float>(embeddingDimension, 0.0f);
	}

	return embeddings[tokenId];
}
