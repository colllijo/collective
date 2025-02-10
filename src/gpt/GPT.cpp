#include "gpt/GPT.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>

#include "embedding/PositionalEncoding.hpp"
#include "matrix/Matrix.hpp"

GPT::GPT(int vocabularySize, int maxSequenceLength, int dModel, int numLayers, int dFF)
	: numLayers(numLayers), maxSequenceLength(maxSequenceLength), dModel(dModel), embedding(vocabularySize, dModel), positionalEncoding(maxSequenceLength, dModel)
{
	for (int i = 0; i < numLayers; i++)
	{
		layers.emplace_back(TransformerBlock(dModel, dFF));
	}
}

GPT::~GPT() = default;

std::vector<std::vector<float>> GPT::forward(const std::vector<int>& tokenIds) const
{
	auto embeddings = applyPositionalEncoding(tokenIds, embedding, positionalEncoding);

	for (const auto& layer : layers)
	{
		embeddings = layer.forward(embeddings);
	}

	return embeddings;
}

int GPT::predictNextToken(const std::vector<int>& tokenIds, float temperature) const
{
	std::vector<std::vector<float>> output = forward(tokenIds);

	std::vector<float> logbits = output.back();
	std::vector<float> probabilities = softmax(logbits, temperature);

	// Debugging: Display probabilities for all tokens
	std::cout << "Token Probabilities (T = " << temperature << "):\n";
    for (size_t i = 0; i < probabilities.size(); i++) {
        std::cout << "Token " << i << ": " << probabilities[i] << std::endl;
    }

	std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

	return distribution(gen);
}
