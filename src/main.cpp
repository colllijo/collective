#include "embedding/EmbeddingLayer.hpp"
#include "embedding/PositionalEncoding.hpp"
#include "tokenization/Tokenizer.hpp"

#define VOCABULARY_SIZE 50
#define EMBEDDING_DIMENSION 512
#define MAX_SEQUENCE_LENGTH 100

int main()
{
	std::string corpus =
		"The quick, brown fox jumps over the lazy dog."
		"The slow-moving river flows through the mountains.";

	Tokenizer tokenizer;
	EmbeddingLayer embeddingLayer(VOCABULARY_SIZE, EMBEDDING_DIMENSION);
	PositionalEncoding positionalEncoding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIMENSION);

	tokenizer.train(corpus, VOCABULARY_SIZE);

	auto tokenIds = tokenizer.apply("The quick brown fox.");
	auto embeddings = applyPositionalEncoding(tokenIds, embeddingLayer, positionalEncoding);

	return 0;
}
