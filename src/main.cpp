#include <iostream>

#include "tokenization/tokenizer.hpp"

int main()
{
	Tokenizer tokenizer;

	std::string corpus =
		"The quick, brown fox jumps over the lazy dog. "
		"The slow-moving river flows through the mountains.";

	tokenizer.train(corpus, 50);
	auto tokens = tokenizer.apply("The quick brown fox!");

	for (const auto& token : tokens)
	{
		std::cout << token << " ";
	}
	std::cout << std::endl;

	return 0;
}
