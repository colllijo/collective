#include <iostream>

#include "gpt/GPT.hpp"
#include "tokenization/BPETokenizer.hpp"

#define VOCABULARY_SIZE 1000
#define MAX_SEQUENCE_LENGTH 50
#define EMBEDDING_DIMENSION 128
#define dFF 256
#define LAYERS 6

int main(int argc, char* argv[])
{
	std::string corpus =
		"Once upon a time, in a land far, far away, there lived a king and a queen.\n"
		"The king was wise and just, and the queen was kind and beautiful.\n"
		"They had a daughter who was as brave as she was clever.\n"
		"One day, a dragon appeared in the kingdom, causing fear and chaos.\n"
		"The brave princess decided to confront the dragon and save her people.\n"
		"With courage and wit, she outsmarted the dragon and restored peace to the land.\n"
		"The kingdom rejoiced, and the princess became a hero.\n"
		"\n"
		"Years passed, and the kingdom flourished under the rule of the wise king and queen.\n"
		"The princess grew into a strong and compassionate leader, loved by all.\n"
		"She traveled to distant lands, forging alliances and bringing prosperity to her people.\n"
		"In her travels, she encountered many challenges, but her bravery and intelligence always prevailed.\n"
		"She faced treacherous mountains, crossed vast deserts, and sailed across stormy seas.\n"
		"Everywhere she went, she spread kindness and wisdom, earning the respect of all she met.\n"
		"\n"
		"One fateful day, a dark sorcerer threatened the peace of the kingdom.\n"
		"The sorcerer cast a spell that brought darkness and despair to the land.\n"
		"The princess, undeterred by the sorcerer's power, set out on a quest to break the spell.\n"
		"With the help of her loyal friends and allies, she ventured into the heart of the sorcerer's lair.\n"
		"Through trials and tribulations, she fought bravely and used her wit to overcome the sorcerer's traps.\n"
		"In the end, she confronted the sorcerer and, with a final act of courage, broke the spell.\n"
		"The kingdom was saved once again, and the princess's legend grew even greater."
		"\n"
		"The princess knew that learning was the key to her success and the prosperity of her kingdom.\n"
		"She dedicated herself to learning new skills and knowledge every day.\n"
		"Learning from scholars, warriors, and travelers, she expanded her understanding of the world.\n"
		"Her passion for learning inspired her people to value education and wisdom.\n"
		"Through learning, she discovered innovative solutions to the kingdom's challenges.\n"
		"Learning became a cornerstone of the kingdom's culture, leading to a golden age of enlightenment and progress.\n"
		"The princess's commitment to learning ensured that her legacy would endure for generations to come.";

	BPETokenizer tokenizer;
	GPT gpt(VOCABULARY_SIZE, MAX_SEQUENCE_LENGTH, EMBEDDING_DIMENSION, LAYERS, dFF);

	tokenizer.train(corpus, VOCABULARY_SIZE);
	// tokenizer.saveVocabulary("vocabulary.dat");

	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " <input>\n";
		return 1;
	}

	std::string input = argv[1];
	auto tokenIds = tokenizer.encode(input);

	for (const auto& token : tokenIds) std::cout << token << ": " << tokenizer.decode({token}) << "\n";
	// for (const auto& token : tokenIds)
	// 	std::cout << token << " ";
	// std::cout << "\n";
	// for (const auto& token : tokenIds)
	// 	std::cout << tokenizer.decode({token}) << " ";
	// std::cout << "\n";

	// int nextToken = gpt.predictNextToken(tokenIds);
	//
	// for (const auto& token : tokenIds)
	// {
	// 	std::cout << token << " ";
	// }
	// std::cout << std::endl;
	// std::cout << nextToken << std::endl;
	//
	// std::cout << tokenizer.decode(tokenIds) << std::endl;
	//
	// tokenIds.push_back(nextToken);
	//
	// std::cout << tokenizer.decode(tokenIds) << std::endl;

	return 0;
}
