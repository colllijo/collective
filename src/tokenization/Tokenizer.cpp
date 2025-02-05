#include "tokenization/Tokenizer.hpp"

#include <functional>
#include <regex>

Tokenizer::Tokenizer() : vocabulary{}, merges{}, nextTokenId(0), tokenIdMap{}, idTokenMap{} {}
Tokenizer::~Tokenizer() = default;

void Tokenizer::train(const std::string& corpus, int vocabularySize)
{
	std::vector<std::string> tokens = tokenizeText(corpus);
	int mergeCount = vocabularySize - vocabulary.size();

	vocabulary.insert(tokens.begin(), tokens.end());

	for (int i = 0; i < mergeCount; ++i)
	{
		auto stats = getStats(tokens);
		if (stats.empty()) break;

		auto mostFrequentPair = std::max_element(stats.begin(), stats.end(), std::greater<>())->first;
		auto mergeToken = mostFrequentPair.first + mostFrequentPair.second;

		tokens = mergeTokens(tokens, mostFrequentPair, mergeToken);

		merges[mostFrequentPair] = mergeToken;
		vocabulary.insert(mergeToken);
	}

	buildVocabulary();
}

std::vector<int> Tokenizer::apply(const std::string& text) const
{
	std::vector<std::string> tokens = tokenizeText(text);

	for (const auto& [pair, mergeToken] : merges)
	{
		tokens = mergeTokens(tokens, pair, mergeToken);
	}

	return tokensToIds(tokens);
}

std::string Tokenizer::decodeTokens(const std::vector<int>& tokenIds) const
{
	std::string text;
	for (const auto& tokenId : tokenIds)
	{
		if (idTokenMap.find(tokenId) != idTokenMap.end())
		{
			text += idTokenMap.at(tokenId) + " ";
		}
	}

	return text;
}

std::vector<std::string> Tokenizer::tokenizeText(const std::string& text) const
{
	std::regex tokenRegex(R"(\w+|[^\w\s]|\s)");
	std::sregex_token_iterator iter(text.begin(), text.end(), tokenRegex);
	std::sregex_token_iterator end;
	std::vector<std::string> tokens(iter, end);

	return tokens;
}

std::map<std::pair<std::string, std::string>, int> Tokenizer::getStats(const std::vector<std::string>& tokens) const
{
	std::map<std::pair<std::string, std::string>, int> stats{};

	for (size_t i = 0; i < tokens.size() - 1; ++i)
	{
		stats[std::make_pair(tokens[i], tokens[i + 1])]++;
	}

	return stats;
}

std::vector<std::string> Tokenizer::mergeTokens(const std::vector<std::string>& tokens, const std::pair<std::string, std::string>& pair, const std::string& mergeToken) const
{
	std::vector<std::string> mergedTokens;
	size_t i = 0;
	while (i < tokens.size())
	{
		if (i < tokens.size() - 1 && tokens[i] == pair.first && tokens[i + 1] == pair.second)
		{
			mergedTokens.push_back(mergeToken);
			i += 2;
		}
		else
		{
			mergedTokens.push_back(tokens[i]);
			i++;
		}
	}
	return mergedTokens;
}

void Tokenizer::buildVocabulary()
{
	// Special tokens
	tokenIdMap["<unk>"] = nextTokenId++;
	tokenIdMap["<bos>"] = nextTokenId++;
	tokenIdMap["<eos>"] = nextTokenId++;

	for (const auto& token : vocabulary)
	{
		if (tokenIdMap.find(token) == tokenIdMap.end())
		{
			tokenIdMap[token] = nextTokenId;
			idTokenMap[nextTokenId] = token;
			nextTokenId++;
		}
	}
}

std::vector<int> Tokenizer::tokensToIds(const std::vector<std::string>& tokens) const
{
	std::vector<int> tokenIds;
	for (const auto& token : tokens)
	{
		if (tokenIdMap.find(token) == tokenIdMap.end())
		{
			tokenIds.push_back(tokenIdMap.at("<unk>"));
		}
		else
		{
			tokenIds.push_back(tokenIdMap.at(token));
		}
	}

	return tokenIds;
}
