#include "tokenization/BPETokenizer.hpp"

#include <codecvt>
#include <deque>
#include <fstream>
#include <iostream>
#include <locale>

void printVocabulary(std::unordered_map<int, std::string> idToTokenMap)
{
	for (size_t i = 0; i < idToTokenMap.size(); ++i)
		std::cout << i << ": " << idToTokenMap.at(i) << "\n";
	std::cout << std::endl;
}

BPETokenizer::BPETokenizer() : nextTokenId(0), tokenToIdMap{}, idToTokenMap{}, merges{} {}
BPETokenizer::~BPETokenizer() = default;

std::vector<int> BPETokenizer::encode(const std::string& text) const
{
	std::vector<std::string> tokens;

	std::string current;
	for (char c : text)
	{
		if (c == ' ')
		{
			if (!current.empty())
			{
				tokens.push_back(current);
				current.clear();
			}
			tokens.push_back(" ");
		}
		else
		{
			current += c;
		}
	}

	if (!current.empty()) tokens.push_back(current);

	std::vector<int> tokenIds;
	for (const auto& token : tokens)
	{
		if (tokenToIdMap.find(token) != tokenToIdMap.end())
		{
			tokenIds.push_back(tokenToIdMap.at(token));
		}
		else
		{
			auto subTokens = applyBPE(token);
			tokenIds.insert(tokenIds.end(), subTokens.begin(), subTokens.end());
		}
	}

	return tokenIds;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const
{
	std::string text;

	for (const auto& token : tokens)
	{
		text += idToTokenMap.at(token);
	}

	return text;
}

void BPETokenizer::train(const std::string& corpus, int vocabularySize, const std::unordered_set<std::string>& specialTokens)
{
	// Build basic vocabulary from ASCII characters and corpus
	std::set<std::string> vocabulary{};

	for (int i = 0; i < 128; ++i) vocabulary.insert(std::string(1, static_cast<char>(i)));

	auto characters = splitText(corpus);
	vocabulary.insert(characters.begin(), characters.end());
	vocabulary.insert(specialTokens.begin(), specialTokens.end());

	buildBasicVocabulary(vocabulary);

	// Map the corpus to tokens
	auto tokens = tokenizeText(characters);

	// Byte pair encoding (BPE)
	for (size_t i = 0; i < vocabularySize - tokenToIdMap.size(); ++i)
	{
		auto pair = mostFrequentPair(tokens);
		if (!pair.has_value()) break;

		tokens = mergeTokens(tokens, pair.value(), nextTokenId);

		// Update merges
		merges[pair.value()] = nextTokenId;

		// Update vocabulary
		std::string mergeToken = idToTokenMap.at(pair.value().first) + idToTokenMap.at(pair.value().second);
		tokenToIdMap[mergeToken] = nextTokenId;
		idToTokenMap[nextTokenId] = mergeToken;

		nextTokenId++;
	}

	printVocabulary(idToTokenMap);
}

void BPETokenizer::saveVocabulary(const std::string& filename) const
{
	std::ofstream file(filename, std::ios::binary);

	if (file.is_open())
	{
		size_t size = tokenToIdMap.size();
		file.write(reinterpret_cast<const char*>(&size), sizeof(size));

		// Write the vocabulary to the file
		for (const auto& [token, id] : tokenToIdMap)
		{
			size_t tokenSize = token.size();
			file.write(reinterpret_cast<const char*>(&tokenSize), sizeof(tokenSize));
			file.write(token.c_str(), tokenSize);
			file.write(reinterpret_cast<const char*>(&id), sizeof(id));
		}

		size = merges.size();
		file.write(reinterpret_cast<const char*>(&size), sizeof(size));

		// Write the merges to the file
		for (const auto& [pair, mergeToken] : merges)
		{
			file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
			file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
			file.write(reinterpret_cast<const char*>(&mergeToken), sizeof(mergeToken));
		}

		file.close();
	}
}

void BPETokenizer::loadVocabulary(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);

	if (file.is_open())
	{
		size_t size;
		file.read(reinterpret_cast<char*>(&size), sizeof(size));

		for (size_t i = 0; i < size; ++i)
		{
			size_t tokenSize;
			file.read(reinterpret_cast<char*>(&tokenSize), sizeof(tokenSize));

			std::string token(tokenSize, '\0');
			file.read(token.data(), tokenSize);

			int id;
			file.read(reinterpret_cast<char*>(&id), sizeof(id));

			tokenToIdMap[token] = id;
			idToTokenMap[id] = token;
		}

		file.read(reinterpret_cast<char*>(&size), sizeof(size));

		for (size_t i = 0; i < size; ++i)
		{
			int left, right, mergeToken;

			file.read(reinterpret_cast<char*>(&left), sizeof(left));
			file.read(reinterpret_cast<char*>(&right), sizeof(right));
			file.read(reinterpret_cast<char*>(&mergeToken), sizeof(mergeToken));

			merges[std::make_pair(left, right)] = mergeToken;
		}

		file.close();
	}
}

std::vector<int> BPETokenizer::applyBPE(const std::string& text) const
{
	auto tokens = tokenizeText(splitText(text));

	bool canMerge = true;
	while (canMerge)
	{
		canMerge = false;
		std::vector<int> mergedTokens;

		size_t i = 0;
		while (i < tokens.size() - 1)
		{
			auto pair = std::make_pair(tokens[i], tokens[i + 1]);

			if (merges.find(pair) != merges.end())
			{
				mergedTokens.push_back(merges.at(pair));
				i += 2;
				canMerge = true;
			}
			else
			{
				mergedTokens.push_back(tokens[i]);
				i++;
			}
		}
		if (i < tokens.size()) mergedTokens.push_back(tokens[i]);

		tokens = mergedTokens;
	}

	return tokens;
}

std::vector<int> BPETokenizer::tokenizeText(const std::vector<std::string>& text) const
{
	std::vector<int> tokens{};

	for (const auto& c : text)
	{
		tokens.push_back(tokenToIdMap.at(c));
	}

	return tokens;
}

std::optional<std::pair<int, int>> BPETokenizer::mostFrequentPair(const std::vector<int>& tokens) const
{
	std::map<std::pair<int, int>, int> pairFrequency{};

	for (size_t i = 0; i < tokens.size() - 1; ++i) pairFrequency[std::make_pair(tokens[i], tokens[i + 1])]++;

	int maxFrequency = 0;
	std::pair<int, int> mostFrequentPair;
	for (const auto& [pair, frequency] : pairFrequency)
	{
		if (frequency > maxFrequency)
		{
			maxFrequency = frequency;
			mostFrequentPair = pair;
		}
	}

	if (maxFrequency == 1) return std::nullopt;
	return mostFrequentPair;
}

std::vector<int> BPETokenizer::mergeTokens(const std::vector<int>& tokens, const std::pair<int, int>& pair, int mergeToken) const
{
	std::deque<int> queue(tokens.begin(), tokens.end());
	std::vector<int> mergedTokens;

	while (!queue.empty())
	{
		int current = queue.front();
		queue.pop_front();

		if (!queue.empty() && (current == pair.first && queue.front() == pair.second))
		{
			queue.pop_front();
			mergedTokens.push_back(mergeToken);
		}
		else
		{
			mergedTokens.push_back(current);
		}
	}

	return mergedTokens;
}

void BPETokenizer::buildBasicVocabulary(const std::set<std::string>& tokens)
{
	for (const auto& token : tokens)
	{
		tokenToIdMap[token] = nextTokenId;
		idToTokenMap[nextTokenId] = token;
		nextTokenId++;
	}
}

std::vector<std::string> BPETokenizer::splitText(const std::string& text) const
{
	std::vector<std::string> tokens;

	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
	std::wstring wideText = converter.from_bytes(text);

	for (wchar_t wc : wideText) tokens.push_back(converter.to_bytes(wc));

	return tokens;
}
