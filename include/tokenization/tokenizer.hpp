#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class Tokenizer
{
public:
	Tokenizer();
	~Tokenizer();

	void train(const std::string& corpus, int vocabularySize = 50);
	std::vector<int> apply(const std::string& text) const;

	std::string decodeTokens(const std::vector<int>& tokenIds) const;
private:
	std::unordered_set<std::string> vocabulary;
	std::map<std::pair<std::string, std::string>, std::string> merges;

	int nextTokenId;
	std::unordered_map<std::string, int> tokenIdMap;
	std::unordered_map<int, std::string> idTokenMap;

	std::vector<std::string> tokenizeText(const std::string& text) const;
	std::map<std::pair<std::string, std::string>, int> getStats(const std::vector<std::string>& tokens) const;
	std::vector<std::string> mergeTokens(const std::vector<std::string>& tokens, const std::pair<std::string, std::string>& pair, const std::string& mergeToken) const;

	void buildVocabulary();

	std::vector<int> tokensToIds(const std::vector<std::string>& tokens) const;
};
