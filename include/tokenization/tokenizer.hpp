#pragma once

#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class Tokenizer
{
public:
	Tokenizer();
	~Tokenizer();

	void train(const std::string& corpus, int vocabularySize = 50);
	std::vector<std::string> apply(const std::string& text) const;

private:
	std::unordered_set<std::string> vocabulary;
	std::map<std::pair<std::string, std::string>, std::string> merges;

	std::vector<std::string> tokenizeText(const std::string& text) const;
	std::map<std::pair<std::string, std::string>, int> getStats(const std::vector<std::string>& tokens) const;
	std::vector<std::string> mergeTokens(const std::vector<std::string>& tokens, const std::pair<std::string, std::string>& pair, const std::string& mergeToken) const;
};
