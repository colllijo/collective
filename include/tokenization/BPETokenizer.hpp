#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

class BPETokenizer
{
public:
	BPETokenizer();
	~BPETokenizer();

	std::vector<int> encode(const std::string& text) const;
	std::string decode(const std::vector<int>& tokenIds) const;

	void train(const std::string& corpus, int vocabularySize = 50, const std::unordered_set<std::string>& specialTokens = {});

	void saveVocabulary(const std::string& filename) const;
	void loadVocabulary(const std::string& filename);

private:
	int nextTokenId;
	std::unordered_map<std::string, int> tokenToIdMap;
	std::unordered_map<int, std::string> idToTokenMap;

	std::map<std::pair<int, int>, int> merges;

	std::vector<int> applyBPE(const std::string& text) const;

	std::optional<std::pair<int, int>> mostFrequentPair(const std::vector<int>& tokens) const;
	std::vector<int> mergeTokens(const std::vector<int>& tokens, const std::pair<int, int>& pair, int mergeToken) const;

	void buildBasicVocabulary(const std::set<std::string>& tokens);
	void buildMergeVocabulary();

	std::vector<std::string> splitText(const std::string& text) const;
	std::vector<int> tokenizeText(const std::vector<std::string>& text) const;
};
