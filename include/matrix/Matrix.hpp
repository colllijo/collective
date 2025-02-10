#pragma once

#include <vector>

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B);

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& A);

std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& x);

std::vector<float> softmax(const std::vector<float>& logits);

std::vector<float> softmax(const std::vector<float>& logits, float temperature);

std::vector<std::vector<float>> selfAttention(const std::vector<std::vector<float>>& queries, const std::vector<std::vector<float>>& keys,
											  const std::vector<std::vector<float>>& values, const std::vector<float>& bq, const std::vector<float>& bk,
											  const std::vector<float>& bv);

std::vector<float> feedForward(const std::vector<float>& x, const std::vector<std::vector<float>>& W1, const std::vector<float>& b1, const std::vector<std::vector<float>>& W2,
							   const std::vector<float>& b2);
