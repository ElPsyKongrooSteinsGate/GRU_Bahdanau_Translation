#pragma once
#include <string>
#include <vector>
#include <sqlite3.h>

struct TranslationPair {
    std::string source;
    std::string target;
};

class TranslationDB {
public:
    explicit TranslationDB(const std::string& path);
    ~TranslationDB();
    TranslationDB(const TranslationDB&) = delete;
    TranslationDB& operator=(const TranslationDB&) = delete;

    void initialize();
    void add_pair(const std::string& source, const std::string& target);
    void import_tsv(const std::string& path);
    std::vector<TranslationPair> load_all() const;

private:
    sqlite3* db_ = nullptr;
};
