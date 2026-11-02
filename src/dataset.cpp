#include "dataset.h"
#include <fstream>
#include <stdexcept>

static void check_sql(int rc, sqlite3* db, const char* action) {
    if (rc != SQLITE_OK)
        throw std::runtime_error(std::string(action) + ": " + sqlite3_errmsg(db));
}

TranslationDB::TranslationDB(const std::string& path) {
    const int rc = sqlite3_open(path.c_str(), &db_);
    if (rc != SQLITE_OK) {
        std::string msg = db_ ? sqlite3_errmsg(db_) : "could not allocate SQLite handle";
        if (db_) sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("SQLite open failed: " + msg);
    }
}
TranslationDB::~TranslationDB() { if (db_) sqlite3_close(db_); }

void TranslationDB::initialize() {
    const char* sql =
        "CREATE TABLE IF NOT EXISTS translations ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "source TEXT NOT NULL,"
        "target TEXT NOT NULL);";
    char* err = nullptr;
    int rc = sqlite3_exec(db_, sql, nullptr, nullptr, &err);
    if (rc != SQLITE_OK) {
        std::string msg = err ? err : sqlite3_errmsg(db_);
        sqlite3_free(err);
        throw std::runtime_error("SQLite schema error: " + msg);
    }
}

void TranslationDB::add_pair(const std::string& source, const std::string& target) {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "INSERT INTO translations(source,target) VALUES(?,?);";
    check_sql(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr), db_, "prepare insert");
    sqlite3_bind_text(stmt, 1, source.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, target.c_str(), -1, SQLITE_TRANSIENT);
    int rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) throw std::runtime_error("SQLite insert failed: " + std::string(sqlite3_errmsg(db_)));
}

void TranslationDB::import_tsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open TSV file: " + path);
    std::string line;
    std::size_t count = 0;
    while (std::getline(in, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        auto source = line.substr(0, tab);
        auto target = line.substr(tab + 1);
        if (!source.empty() && !target.empty()) {
            add_pair(source, target);
            ++count;
        }
    }
    if (count == 0) throw std::runtime_error("No tab-separated source<TAB>target rows found.");
}

std::vector<TranslationPair> TranslationDB::load_all() const {
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT source,target FROM translations ORDER BY id;";
    check_sql(sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr), db_, "prepare select");
    std::vector<TranslationPair> rows;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const auto* s = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const auto* t = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        rows.push_back({s ? s : "", t ? t : ""});
    }
    sqlite3_finalize(stmt);
    return rows;
}
