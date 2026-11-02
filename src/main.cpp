#include "dataset.h"
#include "trainer.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

static void usage() {
    std::cout <<
      "GRU + Bahdanau translation starter\n\n"
      "Commands:\n"
      "  translate init-db <database.db>\n"
      "  translate import-tsv <database.db> <parallel.tsv>\n"
      "  translate add <database.db> <source sentence> <target sentence>\n"
      "  translate train <database.db> <model-directory> [epochs]\n"
      "  translate infer <model-directory> <source sentence>\n\n"
      "TSV format: one source sentence, TAB, one target sentence per line.\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) { usage(); return 1; }
        const std::string command = argv[1];

        if (command == "init-db") {
            TranslationDB db(argv[2]);
            db.initialize();
            std::cout << "Initialized database: " << argv[2] << '\n';
        } else if (command == "import-tsv") {
            if (argc < 4) { usage(); return 1; }
            TranslationDB db(argv[2]);
            db.initialize();
            db.import_tsv(argv[3]);
            std::cout << "Imported TSV pairs.\n";
        } else if (command == "add") {
            if (argc < 5) { usage(); return 1; }
            TranslationDB db(argv[2]);
            db.initialize();
            db.add_pair(argv[3], argv[4]);
            std::cout << "Added translation pair.\n";
        } else if (command == "train") {
            if (argc < 4) { usage(); return 1; }
            TrainConfig cfg;
            if (argc >= 5) cfg.epochs = std::stoll(argv[4]);
            train_model(argv[2], argv[3], cfg);
        } else if (command == "infer") {
            if (argc < 4) { usage(); return 1; }
            std::cout << translate_text(argv[2], argv[3]) << '\n';
        } else {
            usage();
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 2;
    }
    return 0;
}
