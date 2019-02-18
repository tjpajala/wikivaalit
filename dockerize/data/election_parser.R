#! /usr/bin/Rscript
f <- read.csv2("/usr/src/data/candidate_answer_data_kokomaa11042017.csv")
readr::write_csv2(f, "/usr/src/data/candidate_answers_utf8.csv")