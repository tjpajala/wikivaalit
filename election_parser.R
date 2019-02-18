#! /usr/bin/Rscript
f <- read.csv2("./data/candidate_answer_data_kokomaa11042017.csv")
readr::write_csv2(f, "./data/candidate_answers_utf8.csv")
#f2<-read.csv2("./data/candidate_answers_utf8.csv")
#head(f2[,1:5])
