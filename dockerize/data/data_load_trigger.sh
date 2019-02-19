#!/bin/sh

echo "Waiting for data to load..."

python /usr/src/data/get_data_files.py

ls

while [ ! -f /usr/src/data/candidate_answer_data_kokomaa11042017.csv ];
do
  sleep 30s
done

echo "data loaded"

Rscript election_parser.R
mkdir result
cp /usr/src/data/candidate_answers_utf8.csv /usr/src/result/candidate_answers_utf8.csv
cp /usr/src/data/fiwiki-20190101-pages-articles.xml.bz2 /usr/src/result/fiwiki-20190101-pages-articles.xml.bz2
