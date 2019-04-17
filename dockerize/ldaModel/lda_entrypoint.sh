#!/bin/sh


echo "Waiting for data to load..."

while [ ! -f /usr/src/data/candidate_answers_utf8.csv ];
do
  echo "waiting for candidate_answers_utf8.csv"
  sleep 10s
done

while [ ! -f /usr/src/data/fiwiki-20190101-pages-articles.xml.bz2 ];
do
  echo "waiting for fiwiki-20190101-pages-articles.xml.bz2"
  sleep 10s
done

while [ ! -f /usr/src/data/categories.txt ];
do
  echo "waiting for categories.txt"
  sleep 10s
done


echo "data loaded"

python /usr/src/ldaModel/election_preprocessing.py

#mkdir result
#cp /usr/src/data/candidate_answers_utf8.csv /usr/src/result/candidate_answers_utf8.csv
