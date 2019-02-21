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
if [ ! -f /usr/src/data/candidate_answers_utf8.csv ];
then
  echo "candidate_answers_utf8.csv is missing from ./data, please run data container"
  exit 1
fi

if [ ! -f /usr/src/data/fiwiki-20190101-pages-articles.xml.bz2 ];
then
  echo "fiwiki-20190101-pages-articles.xml.bz2 is missing from ./data, please run data container"
  exit 2
fi
if [ ! -f /usr/src/data/categories.txt ];
then
  echo "categories.txt is missing from ./data, please run db container"
  exit 3
fi
echo "data ok"

python /usr/src/ldaModel/election_preprocessing.py

#mkdir result
#cp /usr/src/data/candidate_answers_utf8.csv /usr/src/result/candidate_answers_utf8.csv
