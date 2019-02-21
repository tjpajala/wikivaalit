#!/bin/sh

echo "Waiting for mysql..."

while ! nc -z db 3306; do
  sleep 1s
done

echo "mysql started"



echo "Waiting for data to load..."

while [ ! -f /usr/src/data/cand_df.pkl ];
do
  sleep 10s
done

echo "data loaded"
python app.py run -h 0.0.0.0
