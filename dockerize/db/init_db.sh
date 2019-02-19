#!/bin/bash
/usr/bin/mysqld_safe --verbose --port=3306&
#mysql start --verbose &&
sleep 5

echo "Waiting for mysql..."

#mysql -h 127.0.0.1 -P 3306 -protocol=tcp -e "SHOW databases;" -u root -p root


while ! nc -z -v -w30 localhost 3306; do
  sleep 5s
done

echo "mysql started"

#python3 wiki_sql_generator.py

#mysql -h 127.0.0.1 -P 3306 -protocol=tcp -e "CREATE DATABASE wikidump;" -u root
#mysql -h 127.0.0.1 -P 3306 -protocol=tcp -e "USE wikidump;" -u root
#mysql -h 127.0.0.1 -P 3306 -protocol=tcp -e "SHOW DATABASES;" -uroot -proot

#mysql -h 0.0.0.0 -P 3306 -protocol=tcp -e "CREATE DATABASE wikidump;" -u root
#mysql -h 0.0.0.0 -P 3306 -protocol=tcp -e "USE wikidump;" -u root
#mysql -h 0.0.0.0 -P 3306 -protocol=tcp -e "SHOW TABLES;" -u root