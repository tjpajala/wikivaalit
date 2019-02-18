#!/bin/bash
/usr/bin/mysqld_safe --skip-grant-tables &
sleep 5
#mysql -h 127.0.0.1 -P 3306 -protocol=tcp -e "CREATE DATABASE wikidump;" -u root
#mysql -h 127.0.0.1 -P 3306 -protocol=tcp -e "USE wikidump;" -u root
#mysql -h 127.0.0.1 -P 3306 -protocol=tcp -e "SHOW TABLES;" -u root

#mysql -h 0.0.0.0 -P 3306 -protocol=tcp -e "CREATE DATABASE wikidump;" -u root
#mysql -h 0.0.0.0 -P 3306 -protocol=tcp -e "USE wikidump;" -u root
#mysql -h 0.0.0.0 -P 3306 -protocol=tcp -e "SHOW TABLES;" -u root