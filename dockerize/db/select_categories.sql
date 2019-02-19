USE wikidump;
#CREATE TABLE links AS
#SELECT cl_from, cl_to, cl_type, page_title
#FROM categorylinks JOIN page ON cl_from = page_id
#WHERE ((cl_to LIKE "%yhteiskunta%" COLLATE latin1_general_ci OR cl_to LIKE "%politiikka%" COLLATE latin1_general_ci)
#AND cl_to NOT LIKE "%tyng%");
SELECT * FROM category
INTO OUTFILE '/usr/src/db/result/categories.txt'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';