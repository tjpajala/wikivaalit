from sqlalchemy import create_engine
import pandas as pd
import pymysql
from pprint import pprint



#set display width
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option("display.max_columns", 10)


#run sql commands on command line
# mysql -u root wikidump < ~/PycharmProjects/wikivaalit/dockerize/db/fiwiki-20190101-categorylinks.sql
# mysql -u root wikidump < ~/PycharmProjects/wikivaalit/dockerize/db/fiwiki-20190101-category.sql -p rootpass
# mysql -u root wikidump < ~/PycharmProjects/wikivaalit/dockerize/db/fiwiki-20190101-page.sql -p rootpass


#try pymysql
engine=create_engine('mysql+pymysql://root:root@localhost')
# Connect to the database
engine.connect()


existing_databases = engine.execute("SHOW DATABASES;")
existing_databases = [d[0] for d in existing_databases]
existing_databases
database = 'wikidump'
if database not in existing_databases:
    engine.execute("CREATE DATABASE {0}".format(database))
    print("Created database {0}".format(database))

engine.execute("USE wikidump") # select new db
show_tables = engine.execute("SHOW TABLES;")
show_tables = [d[0] for d in show_tables]
show_tables

categories = pd.read_sql("SELECT * FROM category",engine)
#drop categories without members
categories = categories[categories.cat_pages>0]
links = pd.read_sql("SELECT cl_from, cl_to, cl_type, page_title FROM categorylinks JOIN page ON cl_from = page_id",engine)
links["category"]=links.cl_to.str.decode('UTF-8')
links = links[links.category.str.contains('yhteiskunta|politiikka', case=False, regex=True)]
links = links[~links.category.str.contains('tyngät', case=False, regex=False)]
#save pages to retain in pickle file
links.to_pickle("wiki_cats.pkl")
