#Wikivaalit

This package provides tools to pull Finnish Wikipedia articles, and YLE Kuntavaalit 2017
data from the web, perform an LDA model on that corpus, and look at the closest matching
wikipedia pages for all candidates and parties. The results themselves are not that interesting -
this project was more an exercise of learning NLP techniques and Docker orchestration.

Included Docker services:

###data (runtime approx. 6 minutes)
Pulls the Kuntavaalit and Finnish Wikipedia datasets, and does a little preprocessing on the 
election data.

###db (runtime approx. 5 minutes)
Starts a MySQL server, searches for Wikipedia categories based on keywords, and outputs 
a list of those selected categories.

###ldaModel (runtime 10-60 minutes, depends on parameters)

Takes in the raw data and selected Wikipedia categories, runs a LDA model on the corpus
(with gensim), and produces a results file.

###app

Takes in the results file, and makes a Plotly Dash dashboard app with that.

All of the containers hence depend on the results of the previous container. They are all defined
in the docker compose file, and can be run either in sequence or individually (because
you probably don't want to wait for the download of 650MB of Wikipedia articles every time).

##Getting started

Get all the files into your chosen directory with git:

`git pull tjpajala/wikivaalit`

Run the sequence of containers with:

`cd dockerize`

`docker-compose -f docker-compose-dev.yml up --build`

If you only want to run some containers, you can do it so:

`docker-compose -f docker-compose-dev.yml up --build db`

This runs that and all preceding containers. If you only want to run a certain container 
(without the previous ones), instead run with:

`docker-compose -f docker-compose-dev.yml up --build --no-deps db`
