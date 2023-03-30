# Bio-SQuAREL
### Biomedical Simple Question Answering using Relation and Entity Linking

This contains code for the bio-squarel methodology for answering natural language questions using Wikidata. 

This codebase has been tested to run on Linux Ubuntu systems, and multiple tweaks need to be made to run on Windows or Mac systems.

In order to run the methodology of this repo, the following steps need to be followed:

## 0. Set up the environment:

Create a new environment using the requirements.txt file. The recommended way is to use conda to manage and create a new environment:<br>
    `conda create --name <ENV_NAME> conda create -n bst python=3.9

    conda activate <ENV_NAME>

    conda install jupyter

    pip install -r requirements.txt
    
    python -m spacy download en_core_web_trf`


## 1. Get the biomedical sub Knowledge Graph of Wikidata along with associated files from here: 

1. Download the data from here: https://figshare.com/articles/dataset/Wikidata_Biomedical_SubGraph/22347790
2. Extract the data if needed. 

The downloaded files are essentially the KG triples needed for the construction of the knowledge graph.


## 2. Set-up the relation and entity label search method:
1. Download the entity labels from the following repo: https://figshare.com/articles/dataset/Wikidata_Biomedical_Labels/22360486
2. Download and install elasticsearch (version 7.6.2) and start elasticsearch.
    1. Download elasticsearch from here: https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz
        `wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.6.2-linux-x86_64.tar.gz`
    2. Install:
        `tar -xzf elasticsearch-7.6.2-linux-x86_64.tar.gz`
    3. Start elasticsearch:
        `elasticsearch-7.6.2/bin/elasticsearch`    
3. Go to the search_index_creation folder.
4. Follow the notebook -- create_entity_index.ipynb to create the elasticsearch indexes for entity labels.  
5. Follow the notebook -- create_relation_index.ipynb to create the elasticsearch indexes for relation labels.


These search indexes can be used for efficiently searching for entities' and predicates' labels using their surface forms. 

## 3. Create the Wikidata biomedical KG using GraphDB backend.

Easiest Way to do this is using the GraphDB docker image: 

1. Install docker in your system. Follow these instructions: https://docs.docker.com/engine/install/ubuntu/

2. Start graphdb docker container using the following commands:<br>
    
    `$docker pull dhlabbasel/graphdb-free`

    `$docker run -it -p 7200:7200 --name graphdb -v <FOLDER CONTAINING KG TRIPLE FILES>:/opt/graphdb/home -v <PATH TO FOLDER CONTAINING KG TRIPLE FILES>/graphdb/:/graphdb/data/ -t dhlabbasel/graphdb-free` 

    Do not close the terminal on which this is running. It will stop the container.

3. Go to localhost:7200 in your browser. If running on a remote server, put the IP of the remote server in place of "localhost". It will open up GraphDB Workbench (Web Based GUI). There, set up a GraphDB repository called "wikidata_life_sciences"

    For more information refer the GraphDB documentation: 
    
4. Upload the data into the wikidata_life_sciences repository using either GUI or command line.
    Uploading using GUI is quite simple. Refer to GraphDB documentation for any queries. 
    
    For uploading using command line, follow this process:
    1. Go to the terminal tab in which the GraphDB container was started and close the contained using Ctrl+C
    2. Again start the docker container using:<br> 
        `$docker start graphdb`
    3. Go to the docker container shell using:<br>
        `$docker exec –it graphdb bash` 
    4. Go to the root of the container:<br>
        `$cd /`
    5. Run the command to load the data:<br>
        `$graphdb/bin/loadrdf -f -i wikidata_life_sciences -m parallel opt/graphdb/home/<FOLDER CONTAINING KG TRIPLE FILES>/*` 
    
    Using the above procedure, the data will be loaded into a single KG (repository) called wikidata_life_sciences.
    Users can query the KG using either the GUI (GraphDB Workbench) or API.

## 4. Create the search indexes required for Entity and Relation Linking

A search engine using the entity and relation labels of Wikidata needs to be created. 
    
First, the files for creating these indexes need to be downloaded from here: 
The extracted *.rdf files should be placed in the elasticsearch_indexes folder.
Next, the search index should be created using them:

1. For the entity label search index, follow the notebook elasticsearch_indexes/entity_to_elasticsearch_index.ipynb

2. For the relation or predicate label search index, follow the notebook elasticsearch_indexes/relation_to_elasticsearch_index.ipynb

## 5. All the building blocks are set up. Now evaluation can be carried out.

1. Go to the codebase folder.
2. Run the evaluate.ipynb file. This contains the codes for evaluating the method on all the questions from the test dataset.

