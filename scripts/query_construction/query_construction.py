import pandas as pd

import numpy as np
from subgraph_gen.subgraph_gen import *
from sentence_processing.sentence_processing import *
from relation_linking.relation_linking import *

from SPARQLWrapper import SPARQLWrapper, JSON
sparql = SPARQLWrapper("http://localhost:7200/repositories/wikidata_bio_2")


def create_final_query(focus, target, relation, direction):
    # create primary clause involving focus term
    primary_clause = create_primary_clause(focus, relation, direction)
    
    # create secondary clause involving target term
    secondary_clause = create_secondary_clause(target)
    
    # combine primary and secondary clause
    header = """
    PREFIX wd: <https://www.wikidata.org/wiki/>
    PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
    """
    
    total_query = header + "select ?item where \n" + "{\n" + primary_clause + "\n" + secondary_clause + "\n}"
    return total_query
def create_secondary_clause(target):
    # if target is None return nothing, no secondary clause
    if target is None:
        return ""
    # if target is protein class, add 361
    elif is_protein_family(target)[0]:
        return "?item (wdt:P361|wdt:P279|wdt:P31)* wd:{} .".format(target)
    # if target is a taxon, add 171
    elif is_taxon(target):
        return "?item wdt:P171* wd:{} .".format(target)
    # else use only 31|279
    else:
        return "?item (wdt:P279|wdt:P31)* wd:{} .".format(target)
    
def create_primary_clause(focus, relation, direction):
    # if relation is subclass/subtype, then make appropriate relation
    if "P31" in relation or "P279" in relation:
        relation = "wdt:P31|wdt:P279"
    # else make the complete relation URI
    else:
        relation = "wdt:{}".format(relation)
    # if direction is forward, return appropriate query
    if direction == "forward":
        return "wd:{} {} ?item .".format(focus, relation)
    # if direction is backward, return appropriate query
    elif direction == "backward":
        return "?item {} wd:{} .".format(relation, focus)
    
def run_final_query(query):
    sparql = SPARQLWrapper("http://localhost:7200/repositories/wikidata_bio_2")

    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    ndf = pd.DataFrame(results['results']['bindings'])
#     print(ndf)
    if len(ndf) == 0:
        return []
    else:
        ndfn = ndf.copy()
        try:
            ndf = ndf.applymap(lambda x: x['value'] if x["type"] == "uri" else np.nan)
            ndf["item"] = ndf["item"].apply(lambda x: "wd:" + x.split("/")[-1])
            ret = list(set(ndf["item"].values))
        except:
            ndf = ndfn.applymap(lambda x: x['value'] if x["type"] == "literal" else np.nan)
            ret = list(set(ndf["item"].values))
        return ret
        
def make_gold_query(query):
    header = """
    PREFIX wd: <https://www.wikidata.org/wiki/>
    PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
    """
    
    total_query = header + "\n select ?item where \n" + "{"
    for clause in query:
        total_query = total_query + "\n" + clause + " ."
    total_query = total_query + "\n}"
    return total_query