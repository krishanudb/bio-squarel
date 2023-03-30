import pandas as pd
import numpy as np
from relation_linking.ml import *
from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_processing.sentence_processing import find_class_of
from Elastic.searchIndex import *


relation_index = "wikidata_bio_subset_3_relation_updated_index"
sparql = SPARQLWrapper("http://localhost:7200/repositories/wikidata_bio_2")


def get_domain_range_df(domains_file, ranges_file):
    
    domain_df, range_df = pd.read_csv(domains_file, index_col = 0), pd.read_csv(ranges_file, index_col = 0)
    domain_df = domain_df.applymap(lambda x: 1.0 if x > 1.0 else 0.001)
    range_df = range_df.applymap(lambda x: 1.0 if x > 1.0 else 0.001)
#     domain_df = domain_df.div(domain_df.sum(0))
#     range_df = range_df.div(range_df.sum(0))
    domain_df.index = [x.split("/")[-1] for x in domain_df.index.values]
    range_df.index = [x.split("/")[-1] for x in range_df.index.values]

    return domain_df, range_df

def calculate_probability_distribution_multi_element(elements, df):
    elements = [x for x in elements if x in df.index]
    tdf = df.loc[elements]
    tdf = tdf.div(tdf.sum(1), axis = 0)

    ptdf = np.log(tdf).sum(axis = 0)
    return np.exp(ptdf) / np.exp(ptdf).sum()

def calculate_probability_distribution_single_element(element, df):
    if element not in df.index:
        return None
    tdf = df.loc[[element]]
    tdf = tdf.div(tdf.sum(1), axis = 0)
    tdf = tdf.T
    tdf.columns = ["score"]
    return tdf["score"]

def calculate_probability_distribution_domain_range(domain, range, domain_df, range_df):
    if domain is None and range is not None:
        return calculate_probability_distribution_single_element(range, range_df)
    
    if range is None and domain is not None:
        return calculate_probability_distribution_single_element(domain, domain_df)
        
    if domain is None and range is None:
        return None

        
    tdfd = calculate_probability_distribution_single_element(domain, domain_df)
    
    tdfr = calculate_probability_distribution_single_element(range, range_df)
    
    if tdfr is None and tdfd is None:
        return None
    elif tdfr is None:
        return tdfd
    elif tdfd is None:
        return tdfr
    else:

        return tdfr * tdfd
    
    
def calculate_probability_distribution_only_domain(domain, domain_df):
    
    tdfd = calculate_probability_distribution_single_element(domain, domain_df).T
    tdfd = tdfd[tdfd.columns[0]]

    return tdfd

def calculate_probability_distribution_only_range(range, range_df):
    
    tdfr = calculate_probability_distribution_single_element(range, range_df).T
    tdfr = tdfr[tdfr.columns[0]]

    return tdfr



def query_and_find_relation_forward(relation_probs, focus, target):
    final_rel = []
    for rel in relation_probs.index:
        if good_query_forward(focus, rel, target):
            final_rel.append([rel, relation_probs[rel]])
    return final_rel

def query_and_find_relation_backward(relation_probs, focus, target):
    final_rel = []
    for rel in relation_probs.index:
        if good_query_backward(focus, rel, target):
            final_rel.append([rel, relation_probs[rel]])
    return final_rel

def good_query_forward(focus, rel, target):
    if target is not None and focus is not None:
        query = """
            PREFIX wd: <https://www.wikidata.org/wiki/>
            PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX schema: <http://www.schema.org/>

            ASK 
            {{
                wd:{} wdt:{} ?item .
                ?item (wdt:P31|wdt:P279|wdt:P31|wdt:P171)* wd:{} .
            }}
            """.format(focus, rel, target)
        sparql.setQuery(query)
        sparql.method = 'GET'
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if results["boolean"]:
            return True
        return False
    else:
        query = """
            PREFIX wd: <https://www.wikidata.org/wiki/>
            PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX schema: <http://www.schema.org/>

            ASK 
            {{
                wd:{} wdt:{} ?item .
            }}
            """.format(focus, rel)
        sparql.setQuery(query)
        sparql.method = 'GET'
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if results["boolean"]:
            return True
        return False
    return False
    
def good_query_backward(focus, rel, target):
    if target is not None and focus is not None:
        query = """
            PREFIX wd: <https://www.wikidata.org/wiki/>
            PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX schema: <http://www.schema.org/>

            ASK 
            {{
                ?item wdt:{} wd:{} .
                ?item (wdt:P31|wdt:P279|wdt:P31|wdt:P171)* wd:{} .
            }}
            """.format(rel, focus, target)
        sparql.setQuery(query)
        sparql.method = 'GET'
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if results["boolean"]:
            return True
        return False
    if focus is not None:
        query = """
            PREFIX wd: <https://www.wikidata.org/wiki/>
            PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX schema: <http://www.schema.org/>

            ASK 
            {{
                ?item wdt:{} wd:{} .
            }}
            """.format(rel, focus)
        sparql.setQuery(query)
        sparql.method = 'GET'
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        if results["boolean"]:
            return True
        return False
    return False



def property_label_search(query):
    final_results = []
    results = propertySearch(query, relation_index)

    for result in results:
        final_results.append([result[1], result[2]])
        
    return final_results


def combine_property_search_ml(results, df):
    if len(results) == 0:
        return df
    
    results = pd.DataFrame(results)
    results.columns = ["prop", "score"]
    results["score"] = results["score"] / results["score"].sum()
    
    
    for row in results.index:
        prop = results.loc[row, "prop"][1:-1].split("/")[-1]
        if prop in df:
            df[prop] += results.loc[row, "score"]
        
    df = df / df.sum()
    return df


def predict_uniform_probs():
    df = pd.read_csv("properties_better_list_improved.csv")[["0"]]
    df["score"] = 1./len(df)
    ser = pd.Series(index = df["0"].values, data = df["score"].values)
    return ser



domain_df, range_df = get_domain_range_df("domain_counts_df_no_prior.csv", "range_counts_df_no_prior.csv")

feature_extractor, model = make_model_fit(X, y, "NB", nb_use_prior=False)


def find_relation(focus, target, predicate_phrase_final, model, index_name, text = True, semantic = True, domain_range = True, query_correctness = True, relaxation = True):
    
    if focus is not None:
        focus_class = find_class_of(focus)[0]
    else:
        focus_class = None
    
#     predicate_phrase_final = predicate_phrases[0]
        
#     print(focus_class)    
    if domain_range:
        probs_domain_range_forward = calculate_probability_distribution_domain_range(focus_class, target, domain_df, range_df)    
        probs_domain_range_backward = calculate_probability_distribution_domain_range(target, focus_class, domain_df, range_df)    
    else:
        probs_domain_range_forward = predict_uniform_probs()    
        probs_domain_range_backward = predict_uniform_probs()    
        
#     print(type(probs_domain_range_forward))
#     print(type(probs_domain_range_backward))    
#     print(probs_domain_range_forward)
#     print(probs_domain_range_backward)    
        
    if text:
        text_based_probs, preds = make_predictions(predicate_phrase_final, feature_extractor, model)

        text_based_predictions_series = pd.Series(index = list(preds), data = list(text_based_probs[0]))
        text_based_predictions_series = text_based_predictions_series.div(text_based_predictions_series.sum())
    else:
        text_based_predictions_series = predict_uniform_probs()
        
    if semantic:
        property_label_results = property_label_search(predicate_phrase_final, relation_index)
    else:
        property_label_results = predict_uniform_probs()
        property_label_results = pd.DataFrame(property_label_results)
        property_label_results.columns = ["score"]
        property_label_results["label"] = property_label_results.index
        property_label_results = property_label_results[["label", "score"]]
        
    text_based_predictions_series = combine_property_search_ml(property_label_results, text_based_predictions_series)

#     print((type(text_based_predictions_series)))

#     print(text_based_predictions_series)
    
    if probs_domain_range_forward is not None:
        final_probs_forward = probs_domain_range_forward.multiply(text_based_predictions_series)
    else:
        final_probs_forward = text_based_predictions_series
        
        
    if probs_domain_range_backward is not None:
        final_probs_backward = probs_domain_range_backward.multiply(text_based_predictions_series)
    else:
        final_probs_backward = text_based_predictions_series
    
#     print(final_probs_forward)
#     print(final_probs_backward)
    
    final_probs_forward = final_probs_forward.sort_values(ascending=False)

    final_probs_backward = final_probs_backward.sort_values(ascending=False)
    
#     print(final_probs_forward)
#     print(final_probs_backward)
    
#     print(final_probs_forward["P2293"])
#     print(final_probs_backward["P2293"])
    
    if query_correctness:
        relation_forward = query_and_find_relation_forward(final_probs_forward, focus, target)
        relation_backward = query_and_find_relation_backward(final_probs_backward, focus, target)
        
        if len(relation_forward) == 0 and len(relation_backward) == 0 and relaxation:
            relation_forward = query_and_find_relation_forward(final_probs_forward, focus, None)
            relation_backward = query_and_find_relation_backward(final_probs_backward, focus, None)
            target = None
    else:
        relation_forward = query_and_find_relation_forward(final_probs_forward, focus, None)
        relation_backward = query_and_find_relation_backward(final_probs_backward, focus, None)


    
    direction = None
    
    if len(relation_backward) and len(relation_forward):
        relation_backward = relation_backward[0]
        relation_forward = relation_forward[0]
        
        relation = None

        if relation_backward[0] == relation_forward[0]:
            relation = relation_forward[0]
            direction = "forward"
        else:
            if relation_forward[1] >= relation_backward[1]:
                relation = relation_forward[0]
                direction = "forward"
            else:
                relation = relation_backward[0]
                direction = "backward"
    elif len(relation_backward):
        relation = relation_backward[0][0]
        direction = "backward"

    elif len(relation_forward):
        relation = relation_forward[0][0]
        direction = "forward"
    else:
        relation = None
    
    if relation is not None:
        if "P31" in relation or "P279" in relation:
            relation = "P31|P279"
        
    return relation, direction, target



def find_relation_ablation(focus, target, predicate_phrase_final, model, index_name, text = True, semantic = True, domain_range = True, query_correctness = True, relaxation = True):
    
    if focus is not None:
        focus_class = find_class_of(focus)[0]
    else:
        focus_class = None
    
#     predicate_phrase_final = predicate_phrases[0]
        
#     print(focus_class)    
    if domain_range:
        probs_domain_range_forward = calculate_probability_distribution_domain_range(focus_class, target, domain_df, range_df)    
        probs_domain_range_backward = calculate_probability_distribution_domain_range(target, focus_class, domain_df, range_df)    
    else:
        probs_domain_range_forward = predict_uniform_probs()    
        probs_domain_range_backward = predict_uniform_probs()    
        
#     print(type(probs_domain_range_forward))
#     print(type(probs_domain_range_backward))    
#     print(probs_domain_range_forward)
#     print(probs_domain_range_backward)    
        
    if text:
        text_based_probs, preds = make_predictions(predicate_phrase_final, feature_extractor, model)

        text_based_predictions_series = pd.Series(index = list(preds), data = list(text_based_probs[0]))
        text_based_predictions_series = text_based_predictions_series.div(text_based_predictions_series.sum())
    else:
        text_based_predictions_series = predict_uniform_probs()
        
    if semantic:
        property_label_results = property_label_search(predicate_phrase_final, relation_index)
    else:
        property_label_results = predict_uniform_probs()
        property_label_results = pd.DataFrame(property_label_results)
        property_label_results.columns = ["score"]
        property_label_results["label"] = property_label_results.index
        property_label_results = property_label_results[["label", "score"]]
        
    text_based_predictions_series = combine_property_search_ml(property_label_results, text_based_predictions_series)

#     print((type(text_based_predictions_series)))

#     print(text_based_predictions_series)
    
    if probs_domain_range_forward is not None:
        final_probs_forward = probs_domain_range_forward.multiply(text_based_predictions_series)
    else:
        final_probs_forward = text_based_predictions_series
        
        
    if probs_domain_range_backward is not None:
        final_probs_backward = probs_domain_range_backward.multiply(text_based_predictions_series)
    else:
        final_probs_backward = text_based_predictions_series
    
#     print(final_probs_forward)
#     print(final_probs_backward)
    
    final_probs_forward = final_probs_forward.sort_values(ascending=False)

    final_probs_backward = final_probs_backward.sort_values(ascending=False)
    
#     print(final_probs_forward)
#     print(final_probs_backward)
    
#     print(final_probs_forward["P2293"])
#     print(final_probs_backward["P2293"])
    
    if query_correctness:
        relation_forward = query_and_find_relation_forward(final_probs_forward, focus, target)
        relation_backward = query_and_find_relation_backward(final_probs_backward, focus, target)
        
    else:
        relation_forward = query_and_find_relation_forward(final_probs_forward, focus, None)
        relation_backward = query_and_find_relation_backward(final_probs_backward, focus, None)


    
    direction = None
    
    if len(relation_backward) and len(relation_forward):
        relation_backward = relation_backward[0]
        relation_forward = relation_forward[0]
        
        relation = None

        if relation_backward[0] == relation_forward[0]:
            relation = relation_forward[0]
            direction = "forward"
        else:
            if relation_forward[1] >= relation_backward[1]:
                relation = relation_forward[0]
                direction = "forward"
            else:
                relation = relation_backward[0]
                direction = "backward"
    elif len(relation_backward):
        relation = relation_backward[0][0]
        direction = "backward"

    elif len(relation_forward):
        relation = relation_forward[0][0]
        direction = "forward"
    else:
        relation = None
    
    if relation is not None:
        if "P31" in relation or "P279" in relation:
            relation = "P31|P279"
    
    if relation is None and relaxation:
        return find_relation_ablation(focus, None, predicate_phrase_final, model, relaxation = False)
    
    return relation, direction, target