from elasticsearch import Elasticsearch
import editdistance
import inflect
import re


p = inflect.engine()

es = Elasticsearch([{'host': 'localhost', 'port':9200, 'scheme':'http'}], timeout=300)
docType = "doc"


def entitySearch(query, indexName, cutoff = 85):
    # print(query)
    results=[]
    sing_query = p.singular_noun(query)
    ###################################################
    elasticResults=es.search(index=indexName, body={
        "from": 0,
        "size": 100,
        "query": {
            "match": {
                "label": query
            }
        }
    })

    # print("ElasticResults: ", elasticResults)
    for result in elasticResults['hits']['hits']:
        results.append(result)

    ###################################################
    
    elasticResults=es.search(index=indexName, body={
    "query": {
        "match" : {
            "label" : {
                "query" : query,
                "fuzziness": "AUTO"
                }
            }
        },"size":100
    })

    for result in elasticResults['hits']['hits']:
        results.append(result)


    if sing_query:
        elasticResults=es.search(index=indexName, body={
        "query": {
            "match" : {
                "label" : {
                    "query" : sing_query,
                    "fuzziness": "AUTO"
                    }
                }
            },"size":100
        })

        for result in elasticResults['hits']['hits']:
            results.append(result)


    # print(results)
    new_results = []

    for result in results:
        dist = editdistance.eval(result["_source"]["label"].lower(), query.lower())


## BIG CHANGE
        # result1 = re.sub(r'[\W]+', '', result["_source"]["label"])
        # result2 = re.sub(r'[\W]+', ' ', result["_source"]["label"])
        # query_temp = re.sub(r'[\W]+', ' ', query)

        # dist1 = editdistance.eval(result1.lower(), query.lower())
        # dist2 = editdistance.eval(result2.lower(), query_temp.lower())

        # if sing_query:
        #     dist3 = editdistance.eval(result["_source"]["label"].lower(), sing_query.lower())
            
        #     sing_query_temp = re.sub(r'[\W]+', ' ', query)

        #     dist4 = editdistance.eval(result1.lower(), sing_query.lower())
        #     dist5 = editdistance.eval(result2.lower(), sing_query_temp.lower())
            
        #     dist = min(dist, dist3, dist4, dist5)

        # dist = min(dist, dist1, dist2)

## BIG CHANGE ENDS



        # print(query, result, dist, "\n")

        if dist <= len(query) / 4:
            dist = (1 - dist / len(query)) * 100
            new_results.append([result["_source"]["label"], result["_source"]["uri"], dist, dist])


    results = sorted(new_results, key = lambda x: -x[2])

    already_added = []

    new_results = []

    # print(query)
    # print(results)
    # print(results)
    if len(results):
        highest_score = results[0][2]

        for result in results:
            if result[1] not in already_added and result[2] >= highest_score * 0.8:
                new_results.append(result)
                already_added.append(result[1])

    return new_results

    
def propertySearch(query, indexName):
    # print("Relation Query: ", query)

    # indexName = "wikidata_bio_subset_2_relation_index"
    results = []
    ###################################################
    elasticResults = es.search(index=indexName, body={
        "query": {
            "match": {"label": query}
        }
        , "size": 100
    })

    # print("ElasticResults: ", elasticResults)
    for result in elasticResults['hits']['hits']:
        if result["_source"]["label"].lower().replace('.','').strip()==query.lower().strip():
            results.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 50, 40])
        else:
            results.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 40, 0])

    ###################################################
    elasticResults=es.search(index=indexName, body={
    "query": {
        "match" : {
            "label" : {
                "query" : query,
                "fuzziness": "AUTO"
                
            }
        }
    },"size":100
            }
           )
    # print("ElasticResults: ", elasticResults)

    for result in elasticResults['hits']['hits']:
        edit_distance=editdistance.eval(result["_source"]["label"].lower().replace('.','').strip(), query.lower().strip())
        if edit_distance<=1:
            results.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 50, 40])
        else:
            results.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 25, 0])

    # print("Results Intermediate: ", results)
            
            
    results = sorted(results, key = lambda x: -x[2])

    # print("Property Search Results", results)

    return results[:15]

def propertySearchExactmatch(query):
    # indexName = "wikidata_bio_subset_2_relation_index"
    ###################################################
    # indexName = "wikidata_bio_subset_2_relation_index"
    # elasticResults = es.search(index=indexName, body={
    #     "query": {
    #         "match": {"label": query}
    #     }
    # }

    # elasticResults=es.search(index=indexName, body={
    # "query": {
    #     "match" : {
    #         "label" : {
    #             "query" : query,
    #             "fuzziness": "AUTO"
                
    #         }
    #     }}})

    # # print(query, elasticResults['hits']['hits'])
    # for result in elasticResults['hits']['hits']:

    #     if query.lower().strip() in result["_source"]["label"].lower().replace('.','').strip():
    #         return True


    # return False

    # print("Relation Query: ", query)

    indexName = "wikidata_bio_subset_2_relation_index"
    results = []
    elasticResults = es.search(index=indexName, body={
        "query": {
            "match": {"label": query}
        }
        , "size": 100
    })

    for result in elasticResults['hits']['hits']:
        edit_distance = 100 * (1 - editdistance.eval(result["_source"]["label"].lower(), query.lower()) / max(len(result["_source"]["label"]), len(query)))
        if edit_distance >= 60:
            results.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 50, edit_distance])   

    ###################################################
    elasticResults=es.search(index=indexName, body={
    "query": {
        "match" : {
            "label" : {
                "query" : query,
                "fuzziness": "AUTO"
                
            }
        }
    },"size":100
            }
           )
    # print("ElasticResults: ", elasticResults)

    for result in elasticResults['hits']['hits']:
        edit_distance = 100 * (1 - editdistance.eval(result["_source"]["label"].lower(), query.lower()) / max(len(result["_source"]["label"]), len(query)))
        if edit_distance >= 60:
            results.append([result["_source"]["label"], result["_source"]["uri"], result["_score"] * 50, edit_distance])            
            
    results = sorted(results, key = lambda x: -x[2])

    # print("Property Search Results", results)

    if len(results):
        return True, results
    return False, results

