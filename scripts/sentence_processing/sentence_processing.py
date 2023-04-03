import spacy   
from spacy.matcher import Matcher
import tqdm.notebook as tq
import itertools
import networkx as nx
import re
from subgraph_gen.subgraph_gen import *


from Elastic import searchIndex as wiki_search_elastic
from SPARQLWrapper import SPARQLWrapper, JSON

entity_index = "wikidata_bio_entity_index"

SPARQL_endpoint = "http://localhost:7200/repositories/wikidata_bio_2"

sparql = SPARQLWrapper(SPARQL_endpoint)




nlp = spacy.load('en_core_web_trf')


pattern_verb = [{'POS': 'AUX', 'OP': "?"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'ADP', 'OP': '?'}]

pattern_verb2 = [{'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': "+"},
                {'POS': 'VERB', 'OP': '+'}]


pattern_verb3 = [{'POS': 'VERB', 'OP': "+"},
                {'POS': 'ADP', 'OP': "+"},
                {'POS': 'VERB', 'OP': '+'}]

pattern_verb_verb_prep1 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'ADP', 'OP': '+'},
                {'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'ADP', 'OP': '+'}]

pattern_verb_verb_prep2 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': '+'},
                {'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': '+'}]

pattern_verb_verb_prep3 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': '+'},
                {'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'ADP', 'OP': '+'}]

pattern_verb_verb_prep4 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'ADP', 'OP': '+'},
                {'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': '+'}]


pattern_adj_prep = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'ADJ', 'OP': "+"},
                {'POS': 'ADP', 'OP': '+'}]

pattern_verb_noun_prep1 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'ADP', 'OP': '+'},
                {'POS': 'NOUN'},
                {'POS': 'ADP'}]

pattern_verb_noun_prep2 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': '+'},
                {'POS': 'NOUN'},
                {'POS': 'ADP'}]


pattern_verb_noun_prep3 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': '+'},
                {'POS': 'NOUN'},
                {'POS': 'PART'}]

pattern_verb_noun_prep4 = [{'POS': 'AUX', 'OP': "+"},
                {'POS': 'VERB', 'OP': "+"},
                {'POS': 'PART', 'OP': '+'},
                {'POS': 'NOUN'},
                {'POS': 'ADP'}]


pattern_noun_prep_2 = [{'POS': 'ADJ', 'OP': '*'},
                       {'POS': 'NOUN', 'OP': '*'},
                       {'POS': 'NOUN'},
                       {'POS': 'ADP'},
                       {'POS': 'NOUN'},
                       {'POS': 'ADP'}]

pattern_noun_prep = [{'POS': 'ADJ', 'OP': '*'},
                     {'POS': 'NOUN', 'OP': '*'},
                     {'POS': 'PROPN', 'OP': '*'},
                     {'POS': 'NOUN'},
                     {'POS': 'ADP'}]

pattern_noun_only_4 = [{'POS': 'NOUN'},
           {'POS': 'NOUN'},
           {'POS': 'NOUN'},
           {'POS': 'NOUN'}]

pattern_noun_only_3 = [{'POS': 'NOUN'},
           {'POS': 'NOUN'},
           {'POS': 'NOUN'}]

pattern_noun_chunk = [{'POS': 'DET', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                     {'POS': 'NOUN', 'OP': '*'},
                     {'POS': 'PROPN', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                      {'POS': 'PUNCT', 'OP': '*'},
                      {'POS': 'NUM', 'OP': '*'},
                     {'POS': 'NOUN'},
                     {'POS': 'NUM', 'OP': '*'}]

pattern_noun_chunk1 = [{'POS': 'DET', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                     {'POS': 'NOUN', 'OP': '*'},
                     {'POS': 'PROPN', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                      {'POS': 'PUNCT', 'OP': '*'},
                      {'POS': 'NUM', 'OP': '*'},
                     {'POS': 'PROPN'},
                     {'POS': 'NUM', 'OP': '*'}]

pattern_noun_chunk2 = [{'POS': 'DET', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                     {'POS': 'PROPN', 'OP': '*'},
                     {'POS': 'NOUN', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                      {'POS': 'PUNCT', 'OP': '*'},
                      {'POS': 'NUM', 'OP': '*'},
                     {'POS': 'PROPN'},
                     {'POS': 'NUM', 'OP': '*'}]

pattern_noun_chunk3 = [{'POS': 'DET', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                     {'POS': 'PROPN', 'OP': '*'},
                     {'POS': 'NOUN', 'OP': '*'},
                      {'POS': 'ADJ', 'OP': '*'},
                      {'POS': 'PUNCT', 'OP': '*'},
                      {'POS': 'NUM', 'OP': '*'},
                     {'POS': 'NOUN'},
                     {'POS': 'NUM', 'OP': '*'}]

pattern_noun_chunk_prep = [{'POS': 'NOUN', 'OP': '*'},
                     {'POS': 'PROPN', 'OP': '*'},
                      {'POS': 'ADP', 'OP': '+'},
                      {'POS': 'NOUN', 'OP': '*'},
                     {'POS': 'PROPN', 'OP': '*'},
                     {'POS': 'NUM', 'OP': '*'},
                      {'POS': 'ADP', 'OP': '!'},
                      {'POS': 'NUM', 'OP': '*'},
]

pattern_human = [{'LOWER': 'human'}]

# instantiate a Matcher instance

matcher = Matcher(nlp.vocab)
matcher.add("Verb phrase", [pattern_verb, pattern_verb2, pattern_verb3])
matcher.add("Verb phrase", [pattern_adj_prep])
matcher.add("Noun phrase", [pattern_noun_chunk, pattern_noun_chunk1, pattern_noun_chunk2, 
                            pattern_noun_chunk3, pattern_human, pattern_noun_only_4, pattern_noun_only_3])
matcher.add("Noun prepo noun", [pattern_noun_chunk_prep])
matcher.add("Noun prepo", [pattern_noun_prep, pattern_noun_prep_2, pattern_verb_noun_prep1, pattern_verb_noun_prep2, pattern_verb_noun_prep3, pattern_verb_noun_prep4,
    pattern_verb_verb_prep1, pattern_verb_verb_prep2, pattern_verb_verb_prep3, pattern_verb_verb_prep4])



def contains_noun(phrase):
    for y in phrase:
        if y.pos_ in ["NOUN", "PROPN"]:
            return True
    return False



def reduce_noise_relations(potential_relation_phrases):
    """This function processes the potential predicate phrases and removes the question words (Which, what, etc.) 
    and the determiner (the, a) terms"""
        
    reduced_relation_phrases = []
    
    present_strings = []

    potential_relation_phrases = [x for x in potential_relation_phrases]

    for x in potential_relation_phrases:        
        filtered1 = [y for y in x if y.pos_ not in ['DET', 'PUNCT']] # Remove words like a/the
        
        filtered2 = [y for y in filtered1 if y.text.lower() not in ["which", "where", "what", "when", "how", "who", "list", "name", "patients", "mutation"]]
        
        if len(filtered2):
            filtered2 = [y.text for y in filtered2]

            if " ".join(filtered2) not in present_strings:
                reduced_relation_phrases.append(filtered2)
                present_strings.append(" ".join(filtered2))
                    
    return reduced_relation_phrases




def reduce_noise(potential_entity_phrases):
    """This function processes the potential entities and removes the question words (Which, what, etc.) 
    and the determiner (the, a) terms"""
    
    reduced_entity_phrases = []
    
    present_strings = []
    
    potential_entity_phrases = [x for x in potential_entity_phrases if contains_noun(x)]

    for x in potential_entity_phrases:        
        filtered1 = [y for y in x if y.pos_ not in ['DET', 'PUNCT']] # Remove words like a/the
        
        # Remove words like when, where, how etc.
        filtered2 = [y for y in filtered1 if y.text.lower() not in ["which", "where", "what", "when", "how", "who", "list", "name", "patients", "mutations", "mutation"]]
        
        while len(filtered2) and filtered2[0].pos_ == "ADP":
            filtered2 = filtered2[1:]
        
        if len(filtered2):
            filtered2 = [y.text for y in filtered2]

            if " ".join(filtered2) not in present_strings:
                reduced_entity_phrases.append(filtered2)
                present_strings.append(" ".join(filtered2))
                    
    return reduced_entity_phrases
        

def greedy_selection(G):
    # print(G.nodes)
    nodes = [[node, int(G.nodes[node]["weight"])] for node in G.nodes]
    sorted_nodes = sorted(nodes, key = lambda x: -x[1])
    
    # print(sorted_nodes)
    selected_nodes = []

    for node in sorted_nodes:
        if node[1] > 0:
            others = [x for x in G.neighbors(node[0])]
            if len(set(others).intersection(set(selected_nodes))) == 0:
                selected_nodes.append(node[0])

    return selected_nodes


def greedy_selection_multiple(G):
    connected_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    combinations = []

    for Gc in connected_components:

        nodes = [[node, int(Gc.nodes[node]["weight"])] for node in Gc.nodes]
        sorted_nodes = sorted(nodes, key = lambda x: -x[1])
        selected_nodes = []

        for i in range(len(sorted_nodes)):
            srt_nodes = sorted_nodes[i:] + sorted_nodes[:i]
            temp_selected_nodes = []

            for node in srt_nodes:
                if node[1] > 0 and node[0].strip() != "":
                    others = [x for x in Gc.neighbors(node[0])]
                    if len(set(others).intersection(set(temp_selected_nodes))) == 0:
                        temp_selected_nodes.append(node[0])
            temp_selected_nodes.sort()
            selected_nodes.append("|".join(temp_selected_nodes))

        selected_nodes.sort()
        selected_nodes = list(k for k,_ in itertools.groupby(selected_nodes))

        combinations.append(selected_nodes)

    combinations = [sorted(x) for x in combinations]
    combinations.sort()
    combinations = list(k for k,_ in itertools.groupby(combinations))

    all_possible_combinations = []
    for combs in itertools.product(*combinations):
        temp_combs = []
        for comb in combs:
            temp_combs += [x for x in comb.split("|") if len(x.strip()) > 0]
        all_possible_combinations.append(temp_combs)

    # print(all_possible_combinations)
    all_possible_combinations_with_scores = []

    for combinations in all_possible_combinations:
        score = 0
        for node in combinations:
            score += G.nodes[node]["weight"]
        all_possible_combinations_with_scores.append([combinations, score])

    return all_possible_combinations_with_scores



def reduce_redundancy(potential_entity_phrases, alpha, beta, method):
    """This function processes the potential entities and removes repetitive ones, using the Maximum Weighted
    Independent Set Algorithm. Essentially, we create a graph with phrases as nodes, where every two
    entity phrases / nodes have an edge if there is an overlap between them. Every phrase gets a weight
    which is size(phrase) * alpha * highest_matching_score(phrase).
    
    We calculate the maximum weighted independent score by calculating the complement graph and then 
    taking the maximum weighted clique. This can be done in networkx"""
    # if len(potential_entity_phrases) <= 1:
    #     return potential_entity_phrases
    
    G = nx.Graph()
    
    for phrase in potential_entity_phrases:
        phrase_text = " ".join(phrase)
        matches = wiki_search_elastic.entitySearch(phrase_text, entity_index)
        
        # print("Matches:", matches)

        if len(matches):
            # print("Match: ", phrase_text, matches, "Full")
            # weight = ((beta * len(phrase)) ** alpha) * matches[0][2]
            weight = (len(phrase_text) ** alpha) * matches[0][2] / 100            
        else:
            # matches = wiki_search_elastic.entitySearch(phrase_text, entity_index, cutoff = 70)
            # if len(matches):
            #     print("Match: ", phrase_text, matches[0], "Part")
            #     weight = ((beta * len(phrase) * (70 / 85) * 0.5) ** alpha) * matches[0][2] * (70 / 85)
            # else:
            weight = 0

        G.add_node(phrase_text.replace(" ", "_"), weight = weight)
        
    for phrase1 in potential_entity_phrases:
        for phrase2 in potential_entity_phrases:
            
            if len(set(phrase1) - set(phrase2)) != 0 and len(set(phrase1).intersection(set(phrase2))) != 0:
                
                phrase_text1 = " ".join(phrase1)
                phrase_text2 = " ".join(phrase2)
                
                G.add_edge(phrase_text1.replace(" ", "_"), phrase_text2.replace(" ", "_"))



    # print([[node, G.nodes[node]["weight"]] for node in G.nodes])

    if method == "mwis":
        Gc = nx.complement(G)

        for node in Gc.nodes:
            Gc.nodes[node]["weight"] = int(G.nodes[node]["weight"] * 100)
    
        # import pylab
    
        # print(Gc.nodes)

        # labels = nx.get_node_attributes(Gc, 'weight') 
        # nx.draw(Gc,labels=labels,node_size=1000)
        
        # pylab.show()

        max_nodes = nx.max_weight_clique(Gc)
        potential_entity_phrases = [str(x).replace("_", " ") for x in max_nodes[0]]

    elif method == "greedy":
        max_nodes = greedy_selection(G)        
        potential_entity_phrases = [str(x).replace("_", " ") for x in max_nodes]
        
    return potential_entity_phrases



def reduce_redundancy_multiple(potential_entity_phrases, alpha, beta):
    """This function processes the potential entities and removes repetitive ones, using the Maximum Weighted
    Independent Set Algorithm. Essentially, we create a graph with phrases as nodes, where every two
    entity phrases / nodes have an edge if there is an overlap between them. Every phrase gets a weight
    which is size(phrase) * alpha * highest_matching_score(phrase).
    
    We calculate the maximum weighted independent score by calculating the complement graph and then 
    taking the maximum weighted clique. This can be done in networkx"""
    
    import networkx as nx
    
    G = nx.Graph()
    
    for phrase in potential_entity_phrases:

        phrase_text = " ".join(phrase)
        matches = wiki_search_elastic.entitySearch(phrase_text, entity_index)
        
        if len(matches):
            weight = ((beta * len(phrase) * matches[0][2]) ** alpha)
        else:
            weight = 0
        G.add_node(phrase_text.replace(" ", "_"), weight = weight)
        
    for phrase1 in potential_entity_phrases:
        for phrase2 in potential_entity_phrases:
            
            if len(set(phrase1) - set(phrase2)) != 0 and len(set(phrase1).intersection(set(phrase2))) != 0:
                
                phrase_text1 = " ".join(phrase1)
                phrase_text2 = " ".join(phrase2)
                
                G.add_edge(phrase_text1.replace(" ", "_"), phrase_text2.replace(" ", "_"))
                

    max_nodes = greedy_selection_multiple(G)        
    potential_entity_phrases = [[[str(x).replace("_", " ") for x in y[0]], y[1]] for y in max_nodes]
        
    return potential_entity_phrases



def reduce_redundancy_relations(potential_relation_phrases):
    """This function processes the potential entities and removes repetitive ones, using the Maximum Weighted
    Independent Set Algorithm. Essentially, we create a graph with phrases as nodes, where every two
    entity phrases / nodes have an edge if there is an overlap between them. Every phrase gets a weight
    which is size(phrase) * alpha * highest_matching_score(phrase).
    
    We calculate the maximum weighted independent score by calculating the complement graph and then 
    taking the maximum weighted clique. This can be done in networkx"""
    
    import networkx as nx
    
    G = nx.Graph()
    
    for phrase in potential_relation_phrases:
        phrase_text = " ".join(phrase)
        
        weight = len(phrase_text)

        G.add_node(phrase_text.replace(" ", "_"), weight = weight)
        
    for phrase1 in potential_relation_phrases:
        for phrase2 in potential_relation_phrases:
            
            phrase1lt = [x for x in phrase1]
            phrase2lt = [x for x in phrase2]
            
            if len(set(phrase1lt) - set(phrase2lt)) != 0 and len(set(phrase1lt).intersection(set(phrase2lt))) != 0:
                
                phrase_text1 = " ".join(phrase1)
                phrase_text2 = " ".join(phrase2)
                
                G.add_edge(phrase_text1.replace(" ", "_"), phrase_text2.replace(" ", "_"))
                
    Gc = nx.complement(G)
    
    for node in Gc.nodes:
        Gc.nodes[node]["weight"] = int(G.nodes[node]["weight"])
    
    max_nodes = nx.max_weight_clique(Gc)
    
    potential_relation_phrases = [str(x).replace("_", " ") for x in max_nodes[0]]
        
    return potential_relation_phrases, max_nodes[1]
    
def remove_predicate_overlapping_entities(entity_matches, predicate_matches):
    final_entity_matches = []

    overlapping = []


    for em in entity_matches:
        overlapping_entity = False
        eml = [x for x in em.split(" ") if len(x) >= 3]
        for pm in predicate_matches:
            pml = [x for x in pm.split(" ") if len(x) >= 3]
            if len(set(pml).intersection(set(eml))) / len(set(pml).union(set(eml))) > 0.2:
                overlapping_entity = True
                overlapping.append([em, pm])

        if not overlapping_entity:
            final_entity_matches.append(em)

    # print(overlapping)

    if len(entity_matches) <= 2: # This makes sure that if only one entity is found, it is not removed.
        if len(overlapping) > 0 and len(predicate_matches) > 1:
            predicate_matches = [x for x in predicate_matches if x not in [y[1] for y in overlapping]]

        return entity_matches, predicate_matches

    if len(entity_matches) > 2:
        if len(overlapping) > 0 and len(predicate_matches) > 1:
            predicate_matches = [x for x in predicate_matches if x not in [y[1] for y in overlapping]]
        else:
            entity_matches = [x for x in entity_matches if x not in [y[0] for y in overlapping]]


        return entity_matches, predicate_matches


    return final_entity_matches, predicate_matches


def remove_predicate_overlapping_entities_multiple(entity_matches, predicate_matches):
    final_entity_matches = []
    for em in entity_matches:
        temp_ent_matches = []
        for e in em[0]:
            overlapping_entity = False
            eml = [x for x in e.split(" ") if len(x) >= 3]
            for pm in predicate_matches:
                pml = [x for x in pm.split(" ") if len(x) >= 3]
                if len(set(pml).intersection(set(eml))) / len(set(pml).union(set(eml))) > 0.2:
                    overlapping_entity = True
            if not overlapping_entity:
                temp_ent_matches.append(e)
        final_entity_matches.append([temp_ent_matches, em[1]])
    return final_entity_matches

def extract_bracket_terms(question):
    m = re.findall(r'\(.*?\)', question)
    m = [x[1:-1] for x in m]
    q = re.sub(r'\(.*?\)', '', question)
    q = re.sub(r'\s+', ' ', q)
    return q, m    


def process_question(question, nlp_processor, alpha = 2, beta = 2, method = "mwis"):
    potential_predicate_phrases = []
    potential_entity_phrases = []
    
    question, bracket_terms = extract_bracket_terms(question)

    x = nlp_processor(question)
    
    matches = matcher(x)
    
    spans = [[x.vocab.strings[typ], x[start:end]] for typ, start, end in matches]
    
    for s in spans:
        if s[0] == "Noun phrase" or s[0] == "Noun prepo noun":
            potential_entity_phrases.append(s[1])
        else:
            potential_predicate_phrases.append(s[1])
            
    for y in x.noun_chunks:
        potential_entity_phrases.append(y)

    potential_entity_phrases = reduce_noise(potential_entity_phrases)
    
    filtered_entity_phrases = reduce_redundancy(potential_entity_phrases, alpha = alpha, beta = beta, method = method)

    potential_predicate_phrases = reduce_noise_relations(potential_predicate_phrases)

    filtered_predicate_phrases, score = reduce_redundancy_relations(potential_predicate_phrases)

    filtered_entity_phrases_predicate_removed, filtered_predicate_phrases = remove_predicate_overlapping_entities(filtered_entity_phrases, filtered_predicate_phrases)

    filtered_entity_phrases_predicate_removed += bracket_terms

    return filtered_entity_phrases_predicate_removed, filtered_predicate_phrases, filtered_entity_phrases, potential_predicate_phrases





def process_question_multiple(question, nlp_processor, alpha = 4, beta = 2):
    potential_predicate_phrases = []
    potential_entity_phrases = []
    
    x = nlp_processor(question)
    
    matches = matcher(x)
    
    spans = [[x.vocab.strings[typ], x[start:end]] for typ, start, end in matches]
    


    for s in spans:
        if s[0] == "Noun phrase" or s[0] == "Noun prepo noun":
            potential_entity_phrases.append(s[1])
        else:
            potential_predicate_phrases.append(s[1])
            
    for y in x.noun_chunks:
        potential_entity_phrases.append(y)
    
    potential_entity_phrases = reduce_noise(potential_entity_phrases)
    
#     print(potential_entity_phrases)
    filtered_entity_phrases = reduce_redundancy_multiple(potential_entity_phrases, alpha = alpha, beta = beta)
#     print("\n")
#     print(potential_entity_phrases, score)
    filtered_predicate_phrases, score = reduce_redundancy_relations(potential_predicate_phrases)
    
    filtered_entity_phrases = remove_predicate_overlapping_entities_multiple(filtered_entity_phrases, filtered_predicate_phrases)

    return filtered_entity_phrases, filtered_predicate_phrases, potential_entity_phrases, potential_predicate_phrases



def find_ER_candidates_from_questions(q, nlp, model, feature_extractor, alpha = 1.1, beta = 1.2, method = "mwis"): #alpha was previously 4
    entity_phrases, predicate_phrases, potential_entity_phrases, potential_predicate_phrases = process_question(q, nlp, alpha, beta, method)
    entity_matches = []
    
    for term in entity_phrases:
        # print([x[1] for x in wiki_search_elastic.entitySearch(term, entity_index)[:10]])
        matches = [[x[1][1:-1].split("/")[-1], x[2]] for x in wiki_search_elastic.entitySearch(term, entity_index)[:10]]
        if len(matches) == 0:
            matches = [[x[1][1:-1].split("/")[-1], x[2]] for x in wiki_search_elastic.entitySearch(term, entity_index, 70)[:10]]

        for ent in matches:
            if ent not in [x[0] for x in entity_matches]:
                entity_matches.append([ent[0], term, ent[1]])


    return entity_matches, entity_phrases, predicate_phrases, potential_entity_phrases, potential_predicate_phrases



def find_ER_candidates_from_questions_multiple_sets(q, nlp, model, feature_extractor, alpha = 1.0, beta = 2):
    """This is the modified code which find multiple sets of possible matches"""
    entity_phrases_multiple, predicate_phrases, potential_entity_phrases, potential_predicate_phrases = process_question_multiple(q, nlp, alpha, beta)

    # entity_phrases_multiple = [sorted(x) for x in entity_phrases_multiple]
    # entity_phrases_multiple.sort()
    # entity_phrases_multiple = list(k for k,_ in itertools.groupby(entity_phrases_multiple))

    entity_matches_multiple = []
    
    for entity_phrases in entity_phrases_multiple:
        entity_matches = []
        for term in entity_phrases[0]:
            matches = [x[1][1:-1].split("/")[-1] for x in wiki_search_elastic.entitySearch(term, entity_index)[:10]]
            for ent in matches:
                if ent not in [x[0] for x in entity_matches]:
                    entity_matches.append([ent, term])
        entity_matches_multiple.append([entity_matches, entity_phrases[1]])

    return entity_matches_multiple, entity_phrases_multiple, predicate_phrases, potential_entity_phrases, potential_predicate_phrases

#     print(entity_phrases, predicate_phrases)


#     predicate_matches = []
#     for pred in predicate_phrases:
# #         print(pred)
#         present, matches = wiki_search_elastic.propertySearchExactmatch(pred)
#         if present:
#             for x in matches:
#                 prop = x[1][1:-1].split("/")[-1]
#                 if prop not in [x[0] for x in predicate_matches]:
#                     predicate_matches.append([prop, pred])
# #     print(predicate_matches)
#     for pred in predicate_phrases:
#         propertyMLResults = make_predictions(pred, feature_extractor, model)
        
#         for prop in propertyMLResults[:10]:
#             if prop not in [x[0] for x in predicate_matches]:
#                 predicate_matches.append([prop, pred])
    
#     return entity_matches, predicate_matches, entity_phrases, predicate_phrases, potential_entity_phrases, potential_predicate_phrases




import itertools
from fuzzysearch import find_near_matches

# def find_target(entity_matches, question):
    

    
    
def is_class(entity, limit = 20):
    query = """
        PREFIX wd: <https://www.wikidata.org/wiki/>
        PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX schema: <http://www.schema.org/>

        SELECT DISTINCT ?item WHERE 
        {{
            ?item (wdt:P31|wdt:P279)* wd:{} .
        }}
        """.format(entity)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    ndf = pd.DataFrame(results['results']['bindings'])
    ndf = ndf.applymap(lambda x: x['value'] if x["type"] == "uri" else np.nan)
    
    if len(ndf) > limit:
        return True, len(ndf)
    else:
        return False, 0
    
    
def is_protein_family(entity):
    query = """
        PREFIX wd: <https://www.wikidata.org/wiki/>
        PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX schema: <http://www.schema.org/>

        ASK 
        {{
            wd:{} (wdt:P31|wdt:P279)* ?pf .
            filter(?pf in (wd:Q84467700, wd:Q898273, wd:Q7251477, wd:Q417841))
        }}
        """.format(entity)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    if results["boolean"]:
        return True, 20
    return False, 0



def is_class_or_protein_family(entity):
    clas, num1  = is_class(entity)
    fam, num2 = is_protein_family(entity)
    if clas:
        return clas, num1
    else:
        return fam, num2


    
def merge_type_terms(entity_matches, question):
    if len(entity_matches) <= 1:
        return entity_matches
    perms = list(itertools.permutations(list(set([x[1] for x in entity_matches])), 2))
    
#     print(perms)
    
    perms = {" ".join(x): x for x in perms}
    terms_dict = {x[1]: x[0] for x in entity_matches}
    
    for_removal = []
#     print(perms)
    must_keep = []
    for perm, constituent in perms.items():
        match = find_near_matches(perm.lower(), question.lower(), max_l_dist=int(len(perm) / 5))
#         print(match)
        if len(match) and abs(len(match[0].matched) - len(perm)) < 2:
            if terms_dict[constituent[0]] == terms_dict[constituent[1]]:
                if len(constituent[0]) > len(constituent[1]):
                    must_keep.append(constituent[0])
                else:
                    must_keep.append(constituent[1])
    
    
    for perm, constituent in perms.items():
#         if perm in question:.match('(amazing){e<=1}', 'amaging')
        match = find_near_matches(perm.lower(), question.lower(), max_l_dist=int(len(perm) / 5))
#         print(match)
        if len(match) and abs(len(match[0].matched) - len(perm)) < 2:
#             print("entering")
            keep_both, remove_this = check_type(constituent, terms_dict)
            if not keep_both:
                for_removal.append(remove_this)
        else:
            pass
#             print("not entering")
    final_entity_matches = [x for x in entity_matches if x[1] not in for_removal]
    final_entity_matches += [x for x in entity_matches if x not in final_entity_matches and x[1] in must_keep]
    return final_entity_matches
                
            
def check_type(constituent, terms_dict):
    if is_type(constituent[0], constituent[1], terms_dict):
        return False, constituent[1]
    elif is_type(constituent[1], constituent[0], terms_dict):
        return False, constituent[0]
    return True, None

def is_type(term1, term2, terms_dict):
    ent1 = terms_dict[term1]
    ent2 = terms_dict[term2]
    
    query = """
        PREFIX wd: <https://www.wikidata.org/wiki/>
        PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX schema: <http://www.schema.org/>

        ASK {{wd:{} wdt:P31|wdt:P279 wd:{}}}
        """.format(ent1, ent2)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["boolean"]:
        return True

    query = """
        PREFIX wd: <https://www.wikidata.org/wiki/>
        PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX schema: <http://www.schema.org/>

        ASK {{wd:{} wdt:P31|wdt:P279 ?x .
            ?x wdt:P31|wdt:P279|wdt:P8225 wd:{}}}
        """.format(ent1, ent2)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["boolean"]:
        return True

    query = """
        PREFIX wd: <https://www.wikidata.org/wiki/>
        PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX schema: <http://www.schema.org/>

        ASK {{wd:{} wdt:P171* wd:{}}}
        """.format(ent1, ent2)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["boolean"]:
        return True    
    
    query = """
    PREFIX wd: <https://www.wikidata.org/wiki/>
    PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX schema: <http://www.schema.org/>

    ASK {{wd:{} wdt:P2868 wd:{}}}
    """.format(ent1, ent2)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["boolean"]:
        return True
    
    
    query = """
    PREFIX wd: <https://www.wikidata.org/wiki/>
    PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX schema: <http://www.schema.org/>

    ASK {{
        wd:{} wdt:P31 wd:Q8054 .
        wd:{} wdt:P361 ?x .
        ?x (wdt:P31|wdt:P279)* wd:{} .
    }}
    """.format(ent1, ent1, ent2)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if results["boolean"]:
        return True
    
    
    return False

def final_filter(entities):
    for ent in entities:
        if ent[0] == "Q45314346":
            ent[0] = "Q12136"
    
    return entities


def find_class_of(entity):
    query = """
        PREFIX wd: <https://www.wikidata.org/wiki/>
        PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX schema: <http://www.schema.org/>

        SELECT DISTINCT ?item WHERE 
        {{
            wd:{} wdt:P31 ?item .
        }}
        """.format(entity)
    sparql.setQuery(query)
    sparql.method = 'GET'
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    ndf = pd.DataFrame(results['results']['bindings'])
    ndf = ndf.applymap(lambda x: x['value'] if x["type"] == "uri" else np.nan)
    if len(ndf):
        return [x.split("/")[-1] for x in list(ndf["item"].values)]
    else:
        return [None]


def find_focus_constraint(entity_matches, question, verbose = False):
    if not len(entity_matches):
        return None, None

    entity_matches, tdfs, spokes = find_final_entities(entity_matches, False)
    
    filtered_entity_matches = final_filter(entity_matches)

    filtered_entity_matches = merge_type_terms(filtered_entity_matches, question)
    
    init_entities = [x for x in entity_matches]
    final_entities = [x for x in filtered_entity_matches]
    
    probable_class_entities = []
    probable_non_class_entities = []

    for x in final_entities:
        clas, num = is_class_or_protein_family(x[0])
        if clas:
            probable_class_entities.append([x[0], num])
        else:
            probable_non_class_entities.append(x[0])

    probable_class_entities = [x[0] for x in sorted(probable_class_entities, key=lambda x: x[1], reverse = True)]

    probable_focus_entities = probable_non_class_entities
    probable_target_entities = probable_class_entities
    
    if verbose:
        print("Probable Focus Entities: ", probable_focus_entities)
        print("Probable Target Entities: ", probable_target_entities)
    
    if len(probable_focus_entities):
        focus = probable_focus_entities[0]
    elif len(probable_target_entities) == 1:
        focus = probable_target_entities[0]
    elif len(probable_target_entities) > 1:
        focus = probable_target_entities[1]
    else:
        focus = None
        
    if len(probable_target_entities):
        target = probable_target_entities[0]
    else:
        target = None

    if focus == target:
        target = None

    return focus, target, final_entities









def find_focus_constraint_without_disambiguation(entity_matches, question):
    filtered_entity_matches = final_filter(entity_matches)

    filtered_entity_matches = merge_type_terms(filtered_entity_matches, question)
    
    init_entities = [x for x in entity_matches]
    final_entities = [x for x in filtered_entity_matches]
    
    probable_class_entities = []
    probable_non_class_entities = []

    for x in final_entities:
        clas, num = is_class_or_protein_family(x[0])
        if clas:
            probable_class_entities.append([x[0], num])
        else:
            probable_non_class_entities.append(x[0])

    probable_class_entities = [x[0] for x in sorted(probable_class_entities, key=lambda x: x[1], reverse = True)]

    probable_focus_entities = probable_non_class_entities
    probable_target_entities = probable_class_entities

    
    if len(probable_focus_entities):
        focus = probable_focus_entities[0]
    elif len(probable_target_entities) == 1:
        focus = probable_target_entities[0]
    elif len(probable_target_entities) > 1:
        focus = probable_target_entities[1]
    else:
        focus = None
        
    if len(probable_target_entities):
        target = probable_target_entities[0]
    else:
        target = None

    if focus == target:
        target = None
#     final_entities = [x[0] for x in final_entities]
    return focus, target, final_entities

def find_final_matches_no_disambiguation(entity_matches, entity_phrases):    
    matches = []
    for phrase in entity_phrases:
        match = [[x[0], x[2]] for x in entity_matches if x[1] == phrase]
        match = sorted(match, key = lambda x: x[1], reverse = True)
        if len(match):
            matches.append([match[0][0], phrase, match[0][1]])
    return matches