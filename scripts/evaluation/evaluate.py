from Elastic import searchIndex as wiki_search_elastic
from Elastic.searchIndex import *

from subgraph_gen.subgraph_gen import *
from sentence_processing.sentence_processing import *
from relation_linking.relation_linking import *
from query_construction.query_construction import *

def evaluate_method(simple_questions = None, simple_constrained_questions = None, disambiguation = True, text = True, semantic = True, domain_range = True, query_correctness = True):
    
    total_true_pos = 0.000001
    total_false_pos = 0.000001
    total_false_neg = 0.000001


    total_true_pos_rel = 0.000001
    total_false_pos_rel = 0.000001
    total_false_neg_rel = 0.000001

    total_acc_primary = []
    total_acc_constraint = []
    total_acc_relation = []

    total_true_pos_ans = 0.000001
    total_false_pos_ans = 0.000001
    total_false_neg_ans = 0.000001
    
    
    if simple_questions is not None:
        res = evaluate_simple_questions(simple_questions, disambiguation = disambiguation, text = text, semantic = semantic, domain_range = domain_range, query_correctness = query_correctness)
        
        total_true_pos += res[0]
        total_false_pos += res[1]
        total_false_neg += res[2]


        total_true_pos_rel += res[3]
        total_false_pos_rel += res[4]
        total_false_neg_rel += res[5]

        total_acc_primary += res[6]
        total_acc_constraint += res[7]
        total_acc_relation += res[8]

        total_true_pos_ans += res[9]
        total_false_pos_ans += res[10]
        total_false_neg_ans += res[11]
        
    if simple_constrained_questions is not None:
        res = evaluate_simple_constrained_questions(simple_constrained_questions, disambiguation = disambiguation, text = text, semantic = semantic, domain_range = domain_range, query_correctness = query_correctness)
        
        total_true_pos += res[0]
        total_false_pos += res[1]
        total_false_neg += res[2]


        total_true_pos_rel += res[3]
        total_false_pos_rel += res[4]
        total_false_neg_rel += res[5]

        total_acc_primary += res[6]
        total_acc_constraint += res[7]
        total_acc_relation += res[8]

        total_true_pos_ans += res[9]
        total_false_pos_ans += res[10]
        total_false_neg_ans += res[11]
        
    
    
    
    
    total_precision = total_true_pos / (total_true_pos + total_false_pos)

    total_recall = total_true_pos / (total_true_pos + total_false_neg)

    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)


    total_precision_rel = total_true_pos_rel / (total_true_pos_rel + total_false_pos_rel)

    total_recall_rel = total_true_pos_rel / (total_true_pos_rel + total_false_neg_rel)

    total_f1_rel = 2 * total_precision_rel * total_recall_rel / (total_precision_rel + total_recall_rel)

    total_precision_ans = total_true_pos_ans / (total_true_pos_ans + total_false_pos_ans)

    total_recall_ans = total_true_pos_ans / (total_true_pos_ans + total_false_neg_ans)

    total_f1_ans = 2 * total_precision_ans * total_recall_ans / (total_precision_ans + total_recall_ans)

    return {
        "total_precision_ent": total_precision,
        "total_recall_ent": total_recall,
        "total_f1_ent": total_f1,
        "total_precision_rel": total_precision_rel,
        "total_recall_rel": total_recall_rel,
        "total_f1_rel": total_f1_rel,
        "total_precision_ans": total_precision_ans,
        "total_recall_ans": total_recall_ans,
        "total_f1_ans": total_f1_ans
    }



def get_entities_relations_answers(question, nlp, model, feature_extractor, text, semantic, domain_range, query_correctness, method = "mwis", disambiguation = True):
    
#     print("disambiguation ", disambiguation)
    
    if disambiguation:
    
        entity_matches, entity_phrases, predicate_phrases, pot_entity_phrases, pot_predicate_phrases = find_ER_candidates_from_questions(question, nlp, model, feature_extractor, method = "mwis")

        focus, target, final_entities = find_focus_constraint(entity_matches, question)

        final_entities = [x[0] for x in final_entities]
    else:
        entity_matches, entity_phrases, predicate_phrases, pot_entity_phrases, pot_predicate_phrases = find_ER_candidates_from_questions(question, nlp, model, feature_extractor, method = "mwis")

        final_matches = find_final_matches_no_disambiguation(entity_matches, entity_phrases)

        focus, target, final_entities = find_focus_constraint_without_disambiguation(final_matches, question)

        final_entities = [x[0] for x in final_entities]

    if len(predicate_phrases):
        predicate_phrase = predicate_phrases[0]
    else:
        predicate_phrase = " "
    relation, direction, target = find_relation_ablation(focus, target, predicate_phrase, model, text = text, semantic = semantic, domain_range = domain_range, query_correctness = query_correctness, relaxation = True)

    if relation is not None:
        predicted_answers = run_final_query(create_final_query(focus, target, relation, direction))
        final_relation = [relation]
#         print("Queries:\n", create_final_query(focus, target, relation, direction))
    else:
        predicted_answers = []
        final_relation = []
        
    return focus, target, final_entities, relation, direction, final_relation, predicted_answers

#         print("Queries:\n", "")




def evaluate_simple_constrained_questions(df, disambiguation = True, text = True, semantic = True, domain_range = True, query_correctness = True):

    total_true_pos = 0.000001
    total_false_pos = 0.000001
    total_false_neg = 0.000001


    total_true_pos_rel = 0.000001
    total_false_pos_rel = 0.000001
    total_false_neg_rel = 0.000001

    total_acc_primary = []
    total_acc_constraint = []
    total_acc_relation = []

    total_true_pos_ans = 0.000001
    total_false_pos_ans = 0.000001
    total_false_neg_ans = 0.000001


    for row in tq.tqdm(df.index):
        question = df.loc[row, "0"]
        primary_rel = df.loc[row, "primary_rel"]

        ents = df.loc[row, "entities"]

        primary = df.loc[row, "primary"]
        primary = primary.split(":")[-1]

        constraint = df.loc[row, "target"]
        constraint = constraint.split(":")[-1]

        query = df.loc[row, "query"]
        if "P31" in primary_rel or "P279" in primary_rel:
            primary_rel = "P31|P279"
        else:
            primary_rel = primary_rel.split(":")[-1]

        ents = [x.split(":")[-1] for x in ents]

        primary_rel = [primary_rel]    

        gold_answers = run_final_query(make_gold_query([x[1:-1] for x in query[1:-1].split(", ")]))
        
        focus, target, final_entities, relation, direction, final_relation, predicted_answers = get_entities_relations_answers(question, nlp, model, feature_extractor, text, semantic, domain_range, query_correctness, method = "mwis", disambiguation = disambiguation)

        
#         COMPUTE METRICS
        tp = len(set(final_entities).intersection(set(ents)))
        fp = len(set(final_entities)) - tp
        fn = len(set(ents)) - tp

        total_true_pos += tp
        total_false_pos += fp
        total_false_neg += fn


    ##################
        tp_rel = len(set(final_relation).intersection(set(primary_rel)))
        fp_rel = len(set(final_relation)) - tp_rel
        fn_rel = len(set(primary_rel)) - tp_rel


        total_true_pos_rel += tp_rel
        total_false_pos_rel += fp_rel
        total_false_neg_rel += fn_rel

    ###################
        tp_ans = len(set(predicted_answers).intersection(set(gold_answers)))
        fp_ans = len(set(predicted_answers)) - tp_ans
        fn_ans = len(set(gold_answers)) - tp_ans

        total_true_pos_ans += tp_ans
        total_false_pos_ans += fp_ans
        total_false_neg_ans += fn_ans


        if primary_rel[0] in final_relation:
            total_acc_relation.append(1)
        else:
            total_acc_relation.append(0)

        if focus == primary:
            total_acc_primary.append(1)
        else:
            total_acc_primary.append(0)

        if target == constraint:
            total_acc_constraint.append(1)
        else:
            total_acc_constraint.append(0)

    return total_true_pos, total_false_pos, total_false_neg, total_true_pos_rel, total_false_pos_rel, total_false_neg_rel, total_acc_primary, total_acc_constraint, total_acc_relation, total_true_pos_ans, total_false_pos_ans, total_false_neg_ans, 




def evaluate_simple_questions(df, disambiguation = True, text = True, semantic = True, domain_range = True, query_correctness = True):

    total_true_pos = 0.000001
    total_false_pos = 0.000001
    total_false_neg = 0.000001


    total_true_pos_rel = 0.000001
    total_false_pos_rel = 0.000001
    total_false_neg_rel = 0.000001

    total_acc_primary = []
    total_acc_constraint = []
    total_acc_relation = []

    total_true_pos_ans = 0.000001
    total_false_pos_ans = 0.000001
    total_false_neg_ans = 0.000001


    for row in tq.tqdm(df.index):
        question = df.loc[row, "0"]
        primary_rel = df.loc[row, "primary_rel"]

        ents = df.loc[row, "entities"]

        primary = df.loc[row, "primary"]
        primary = primary.split(":")[-1]


        constraint = None
        query = df.loc[row, "query"]
        if "P31" in primary_rel or "P279" in primary_rel:
            primary_rel = "P31|P279"
        else:
            primary_rel = primary_rel.split(":")[-1]

        ents = [x.split(":")[-1] for x in ents]

        primary_rel = [primary_rel]    

        gold_answers = run_final_query(make_gold_query([x[1:-1] for x in query[1:-1].split(", ")]))

        focus, target, final_entities, relation, direction, final_relation, predicted_answers = get_entities_relations_answers(question, nlp, model, feature_extractor, text, semantic, domain_range, query_correctness, method = "mwis", disambiguation = disambiguation)

        tp = len(set(final_entities).intersection(set(ents)))
        fp = len(set(final_entities)) - tp
        fn = len(set(ents)) - tp

        total_true_pos += tp
        total_false_pos += fp
        total_false_neg += fn


    ##################
        tp_rel = len(set(final_relation).intersection(set(primary_rel)))
        fp_rel = len(set(final_relation)) - tp_rel
        fn_rel = len(set(primary_rel)) - tp_rel


        total_true_pos_rel += tp_rel
        total_false_pos_rel += fp_rel
        total_false_neg_rel += fn_rel

    ###################
        tp_ans = len(set(predicted_answers).intersection(set(gold_answers)))
        fp_ans = len(set(predicted_answers)) - tp_ans
        fn_ans = len(set(gold_answers)) - tp_ans

        total_true_pos_ans += tp_ans
        total_false_pos_ans += fp_ans
        total_false_neg_ans += fn_ans

        if primary_rel[0] in final_relation:
            total_acc_relation.append(1)
        else:
            total_acc_relation.append(0)

        if focus == primary:
            total_acc_primary.append(1)
        else:
            total_acc_primary.append(0)

        if target == constraint:
            total_acc_constraint.append(1)
        else:
            total_acc_constraint.append(0)

    return total_true_pos, total_false_pos, total_false_neg, total_true_pos_rel, total_false_pos_rel, total_false_neg_rel, total_acc_primary, total_acc_constraint, total_acc_relation, total_true_pos_ans, total_false_pos_ans, total_false_neg_ans




def find_relation_ablation(focus, target, predicate_phrase_final, model, text = True, semantic = True, domain_range = True, query_correctness = True, relaxation = True):
    
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
        property_label_results = property_label_search(predicate_phrase_final)
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