from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import numpy as np

import networkx as nx
import itertools


SPARQL_endpoint = "http://localhost:7200/repositories/wikidata_bio_2"

sparql = SPARQLWrapper(SPARQL_endpoint)

import tqdm.notebook as tq



def find_outgoing_nodes_df(entity, limit):
	query = """
		PREFIX wd: <https://www.wikidata.org/wiki/>
		PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
		PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
		PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
		PREFIX schema: <http://www.schema.org/>


		select distinct ?a ?b ?c where {{{{
		?a ?b ?c .

		filter(?a = wd:{}) .
		filter(?b not in (skos:prefLabel, skos:altLabel, schema:description)) .
		}}}}
		limit {}""".format(entity, limit)
# 	print(query)
	sparql.setQuery(query)
	sparql.method = 'GET'
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()

	# print(results)
	ndf = pd.DataFrame(results['results']['bindings'])
	ndf = ndf.applymap(lambda x: x['value'] if x["type"] == "uri" else np.nan)
	try:
		ndf = ndf[["a", "b", "c"]]
		ndf.columns = ["1", "2", "3"]

		ndf = ndf.dropna()

		ndf = ndf.drop_duplicates()

		return ndf
	except Exception as e:
		return pd.DataFrame()


def find_incoming_nodes_df(entity, limit):
	query = """
		PREFIX wd: <https://www.wikidata.org/wiki/>
		PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
		PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
		PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
		PREFIX schema: <http://www.schema.org/>


		select distinct ?c ?b ?a where {{{{
		?c ?b ?a .

		filter(?a = wd:{}) .
		filter(?b not in (skos:prefLabel, skos:altLabel, schema:description)) .
		}}}}
		limit {}""".format(entity, limit)
	sparql.setQuery(query)
	sparql.method = 'GET'
	sparql.setReturnFormat(JSON)
	try:
		results = sparql.query().convert()
		ndf = pd.DataFrame(results['results']['bindings'])
		ndf = ndf.applymap(lambda x: x['value'] if x["type"] == "uri" else np.nan)

		ndf = ndf[["c", "b", "a"]]
		ndf.columns = ["1", "2", "3"]

		ndf = ndf.dropna()

		ndf = ndf.drop_duplicates()

		return ndf
	except Exception as e:
		return pd.DataFrame()

def find_neighbours(entity, cutoff_hub_spoke = 200):
	ndf = pd.concat([find_incoming_nodes_df(entity, 1000000), find_outgoing_nodes_df(entity, 1000000)])
	# print(ndf.columns)	
	if len(ndf) < cutoff_hub_spoke:
		return find_second_neighbours(entity, ndf, cutoff_hub_spoke), True
	else:
		return ndf, False
		
def find_second_neighbours(entity, ndf, cutoff_hub_spoke):
	# print(ndf.columns)
	first_neighbours = list(set(ndf["1"].values).union(set(ndf["3"].values)) - set(["https://www.wikidata.org/wiki/" + entity]))
	no_dfs = []
	for neighbour in first_neighbours:
		neighbour_entity = neighbour.split("/")[-1]
		
		ntdf1 = find_incoming_nodes_df(neighbour_entity, cutoff_hub_spoke + 1)
		ntdf2 = find_outgoing_nodes_df(neighbour_entity, cutoff_hub_spoke + 1)

		
## BIG CHANGE
		ntdf3 = find_superclasses(neighbour_entity)

		no_dfs.append(ntdf1)
## BIG CHANGE ENDS


#		 print(ntdf)
		if len(ntdf1) < cutoff_hub_spoke and len(ntdf1):
			no_dfs.append(ntdf1)
			
		if len(ntdf2) < cutoff_hub_spoke and len(ntdf2):
			no_dfs.append(ntdf2)
			
	n2df = pd.concat(no_dfs + [ndf])
	n2df = n2df.drop_duplicates()
	
	return n2df


def is_taxon(entity):
	
	query = """
	PREFIX wd: <https://www.wikidata.org/wiki/>
	PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
	ASK where {{ 
		wd:{} (wdt:P31|wdt:P279)* wd:Q21871294  .
	}}
	""".format(entity)
	
	sparql.setQuery(query)
	sparql.method = 'GET'
	sparql.setReturnFormat(JSON)
	result = sparql.query().convert()["boolean"]
	return result

def find_superclasses(entity):
	query = """
	PREFIX wd: <https://www.wikidata.org/wiki/>
	PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
	select ?c where {{ 
	wd:{} (wdt:P31|wdt:P279)* ?c  .
	}}
	""".format(entity)
	
	sparql.setQuery(query)
	sparql.method = 'GET'
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()

	ndf = pd.DataFrame(results['results']['bindings'])

	ndf = ndf.applymap(lambda x: x['value'] if x["type"] == "uri" else np.nan)
	ndf["b"] = "(wdt:P31|wdt:P279)*"
	ndf["a"] = "https://www.wikidata.org/wiki/" + entity.split(":")[-1]

	ndf = ndf[["c", "b", "a"]]
	ndf.columns = ["1", "2", "3"]
	ndf = ndf.dropna()
	ndf = ndf.drop_duplicates()
	return ndf


def is_protein_or_gene(ent):
	if ent in ['Q8054', 'Q7187']:
		return True

	query = """
	PREFIX wd: <https://www.wikidata.org/wiki/>
	PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
	PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
	PREFIX schema: <http://www.schema.org/>

	ASK {{
		wd:{} (wdt:P31|wdt:P279)* ?protein_or_gene .
		filter (?protein_or_gene in (wd:Q8054, wd:Q7187)) .
	}}
	""".format(ent)
	# print(query)
	sparql.setQuery(query)
	sparql.method = 'GET'
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	if results["boolean"]:
		return True

def is_related_to_taxon(ent):
	if ent in ['Q8054', 'Q7187']:
		return True
	query = """
	PREFIX wd: <https://www.wikidata.org/wiki/>
	PREFIX wdt: <https://www.wikidata.org/wiki/Property:>
	PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
	PREFIX schema: <http://www.schema.org/>

	ASK {{
		wd:{} ?x ?y .
		?y (wdt:P31|wdt:P279)* ?z.
		filter (?z in (wd:Q7239, wd:Q21871294)) .
	}}
	""".format(ent)
	# print(query)
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
		wd:{} (wdt:P31|wdt:P279) ?protein_or_gene .
		filter (?protein_or_gene in (wd:Q8054, wd:Q7187)) .
	}}
	""".format(ent)
	# print(query)
	sparql.setQuery(query)
	sparql.method = 'GET'
	sparql.setReturnFormat(JSON)
	results = sparql.query().convert()
	if results["boolean"]:
		return True


def find_subgraph(entity_matches):
	dfs = {}
	
	groups = {x[1]: [] for x in entity_matches}   
	spoke = {}
	
	for ent in entity_matches:
		idd = ent[0]
		groups[ent[1]] += [[idd, ent[2]]]
		
		dfs[idd], spoke[idd] = find_neighbours(idd)
		
	taxon_present = False
	for x in entity_matches:
		if is_taxon(x[0]):
			taxon_present = True
	
	protein_or_gene = False
	
	for ent, match in groups.items():
		score = 0
		count = 0
		for m in match:
			if is_related_to_taxon(m[0]):
				score += 1
			count += 1
		if score / count > 0.8:
			protein_or_gene = True



		# if is_protein_or_gene(match[0][0]):
		#	 protein_or_gene_temp = True
		#	 for m in match:
		#		 if m[1] == match[0][1] and not is_protein_or_gene(m[0]):
		#			 protein_or_gene_temp = False
		#	 if protein_or_gene_temp:
		#		 protein_or_gene = True

	if not taxon_present and protein_or_gene:
		groups["no_taxon_so_assumed_human"] = [["Q15978631", 50.]]
		
	return dfs, groups, spoke


def total_path_length(combination, G):
	total_spl_length = 0
	for c1 in combination:
		for c2 in combination:
			spl = nx.shortest_path_length(G, source = 'https://www.wikidata.org/wiki/' + c1, target = 'https://www.wikidata.org/wiki/' + c2)
			total_spl_length += spl

	return total_spl_length

def min_total_path_length(combinations, G):
	min_total_path_length = 1000
	max_comb = max([x[1] for x in combinations])
	filtered = []

	all_combs = []

	for comb in combinations:
		# print(comb)
		try:
			tpl = (total_path_length(comb[0], G) + 0.001) / ((comb[1] / max_comb))
		except:
			tpl = (len(comb[0]) * 2 + 1 + 0.001) / ((comb[1] / max_comb))
		# print(tpl)
		if tpl < min_total_path_length:
			filtered = [comb[0]]
			min_total_path_length = tpl
		elif tpl == min_total_path_length:
			filtered += [comb[0]]
			min_total_path_length = tpl

		all_combs.append([comb, tpl])

	return filtered, all_combs

def final_entities(combinations, G, groups, tdfs, spokes):
	# print(combinations)	
	filtered_entities, all_combs = min_total_path_length(combinations, G)

	final_entities = []
	final_filtered_entities = []
	# print(tdfs)
	if len(filtered_entities) >= 2:
		max_score = 0
		for ent_set in filtered_entities:
			score = sum([len(tdfs[x]) for x in ent_set if x in tdfs.keys()])
			if score > max_score:
				final_filtered_entities = ent_set
				max_score = score
	else:
		final_filtered_entities = filtered_entities[0]

	# print(final_filtered_entities)
	# for filtered_entities_set in filtered_entities:
	new_tdfs = {}
	new_spokes = {}
	for k, l in zip(groups.keys(), final_filtered_entities):
		if k != "no_taxon_so_assumed_human":
			if [l, k] not in final_entities:
				final_entities.append([l, k])
				new_tdfs[l] = tdfs[l]
				new_spokes[l] = spokes[l]
	return final_entities, new_tdfs, new_spokes

def find_final_entities(entity_matches, verbose = True):
	if verbose:
		print("Finding SubGraph Edges")
	tdfs, groups, spokes = find_subgraph(entity_matches)

	# print(groups)
	tdf = pd.concat(list(tdfs.values()))
	if verbose:
		print("Making SubGraph")
	G = nx.from_pandas_edgelist(tdf, "1", "3", edge_attr='2', create_using=nx.MultiGraph())
	combinations = [p for p in itertools.product(*groups.values())]
	# print(combinations)
	combinations = [[[x[0] for x in c], sum([x[1] for x in c])] for c in combinations]
	# print(combinations)
	if verbose:
		print("Filtering Entities")
		
	return final_entities(combinations, G, groups, tdfs, spokes)



