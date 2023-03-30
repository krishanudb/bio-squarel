## Contents

This folder contains the training and testing data of BioASQ-WD

The annotated questions are divided into 4 files:
1. Simple Questions Training Set --> simple_questions_train.csv
2. Constrained Simple Questions Training Set --> constrained_simple_questions_train.csv
3. Simple Questions Test Set --> simple_questions_test.csv
4. Constrained Simple Questions Test Set --> constrained_simple_questions_test.csv

These splits do not contain explicit Dev Set. But users can create their own random splits from the Training Set and use it as a Dev Set.

Each of these files contain the natural language questions, their corresponding SPARQL queries, the entities mentioned in the question, the relation that can answer the question and so on.