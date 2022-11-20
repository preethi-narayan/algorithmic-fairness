# algorithmic-fairness
Incorporating fairness-enhancing interventions into machine learning pipelines, observing the effect of hyperparameter tuning on performance, in terms of both accuracy and fairness
## Objectives
Understand that different notions of fairness correspond to points of view of different
stakeholders, and are often mutually incompatible. <br>
Gain hands-on experience with incorporating fairness-enhancing interventions into
machine learning pipelines. <br>
Learn about the trade-offs between fairness and accuracy. <br>
Observe the effect of hyperparameter tuning on performance, in terms of both accuracy
and fairness. <br>
### Fairness from the point of view of different stakeholders
#### Part A
Consider the COMPAS investigation by ProPublica and Northpointe’s response.
(You may also wish to consult Northpointe’s report.) For each metric A-E below, explain in 1-2
sentences which stakeholders would benefit from a model that optimizes that metric, and why.
If you believe that it would not be reasonable to optimize that metric in this case, state so. <br>
1. Accuracy <br>
2. Positive predictive value <br>
3. False positive rate <br>
4. False negative rate <br>
5. Statistical parity (demographic parity among the individuals receiving any prediction) <br>
#### Part B
Consider a hypothetical scenario in which TechCorp, a large technology company,
is hiring for data scientist roles. Alex, a recruiter at TechCorp, uses a resume screening tool
called Prophecy to help identify promising candidates. Prophecy takes applicant resumes as
input and returns them in ranked (sorted) order, with the more promising applicants (according
to the tool) appearing closer to the top of the ranked list. Alex takes the output of the Prophecy
tool under advisement when deciding whom to invite for a job interview. <br>
In their 1996 paper “Bias in computer systems”, Friedman & Nissenbaum discuss three types of
bias: A. pre-existing, B. technical, and C. emergent. We also discussed these types of bias in
class and in the “All about that Bias” comic. <br>
For each type of bias:
1. Give an example of how this type of bias may arise in the scenario described above; <br>
2. Name a stakeholder group that may be harmed by this type of bias; and <br> 
3. Propose an intervention that may help mitigate this type of bias. <br>
#### Part C
Consider a hypothetical scenario in which an admissions officer at Best University
is evaluating applicants based on 3 features: SAT score, high school GPA, and family income
bracket (low, medium, high). We discussed several equality of opportunity (EO) doctrines in
class and in the “Fairness and Friends” comic: formal, substantive / luck egalitarian, and
substantive / Rawlsian. <br>
1. In a selection procedure that is fair according to formal EO, which of these features
would the admissions officer use? Briefly justify your answer. <br>
2. Suppose that income-based differences are observed in applicants’ SAT scores: the
median SAT score is lower for applicants from low-income families, as compared to
those from medium- and high-income families. Which EO doctrine(s) is/are consistent
with the goal of correcting such differences in the applicant pool? Briefly justify your
answer. <br>
3. Describe an applicant selection procedure that is fair according to luck-egalitarian EO. <br>
### Fairness-enhancing interventions in machine learning pipelines
In this part of the assignment you will use AIF360 to incorporate fairness-enhancing
interventions into binary classification pipelines. The ASCIncome dataset has been preloaded from Folktables. Sex has been selected as the
sensitive attribute to analyze throughout this question. <br>
For the below tasks, split your data into 80% training and 20% test. Report all results on the
withheld test dataset. <br>
You will evaluate performance using the following metrics: <br>
(i) Overall accuracy <br>
(ii) Accuracy for the privileged group <br>
(iii) Accuracy for the unprivileged group <br>
(iv) Disparate Impact <br>
(v) False positive rate difference <br>
#### Part A 
Train a baseline random forest model to predict income. Use the
hyperparameters in the provided notebook. Report performance on the metrics listed
above on the test set. Discuss your results in the report.
#### Part B
Consider Disparate Impact Remover (DI-Remover), a pre-processing
fairness-enhancing intervention by Feldman et al., 2015 (here) that is implemented in
AIF360. This algorithm provides a parameter called the repair level that controls the
trade-off between fairness and accuracy. In this question, you will measure the impact of
the repair level on fairness and accuracy.
Transform the original dataset using DI-Remover with five different values of the
repair level: [0, 0.25, 0.5, 0.75, 1]. Train a random forest model on each transformed
dataset using the same hyperparameters and train/test split proportions that you used in
part (a). Report the same five metrics again for each trained model.
Discuss in your report how these results compare with the metrics from the baseline
random forest model in part (a), paying particular attention to the impact of repair level.
You may wish to plot each metric against the repair level.
#### Part C
Train a model using the Prejudice Remover in-processing technique by
Kamishima et al. 2012 (link) that is implemented in AIF360. This algorithm provides a
parameter called eta, which controls the fairness regularization weight. Use the values
[0.01, 0.1, 1] for the eta parameter. Plot both the accuracy and disparate impact as you
adjust this parameter and discuss the results.
Discuss in your report how the effect of the eta parameter compares to what you
observed for DI-Remover. (Remember: Prejudice Remover is not a pre-processing
method that is combined with an existing Random Forest model. It’s a different model
altogether, which fits a Logistic Regression under the hood.)
