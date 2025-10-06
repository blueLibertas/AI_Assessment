import prompt_final_CoT_4_1 as CoT4
import prompt_final_CoT_5 as CoT5
import prompt_final_nonCoT_4_1 as nCoT4
import prompt_final_nonCoT_5 as nCoT5
import prompt_final_SR_4_1 as SR4
import prompt_final_SR_5 as SR5
import os
import pandas as pd


criterions = ["Content Quality", "Content Coverage", "Content Coherence", "Argument"]
definitions = ["The degree to which ideas in the summary are related to the topic",
               "The degree to which central ideas from the article are clearly expressed",
               "The degree to which ideas in the summary are related to one another and coherently organized",
               "The degree to which the summary states and elaborates a clear claim"]
score_guides = [
"""0: Most of the ideas in the summary and argument are not central to the topic, not expressed clearly, or are vague or repetitive.
1: Many of the ideas in the summary and argument relate to the topic, but only a few of them are central to the topic, which may be due to vagueness, repetition, lack of clarity, or failure to express central ideas.
2: About half the ideas in the summary and argument are expressed clearly and are central to the topic, but about half the ideas do not meet this combined criterion of clarity and centrality, which may be due to vagueness, repetition, lack of clarity, or failure to identify central ideas.
3: About half the ideas in the summary and argument are expressed clearly and are central to the topic, and there is little to no vagueness or repetition. However, some ideas are either unclear, or not central to the topic, or some combination.
4: Most of the ideas in the summary and argument are expressed clearly and are central to the topic; there is little to no vagueness or repetition. However, a few ideas are either unclear or not central to the topic.
5: All or nearly all the ideas in the summary and argument are related to the topic, most of them are central to it and all or nearly all are expressed clearly, with little or no vagueness or repetition.
""",
"""0: Most of the central ideas from the article(s) are not expressed clearly in the summary and argument, and ideas from the article(s) that are included are expressed in a way that is unclear, vague, or repetitive.
1: Some of the central ideas from the article(s) are expressed clearly in the summary and argument, but many of the central ideas from the article(s) are missing, or are included in a way that is unclear, vague, or repetitive. Most ideas from the article(s) that are expressed clearly in the summary and argument are not central to the topic.
2: Many of the central ideas from the article(s) are expressed clearly in the summary and argument, but many of the central ideas from the article(s) are missing, or are included in a way that is unclear, vague, or repetitive. Many ideas from the article that are expressed clearly in the summary and argument are not central to the topic.
3: Most of the central ideas from the article(s) are expressed clearly in the summary and argument. The remaining ideas from the article that are expressed in the summary and argument are either not central, not clear, vague or repetitive.
4: Most of the central ideas from the article(s) are expressed clearly in the summary and argument. Nearly all ideas from the article(s) expressed in the summary and argument are related to the topic, and are expressed clearly, with little vagueness or repetition.
5: All or nearly all of the central ideas from the article are expressed clearly in the summary and argument. Very few of the ideas from the article(s) are expressed in a way that is unclear, vague or repetitive.
""",
"""0: The ideas expressed in the summary and argument are not easy to follow, and do not relate well to one another.
1: Some of the ideas expressed in the summary and argument relate well to one another, but most do not relate well to one another, and are not easy to follow.
2: Many of the ideas expressed in the summary and argument relate well to one another, making it fairly easy to follow much of the discussion. But many of the ideas expressed in the summary and argument do not relate well to one another, so it is difficult to form a coherent understanding of the whole.
3: Most of the ideas expressed in the summary and argument relate well to one another, and the discussion as a whole is fairly easy to follow. However, some of the ideas do not relate well and, as a result, part of the discussion is hard to follow.
4: Most of the ideas expressed in the summary and argument relate well to one another, and the discussion as a whole is easy to follow. A few ideas seem out of place or less well integrated into the overall organization.
5: All or nearly all of the ideas expressed in the summary and argument relate well to one another, so the discussion as a whole flows well from one idea to the next, and the overall organization is very coherent.
""",
"""0: Essay responds to the topic in some way but does not state a claim on the topic.
1: Essay states a claim, but no reasons are given to support the claim, or the reasons given are unrelated to or inconsistent with the claim, or they are incoherent.
2: Essay states a clear claim and gives one or two reasons to support the claim, but the reasons are not explained or supported in any coherent way. The reasons may be limited plausibility, and inconsistencies may be present.
3: Essay states a claim and gives reason(s) to support the claim, plus some explanation or elaboration of the reasons. The reasons though are not enough explanation of the information provided. There may be some inconsistencies, irrelevant information, or problems with organization and clarity.
4: Essay states a clear claim and gives reasons to support the claim. The reasons are explained clearly and logically, with few minor inconsistencies. Organization of the essay is generally good but is missing a concluding statement, or there are inconsistencies or irrelevances that weaken the argument.
5: Meets the criteria for previous level. In addition, the essay is generally well organized, includes a concluding statement. The writing is clear and logical, and irrelevances that would weaken the argument.
""",]

data = {
    "ContentQuality": {"score": [], "reasoning": [], "strength": [], "improvement": []},
    "ContentCoverage": {"score": [], "reasoning": [], "strength": [], "improvement": []},
    "ContentCoherence": {"score": [], "reasoning": [], "strength": [], "improvement": []},
    "Argument": {"score": [], "reasoning": [], "strength": [], "improvement": []}
}
mapping = {0: "ContentQuality", 1: "ContentCoverage", 2: "ContentCoherence", 3: "Argument"}

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
os.chdir(parent_dir)

f = open("data/Chapter10Evaluation.txt", "r", encoding='utf-8')
lm_85 = f.read()
f.close()

f = open("data/Chapter12Learning Analytics.txt", "r", encoding='utf-8')
lm_106 = f.read()
f.close()

expert85 = 'Evaluation is a fundamental component of the instructional design. Evaluation is the process of determining merit, worth, and value of things. The types of evaluation include confirmatory evaluation, formative evaluation, and summative evaluation. For example, formative evaluation supports the process of improvement, focusing on learner ability. Summative evaluation focuses on the overall effectiveness, usefulness, or worth of the instruction. This chapter introduces several evaluation models. Stufflebeam proposes the CIPP model that stands for context, input, process, and product evaluation. CIPP model influenced program planning, program structuring, implementation decisions. In the CIPP model, an evaluator often participates in a project as a member of the project team. From a broad perspective, Rossi views that evaluation can include needs assessment, theory assessment, implementation assessment, impact, and efficiency assessment. Chen proposes theory-driven evaluation in which evaluators and stakeholders work together. The important role of an evaluator is to help articulate, evaluate, and improve the program theory including an action model and change model. Kirkpatrick suggests that training evaluation should exam four levels of the outcomes including reaction, learning, behavior, and business results. Brinkerhoff emphasizes the use of success case to evaluate a program. He suggests that an organization can gain profits by applying knowledge learned from success cases. Lastly, Patton views that the use of evaluation findings is critical, and thus his evaluation model focuses on producing evaluation use. The utility of evaluation is judged by the degree of use. The use of evaluation findings can increase when stakeholders become active participants in the evaluation process.'
expert106 = 'Learning analytics involves collecting and exploring data sets to search for meaningful patterns. Data is collected and analyzed to support various decision-making across educational institutions. The goal of learning analytics is to improve the learning experience. To this end, learning analytics use tools for extracting, tracking, and analyzing learner behaviors and performance. The outcomes include descriptive analytics that describes the state of learning, diagnostic analytics that attempts to understand why things happened, predictive analytics that attempts to describe what will happen next, and prescriptive analytics that suggest solutions to specific issues. The data used for learning analytics may come from national databases, institutional data, learning systems, and instructors. The results of learning analytics can be used for dashboards, institutional tools, and database tools to improve learning performance. For example, the outcomes of learning analytics can drive formative feedback for student success and recommendations for design improvement. There are sociocultural issues and legal issues around the use of these tools, such as protection of personal data.'

keyConcepts85 = ["process", "training evaluation", "use", "evalutation model", "cipp model", "stakeholder", "success case", "evaluation finding", "evaluation", "summative evaluation", "theory-driven evaluation", "formative evaluation", "evaluator"]
keyConcepts106 = ["datum", "various decision-making", "descriptive anlaytic", "recommendation", "sociocultural issue", "design improvement", "pedictive analytic", "tool", "performance", "learning analytic", "learner behavior", "analytic", "outcome"]

# Each learner summary

df = pd.read_excel('./data/grades_with_summary.xlsx', sheet_name='WithSummary')


### import prompt_final_CoT_4_1 as CoT4
### import prompt_final_CoT_5mini as CoT5
### import prompt_final_nonCoT_4_1 as nCoT4
## import prompt_final_nonCoT_5mini as nCoT5
### import prompt_final_SR_4_1 as SR4
# import prompt_final_SR_5mini as SR5

for idx in range(len(df)):
    learner_summary = df.iloc[idx]['summary']
    assignmentID = df.iloc[idx]['AssignmentID']
    # print(df.iloc[idx]['AssignmentID']==85)
    # print(f"#{idx:3d}...", end="\t")
    # For each criteria
    for i in range (len(criterions)):
        # print(f"{i:3d}...", end="\t")

        if (df.iloc[idx]['AssignmentID'] == 85):
            # print(85, end='\t')
            result = SR5.evaluate_text(criterion = criterions[i], 
                                    definition = definitions[i], 
                                    score_guide = score_guides[i], 
                                    learning_material = lm_85, 
                                    expert_summary = expert85, 
                                    key_concepts = keyConcepts85, 
                                    learner_summary = learner_summary)
        
        else:
            # print(106, end='\t')
            result = SR5.evaluate_text(criterion = criterions[i], 
                                    definition = definitions[i], 
                                    score_guide = score_guides[i], 
                                    learning_material = lm_106, 
                                    expert_summary = expert106, 
                                    key_concepts = keyConcepts106, 
                                    learner_summary = learner_summary)

        key = mapping.get(i, "Argument")
        data[key]["score"].append(result["score"])
        data[key]["reasoning"].append(result["reasoning"])
        data[key]["strength"].append(result["strength"])
        data[key]["improvement"].append(result["improvement"])
        
    if idx % 5 == 0:
        print("Evaluating....{}".format(idx))
        # print(f"CRITERION: {result['criterion']}")
        # print(f"- Score: {result['score']}")
        # print(f"- Reasoning: {result['reasoning']}")
        # print(f"- Strength: {result['strength']}")
        # print(f"- Improvement: {result['improvement']}")
        # print("="*50)


# Convert collected data into DataFrame columns
results_df = pd.DataFrame({
    "ContentQuality_Score": data["ContentQuality"]["score"],
    "ContentQuality_Reasoning": data["ContentQuality"]["reasoning"],
    "ContentQuality_Strength": data["ContentQuality"]["strength"],
    "ContentQuality_Improvement": data["ContentQuality"]["improvement"],

    "ContentCoverage_Score": data["ContentCoverage"]["score"],
    "ContentCoverage_Reasoning": data["ContentCoverage"]["reasoning"],
    "ContentCoverage_Strength": data["ContentCoverage"]["strength"],
    "ContentCoverage_Improvement": data["ContentCoverage"]["improvement"],

    "ContentCoherence_Score": data["ContentCoherence"]["score"],
    "ContentCoherence_Reasoning": data["ContentCoherence"]["reasoning"],
    "ContentCoherence_Strength": data["ContentCoherence"]["strength"],
    "ContentCoherence_Improvement": data["ContentCoherence"]["improvement"],

    "Argument_Score": data["Argument"]["score"],
    "Argument_Reasoning": data["Argument"]["reasoning"],
    "Argument_Strength": data["Argument"]["strength"],
    "Argument_Improvement": data["Argument"]["improvement"]
})

# Merge new columns into original df
final_df = pd.concat([df, results_df], axis=1)
final_df.to_excel("data/SR5_2.xlsx", index=False)
