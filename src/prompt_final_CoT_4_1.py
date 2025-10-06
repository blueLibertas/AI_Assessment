from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

OPENAI_API_KEY = ""

# Initialize OpenAI
chat = ChatOpenAI(model="gpt-4.1", temperature=0.3, openai_api_key = OPENAI_API_KEY)


# Define desired data structure.
class Evaluation(BaseModel):
    criterion: str = Field(description="The used evaluation criterion")
    score: str = Field(description="Numeric score (0 to 5) of the text")
    reasoning: str = Field(description="Detailed reasoning for the score")
    strength: str = Field(description="Strength of the learner's summary")
    improvement: str = Field(description="Actionable improvement suggestion")    

# Define output parser
output_parser = JsonOutputParser(pydantic_object=Evaluation)
format_instructions = output_parser.get_format_instructions()
# print(format_instructions)

# Prompt
def build_prompt(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary):
    return f"""
You are evaluating a learner's summary based ONLY on **{criterion}**.

Definition:
{definition}

Score Guide:
{score_guide}

1. Think step by step and analyze the learner's summary vs. expert summary and key concepts internally.
2. DO NOT show these reasoning steps.
3. After reasoning, output ONLY JSON in this format:
{{
  "criterion": "{criterion}",
  "score": <integer 0-5>,
  "reasoning": "<summarized reasoning>",
  "strength": "<strengths>",
  "improvement": "<actionable improvement suggestion>"
}}

Inputs:
- Full Material: {learning_material}
- Expert Summary: {expert_summary}
- Key Concepts: {", ".join(key_concepts)}
- Learner Summary: {learner_summary}

{format_instructions}

"""
def evaluate_text(criterion: str, definition: str, score_guide: str, learning_material: str, expert_summary: str, key_concepts: list, learner_summary: str) -> dict:
    """
    Evaluates a learner's summary based on the given criterion.

    Args:
        criterion (str): The evaluation criterion.
        definition (str): The definition of the criterion.
        score_guide (str): The scoring rubric/guide.
        learning_material (str): The original learning material.
        expert_summary (str): The instructor/expert summary.
        key_concepts (list): Key concepts identified by the instructor.
        learner_summary (str): The learner's summary to evaluate.

    Returns:
        dict: A dictionary with keys: criterion, score, reasoning, strength, improvement.
    """
    full_prompt = build_prompt(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary)
    formatted_prompt = [ {"role": "user", "content": full_prompt} ]
    response = chat.invoke(formatted_prompt)
    return output_parser.parse(response.content)

# Role and Objective
# Instructions
## Sub-categories for more detailed instructions
# Reasoning Steps
# Output Format
# Examples
## Example 1
# Context
# Final instructions and prompt to think step by step
