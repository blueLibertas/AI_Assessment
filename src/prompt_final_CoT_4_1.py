from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import random

OPENAI_API_KEY = ""

# Initialize OpenAI
chat = ChatOpenAI(model="gpt-4.1", temperature=0.1, openai_api_key=OPENAI_API_KEY)

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

def build_prompt(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary):
    """Improved CoT prompt with explicit reasoning steps, examples, and validation"""
    
    # Criterion-specific reasoning templates
    reasoning_templates = {
        "Content Quality": """
        **Step 1: Identify Topic-Related Ideas**
        - Extract all ideas from the learner summary
        - Check if each idea relates to the topic (evaluation/learning analytics)
        - Note clarity of expression for each idea
        
        **Step 2: Assess Clarity and Specificity**
        - Rate each idea on clarity (specific vs vague)
        - Check for repetition or redundancy
        - Compare idea clarity against expert summary
        
        **Step 3: Count and Score**
        - Count ideas that are both relevant AND clearly expressed
        - Calculate percentage of clear, relevant ideas
        - Apply rubric scoring based on this percentage
        """,
        "Content Coverage": """
        **Step 1: Extract Central Ideas from Expert Summary**
        - List all central concepts from expert summary
        - Note how clearly each is expressed
        
        **Step 2: Check Learner Coverage**
        - Identify which central ideas appear in learner summary
        - Assess clarity of expression for each covered idea
        - Note missing central ideas
        
        **Step 3: Calculate Coverage Score**
        - Count central ideas present in learner summary
        - Rate clarity of expression for present ideas
        - Apply coverage rubric based on completeness and clarity
        """,
        "Content Coherence": """
        **Step 1: Map Idea Relationships**
        - Identify how ideas connect in learner summary
        - Check for logical flow and transitions
        - Note disconnected or poorly connected ideas
        
        **Step 2: Compare Organization**
        - Compare learner organization to expert summary structure
        - Assess overall readability and flow
        
        **Step 3: Apply Coherence Scoring**
        - Rate quality of idea relationships
        - Apply rubric based on logical flow and organization
        """,
        "Argument": """
        **Step 1: Identify Main Claim**
        - Find the main thesis/claim in learner summary
        - Check if claim is clear and specific
        
        **Step 2: Check Supporting Evidence**
        - Look for reasons supporting the claim
        - Assess quality and relevance of evidence
        - Check for logical consistency
        
        **Step 3: Evaluate Conclusion**
        - Check if conclusion follows from evidence
        - Assess overall argument strength
        - Apply argument rubric criteria
        """
    }
    
    # Concrete examples from the data
    examples = {
        "Content Quality": """
        **Example - Content Quality:**
        Expert Summary: "Evaluation involves determining merit, worth, and value through systematic processes like formative and summative evaluation."
        Learner Summary: "Evaluation is about checking if things work good and finding problems."
        
        **Reasoning Process:**
        1. Expert ideas: merit/worth/value, systematic processes, formative/summative types
        2. Learner ideas: basic "checking" concept, vague "work good", finding problems
        3. Learner ideas are relevant but lack clarity and specificity
        4. Missing systematic approach and specific evaluation types
        5. Only 1/3 ideas are both relevant and clearly expressed
        6. Score: 2/5 (some relevance but lacks depth and precision)
        """,
        "Content Coverage": """
        **Example - Content Coverage:**
        Expert Summary: "Learning analytics uses data to improve education through descriptive, diagnostic, predictive, and prescriptive analytics."
        Learner Summary: "Learning analytics helps teachers understand students better."
        
        **Reasoning Process:**
        1. Expert central ideas: data use, education improvement, 4 types of analytics
        2. Learner mentions: data use, education improvement
        3. Missing: specific analytics types (descriptive, diagnostic, etc.)
        4. Present ideas are clear but incomplete
        5. Coverage: 2/4 central ideas (50%)
        6. Score: 2/5 (partial coverage, missing key concepts)
        """,
        "Content Coherence": """
        **Example - Content Coherence:**
        Expert Summary: "Evaluation is the process of determining merit, worth, and value. It includes formative and summative evaluation, with systematic approaches like CIPP model."
        Learner Summary: "Evaluation is important. CIPP model has four parts. Formative evaluation helps improve things. Summative evaluation checks results."
        
        **Reasoning Process:**
        1. Ideas present: evaluation definition, CIPP model, formative/summative
        2. Flow: definition → CIPP → formative → summative (logical progression)
        3. Transitions: clear connections between ideas
        4. Organization: follows logical sequence
        5. Score: 4/5 (good flow, clear connections, minor gaps)
        """,
        "Argument": """
        **Example - Argument:**
        Expert Summary: "Learning analytics improves education by using data to identify at-risk students and provide targeted interventions."
        Learner Summary: "Learning analytics is good because it helps students. Teachers can see who needs help and give them support."
        
        **Reasoning Process:**
        1. Claim: "Learning analytics is good"
        2. Reason: "helps students"
        3. Evidence: "teachers can see who needs help and give support"
        4. Logic: Reason supports claim, evidence supports reason
        5. Conclusion: Implied but not explicit
        6. Score: 3/5 (clear claim and reason, basic evidence, missing explicit conclusion)
        """
    }
    
    reasoning_template = reasoning_templates.get(criterion, reasoning_templates["Content Quality"])
    example = examples.get(criterion, examples["Content Quality"])
    
    return f"""
You are an expert educational assessor. Follow this systematic evaluation process:

**Criterion: {criterion}**
**Definition: {definition}**

**EXPLICIT REASONING STEPS:**
{reasoning_template}

**CONCRETE EXAMPLE:**
{example}

**NOW EVALUATE THIS LEARNER SUMMARY:**

**Expert Summary:** {expert_summary}
**Key Concepts:** {", ".join(key_concepts)}
**Learner Summary:** {learner_summary}

**Scoring Rubric:**
{score_guide}

**VALIDATION STEPS:**
1. Double-check your score against the rubric descriptions
2. Verify your reasoning addresses the specific criterion
3. Ensure evidence supports your score
4. Consider alternative interpretations
5. Confirm score aligns with both strengths and improvements identified

**Your Analysis:**
Follow the reasoning process above, provide specific evidence for your score, and validate your assessment.

{format_instructions}
"""
def evaluate_with_self_consistency(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary, num_samples=3):
    """Generate multiple reasoning paths and select most consistent score"""
    scores = []
    reasonings = []
    strengths = []
    improvements = []
    
    for i in range(num_samples):
        # Add slight variation to prompt for different perspectives
        base_prompt = build_prompt(criterion, definition, score_guide, 
                                 learning_material, expert_summary, 
                                 key_concepts, learner_summary)
        
        # Add perspective variation
        perspective_variations = [
            "Focus on precision and accuracy in your assessment.",
            "Emphasize completeness and thoroughness in your evaluation.",
            "Prioritize clarity and coherence in your analysis."
        ]
        
        varied_prompt = base_prompt + f"\n\n**Evaluation Perspective {i+1}:** {perspective_variations[i % len(perspective_variations)]}"
        
        formatted_prompt = [{"role": "user", "content": varied_prompt}]
        response = chat.invoke(formatted_prompt)
        result = output_parser.parse(response.content)
        
        scores.append(int(result["score"]))
        reasonings.append(result["reasoning"])
        strengths.append(result["strength"])
        improvements.append(result["improvement"])
    
    # Select most frequent score (self-consistency)
    most_common_score = max(set(scores), key=scores.count)
    score_count = scores.count(most_common_score)
    
    # Use reasoning from the most common score
    most_common_index = scores.index(most_common_score)
    
    return {
        "criterion": criterion,
        "score": str(most_common_score),
        "reasoning": f"Consensus score {most_common_score} from {score_count}/{num_samples} evaluations. {reasonings[most_common_index]}",
        "strength": strengths[most_common_index],
        "improvement": improvements[most_common_index]
    }

def evaluate_text(criterion: str, definition: str, score_guide: str, learning_material: str, expert_summary: str, key_concepts: list, learner_summary: str) -> dict:
    """
    Evaluates a learner's summary based on the given criterion using improved CoT with self-consistency.

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
    return evaluate_with_self_consistency(criterion, definition, score_guide, 
                                        learning_material, expert_summary, 
                                        key_concepts, learner_summary)

# Improved Chain of Thought Implementation Features:
# 1. Explicit reasoning steps for each criterion
# 2. Concrete examples from actual data
# 3. Self-consistency decoding (3 evaluation perspectives)
# 4. Validation steps to ensure accuracy
# 5. Criterion-specific reasoning templates
# 6. Lower temperature (0.1) for more consistent results
