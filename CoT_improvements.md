# Chain of Thought Prompting Improvements for Educational Assessment

## Current Issues with Chain of Thought Implementation

### **Why CoT is Performing Poorly:**

1. **Vague Instructions**: The current CoT prompt only says "Think step by step and analyze internally" - this is too generic and doesn't provide specific guidance on what steps to follow.

2. **No Examples**: The prompts lack few-shot examples showing proper reasoning chains for educational assessment tasks.

3. **Hidden Reasoning**: The instruction "DO NOT show these reasoning steps" prevents the model from actually demonstrating its reasoning process, which defeats the purpose of CoT.

4. **No Structure**: There's no framework for how to systematically compare learner vs. expert summaries across the four criteria.

5. **Missing Self-Consistency**: No mechanism to validate or cross-check the reasoning process.

## Research-Backed Improvements

### **1. Few-Shot Examples with Explicit Reasoning Chains**

Add concrete examples showing the step-by-step evaluation process:

```python
def build_improved_cot_prompt(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary):
    return f"""
You are an expert educational assessor evaluating learner summaries. Follow this step-by-step process:

**Step 1: Analyze the Expert Summary**
- Identify key concepts and main ideas in the expert summary
- Note the structure and organization patterns

**Step 2: Examine the Learner Summary**  
- Identify what concepts the learner included
- Assess how clearly they're expressed
- Check organization and coherence

**Step 3: Compare Against Criterion**
- For {criterion}: {definition}
- Apply the scoring rubric systematically

**Step 4: Score and Justify**
- Assign score based on rubric
- Provide specific evidence for the score

**Example for Content Quality:**
Expert Summary: "Evaluation involves determining merit, worth, and value through systematic processes like formative and summative evaluation."
Learner Summary: "Evaluation is about checking if things work good and finding problems."

Reasoning:
1. Expert covers key concepts: merit/worth/value, systematic processes, formative/summative
2. Learner only mentions basic concept of "checking if things work" 
3. Missing systematic approach and specific evaluation types
4. Score: 2/5 (some relevance but lacks depth and precision)

Now evaluate this learner summary:

Criterion: {criterion}
Definition: {definition}
Score Guide: {score_guide}

Expert Summary: {expert_summary}
Key Concepts: {", ".join(key_concepts)}
Learner Summary: {learner_summary}

Provide your step-by-step analysis and final evaluation.
"""
```

### **2. Self-Consistency Decoding**

Implement multiple reasoning paths and select the most consistent:

```python
def evaluate_with_self_consistency(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary, num_samples=3):
    """Generate multiple reasoning paths and select most consistent score"""
    scores = []
    reasonings = []
    
    for i in range(num_samples):
        # Add slight variation to prompt
        varied_prompt = build_improved_cot_prompt(criterion, definition, score_guide, 
                                                learning_material, expert_summary, 
                                                key_concepts, learner_summary) + f"\n\nApproach this evaluation from perspective {i+1}."
        
        result = evaluate_single_path(varied_prompt)
        scores.append(int(result["score"]))
        reasonings.append(result["reasoning"])
    
    # Select most frequent score
    most_common_score = max(set(scores), key=scores.count)
    return {
        "score": str(most_common_score),
        "reasoning": f"Consensus from {num_samples} evaluations: {most_common_score}",
        "strength": "Multiple evaluation perspectives considered",
        "improvement": "Based on consistent assessment across approaches"
    }
```

### **3. Progressive-Hint Prompting**

Break down the evaluation into smaller, guided steps:

```python
def progressive_hint_prompt(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary):
    return f"""
**Phase 1: Concept Identification**
First, identify the key concepts in the expert summary: {expert_summary}
Now identify concepts in the learner summary: {learner_summary}

**Phase 2: Quality Assessment** 
For {criterion}: {definition}
Rate the quality of concept expression in the learner summary (1-5 scale).

**Phase 3: Coverage Analysis**
Compare concept coverage between expert and learner summaries.
What percentage of key concepts are present?

**Phase 4: Final Scoring**
Based on your analysis above, assign a score using this rubric:
{score_guide}

Provide your final evaluation with specific evidence.
"""
```

### **4. Rubric-Specific Reasoning Templates**

Create specific reasoning frameworks for each criterion:

```python
def criterion_specific_cot(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary):
    if criterion == "Content Quality":
        reasoning_template = """
        1. Identify topic-related ideas in learner summary
        2. Assess clarity of expression for each idea
        3. Check for vagueness or repetition
        4. Compare against expert summary's clarity
        5. Apply rubric scoring criteria
        """
    elif criterion == "Content Coverage":
        reasoning_template = """
        1. List central ideas from expert summary
        2. Check which central ideas appear in learner summary
        3. Assess how clearly each central idea is expressed
        4. Identify missing or unclear central ideas
        5. Apply coverage rubric criteria
        """
    elif criterion == "Content Coherence":
        reasoning_template = """
        1. Map idea relationships in learner summary
        2. Check logical flow between ideas
        3. Identify disconnected or poorly connected ideas
        4. Compare organization to expert summary
        5. Apply coherence rubric criteria
        """
    elif criterion == "Argument":
        reasoning_template = """
        1. Identify the main claim in learner summary
        2. Check for supporting reasons/evidence
        3. Assess logical consistency of argument
        4. Evaluate conclusion quality
        5. Apply argument rubric criteria
        """
    
    return f"""
    You are evaluating a learner's summary based on **{criterion}**.
    
    Definition: {definition}
    Score Guide: {score_guide}
    
    Follow this reasoning process:
    {reasoning_template}
    
    Expert Summary: {expert_summary}
    Key Concepts: {", ".join(key_concepts)}
    Learner Summary: {learner_summary}
    
    Provide your step-by-step analysis and final evaluation.
    """
```

### **5. Validation and Error Checking**

Add self-validation steps:

```python
def add_validation_steps(prompt):
    return prompt + """

**Validation Steps:**
1. Double-check your score against the rubric descriptions
2. Verify your reasoning addresses the specific criterion
3. Ensure evidence supports your score
4. Consider alternative interpretations
5. Confirm score aligns with both strengths and improvements identified
"""
```

### **6. Complete Improved CoT Implementation**

Here's a complete improved version that combines all the research-backed improvements:

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import random

OPENAI_API_KEY = ""

# Initialize OpenAI
chat = ChatOpenAI(model="gpt-4", temperature=0.1, openai_api_key=OPENAI_API_KEY)

class Evaluation(BaseModel):
    criterion: str = Field(description="The used evaluation criterion")
    score: str = Field(description="Numeric score (0 to 5) of the text")
    reasoning: str = Field(description="Detailed reasoning for the score")
    strength: str = Field(description="Strength of the learner's summary")
    improvement: str = Field(description="Actionable improvement suggestion")

output_parser = JsonOutputParser(pydantic_object=Evaluation)
format_instructions = output_parser.get_format_instructions()

def build_improved_cot_prompt(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary):
    """Improved CoT prompt with explicit reasoning steps and examples"""
    
    # Criterion-specific reasoning templates
    reasoning_templates = {
        "Content Quality": """
        1. Identify all topic-related ideas in the learner summary
        2. Assess clarity of expression for each idea (specific, concrete vs vague)
        3. Check for repetition or redundancy
        4. Compare idea clarity against expert summary
        5. Count ideas that meet both relevance and clarity criteria
        6. Apply rubric scoring based on percentage of clear, relevant ideas
        """,
        "Content Coverage": """
        1. Extract all central ideas from the expert summary
        2. Check which central ideas appear in the learner summary
        3. Assess how clearly each central idea is expressed
        4. Identify missing central ideas
        5. Check for unclear or vague expressions of central ideas
        6. Apply coverage rubric based on completeness and clarity
        """,
        "Content Coherence": """
        1. Map relationships between ideas in learner summary
        2. Check logical flow and transitions between ideas
        3. Identify disconnected or poorly connected ideas
        4. Compare organization structure to expert summary
        5. Assess overall readability and flow
        6. Apply coherence rubric based on idea relationships
        """,
        "Argument": """
        1. Identify the main claim or thesis in learner summary
        2. Check for supporting reasons or evidence
        3. Assess logical consistency and validity
        4. Evaluate quality of conclusion
        5. Check for counterarguments or limitations
        6. Apply argument rubric based on claim clarity and support
        """
    }
    
    # Few-shot examples for each criterion
    examples = {
        "Content Quality": """
        **Example:**
        Expert: "Evaluation involves determining merit, worth, and value through systematic processes like formative and summative evaluation."
        Learner: "Evaluation is about checking if things work good and finding problems."
        
        Reasoning:
        1. Expert ideas: merit/worth/value, systematic processes, formative/summative types
        2. Learner ideas: basic "checking" concept, vague "work good", finding problems
        3. Learner ideas are relevant but lack clarity and specificity
        4. Missing systematic approach and specific evaluation types
        5. Only 1/3 ideas are both relevant and clearly expressed
        6. Score: 2/5 (some relevance but lacks depth and precision)
        """,
        "Content Coverage": """
        **Example:**
        Expert: "Learning analytics uses data to improve education through descriptive, diagnostic, predictive, and prescriptive analytics."
        Learner: "Learning analytics helps teachers understand students better."
        
        Reasoning:
        1. Expert central ideas: data use, education improvement, 4 types of analytics
        2. Learner mentions: data use, education improvement
        3. Missing: specific analytics types (descriptive, diagnostic, etc.)
        4. Present ideas are clear but incomplete
        5. Coverage: 2/4 central ideas (50%)
        6. Score: 2/5 (partial coverage, missing key concepts)
        """
    }
    
    reasoning_template = reasoning_templates.get(criterion, reasoning_templates["Content Quality"])
    example = examples.get(criterion, examples["Content Quality"])
    
    return f"""
You are an expert educational assessor. Evaluate the learner summary using this systematic approach:

**Criterion: {criterion}**
**Definition: {definition}**

**Step-by-Step Reasoning Process:**
{reasoning_template}

{example}

**Now evaluate this learner summary:**

**Expert Summary:** {expert_summary}
**Key Concepts:** {", ".join(key_concepts)}
**Learner Summary:** {learner_summary}

**Scoring Rubric:**
{score_guide}

**Your Analysis:**
1. Follow the reasoning process above
2. Provide specific evidence for your score
3. Identify concrete strengths and improvements
4. Validate your score against the rubric

{format_instructions}
"""

def evaluate_with_self_consistency(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary, num_samples=3):
    """Generate multiple reasoning paths and select most consistent score"""
    scores = []
    reasonings = []
    strengths = []
    improvements = []
    
    for i in range(num_samples):
        # Add slight variation to prompt
        base_prompt = build_improved_cot_prompt(criterion, definition, score_guide, 
                                              learning_material, expert_summary, 
                                              key_concepts, learner_summary)
        
        # Add perspective variation
        varied_prompt = base_prompt + f"\n\n**Evaluation Perspective {i+1}:** Focus on {'precision' if i==0 else 'completeness' if i==1 else 'clarity'} in your assessment."
        
        formatted_prompt = [{"role": "user", "content": varied_prompt}]
        response = chat.invoke(formatted_prompt)
        result = output_parser.parse(response.content)
        
        scores.append(int(result["score"]))
        reasonings.append(result["reasoning"])
        strengths.append(result["strength"])
        improvements.append(result["improvement"])
    
    # Select most frequent score
    most_common_score = max(set(scores), key=scores.count)
    score_count = scores.count(most_common_score)
    
    return {
        "criterion": criterion,
        "score": str(most_common_score),
        "reasoning": f"Consensus score {most_common_score} from {score_count}/{num_samples} evaluations. Reasoning: {reasonings[scores.index(most_common_score)]}",
        "strength": strengths[scores.index(most_common_score)],
        "improvement": improvements[scores.index(most_common_score)]
    }

def evaluate_text(criterion, definition, score_guide, learning_material, expert_summary, key_concepts, learner_summary):
    """Main evaluation function with improved CoT prompting"""
    return evaluate_with_self_consistency(criterion, definition, score_guide, 
                                        learning_material, expert_summary, 
                                        key_concepts, learner_summary)
```

## Research Evidence Supporting These Improvements

### **Key Research Findings:**

1. **Few-shot examples** significantly improve CoT performance (Wei et al., 2022)
2. **Self-consistency** reduces error rates by 20-30% (Wang et al., 2022)  
3. **Structured reasoning templates** improve consistency (Kojima et al., 2022)
4. **Progressive guidance** helps maintain focus on complex tasks (Zhou et al., 2023)
5. **Criterion-specific frameworks** improve domain-specific performance (Zelikman et al., 2022)

### **Expected Improvements:**

- **Consistency**: Self-consistency decoding should reduce score variance
- **Accuracy**: Few-shot examples and structured reasoning should improve alignment with human evaluators
- **Reliability**: Validation steps should catch common errors
- **Transparency**: Explicit reasoning steps make evaluation process auditable

### **Implementation Notes:**

1. **Temperature**: Use lower temperature (0.1) for more consistent reasoning
2. **Model Selection**: GPT-4 performs better than GPT-3.5 for complex reasoning tasks
3. **Validation**: Consider human validation of a sample to ensure improvements
4. **Monitoring**: Track consistency metrics across different evaluation runs

This improved approach addresses the core issues with the current CoT implementation while incorporating research-backed best practices for educational assessment tasks.
