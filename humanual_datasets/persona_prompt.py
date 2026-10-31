import mlflow

persona_prompt_template = """You are an expert at analyzing a {{app_name}} user behavior. You should generate a JSON object to describe user persona based a target user's responses to some contexts. The contexts ONLY provide other people' posts, and you should NOT use them to infer the target user's demographics. You should ONLY use the target user's responses to summarize the persona.

## Context and Responses:
{{comments_text}}

## Aspects to cover:

1. Demographics:
- Use explicit subfields: "age group", "gender", "location", "occupation", "nationality", "other"
- Fill with explicit info if available, otherwise "NA".

2. Interests:
- What subjects or themes do they frequently respond on?

3. Values:
- What opinions, attitudes, or worldviews are reflected in their responses?

4. Communication:
- What are their writing styles and formatting habits?

5. Statistics:
- Average / Minimum / Maximum response length (in words). Most frequent words or phrases. Variations in sentence structure and so on.

## Output (strict JSON):
{
    "analysis": <str>,
    "demographics": {
        "age group": <str>,
        "gender": <str>,
        "location": <str>,
        "occupation": <str>,
        "nationality": <str>,
        "other": <str>
    },
    "interests": <a list of 8-12 phrases>,
    "values": <a list of 8-12 phrases>,
    "communication": <a list of 8-12 phrases>,
    "statistics": <a list of 5-10 phrases>
}

## Instructions:
- [CRITICAL] You MUST always include ALL fields in the JSON output, including "demographics" with ALL its subfields. If demographic information is not explicitly mentioned in the user's responses, set all demographic fields to "NA" but still include them.
- "age group" field: Identify if the user mentioned being X years old in a response from year Y. And find the year of their last response, say Z. Then calculate their age group as (X + (Z - Y)). If no explicit age mentioned, set to "NA".
- "demographics" fields: When extracting demographics, only use explicitly mentioned information. Base your evidence on the user's responses. Do not make assumptions or guesses. If no explicit information is available, use "NA" for each field but ALWAYS include the demographics object.
- [Important!] Other fields: Ensure the phrases are specific, evidence-based, and describe comprehensive aspects of the user. You should quote parts of the user's actual responses as evidence in each phrase without metionining the example index. Avoid vague or generic phrases. Instead, reflect the user's unique traits, behaviors, or preferences.
- "analysis" field: Provide a detailed and step-by-step analysis with the evidence and your reasoning to obtain the user's demongraphics, interests, values, communication style, and statistics.

Your Output:
"""

updated_prompt = mlflow.genai.register_prompt(
    name="persona-prompt-template",  # Specify the existing prompt name
    template=persona_prompt_template,
    commit_message="Original"
)