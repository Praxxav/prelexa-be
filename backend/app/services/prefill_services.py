import json
import logging
import asyncio
from typing import Dict
from fastapi import HTTPException
from backend.db.database import db
from backend.app.agent.prefiller import prefiller_agent
from backend.app.models.models import PrefillRequest , GenerateQuestionsRequest
from backend.app.utils.schemas import TemplateOut
from backend.app.agent.question_generator import question_generator_agent



async def prefill_variables_from_query(request: PrefillRequest):
    """
    
    Uses an LLM to pre-fill template variables based on the initial user query.
    """
    template = await db.template.find_unique(where={"id": request.template_id}, include={"variables": True})
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")

    # Prepare the variables for the agent prompt
    variables_for_prompt = [
        {"key": v.key, "label": v.label, "description": v.description}
        for v in template.variables
    ]

    try:
        detected_variables = await prefiller_agent.process({
            "query": request.query,
            "variables_json": json.dumps(variables_for_prompt, indent=2)
        })

        return {"message": "Prefill successful.", "query": request.query, "detected_variables": detected_variables}

    except Exception as e:
        logging.error(f"Error during prefill for template {request.template_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to prefill variables from query.")

async def generate_questions(request: GenerateQuestionsRequest):
    """
    Generates human-friendly questions for required variables that are not yet filled.
    """
    template = await db.template.find_unique(where={"id": request.template_id}, include={"variables": True})
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")

    missing_vars = [
        var for var in template.variables 
        if var.required and var.key not in request.filled_variables
    ]

    # Create a list of tasks to generate questions concurrently
    question_generation_tasks = []
    for var in template.variables:
        if var.required and var.key not in request.filled_variables:
            task = question_generator_agent.process({
                "label": var.label,
                "description": var.description
            })
            question_generation_tasks.append(task)

    # Execute all question generation tasks in parallel
    generated_questions = await asyncio.gather(*question_generation_tasks)

    # Combine the results with the variable keys
    questions_to_ask = [
        {"key": var.key, "question": question, "example": var.example}
        for var, question in zip(missing_vars, generated_questions)
    ]

    return {"missing_variables_questions": questions_to_ask}
