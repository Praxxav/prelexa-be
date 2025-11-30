import os
from fastapi import UploadFile, File, HTTPException, Depends
from backend.app.agent.templatizer import templatizer_agent
from backend.app.tasks.document_tasks import extract_text_from_file
from backend.app.utils.uploads import UPLOAD_DIR
import yaml
import logging
from backend.app.utils.schemas import TemplateIn, TemplateOut
from backend.db.database import db
from backend.app.agent.bootstrap_agent import bootstrap_agent
from backend.app.models.models import FillTemplateRequest, DraftRequest
from backend.app.utils.dependencies import get_org_id


# -----------------------------------------------------------
# GET ALL TEMPLATES (ORG SCOPED)
# -----------------------------------------------------------
async def get_all_templates(org_id: str):
    templates = await db.template.find_many(
        where={"orgId": org_id},
        order={"createdAt": "desc"},
        include={"variables": True}
    )
    return [TemplateOut.from_orm(t) for t in templates]


# -----------------------------------------------------------
# SEARCH / BOOTSTRAP TEMPLATES (ORG SCOPED)
# -----------------------------------------------------------
async def find_templates(request: DraftRequest, org_id: str):

    query_words = set(request.query.lower().split())

    try:
        templates = await db.template.find_many(
            where={"orgId": org_id},
            include={"variables": True}
        )

        scored_templates = []

        for t in templates:
            score = 0
            searchable_text = " ".join(filter(None, [
                t.title,
                t.fileDescription,
                t.docType,
                t.jurisdiction,
                " ".join(t.similarityTags)
            ])).lower()

            for word in query_words:
                if word in searchable_text:
                    score += 1

            if score > 0:
                scored_templates.append({"template": t, "score": score})

        # If found
        if scored_templates:
            scored_templates.sort(key=lambda x: x["score"], reverse=True)
            return {"status": "found", "results": scored_templates}

        # Otherwise → create using bootstrap agent
        logging.info(f"No local template match. Bootstrapping: {request.query}")
        new_template_data = await bootstrap_agent.bootstrap_template(request.query)

        if not new_template_data:
            return {"status": "not_found", "message": "No templates found online or locally."}

        template_markdown = (
            new_template_data.get("template_markdown") or
            new_template_data.get("full_markdown")
        )

        if not template_markdown:
            logging.error("Bootstrap returned no markdown content.")
            return {"status": "not_found", "message": "Bootstrap returned invalid template data."}

        # Parse YAML front-matter
        if template_markdown.strip().startswith("---"):
            parts = template_markdown.split("---", 2)
            if len(parts) < 3:
                logging.error("Invalid YAML format from bootstrap.")
                return {"status": "error", "message": "Invalid bootstrap template format."}

            yaml_part = parts[1].strip()
            body_md = parts[2].strip()
        else:
            yaml_part = ""
            body_md = template_markdown

        # Parse YAML metadata
        try:
            meta = yaml.safe_load(yaml_part) if yaml_part else {}
        except yaml.YAMLError:
            meta = {}

        variables_data = meta.get("variables", [])
        if not isinstance(variables_data, list):
            variables_data = []

        # SAVE BOOTSTRAPPED TEMPLATE (FIX: INCLUDE ORG ID)
        new_template = await db.template.create(
            data={
                "title": meta.get("title") or new_template_data.get("title") or f"Template for {request.query}",
                "fileDescription": meta.get("file_description") or "",
                "jurisdiction": meta.get("jurisdiction") or "",
                "docType": meta.get("doc_type") or "",
                "similarityTags": meta.get("similarity_tags") or [],
                "bodyMd": body_md,
                "orgId": org_id,  # FIX ADDED
                "variables": {
                    "create": [
                        {
                            "key": v.get("key", f"field_{idx}"),
                            "label": v.get("label", f"Field {idx}"),
                            "description": v.get("description", ""),
                            "example": v.get("example", ""),
                            "required": v.get("required", True),
                            "type": v.get("type", "string"),
                        }
                        for idx, v in enumerate(variables_data)
                        if isinstance(v, dict) and v.get("key")
                    ]
                }
            }
        )

        return {
            "status": "bootstrapped",
            "source_url": new_template_data.get("source_url"),
            "source_title": new_template_data.get("source_title"),
            "template": new_template
        }

    except Exception as e:
        logging.error("Error in find_templates:", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------
# CREATE TEMPLATE FROM UPLOADED FILE
# -----------------------------------------------------------
async def create_template_from_upload(file: UploadFile, org_id: str):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".docx", ".txt"]:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Save temporarily
    file_path = os.path.join(UPLOAD_DIR, f"temp_{file.filename}")
    with open(file_path, "wb") as f:
        f.write(await file.read())

    text_content = await extract_text_from_file(file_path)
    os.remove(file_path)

    template_markdown = await templatizer_agent.process(text_content)

    return {
        "message": "Template extraction successful.",
        "template_markdown": template_markdown,
        "orgId": org_id,  # include in response for frontend if needed
    }


# -----------------------------------------------------------
# SAVE TEMPLATE FROM MARKDOWN (ORG SCOPED)
# -----------------------------------------------------------
async def save_template(template_data: TemplateIn, org_id: str):

    try:
        parts = template_data.template_markdown.split("---")
        if len(parts) < 3:
            raise HTTPException(status_code=400, detail="Invalid template format.")

        front_matter = parts[1]
        body_md = "---".join(parts[2:]).strip()

        try:
            parsed_yaml = yaml.safe_load(front_matter)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"YAML parse error: {e}")

        # Prepare variables
        variables_to_create = []
        for idx, v in enumerate(parsed_yaml.get("variables", [])):
            if isinstance(v, dict):
                variables_to_create.append({
                    "key": v.get("key", f"field_{idx}"),
                    "label": v.get("label", f"Field {idx}"),
                    "description": v.get("description", ""),
                    "example": v.get("example", ""),
                    "required": v.get("required", True),
                })

        # FIX: orgId NOW INCLUDED
        new_template = await db.template.create(
            data={
                "title": parsed_yaml.get("title", "Untitled Template"),
                "fileDescription": parsed_yaml.get("file_description", ""),
                "jurisdiction": parsed_yaml.get("jurisdiction", ""),
                "docType": parsed_yaml.get("doc_type", ""),
                "similarityTags": parsed_yaml.get("similarity_tags", []),
                "bodyMd": body_md,
                "orgId": org_id,  # FIX ADDED
                "variables": {"create": variables_to_create} if variables_to_create else None,
            }
        )

        return {"message": "Template saved successfully!", "template_id": new_template.id}

    except Exception as e:
        logging.error("Error saving template", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------
# FILL TEMPLATE
# -----------------------------------------------------------
async def fill_template(request: FillTemplateRequest, org_id: str):

    template = await db.template.find_unique(
        where={"id": request.template_id, "orgId": org_id}
    )
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")

    draft = template.bodyMd
    for key, value in request.variables.items():
        draft = draft.replace(f"{{{{{key}}}}}", str(value))

    return {"draft_markdown": draft}


# -----------------------------------------------------------
# GET TEMPLATE BY ID
# -----------------------------------------------------------
async def get_template_by_id(template_id: str, org_id: str):

    try:
        template = await db.template.find_unique(
            where={"id": template_id, "orgId": org_id},
            include={"variables": True}
        )

        if not template:
            raise HTTPException(status_code=404, detail="Template not found")

        return TemplateOut.from_orm(template)

    except Exception as e:
        logging.error(f"Error fetching template {template_id}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch template")
