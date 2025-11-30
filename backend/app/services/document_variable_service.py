from backend.db.database import db

class DocumentVariableService:
    @staticmethod
    async def create_variable(document_id: str, name: str, value: str = None, confidence: float = None):
        """Create a new variable for a document"""
        variable = await db.documentvariable.create(
            data={
                "documentId": document_id,
                "name": name,
                "value": value,
                "confidence": confidence,
            }
        )
        return variable

    @staticmethod
    async def get_variables(document_id: str):
        """Fetch all variables for a given document"""
        try:
            print(f"📄 Fetching variables for document: {document_id}")
            variables = await db.documentvariable.find_many(
                where={"documentId": document_id},
                order={"createdAt": "desc"},  # optional sorting
            )
            print(f"✅ Found {len(variables)} variables")
            return variables
        except Exception as e:
            print(f"❌ Error fetching variables for document {document_id}: {e}")
            raise e

    @staticmethod
    async def update_variable(variable_id: str, value: str = None):
        """Update a specific variable's value"""
        variable = await db.documentvariable.update(
            where={"id": variable_id},
            data={"value": value},
        )
        return variable

    @staticmethod
    async def delete_variable(variable_id: str):
        """Delete a variable"""
        variable = await db.documentvariable.delete(where={"id": variable_id})
        return variable

    @staticmethod
    async def bulk_create_variables(document_id: str, variables: list):
        """Insert multiple variables at once"""
        if not variables:
            return []

        await db.documentvariable.create_many(
            data=[
                {
                    "documentId": document_id,
                    "name": v["name"],
                    "value": v.get("value", ""),
                    "confidence": v.get("confidence"),
                    "editable": v.get("editable", True),
                }
                for v in variables
            ]
        )
        return True
