from fastapi import Header, HTTPException

async def get_org_id(x_org_id: str = Header(None)):
    if x_org_id is None:
        raise HTTPException(status_code=400, detail="Missing x-org-id header")
    return x_org_id
