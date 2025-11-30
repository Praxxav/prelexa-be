from contextlib import asynccontextmanager
from prisma import Prisma

# Create a single Prisma instance
db = Prisma()

async def connect_db():
    """Connect to the Prisma database"""
    if not db.is_connected():
        await db.connect()
        print("✅ Connected to Prisma database")

async def disconnect_db():
    """Disconnect from the Prisma database"""
    if db.is_connected():
        await db.disconnect()
        print("🔌 Disconnected from Prisma database")

# FastAPI lifespan context
@asynccontextmanager
async def lifespan(app):
    """Manages Prisma DB connection lifecycle"""
    await connect_db()
    yield
    await disconnect_db()
