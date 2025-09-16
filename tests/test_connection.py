from falkordb import FalkorDB
from qdrant_client import QdrantClient
import os

# Test FalkorDB connection
falkor = FalkorDB(
    host=os.getenv('RAILWAY_PUBLIC_DOMAIN', 'localhost'),
    port=6379
)
print("✅ FalkorDB connected!")

# Test Qdrant connection  
qdrant = QdrantClient(
    host=os.getenv('QDRANT_HOST'),
    port=6333,
    api_key=os.getenv('QDRANT_API_KEY'),
    https=True
)
print("✅ Qdrant connected!")
print(f"Collections: {qdrant.get_collections()}")

# Create memory graph
graph = falkor.select_graph('memories')
graph.query("CREATE (:Memory {id: 'test', content: 'System initialized'})")
print("✅ Memory graph created!")