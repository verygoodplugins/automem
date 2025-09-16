# app.py - Memory Service API
from flask import Flask, request, jsonify
from falkordb import FalkorDB
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Log startup
logger.info("🧠 FalkorDB Memory API Starting...")

# Initialize FalkorDB
try:
    falkor_host = os.getenv('RAILWAY_PRIVATE_DOMAIN', 'localhost')
    logger.info(f"🔗 Connecting to FalkorDB at {falkor_host}:6379")
    falkordb = FalkorDB(host=falkor_host, port=6379)
    memory_graph = falkordb.select_graph('memories')
    logger.info("✅ FalkorDB connection established")
except Exception as e:
    logger.error(f"❌ FalkorDB connection failed: {e}")
    falkordb = None
    memory_graph = None

# Initialize Qdrant
try:
    qdrant_url = os.getenv('QDRANT_URL', 'https://your-cluster.qdrant.io')
    logger.info(f"🔗 Connecting to Qdrant at {qdrant_url}")
    qdrant = QdrantClient(
        url=qdrant_url,
        api_key=os.getenv('QDRANT_API_KEY'),
        port=6333,
        https=True
    )
    logger.info("✅ Qdrant connection established")
except Exception as e:
    logger.error(f"❌ Qdrant connection failed: {e}")
    qdrant = None

# Ensure Qdrant collection exists
COLLECTION_NAME = "memories"
if qdrant:
    try:
        # Check if collection exists, create if not
        collections = qdrant.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"📦 Creating Qdrant collection: {COLLECTION_NAME}")
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        else:
            logger.info(f"✅ Qdrant collection '{COLLECTION_NAME}' already exists")
    except Exception as e:
        logger.warning(f"⚠️ Qdrant collection setup failed: {e}")
        qdrant = None

@app.route('/health', methods=['GET'])
def health():
    logger.info("🩺 Health check requested")
    
    falkor_status = "connected" if falkordb else "disconnected"
    qdrant_status = "connected" if qdrant else "disconnected"
    
    health_data = {
        "status": "healthy" if (falkordb and qdrant) else "degraded",
        "falkordb": falkor_status,
        "qdrant": qdrant_status,
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv('FLASK_ENV', 'production')
    }
    
    logger.info(f"📊 Health status: {health_data}")
    return jsonify(health_data)

@app.route('/memory', methods=['POST'])
def store_memory():
    """Store a new memory in both FalkorDB and Qdrant"""
    data = request.json
    
    # Generate unique ID
    memory_id = str(uuid.uuid4())
    
    # Extract data
    content = data.get('content')
    embedding = data.get('embedding', np.random.rand(768).tolist())  # Mock embedding
    tags = data.get('tags', [])
    importance = data.get('importance', 0.5)
    
    # Store in FalkorDB (graph structure)
    memory_graph.query("""
        CREATE (m:Memory {
            id: $id,
            content: $content,
            timestamp: $time,
            importance: $importance,
            tags: $tags
        })
    """, {
        'id': memory_id,
        'content': content,
        'time': datetime.now().isoformat(),
        'importance': importance,
        'tags': tags
    })
    
    # Store in Qdrant (vector embeddings)
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=memory_id,
            vector=embedding,
            payload={
                "content": content,
                "tags": tags,
                "importance": importance,
                "timestamp": datetime.now().isoformat()
            }
        )]
    )
    
    return jsonify({
        "status": "success",
        "memory_id": memory_id,
        "message": "Memory stored in both FalkorDB and Qdrant"
    })

@app.route('/recall', methods=['GET'])
def recall_memories():
    """Recall memories using both semantic search and graph traversal"""
    query_text = request.args.get('query', '')
    query_embedding = request.args.get('embedding', '').split(',')
    limit = int(request.args.get('limit', 5))
    
    results = []
    
    # If we have an embedding, search Qdrant
    if query_embedding and len(query_embedding) == 768:
        vector_results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=[float(x) for x in query_embedding],
            limit=limit
        )
        
        # Get memory IDs from vector search
        memory_ids = [hit.id for hit in vector_results]
        
        # Enrich with graph relationships from FalkorDB
        for memory_id in memory_ids:
            graph_result = memory_graph.query("""
                MATCH (m:Memory {id: $id})
                OPTIONAL MATCH (m)-[r:RELATES_TO]-(related:Memory)
                RETURN m, collect({
                    content: related.content,
                    strength: r.strength
                }) as relations
            """, {'id': memory_id})
            
            for row in graph_result.result_set:
                results.append({
                    'memory': row[0],
                    'relations': row[1]
                })
    
    # Also do text-based graph search
    if query_text:
        text_results = memory_graph.query("""
            MATCH (m:Memory)
            WHERE m.content CONTAINS $query
            RETURN m
            ORDER BY m.importance DESC
            LIMIT $limit
        """, {'query': query_text, 'limit': limit})
        
        for row in text_results.result_set:
            results.append({'memory': row[0], 'relations': []})
    
    return jsonify({
        "status": "success",
        "results": results,
        "count": len(results)
    })

@app.route('/associate', methods=['POST'])
def create_association():
    """Create an association between two memories"""
    data = request.json
    memory1_id = data.get('memory1_id')
    memory2_id = data.get('memory2_id')
    strength = data.get('strength', 0.5)
    relation_type = data.get('type', 'RELATES_TO')
    
    result = memory_graph.query(f"""
        MATCH (m1:Memory {{id: $id1}})
        MATCH (m2:Memory {{id: $id2}})
        CREATE (m1)-[r:{relation_type} {{strength: $strength}}]->(m2)
        RETURN r
    """, {
        'id1': memory1_id,
        'id2': memory2_id,
        'strength': strength
    })
    
    return jsonify({
        "status": "success",
        "message": f"Association created between {memory1_id} and {memory2_id}"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8001))  # Use 8001 as default
    logger.info(f"🚀 Starting Flask API on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug to avoid restart conflicts
