# test_falkordb.py
from falkordb import FalkorDB
import os
from datetime import datetime

# Get Railway URL - you'll need to replace this with your actual domain
RAILWAY_URL = os.getenv('RAILWAY_STATIC_URL', 'flask-production-5fcd.up.railway.app')

try:
    # Connect to FalkorDB
    db = FalkorDB(host=RAILWAY_URL, port=6379)
    print(f"✅ Connected to FalkorDB at {RAILWAY_URL}")
    
    # Create a memory graph
    graph = db.select_graph('memories')
    
    # Create first memory
    result = graph.query("""
        CREATE (m:Memory {
            id: 'test-001',
            content: 'FalkorDB successfully deployed on Railway',
            timestamp: $time,
            importance: 0.9,
            tags: ['deployment', 'success', 'infrastructure']
        }) RETURN m
    """, {'time': datetime.now().isoformat()})
    
    print("✅ Created first memory node!")
    
    # Create a related memory
    graph.query("""
        MATCH (m1:Memory {id: 'test-001'})
        CREATE (m2:Memory {
            id: 'test-002',
            content: 'Ready to integrate with Qdrant for vector search',
            timestamp: $time,
            importance: 0.8
        })-[:RELATES_TO {strength: 0.9}]->(m1)
        RETURN m2
    """, {'time': datetime.now().isoformat()})
    
    print("✅ Created related memory with association!")
    
    # Query the graph
    results = graph.query("""
        MATCH (m:Memory)
        RETURN m.content, m.importance
        ORDER BY m.importance DESC
    """)
    
    print("\n📊 Current memories:")
    for row in results.result_set:
        print(f"  - {row[0]} (importance: {row[1]})")
    
    # Test graph traversal
    related = graph.query("""
        MATCH (m1:Memory)-[r:RELATES_TO]-(m2:Memory)
        RETURN m1.content, r.strength, m2.content
    """)
    
    print("\n🔗 Memory associations:")
    for row in related.result_set:
        print(f"  - '{row[0]}' ←→ '{row[2]}' (strength: {row[1]})")
        
    print("\n✨ FalkorDB is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure to run: railway domain")
    print("Then set RAILWAY_STATIC_URL environment variable")
