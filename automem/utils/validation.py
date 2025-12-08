"""Runtime validation utilities for AutoMem configuration."""


def validate_vector_dimensions(qdrant_client=None):
    """
    Validate that Qdrant collection vector dimension matches VECTOR_SIZE config.
    
    Args:
        qdrant_client: Optional QdrantClient instance. If None, validation is skipped.
    
    Raises:
        ValueError: If collection dimension doesn't match VECTOR_SIZE, with migration instructions.
    """
    from automem.config import VECTOR_SIZE, COLLECTION_NAME
    
    if qdrant_client is None:
        return  # No client available, skip validation
    
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        collection_dim = collection_info.config.params.vectors.size
        
        if collection_dim != VECTOR_SIZE:
            raise ValueError(
                f"\n{'='*70}\n"
                f"VECTOR DIMENSION MISMATCH\n"
                f"{'='*70}\n"
                f"Qdrant collection '{COLLECTION_NAME}': {collection_dim}d\n"
                f"Config VECTOR_SIZE: {VECTOR_SIZE}d\n\n"
                f"Your Qdrant collection uses {collection_dim}-dimensional vectors,\n"
                f"but your config expects {VECTOR_SIZE}-dimensional vectors.\n\n"
                f"{'Option 1: Keep existing embeddings (recommended)'}\n"
                f"  Set VECTOR_SIZE={collection_dim} in your .env file\n\n"
                f"{'Option 2: Migrate to ' + str(VECTOR_SIZE) + 'd embeddings'}\n"
                f"  1. Backup:  python scripts/backup_automem.py\n"
                f"  2. Migrate: python scripts/reembed_embeddings.py\n"
                f"  3. Verify:  Check collection dimension in Qdrant\n"
                f"  4. Update:  Add VECTOR_SIZE={VECTOR_SIZE} to .env\n\n"
                f"See docs/MIGRATIONS.md for detailed migration guide.\n"
                f"{'='*70}\n"
            )
    except Exception as e:
        # Collection doesn't exist yet (first run) - this is fine
        if isinstance(e, AttributeError):
            # Likely running with a stubbed client (tests); skip validation
            return
        error_msg = str(e).lower()
        if "not found" not in error_msg and "doesn't exist" not in error_msg:
            # Re-raise unexpected errors
            raise
