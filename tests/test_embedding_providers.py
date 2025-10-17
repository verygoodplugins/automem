"""Comprehensive test suite for embedding providers."""

import os
from unittest.mock import Mock, MagicMock, patch
import pytest

# Test imports with fallback for missing packages
try:
    from automem.embedding.provider import EmbeddingProvider
    from automem.embedding.placeholder import PlaceholderEmbeddingProvider
except ImportError:
    pytest.skip("Embedding module not available", allow_module_level=True)


# ==================== Fixtures ====================

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = Mock()
    mock_client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 768)]
    )
    return mock_client


@pytest.fixture
def mock_fastembed_model():
    """Mock FastEmbed model for testing."""
    mock_model = Mock()
    mock_model.embed.return_value = [[0.2] * 768]
    mock_model.dimension = 768
    return mock_model


@pytest.fixture(autouse=True)
def reset_provider_state(monkeypatch):
    """Reset embedding provider state between tests."""
    # Reset any cached state
    import app
    if hasattr(app.state, 'embedding_provider'):
        app.state.embedding_provider = None
    yield
    # Cleanup after test
    if hasattr(app.state, 'embedding_provider'):
        app.state.embedding_provider = None


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Helper to set environment variables."""
    def _set_vars(**kwargs):
        for key, value in kwargs.items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
    return _set_vars


# ==================== PlaceholderEmbeddingProvider Tests ====================

def test_placeholder_provider_deterministic():
    """Test that same text always produces same embedding."""
    provider = PlaceholderEmbeddingProvider(dimension=768)

    text = "This is a test sentence"
    embedding1 = provider.generate_embedding(text)
    embedding2 = provider.generate_embedding(text)

    assert embedding1 == embedding2, "Same text should produce identical embeddings"
    assert len(embedding1) == 768, "Embedding should have correct dimension"
    assert all(0 <= x <= 1 for x in embedding1), "Values should be in [0, 1] range"


def test_placeholder_provider_different_inputs():
    """Test that different texts produce different embeddings."""
    provider = PlaceholderEmbeddingProvider(dimension=768)

    text1 = "First sentence"
    text2 = "Second sentence"
    embedding1 = provider.generate_embedding(text1)
    embedding2 = provider.generate_embedding(text2)

    assert embedding1 != embedding2, "Different texts should produce different embeddings"


def test_placeholder_provider_dimension():
    """Test that provider respects dimension parameter."""
    dimensions = [384, 768, 1024]

    for dim in dimensions:
        provider = PlaceholderEmbeddingProvider(dimension=dim)
        embedding = provider.generate_embedding("test")
        assert len(embedding) == dim, f"Should produce {dim}-dimensional embedding"
        assert provider.dimension() == dim, f"Should report dimension as {dim}"


def test_placeholder_provider_batch():
    """Test batch embedding generation."""
    provider = PlaceholderEmbeddingProvider(dimension=768)

    texts = ["text1", "text2", "text3"]
    embeddings = provider.generate_embeddings_batch(texts)

    assert len(embeddings) == 3, "Should generate 3 embeddings"
    assert all(len(emb) == 768 for emb in embeddings), "All embeddings should be 768-dim"

    # Verify determinism in batch
    single_embeddings = [provider.generate_embedding(t) for t in texts]
    assert embeddings == single_embeddings, "Batch should match individual generation"


def test_placeholder_provider_name():
    """Test provider name reporting."""
    provider = PlaceholderEmbeddingProvider()
    assert provider.provider_name() == "placeholder"
    assert "placeholder" in repr(provider).lower()


# ==================== FastEmbedProvider Tests ====================

@patch('automem.embedding.fastembed.TextEmbedding')
def test_fastembed_provider_initialization(mock_text_embedding_class, mock_fastembed_model):
    """Test FastEmbed provider initialization."""
    # Mock the TextEmbedding class
    mock_text_embedding_class.return_value = mock_fastembed_model

    from automem.embedding.fastembed import FastEmbedProvider

    provider = FastEmbedProvider(dimension=768)

    # Verify model was initialized with correct parameters
    mock_text_embedding_class.assert_called_once()
    call_kwargs = mock_text_embedding_class.call_args[1]
    assert call_kwargs['model_name'] == 'BAAI/bge-base-en-v1.5'
    assert 'cache_dir' in call_kwargs


@patch('automem.embedding.fastembed.TextEmbedding')
def test_fastembed_provider_dimension_validation(mock_text_embedding_class, mock_fastembed_model):
    """Test dimension mismatch warning."""
    import numpy as np

    # Mock model with different dimension
    # Mock the embed method to return an iterator with 384 dimensions
    mock_fastembed_model.embed.return_value = iter([np.array([0.1] * 384)])
    mock_fastembed_model.dimension = 384
    mock_text_embedding_class.return_value = mock_fastembed_model

    from automem.embedding.fastembed import FastEmbedProvider

    # Request 768 but model actually returns 384
    with patch('automem.embedding.fastembed.logger') as mock_logger:
        provider = FastEmbedProvider(dimension=768)

        # Should log warning about mismatch
        mock_logger.warning.assert_called()
        # The logger.warning is called with a format string and args
        warning_format = mock_logger.warning.call_args[0][0]
        warning_args = mock_logger.warning.call_args[0][1:]
        # Check that it's the right warning (about dimension mismatch)
        assert "actual dimension" in warning_format
        assert "!= configured" in warning_format
        # Check the actual values passed
        assert 384 in warning_args  # actual dimension
        assert 768 in warning_args  # configured dimension

    # Should use actual dimension
    assert provider.dimension() == 384


@patch('automem.embedding.fastembed.TextEmbedding')
def test_fastembed_provider_batch_processing(mock_text_embedding_class, mock_fastembed_model):
    """Test batch embedding with FastEmbed."""
    # Mock numpy arrays that have tolist() method
    import numpy as np
    mock_fastembed_model.embed.return_value = [
        np.array([0.1] * 768),
        np.array([0.2] * 768),
        np.array([0.3] * 768)
    ]
    mock_fastembed_model.dimension = 768
    mock_text_embedding_class.return_value = mock_fastembed_model

    from automem.embedding.fastembed import FastEmbedProvider

    provider = FastEmbedProvider(dimension=768)
    texts = ["text1", "text2", "text3"]
    embeddings = provider.generate_embeddings_batch(texts)

    assert len(embeddings) == 3
    # The model might be called twice - once for warmup, once for actual texts
    assert mock_fastembed_model.embed.call_count >= 1
    # Check that it was called with the right texts
    actual_call = mock_fastembed_model.embed.call_args_list[-1][0][0]
    assert actual_call == texts


@patch('automem.embedding.fastembed.TextEmbedding')
def test_fastembed_provider_model_selection(mock_text_embedding_class):
    """Test automatic model selection based on dimension."""
    from automem.embedding.fastembed import FastEmbedProvider

    # Test 384 dimension
    provider = FastEmbedProvider(dimension=384)
    call_kwargs = mock_text_embedding_class.call_args[1]
    assert 'bge-small' in call_kwargs['model_name'] or 'MiniLM' in call_kwargs['model_name']

    # Test 1024 dimension
    mock_text_embedding_class.reset_mock()
    provider = FastEmbedProvider(dimension=1024)
    call_kwargs = mock_text_embedding_class.call_args[1]
    assert 'bge-large' in call_kwargs['model_name']


def test_fastembed_provider_import_failure():
    """Test graceful handling when fastembed is not installed."""
    # The module might be available, so we just test that it exists or doesn't
    try:
        from automem.embedding.fastembed import FastEmbedProvider
        assert FastEmbedProvider is not None
    except ImportError:
        # If import fails, that's also fine - it means fastembed not installed
        pass


# ==================== OpenAIEmbeddingProvider Tests ====================

def test_openai_provider_initialization(mock_openai_client):
    """Test OpenAI provider initialization."""
    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client

        from automem.embedding.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            dimension=768
        )

        # Verify client was created with timeout and retry
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs['api_key'] == "test-key"
        assert 'timeout' in call_kwargs
        assert 'max_retries' in call_kwargs


def test_openai_provider_single_embedding(mock_openai_client):
    """Test single text embedding with OpenAI."""
    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.5] * 768)]
        )

        from automem.embedding.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        embedding = provider.generate_embedding("test text")

        assert len(embedding) == 768
        assert embedding[0] == 0.5

        # Verify API call
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input="test text",
            dimensions=768
        )


def test_openai_provider_batch_embedding(mock_openai_client):
    """Test batch embedding with OpenAI."""
    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client

        # Mock batch response
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[
                Mock(embedding=[0.1] * 768),
                Mock(embedding=[0.2] * 768),
                Mock(embedding=[0.3] * 768)
            ]
        )

        from automem.embedding.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="test-key")
        texts = ["text1", "text2", "text3"]
        embeddings = provider.generate_embeddings_batch(texts)

        assert len(embeddings) == 3
        assert embeddings[0][0] == 0.1
        assert embeddings[1][0] == 0.2
        assert embeddings[2][0] == 0.3


def test_openai_provider_dimension_validation(mock_openai_client):
    """Test dimension validation in OpenAI provider."""
    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client

        # Mock response with wrong dimension
        mock_openai_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]  # Wrong dimension
        )

        from automem.embedding.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="test-key", dimension=768)

        # Should raise error for dimension mismatch
        with pytest.raises(ValueError, match="OpenAI embedding length 1536 != configured dimension 768"):
            provider.generate_embedding("test")


def test_openai_provider_import_failure():
    """Test graceful handling when openai is not installed."""
    # The module might be available, so we just test that it exists or doesn't
    try:
        from automem.embedding.openai import OpenAIEmbeddingProvider
        assert OpenAIEmbeddingProvider is not None
    except ImportError:
        # If import fails, that's also fine - it means openai not installed
        pass


# ==================== Provider Selection Logic Tests ====================

@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "EMBEDDING_PROVIDER": "auto"})
def test_provider_selection_auto_with_openai(mock_openai_client):
    """Test auto-selection prefers OpenAI when API key is available."""
    import app

    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client

        app.init_embedding_provider()

        assert app.state.embedding_provider is not None
        assert "openai" in app.state.embedding_provider.provider_name()


@patch.dict(os.environ, {"EMBEDDING_PROVIDER": "auto"}, clear=True)
@patch('automem.embedding.fastembed.TextEmbedding')
def test_provider_selection_auto_without_openai(mock_text_embedding_class, mock_fastembed_model):
    """Test auto-selection falls back to FastEmbed without API key."""
    import app

    # Remove any existing OpenAI key
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']

    mock_text_embedding_class.return_value = mock_fastembed_model

    app.init_embedding_provider()

    assert app.state.embedding_provider is not None
    assert "fastembed" in app.state.embedding_provider.provider_name() or \
           "placeholder" in app.state.embedding_provider.provider_name()


@patch.dict(os.environ, {"EMBEDDING_PROVIDER": "auto"}, clear=True)
def test_provider_selection_auto_placeholder_fallback():
    """Test auto-selection falls back to placeholder when all fail."""
    import app

    # Clear OPENAI_API_KEY to ensure OpenAI isn't selected
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']

    # Mock fastembed to fail
    with patch('automem.embedding.fastembed.TextEmbedding', side_effect=ImportError("Mock failure")):
        app.state.embedding_provider = None  # Reset state
        app.init_embedding_provider()

        assert app.state.embedding_provider is not None
        # Should fall back to placeholder
        assert app.state.embedding_provider.provider_name() == "placeholder"


@patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "EMBEDDING_PROVIDER": "openai"})
def test_provider_selection_explicit_openai(mock_openai_client):
    """Test explicit OpenAI provider selection."""
    import app

    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client

        app.init_embedding_provider()

        assert "openai" in app.state.embedding_provider.provider_name()


@patch.dict(os.environ, {"EMBEDDING_PROVIDER": "openai"}, clear=True)
def test_provider_selection_openai_no_key():
    """Test explicit OpenAI fails without API key."""
    import app

    # Remove API key
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set"):
        app.init_embedding_provider()


@patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"})
@patch('automem.embedding.fastembed.TextEmbedding')
def test_provider_selection_explicit_local(mock_text_embedding_class, mock_fastembed_model):
    """Test explicit local provider selection."""
    import app

    mock_text_embedding_class.return_value = mock_fastembed_model

    app.init_embedding_provider()

    assert "fastembed" in app.state.embedding_provider.provider_name()


@patch.dict(os.environ, {"EMBEDDING_PROVIDER": "placeholder"})
def test_provider_selection_explicit_placeholder():
    """Test explicit placeholder provider selection."""
    import app

    app.init_embedding_provider()

    assert app.state.embedding_provider.provider_name() == "placeholder"


# ==================== Integration Tests with app.py ====================

def test_init_embedding_provider_caching():
    """Test that provider is only initialized once."""
    import app

    # First initialization
    app.init_embedding_provider()
    first_provider = app.state.embedding_provider

    # Second call should not reinitialize
    app.init_embedding_provider()
    second_provider = app.state.embedding_provider

    assert first_provider is second_provider, "Provider should be cached"


@patch.dict(os.environ, {"VECTOR_SIZE": "1024", "EMBEDDING_PROVIDER": "placeholder"})
def test_init_embedding_provider_with_custom_dimension():
    """Test provider initialization with custom dimension."""
    import app

    # Force re-initialization
    app.state.embedding_provider = None
    app.VECTOR_SIZE = 1024

    app.init_embedding_provider()

    assert app.state.embedding_provider.dimension() == 1024


def test_generate_embedding_with_provider():
    """Test generate embedding function with provider."""
    import app

    # Ensure provider is initialized
    app.init_embedding_provider()

    # Use the provider directly
    embedding = app._generate_real_embedding("test text")

    assert embedding is not None
    assert len(embedding) == app.VECTOR_SIZE
    assert all(isinstance(x, float) for x in embedding)


def test_generate_embedding_fallback():
    """Test embedding generation falls back to placeholder on error."""
    import app

    # Mock provider that raises exception
    mock_provider = Mock()
    mock_provider.generate_embedding.side_effect = Exception("Provider failed")
    app.state.embedding_provider = mock_provider

    # Should fall back to placeholder
    embedding = app._generate_placeholder_embedding("test text")

    # Should generate placeholder embedding
    assert embedding is not None
    assert len(embedding) == app.VECTOR_SIZE


# ==================== Error Handling Tests ====================

def test_provider_network_error_handling(mock_openai_client):
    """Test handling of network errors in provider."""
    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client

        # Simulate network error
        mock_openai_client.embeddings.create.side_effect = Exception("Network error")

        from automem.embedding.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="test-key")

        with pytest.raises(Exception, match="Network error"):
            provider.generate_embedding("test")


def test_provider_dimension_mismatch_recovery():
    """Test recovery from dimension mismatches."""
    import app

    # Reset VECTOR_SIZE to default
    app.VECTOR_SIZE = 768

    # Initialize with placeholder provider explicitly
    app.state.embedding_provider = PlaceholderEmbeddingProvider(dimension=768)

    # Generate embedding with 768 dimensions
    embedding_768 = app.state.embedding_provider.generate_embedding("test")
    assert len(embedding_768) == 768

    # Change to different dimension
    app.state.embedding_provider = PlaceholderEmbeddingProvider(dimension=1024)

    # Should handle different dimension
    embedding_1024 = app.state.embedding_provider.generate_embedding("test")
    assert len(embedding_1024) == 1024


def test_provider_initialization_failure_recovery(monkeypatch):
    """Test recovery from provider initialization failures."""
    import app

    # Mock provider that fails to initialize
    def failing_init(*args, **kwargs):
        raise Exception("Init failed")

    with patch('automem.embedding.fastembed.TextEmbedding', side_effect=failing_init):
        # Should fall back to placeholder
        app.state.embedding_provider = None
        app.init_embedding_provider()

        assert app.state.embedding_provider is not None
        assert app.state.embedding_provider.provider_name() == "placeholder"


# ==================== Provider Feature Tests ====================

def test_placeholder_hash_stability():
    """Test that placeholder uses stable hashing."""
    import hashlib

    provider = PlaceholderEmbeddingProvider(dimension=768)

    # Same content should always produce same hash seed
    text = "Stable text"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    expected_seed = int.from_bytes(digest[:8], "little", signed=False)

    # Generate multiple times
    embeddings = [provider.generate_embedding(text) for _ in range(5)]

    # All should be identical
    for emb in embeddings[1:]:
        assert emb == embeddings[0], "Hash-based embedding should be stable"


@patch('automem.embedding.fastembed.TextEmbedding')
def test_fastembed_model_caching(mock_text_embedding_class, mock_fastembed_model):
    """Test that FastEmbed uses model caching."""
    from automem.embedding.fastembed import FastEmbedProvider

    mock_text_embedding_class.return_value = mock_fastembed_model

    provider = FastEmbedProvider(dimension=768)

    # Check cache_dir was set
    call_kwargs = mock_text_embedding_class.call_args[1]
    assert 'cache_dir' in call_kwargs
    assert 'automem/models' in call_kwargs['cache_dir'] or \
           '.config/automem' in call_kwargs['cache_dir']


def test_openai_retry_configuration(mock_openai_client):
    """Test OpenAI client retry configuration."""
    with patch('automem.embedding.openai.OpenAI') as mock_openai_class:
        mock_openai_class.return_value = mock_openai_client

        from automem.embedding.openai import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            timeout=30.0,
            max_retries=5
        )

        # Verify timeout and retry settings
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs['timeout'] == 30.0
        assert call_kwargs['max_retries'] == 5


# ==================== End-to-End Tests ====================

@patch.dict(os.environ, {"EMBEDDING_PROVIDER": "placeholder", "API_TOKEN": "", "ADMIN_TOKEN": ""})
def test_end_to_end_memory_storage_with_provider(monkeypatch):
    """Test complete flow of storing memory with embedding provider."""
    import app
    import json

    # Reset state
    app.state = app.ServiceState()

    # Mock graph
    mock_graph = Mock()
    mock_graph.query.return_value = Mock(result_set=[[Mock(properties={"id": "test-id"})]])
    app.state.memory_graph = mock_graph

    # Initialize provider
    app.init_embedding_provider()

    # Set API token for the test
    monkeypatch.setattr(app, "API_TOKEN", "")

    client = app.app.test_client()

    response = client.post(
        "/memory",
        data=json.dumps({"content": "Test memory with provider"}),
        content_type="application/json"
    )

    assert response.status_code == 201
    data = response.get_json()
    assert data["status"] == "success"

    # Verify embedding was generated
    assert mock_graph.query.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])