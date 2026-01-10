# Fix CI Failures - AutoMem/AutoHub

You are fixing CI failures for the AutoMem/AutoHub project. This is a Python Flask application with:
- Flask + SSE (Server-Sent Events)
- FalkorDB (graph database)
- Qdrant (vector database)
- Docker/Docker Compose deployment
- pytest for testing

## Your Task

Analyze the test failures and CodeRabbit comments below, then make the **minimal changes** needed to fix them.

## Rules

1. **Be surgical** - Only change what's necessary to fix the specific issue
2. **Don't refactor** - Resist the urge to "improve" adjacent code
3. **Match existing style** - Follow the patterns already in the codebase
4. **Preserve API** - Don't change endpoints or response formats unless the fix requires it
5. **Keep tests passing** - Your fix should not break other tests

## Common Fixes

- **Import errors**: Check relative imports, ensure modules exist
- **Type hints**: Add proper typing for function signatures
- **Async issues**: Ensure proper async/await usage
- **Database errors**: Check connection handling and query syntax
- **API errors**: Validate request/response handling

## What NOT to Do

- Don't add new features
- Don't refactor working code
- Don't change Docker configuration unless required
- Don't modify environment variable handling unnecessarily
- Don't add excessive logging or comments
