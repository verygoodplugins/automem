FROM falkordb/falkordb:latest

EXPOSE 6379 3000

# FalkorDB runs with Redis + Browser UI
CMD ["redis-server", "--loadmodule", "/FalkorDB/bin/linux-x64-release/src/falkordb.so"]