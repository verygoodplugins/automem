# FalkorDB with persistence and backup support
FROM falkordb/falkordb:latest

# Add backup script
COPY .railway/backup-falkordb.sh /usr/local/bin/backup-falkordb.sh
RUN chmod +x /usr/local/bin/backup-falkordb.sh

# Configure persistence
ENV REDIS_ARGS="--save 900 1 --save 300 10 --save 60 10000 --appendonly yes --dir /data"

# Expose ports
EXPOSE 6379

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD redis-cli ping || exit 1

# Volume for persistent data
VOLUME ["/data"]

CMD ["redis-server", "--loadmodule", "/usr/lib/redis/modules/libgraphcontext.so"]
