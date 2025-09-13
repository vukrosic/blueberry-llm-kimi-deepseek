#!/bin/bash

# Simple training script
mkdir -p data_cache models logs

case ${1:-train} in
    train) docker compose up --build blueberry-train ;;
    dev) docker compose up -d blueberry-dev && docker compose exec blueberry-dev /bin/bash ;;
    tensorboard) docker compose up tensorboard ;;
    clean) docker compose down --volumes && docker system prune -f ;;
    *) echo "Usage: $0 [train|dev|tensorboard|clean]" ;;
esac
