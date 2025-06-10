#!/bin/bash
docker run --name pgvector-db -e POSTGRES_PASSWORD=mysecret -p 5432:5432 -d ankane/pgvector