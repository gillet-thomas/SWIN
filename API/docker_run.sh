#Docker script to deploy
docker build -t swin .
docker run -p 8000:8000 swin
