docker build --pull --rm -f "Dockerfile" -t lateresidualconnections:latest "." 
docker run -d --name lrc --rm -v $PWD:/LateResidualConnections -it lateresidualconnections
docker exec --workdir /late-residual -it lrc /bin/bash