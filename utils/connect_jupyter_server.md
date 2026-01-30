1. Run the docker image with this flags:
`docker run --rm -it --gpus device=$CUDA_VISIBLE_DEVICES -p 50000:50000 llm bash`  
    In this way we connect the port 50000 of the docker container with the 50000 of the host (the Server)

2. Inside the container run 
`pip3 install --upgrade jupyterlab`

3. Once intalled run:
`jupyter-lab --ip 0.0.0.0 --allow-root --port 50000`  
`--ip` tells to accept request outside localhost  
`--allow-root` permits to start the jupyter server as root  
`--port` is just the port

4. Now the magic. Because you have the jupyter notebook on 50000 of the server but you can not access it from your PC.

5. You need a ssh tunnel, you have to map a port of the server on a port of your local machine using ssh. To do so use:
`ssh -N -f -L 50000:localhost:50000 -p 37335 USERNAME@137.204.107.***`  
ssh -N -f -L 50000:localhost:50000 -p 37335 molfetta@137.204.107.153


6. From the container terminal you will see 2 links. Copy and paste the second one. WOOOOW it works (I Hope)