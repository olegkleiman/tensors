---
description: This pages provides a step-by-step tutorial on how to serve the saved TF model
---

# Tutorial

TS Serving provided by Google as Docker image both for CPU and GPU. Currently, only images for Linux or macOS are available.

```bash
$ docker pull tensorflow/serving
```

TF Serving uses the `SavedModel` format for its ML models. \(The process of saving the model into this format is described in section [5. Saving Models ](https://oleg-kleyman.gitbook.io/eml/saving-models/brief)of this workspace\). 

If you haven't yet the saved model, you can get, say,  [ResNet, ](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035)from here:

```bash
$ mkdir /tmp/resnet
$ curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz
```

Now that we have the SavedModel in `/tmp/resnet` directory, serving it with Docker is easy as running it pointing to this directory:

```bash
$ docker run -p 8501:8501 --name tfserving_resnet \
--mount type=bind,source=/tmp/resnet,target=/models/resnet \
-e MODEL_NAME=resnet -t tensorflow/serving &
```

Breaking down the command line arguments, we have

* `-p 8501:8501` : publishing the container's port 8501 \(where TF Serving responds to REST API requests\) to the host's port 8501
* _`-`-`name tfserving_resnet` : get the container the name "tfserving_\_resnet"
* `--mount type=bind, source=/tmp/resnet, target=/models/resnet` : mounting the host local directory `/tmp/resnet` to the container's directory `/models/resnet` so TF Serving can read the model inside the container
* `-e MODEL_NAME=resnet` : environment variable telling TF Serving to load the model named "resnet".
* `-t tensorflow/serving` : run a Docker contained based on the image "tensorflow/serving"

Logs from this run display the open end-points: `0.0.0.0:8500` for gRPC and `localhost:8501` for REST HTTP:

```python
2021-04-20 16:30:23.898742: I tensorflow_serving/model_servers/server.cc:371] 
Running gRPC ModelServer at 0.0.0.0:8500 ...

[warn] getaddrinfo: address family for nodename not supported

[evhttp_server.cc : 238] NET_LOG: Entering the event loop ...

2021-04-20 16:30:23.902030: I tensorflow_serving/model_servers/server.cc:391] 
Exporting HTTP/REST API at:localhost:8501 ...
```

Now, this newly created container should be listed among the others:

```bash
$ docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED        STATUS          PORTS                              NAMES
dc5c61633bec   tensorflow/serving   "/usr/bin/tf_serving…"   20 hours ago   Up 46 minutes   8500/tcp, 0.0.0.0:8501->8501/tcp   tfserving_resnet
e3d3f15c4675   registry:2           "/entrypoint.sh /etc…"   5 months ago   Up 21 hours     0.0.0.0:5000->5000/tcp             registry
```

As said, this Docker app may serve both gRPC and REST requests. Specifically, the following code issues HTTP POST request to localhost according to the following schema:

http://{HOST}:{PORT}/v1/models/{MODEL\_NAME}:{VERB}

`{MODEL_NAME}` here is the same as specified in `-e MODEL_NAME=` when starting Docker app.

`{VERB}` one of the predefined  _predict_, _classify_ or _regress_

```python
SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'
# Compose a JSON Predict request (send JPEG image in base64).
jpeg_bytes = base64.b64encode(dl_request.content).decode('utf-8')
predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes

response = requests.post(SERVER_URL, data=predict_request)
prediction = response.json()['predictions'][0]
prediction['classes']
```

The full client script is [here](https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py).

The general form of JSON payload for gRPC request is:

```javascript
{"signature_name": "<string>",
"instances": <value>
}
```

where `"signature_name"` is usually set to `"serving_default"` or maybe omitted. This signature name is taken from the structure of SavedModel described in [Brief](https://oleg-kleyman.gitbook.io/eml/saving-models/brief) section of "Saving models" part of this workspace.

### TPU

This is a huge topic that should be discovered in separate works. Briefly, there are non-casual similarity exists between the Docker containers running TF Serving and managed by Kubernetes and general TPU architecture. TPU is also accessed by exposed gRPC end-points. 

