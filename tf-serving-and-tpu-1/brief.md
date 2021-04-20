---
description: >-
  Serving is a process of taking a trained and saved model and making it
  available to serve prediction requests. To this extent, TF Serving provided as
  a Docker container accomplishes this mission.
---

# Brief

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
* `-e MODEL_NAME=resnet` : telling TF Serving to load the model named "resnet".
* `-t tensorflow/serving` : run a Docker contained based on the image "tensorflow/serving"

Now, this newly created container should be listed among the others:

```bash
$ docker ps
CONTAINER ID   IMAGE                COMMAND                  CREATED        STATUS          PORTS                              NAMES
dc5c61633bec   tensorflow/serving   "/usr/bin/tf_serving…"   20 hours ago   Up 46 minutes   8500/tcp, 0.0.0.0:8501->8501/tcp   tfserving_resnet
e3d3f15c4675   registry:2           "/entrypoint.sh /etc…"   5 months ago   Up 21 hours     0.0.0.0:5000->5000/tcp             registry
```

