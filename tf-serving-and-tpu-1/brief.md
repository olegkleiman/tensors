---
description: >-
  Serving is a process of taking a trained and saved model and making it
  available to serve prediction requests. To this extent, TF Serving provided as
  a Docker container accomplishes this mission.
---

# Brief

TS Serving provided by Google as Docker image both for CPU and GPU. Currently, only images for Linux or MacOS are available.

```text
$ docker pull tensorflow/serving
```

TF Serving uses the `SavedModel` format for its ML models. Saving the model into this format is described in section 5. Saving Models of this workspace.

