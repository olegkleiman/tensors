# Importing to tensoflow.js

The official Google tutorial on this topic is [here](https://www.tensorflow.org/js/tutorials/conversion/import_keras).

Some notes to this tutorial:

1. Converting .pb to TF.js model

   `tensorflowjs_converter --input_format=tf_saved_model ./saved/pow/1/ ./saved/pow/js/`

2. When converting `SavedModel` to TF.js model, the produced JSON is re-assembles the signature "serving\_default" from .pb:

{% tabs %}
{% tab title="mode.json" %}
```javascript
{
    "format": "graph-model",
    "generatedBy": "2.4.1",
    "convertedBy": "TensorFlow.js Converter v3.5.0",
    "signature": {
        "inputs": {
            "x:0": {
                "name": "x:0",
                "dtype": "DT_FLOAT",
                "tensorShape": {}
            },
            "y:0": {
                "name": "y:0",
                "dtype": "DT_FLOAT",
                "tensorShape": {}
            }
        },
        "outputs": {
            "Identity:0": {
                "name": "Identity:0",
                "dtype": "DT_FLOAT",
                "tensorShape": {}
            }
        }
    },
    "modelTopology": {
        "node": [
            {
                "name": "x",
                "op": "Placeholder",
                "attr": {
                    "shape": {
                        "shape": {}
                    },
                    "dtype": {
                        "type": "DT_FLOAT"
                    }
                }
            },
            {
                "name": "y",
                "op": "Placeholder",
                "attr": {
                    "shape": {
                        "shape": {}
                    },
                    "dtype": {
                        "type": "DT_FLOAT"
                    }
                }
            },
            {
                "name": "PartitionedCall/Pow",
                "op": "Pow",
                "input": [
                    "x",
                    "y"
                ],
                "attr": {
                    "T": {
                        "type": "DT_FLOAT"
                    }
                }
            },
            {
                "name": "Identity",
                "op": "Identity",
                "input": [
                    "PartitionedCall/Pow"
                ],
                "attr": {
                    "T": {
                        "type": "DT_FLOAT"
                    }
                }
            }
        ],
        "library": {},
        "versions": {
            "producer": 561
        }
    },
    "weightsManifest": [
        {
            "paths": [],
            "weights": []
        }
    ]
}
```
{% endtab %}

{% tab title="saved\_model\_cli show" %}
```text
pow> saved_model_cli show --dir ./1 --all
2021-04-25 13:14:00.884613: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_x:0
    inputs['y'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: serving_default_y:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_0'] tensor_info:
        dtype: DT_FLOAT
        shape: ()
        name: PartitionedCall:0
  Method name is: tensorflow/serving/predict

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          x: TensorSpec(shape=(), dtype=tf.float32, name='x')
        Argument #2
          y: TensorSpec(shape=(), dtype=tf.float32, name='y')

C:\Users\olegk\PycharmProjects\tensors\saved\pow>
```
{% endtab %}
{% endtabs %}

In particular, when the custom model discussed at Section "Saving regular functions", based on regular function, is converted to TF.js format \(model.json\), the code to perform predict looks like:

```javascript
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = './model/model.json';
const model = await loadGraphModel(MODEL_URL);
const x = tf.tensor(5.)
const y = tf.tensor(2.)
model.predict(    
    { 
        'x' : tf.tensor(5.),
        'y' : tf.tensor(2.) 
    }
).print();
```







