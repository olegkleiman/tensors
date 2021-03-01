import tensorflow as tf
import tensorflow_datasets as tfds

print(tfds.list_builders())

ds, info = tfds.load("mnist", split="train", shuffle_files=True, with_info=True)
assert isinstance(ds, tf.data.Dataset)
prefetched_ds = ds.batch(32).prefetch(1)

examples = prefetched_ds.take(1)  # Only take a single example
for example in examples:
    print(list(example.keys()))
    image = example["image"]
    label = example["label"]
    print(image.shape, label)

tfds.benchmark(prefetched_ds, batch_size=32)
tfds.benchmark(prefetched_ds, batch_size=32)  # Second epoch should be much faster due to auto-caching

tfds.show_examples(ds, info)
