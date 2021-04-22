# Automatic Differentiation

Generally speaking, if we know a value of a function at the point, it is **impossible** to find its derivative at the same point. But this becomes possible if we know some additional information about the function.

 Consider $$f(x) = x^2$$. 

At point $$x_0 = 3$$ we have $$f(x_0) = 9$$ and $$f'(x_0) = 2x|_{x_0} = 2x|_{3} = 6$$ .

What gives us an answer is an advanced knowledge of the _symbolic_ derivative $$f'(x) = (x^2)' = 2x$$ .

Once we know this rule, given an initial value $$x_0 = 3$$ and $$f(x_0) = 9$$ ,we are able to calculate the derivative. We even don't have to know the closed-form of $$f(x)$$ ! Just these two values and a symbolic derivative.

But how to get this symbolic derivative? The key point here is that at some point of time ago we had to calculate the value of the function. What exactly we did then? We had known how to square the value $$x_0$$. Or in other words, we used an operation "square" defined somewhere for us. Now let's take this operation and extend it not only to produce the calculation result but also to store the defined derivative nearby. Preferably in the predefined property of this operation class. 

Let's extend this way the operations for all [elementary ](https://en.wikipedia.org/wiki/Elementary_function)\(or even [analytic](https://en.wikipedia.org/wiki/Analytic_function)\) functions. We just know its derivatives in advance from the calculus in closed form or, worth case, we know its Taylor series. From now on we'll express these operations not in Python style, but with a help of some framework.

{% tabs %}
{% tab title="TensorFlow" %}
```text
import tensorflow as tf

tf.math.square(x, name=None)
```
{% endtab %}

{% tab title="PyTorch" %}
```text
import torch
 
torch.square(input_tensor)
```
{% endtab %}
{% endtabs %}

 Using such an operation we'll have the derivative stored aside. So when we requested to calculate the derivative at a point, we'll pick up this stored value from there and thus, make up the missed information for the whole calculation.



