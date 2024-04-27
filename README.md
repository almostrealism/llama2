## Llama2 - Java Implementation using Almost Realism HPC Platform

This is a basic implementation of Llama2 with only a single dependency, the
Almost Realism HPC library. This work was inspired by
[llama2.c](https://github.com/karpathy/llama2.c) and
[llama2.java](https://github.com/mukel/llama2.java) and it uses the same format
for the tokenizer and model weights files.

### Requirements

If you are interested in converting models for use with this implementation,
I recommend taking a look at the instructions provided by the author of llama2.c
in the README for that repository. However, below you will find a link to
an example export so that anyone can test this implementation with a small
model.

### Usage

1. Compile the project using `mvn package`
2. Download the model weights for [stories110M.bin](https://www.dropbox.com/scl/fi/romns8veg67agl5czmtww/stories110M.bin?rlkey=sbspy97d2j1p3jilgaff190pz&st=kak6t2uo&dl=1)
3. Run the project using `java -jar target/ar-llama2-0.1.jar` from the directory containing both the model weights and tokenizer.bin (included in this repository)

### Will this use my GPU?

This depends on determinations made by the Almost Realism HPC library. If your system supports
Metal or OpenCL, then yes it should use your GPU for some or all of the computations. The choice
of what to run CPU versus GPU is made dynamically, but it is possible to change it.

You can learn more about the Almost Realism HPC library via the [GitHub repository](https://github.com/almostrealism/common)

### Example Output
Following the usage instructions above should result in something like the output below.

```shell
(rings) Mac-Studio:llama2 michael$ java -jar target/ar-llama2-0.1.jar
Hardware[JNI]: Max RAM is 4096 Megabytes (FP32)
Hardware[MTL]: Max RAM is 4096 Megabytes (FP32)
Hardware[CL]: Using GPU 0 for kernels
Hardware[CL]: 64 cores @ 1.0GHz
Hardware[CL]: 3D work support and 0.25kb work size
Hardware[CL]: 32.0kb local / 96.0gb global (18.0gb allocation limit)
Hardware[CL]: Max RAM is 4096 Megabytes (FP32)
Hardware[JNI]: Enabling shared memory via org.almostrealism.hardware.metal.MetalMemoryProvider@566776ad
Loaded weights in 882ms
<s>
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red apple on a tree. She wanted to eat it, but it was too high up.
Lily asked her friend, a little bird, "Can you help me get the apple?"
The bird said, "Sure, I can fly up and get it for you."
The bird flew up to the apple and pecked it off the tree. Lily was so happy and took a big bite. But then, she saw a bug on the apple. She didn't like bugs, so she threw the apple away.
Later that day, Lily's mom asked her to help with the laundry. Lily saw a shirt that was too big for her. She asked her mom, "Can you make it fit me?"
Her mom said, "Yes, I can make it fit you."
Lily was happy that her shirt would fit her. She learned that sometimes things don't fit, but there is always a way to make them fit.
<s>
Once upon a time, there was a little girl named Lily
tokens per second: 8.559632
[13:22.06] OperationProfile: default - 76.80 seconds:
        attentionKeys layer (12, 64)->(12, 1024): 3072 [27.953s tot | 9.099ms avg] 36%
        softmax2d layer (12, 1024)->(12, 1024): 3072 [15.887s tot | 5.172ms avg] 20%
        ropeRotation layer (12, 32, 2)->(12, 32, 2): 6144 [10.661s tot | 1.735ms avg] 13%
        attentionValues layer (12, 1024)->(768): 3072 [5.759s tot | 1.875ms avg] 7%
        dense 768 layer (768)->(768): 12288 [4.672s tot | 0.38ms avg] 6%
        rmsnorm layer (768)->(768): 6400 [3.402s tot | 0.532ms avg] 4%
        dense 768 layer (768)->(2048): 6144 [2.284s tot | 0.372ms avg] 2%
        dense 2048 layer (2048)->(768): 3072 [1.982s tot | 0.645ms avg] 2%
        accum layer (768)->(768): 6144 [1.475s tot | 0.24ms avg] 1%
        silu layer (2048)->(2048): 3072 [0.726s tot | 0.236ms avg] 0%
        softmax2d layer (Input Record): 3072 [0.707s tot | 0.23ms avg] 0%
        product layer (768)->(2048): 3072 [0.698s tot | 0.227ms avg] 0%
        MemoryDataCopy$$Lambda/0x00000070010e1e50: 52992 [0.409s tot | 0.008ms avg] 0%
        dense 768 layer (768)->(32000): 256 [0.146s tot | 0.571ms avg] 0%
        CollectionReceptor$$Lambda/0x00000070011195b8: 6144 [0.039s tot | 0.006ms avg] 0%

Done
```

The profile information will vary quite a lot depending on your hardware (these results are from an Apple M1).

### Ongoing Development

The Almost Realism HPC platform is a work in progress. If you are interested in a part-time
paid role introducing new model implementations (like this one) built on top of that platform,
please reach out to me through the contact information on my GitHub profile (@ashesfall).

