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
3. Run the project using `java -jar target/ar-llama2-0.3.jar` from the directory containing both the model weights and tokenizer.bin (included in this repository)

### Will this use my GPU?

This depends on determinations made by the Almost Realism HPC library. If your system supports
Metal or OpenCL, then yes it should use your GPU for some or all of the computations. The choice
of what to run CPU versus GPU is made dynamically, but it is possible to change it.

You can learn more about the Almost Realism HPC library via the [GitHub repository](https://github.com/almostrealism/common)

### Example Output
Following the usage instructions above should result in something like the output below.

```shell
Mac-Studio:llama2 michael$ java -jar target/ar-llama2-0.3.jar
Hardware[JNI]: Max RAM is 4096 Megabytes (FP32)
Hardware[MTL]: Max RAM is 4096 Megabytes (FP32)
Hardware[CL]: Using GPU 0 for kernels
Hardware[CL]: 64 cores @ 1.0GHz
Hardware[CL]: 3D work support and 0.25kb work size
Hardware[CL]: 32.0kb local / 96.0gb global (18.0gb allocation limit)
Hardware[CL]: Max RAM is 4096 Megabytes (FP32)
Hardware[JNI]: Enabling shared memory via MetalMemoryProvider
Loaded weights in 743ms
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
tokens per second: 9.184556
[09:13.43] OperationProfile: default - 52.757 seconds:
        softmax2d layer (12, 1024)->(12, 1024) (12, 1024)[axis=1|12x1024]: 3072 [11.734s tot | 3.82ms avg] 22%
        ropeRotation layer (12, 32, 2)->(12, 32, 2) (12, 32, 2)[axis=1|12x64]: 6144 [7.615s tot | 1.239ms avg] 14%
        rmsnorm layer (768)->(768) (768)[axis=1|768x1]: 6400 [6.761s tot | 1.056ms avg] 12%
        dense 768 layer (768)->(768) (768, 1)[axis=1|768x1]: 12288 [6.741s tot | 0.549ms avg] 12%
        attentionValues layer (12, 1024)->(768) (768)[axis=1|768x1]: 3072 [6.618s tot | 2.154ms avg] 12%
        attentionKeys layer (12, 64)->(12, 1024) (12, 1024)[axis=2|12288x1]: 3072 [3.933s tot | 1.28ms avg] 7%
        dense 768 layer (768)->(2048) (2048, 1)[axis=1|2048x1]: 6144 [3.392s tot | 0.552ms avg] 6%
        dense 2048 layer (2048)->(768) (768, 1)[axis=1|768x1]: 3072 [2.004s tot | 0.652ms avg] 3%
        accum layer (768)->(768) (768)[axis=1|768x1]: 6144 [1.327s tot | 0.216ms avg] 2%
        silu layer (2048)->(2048) (2048)[axis=1|2048x1]: 3072 [0.674s tot | 0.219ms avg] 1%
        softmax2d layer (Input Record) (12, 1024)[axis=2|12288x1]: 3072 [0.668s tot | 0.217ms avg] 1%
        product layer (768)->(2048) (2048)[axis=1|2048x1]: 3072 [0.665s tot | 0.216ms avg] 1%
        dense 768 layer (768)->(32000) (32000, 1)[axis=1|32000x1]: 256 [0.201s tot | 0.784ms avg] 0%
        attentionValues layer (Input Record): 3072 [0.101s tot | 0.033ms avg] 0%
        dense 768 layer (Input Record): 18688 [0.084s tot | 0.004ms avg] 0%
        ropeRotation layer (Input Record): 6144 [0.039s tot | 0.006ms avg] 0%
        CollectionReceptor$$Lambda/0x0000007001137630: 6144 [0.036s tot | 0.006ms avg] 0%
        accum layer (Input Record): 6144 [0.033s tot | 0.005ms avg] 0%
        silu layer (Input Record): 3072 [0.028s tot | 0.009ms avg] 0%
        dense 2048 layer (Input Record): 3072 [0.028s tot | 0.009ms avg] 0%
        rmsnorm layer (Input Record): 6400 [0.025s tot | 0.004ms avg] 0%
        attentionKeys layer (Input Record): 3072 [0.019s tot | 0.006ms avg] 0%
        Model Forward Output: 256 [0.018s tot | 0.071ms avg] 0%
        product layer (Input Record): 3072 [0.016s tot | 0.005ms avg] 0%

Done
```

The profile information will vary quite a lot depending on your hardware (these results are from an Apple M1).

### Ongoing Development

The Almost Realism HPC platform is a work in progress. If you are interested in a part-time
paid role introducing new model implementations (like this one) built on top of that platform,
please reach out to me through the contact information on my GitHub profile (@ashesfall).

