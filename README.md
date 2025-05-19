## Fork of BitCrack with minor improvements

- See https://github.com/brichard19/BitCrack for the original non-fork repo.

### Improvements:

- With this fork, the amount of targets is only limited by your VRAM instead of the constant memory limit, which allowed only a few thousand addresses maximum.
- Range system that lets you create a range descriptor with the supplied configuration and addresses and lets you easily load where you left off next time.
- Ranges are randomized and already processed ranges aren't processed the next time.
- Simple progressbar

### Building for CUDA

First make sure you pass the correct <code>COMPUTE_CAP</code> in the makefile and save.

Then:

<pre>bash BUILD_CUDA.sh</pre>

### Example usage

#### Creating range descriptor file

<pre>
  ./cuBitCrack --create-ranges myranges.txt -i in.txt --keyspace 0x0:0xFFFFFFFFFFFFFFFF --range-size 0xFFFFFFFFF
</pre>

#### Continuing from range descriptor file
<pre>
  ./cuBitCrack --process-ranges myranges.txt
</pre>
