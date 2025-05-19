# Fork of BitCrack that creates **ranges** and randomly selects one continously (ignoring the already selected ones).

When processing ranges with `--process-ranges`, progress output includes two
bars. The first bar shows progress in the current range and the second bar shows
overall range progress.

Example output:

```
GeForce GTX 1080   512/8192MB | 1 target 10.00 MKey/s (12,345,678 total) [00:02:10] | [####------] [#---------] 3/15 size 2^44 eta:00:10:22
```
