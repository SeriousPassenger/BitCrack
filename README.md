# Fork of BitCrack that creates **ranges** and randomly selects one continously (ignoring the already selected ones).

When using `--create-ranges` you can control the size of each generated range
with the `--range-size` option. The value may be a decimal number or a power of
two specified as `2^x`.

Examples:

```
./bitcrack --create-ranges ranges.txt --range-size 2^32
./bitcrack --create-ranges ranges.txt --range-size 1000000
```
