# APL

APL is an array programming language developed in the 60s. This project uses it for its close implications regarding parallelization of its operands. The details are better explained in a late 2019 doctoral dissertation: [A data parallel compiler hosted on the GPU](https://scholarworks.iu.edu/dspace/handle/2022/24749).

## Usage

1. To execute APL, [Dyalog](https://www.dyalog.com/download-zone.htm) is a modern APL interpreter. Something similar is Rob Pike's [ivy interpreter](https://godoc.org/robpike.io/ivy) written in Go.
2. To write APL, common idioms are listed on [APLcart](https://aplcart.info/). Here, you can search for idiomatic ways to do things. For notation reference, Dyalog has good examples in its IDE. Similarly, `ivy` lists the commands in its godoc page.
3. To write GPU performant APL, consider [the performance guidelines](https://github.com/Co-dfns/Co-dfns/blob/master/docs/PERFORMANCE.md) of the `co-dfns` project.
4. To benchmark APL on Dyalog, import the [dfns](https://dfns.dyalog.com/) workspace and use the [cmpx](https://dfns.dyalog.com/n_cmpx.htm) and [time](https://dfns.dyalog.com/n_time.htm) commands.
5. To produce executables, see [Dyalog APL: how to write standalone files that can be executed?](https://stackoverflow.com/questions/60698569/dyalog-apl-how-to-write-standalone-files-that-can-be-executed)
