# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

# 3.4

![alt text](image.png)
Timing output:
```
Timing summary
Size: 64
    fast: 0.00294
    gpu: 0.00559
Size: 128
    fast: 0.01394
    gpu: 0.01277
Size: 256
    fast: 0.09210
    gpu: 0.04554
Size: 512
    fast: 1.19863
    gpu: 0.26912
Size: 1024
    fast: 7.63548
    gpu: 0.96579
```
# 3.5

## Simple
```
Epoch  0  loss  6.211235353021375 correct 36
Epoch  10  loss  2.19127950013039 correct 47
Epoch  20  loss  1.4816074335032168 correct 49
Epoch  30  loss  1.0329069114469975 correct 49
Epoch  40  loss  1.283262857380862 correct 50
Epoch  50  loss  0.6453182890291175 correct 48
Epoch  60  loss  0.5572747308230198 correct 49
Epoch  70  loss  0.9376906470248648 correct 50
Epoch  80  loss  0.5906952824734057 correct 50
Epoch  90  loss  0.43853210342986165 correct 50
Epoch  100  loss  0.6598587855959119 correct 50
Epoch  110  loss  0.18293504459030313 correct 50
Epoch  120  loss  0.12430816838788501 correct 50
Epoch  130  loss  0.37958360119650686 correct 50
Epoch  140  loss  0.3178915527772254 correct 50
Epoch  150  loss  0.397342447771347 correct 50
Epoch  160  loss  0.5409547707756727 correct 50
Epoch  170  loss  0.4559593226070752 correct 50
Epoch  180  loss  0.3833895020953361 correct 50
Epoch  190  loss  0.5728523436189813 correct 50
Epoch  200  loss  0.22439762798535148 correct 50
Epoch  210  loss  0.025288997065458903 correct 50
Epoch  220  loss  0.19647686825869184 correct 50
Epoch  230  loss  0.21505935362872458 correct 50
Epoch  240  loss  0.018012555260724667 correct 50
Epoch  250  loss  0.8053052455043793 correct 50
Epoch  260  loss  0.11880522140416375 correct 50
Epoch  270  loss  0.14151656212514044 correct 50
Epoch  280  loss  0.15653940076564915 correct 50
Epoch  290  loss  0.18090740621103063 correct 50
Epoch  300  loss  0.05014822760307206 correct 50
Epoch  310  loss  0.14974098430421368 correct 50
Epoch  320  loss  0.24049842161229923 correct 50
Epoch  330  loss  0.0509718340645229 correct 50
Epoch  340  loss  0.3008884118188292 correct 50
Epoch  350  loss  0.3165934919062321 correct 50
Epoch  360  loss  0.001542285743748907 correct 50
Epoch  370  loss  0.024735058156703005 correct 50
Epoch  380  loss  0.03970698495746322 correct 50
Epoch  390  loss  0.1211959356233181 correct 50
Epoch  400  loss  0.14535265635419758 correct 50
Epoch  410  loss  0.14397131561579116 correct 50
Epoch  420  loss  0.30795122188863727 correct 50
Epoch  430  loss  0.058367531425256663 correct 50
Epoch  440  loss  0.04490204582603363 correct 50
Epoch  450  loss  0.0616295789796114 correct 50
Epoch  460  loss  0.15075067205942172 correct 50
Epoch  470  loss  0.25179170679828056 correct 50
Epoch  480  loss  0.17255739528717023 correct 50
Epoch  490  loss  0.07575385096505952 correct 50
```

## Split

50 Points, 100 Hidden Layers, 0.05 Learning Rate

```
Epoch  0  loss  5.610016303610899 correct 37
Epoch  10  loss  4.563037667191005 correct 39
Epoch  20  loss  3.5261782710546576 correct 45
Epoch  30  loss  3.8064184204571037 correct 49
Epoch  40  loss  1.8278259430511146 correct 47
Epoch  50  loss  2.4137606284325863 correct 49
Epoch  60  loss  2.2798702419854004 correct 49
Epoch  70  loss  1.771652641515164 correct 47
Epoch  80  loss  1.7781667144015723 correct 49
Epoch  90  loss  1.7910854152671178 correct 49
Epoch  100  loss  0.20379882873026628 correct 49
Epoch  110  loss  1.2032430699532324 correct 49
Epoch  120  loss  0.8400600518543537 correct 50
Epoch  130  loss  1.027344770772624 correct 50
Epoch  140  loss  1.0803527885143305 correct 49
Epoch  150  loss  0.7201849316776706 correct 47
Epoch  160  loss  0.6892199419478762 correct 50
Epoch  170  loss  1.3982535536121925 correct 49
Epoch  180  loss  0.30605574591550055 correct 49
Epoch  190  loss  2.2327717976000527 correct 49
Epoch  200  loss  0.42416575118012634 correct 49
Epoch  210  loss  0.13069850932638497 correct 48
Epoch  220  loss  0.3232398282880687 correct 49
Epoch  230  loss  0.1942062761498258 correct 49
Epoch  240  loss  0.33255814694733454 correct 50
Epoch  250  loss  0.4597572773317945 correct 49
Epoch  260  loss  0.4088453252876592 correct 50
Epoch  270  loss  1.1969854283921237 correct 49
Epoch  280  loss  1.6869998369131627 correct 49
Epoch  290  loss  0.8571025718035621 correct 50
Epoch  300  loss  0.2576425394398878 correct 49
Epoch  310  loss  1.0423726843592702 correct 49
Epoch  320  loss  0.3570983154396318 correct 49
Epoch  330  loss  0.8426439270253251 correct 49
Epoch  340  loss  1.1452183406029368 correct 49
Epoch  350  loss  0.6849418677786742 correct 50
Epoch  360  loss  0.2510150462650637 correct 50
Epoch  370  loss  1.4279906256390857 correct 49
Epoch  380  loss  0.044511562760412945 correct 50
Epoch  390  loss  0.23688132895500622 correct 49
Epoch  400  loss  0.4561542232536647 correct 49
Epoch  410  loss  1.2565727552165034 correct 50
Epoch  420  loss  0.3175125855940618 correct 50
Epoch  430  loss  0.8904622908477668 correct 49
Epoch  440  loss  0.4724454073780046 correct 49
Epoch  450  loss  0.16210904088363873 correct 49
Epoch  460  loss  0.06720523074185052 correct 49
Epoch  470  loss  0.8191751294727074 correct 50
Epoch  480  loss  0.2000704731308156 correct 50
Epoch  490  loss  0.1157289077189706 correct 50
```