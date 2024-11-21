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

# 3.5

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