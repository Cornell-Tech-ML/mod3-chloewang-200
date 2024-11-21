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
Epoch 0 | loss 5.936034854415316 | correct 31 | epoch time 13.476916551589966 seconds | total time 13.476919412612915 seconds
Epoch 10 | loss 2.700046986481981 | correct 47 | epoch time 0.09898662567138672 seconds | total time 18.483076095581055 seconds
Epoch 20 | loss 0.7603237879256799 | correct 49 | epoch time 0.09829092025756836 seconds | total time 19.50894331932068 seconds
Epoch 30 | loss 1.6606763319334201 | correct 49 | epoch time 0.10089445114135742 seconds | total time 20.541847229003906 seconds
Epoch 40 | loss 1.152103296255384 | correct 48 | epoch time 0.10601544380187988 seconds | total time 21.62561011314392 seconds
Epoch 50 | loss 0.8240099345265399 | correct 48 | epoch time 0.10050821304321289 seconds | total time 22.65865421295166 seconds
Epoch 60 | loss 0.10180502517245558 | correct 50 | epoch time 0.10088634490966797 seconds | total time 23.68510937690735 seconds
Epoch 70 | loss 2.5110852909396986 | correct 47 | epoch time 0.0993812084197998 seconds | total time 24.712130308151245 seconds
Epoch 80 | loss 0.6824239470365787 | correct 48 | epoch time 0.10245585441589355 seconds | total time 25.735752820968628 seconds
Epoch 90 | loss 0.2043357607425777 | correct 50 | epoch time 0.10235905647277832 seconds | total time 26.775158405303955 seconds
Epoch 100 | loss 0.4487675399131918 | correct 50 | epoch time 0.1890549659729004 seconds | total time 28.10636305809021 seconds
Epoch 110 | loss 1.2221359062727215 | correct 50 | epoch time 0.10323095321655273 seconds | total time 29.917121410369873 seconds
Epoch 120 | loss 0.27046560203939385 | correct 50 | epoch time 0.10037946701049805 seconds | total time 30.94181513786316 seconds
Epoch 130 | loss 0.771008237924816 | correct 50 | epoch time 0.10141396522521973 seconds | total time 31.968180418014526 seconds
Epoch 140 | loss 0.060679234441329465 | correct 48 | epoch time 0.10062360763549805 seconds | total time 33.00263810157776 seconds
Epoch 150 | loss 0.6820491789677086 | correct 50 | epoch time 0.10018277168273926 seconds | total time 34.03198575973511 seconds
Epoch 160 | loss 0.829057235189586 | correct 50 | epoch time 0.09998059272766113 seconds | total time 35.051857709884644 seconds
Epoch 170 | loss 0.9123322224426237 | correct 50 | epoch time 0.10278511047363281 seconds | total time 36.074864864349365 seconds
Epoch 180 | loss 0.20325107603351952 | correct 48 | epoch time 0.10040736198425293 seconds | total time 37.083674907684326 seconds
Epoch 190 | loss 1.5094579370837062 | correct 48 | epoch time 0.10095977783203125 seconds | total time 38.09466290473938 seconds
Epoch 200 | loss 0.24333339931768047 | correct 50 | epoch time 0.10050559043884277 seconds | total time 39.12045407295227 seconds
Epoch 210 | loss 0.46001068317848026 | correct 49 | epoch time 0.19222497940063477 seconds | total time 40.38849997520447 seconds
Epoch 220 | loss 0.07540070808048094 | correct 49 | epoch time 0.23930096626281738 seconds | total time 42.233131647109985 seconds
Epoch 230 | loss 0.49900941687919936 | correct 50 | epoch time 0.09956836700439453 seconds | total time 43.34061145782471 seconds
Epoch 240 | loss 0.7532872334061086 | correct 50 | epoch time 0.10059833526611328 seconds | total time 44.36246299743652 seconds
Epoch 250 | loss 0.8960118859628958 | correct 50 | epoch time 0.09989070892333984 seconds | total time 45.39019012451172 seconds
Epoch 260 | loss 0.20583994914828743 | correct 48 | epoch time 0.17168498039245605 seconds | total time 46.48948049545288 seconds
Epoch 270 | loss 0.08620838827038949 | correct 50 | epoch time 0.10013556480407715 seconds | total time 47.511768102645874 seconds
Epoch 280 | loss 0.08736338894907635 | correct 50 | epoch time 0.09773135185241699 seconds | total time 48.529751777648926 seconds
Epoch 290 | loss 0.6384305207549746 | correct 50 | epoch time 0.09787964820861816 seconds | total time 49.535107135772705 seconds
Epoch 300 | loss 0.778683907452391 | correct 50 | epoch time 0.09914302825927734 seconds | total time 50.552032232284546 seconds
Epoch 310 | loss 0.11724129380134013 | correct 50 | epoch time 0.10170245170593262 seconds | total time 51.62231993675232 seconds
Epoch 320 | loss 0.447798594230346 | correct 50 | epoch time 0.18774104118347168 seconds | total time 52.84891724586487 seconds
Epoch 330 | loss 0.11888290966564187 | correct 50 | epoch time 0.23019170761108398 seconds | total time 54.77495765686035 seconds
Epoch 340 | loss 0.7631027291396756 | correct 50 | epoch time 0.10203003883361816 seconds | total time 55.85623478889465 seconds
Epoch 350 | loss 0.2906549736553348 | correct 49 | epoch time 0.09895944595336914 seconds | total time 56.88109374046326 seconds
Epoch 360 | loss 0.5432164666449915 | correct 50 | epoch time 0.10192322731018066 seconds | total time 57.89941740036011 seconds
Epoch 370 | loss 0.39970138564066665 | correct 49 | epoch time 0.1005704402923584 seconds | total time 58.92042374610901 seconds
Epoch 380 | loss 0.7945691825396242 | correct 50 | epoch time 0.09925723075866699 seconds | total time 59.94179964065552 seconds
Epoch 390 | loss 0.7914018697601806 | correct 50 | epoch time 0.09945344924926758 seconds | total time 60.956660985946655 seconds
Epoch 400 | loss 0.012135936435035629 | correct 48 | epoch time 0.10116338729858398 seconds | total time 61.98042011260986 seconds
Epoch 410 | loss 0.014442397282427808 | correct 50 | epoch time 0.10049986839294434 seconds | total time 63.00088405609131 seconds
Epoch 420 | loss 0.3480845740435951 | correct 50 | epoch time 0.09821915626525879 seconds | total time 64.01518321037292 seconds
Epoch 430 | loss 0.10778790421053787 | correct 49 | epoch time 0.1376042366027832 seconds | total time 65.12557315826416 seconds
Epoch 440 | loss 0.1465467101531052 | correct 50 | epoch time 0.17130780220031738 seconds | total time 66.97771978378296 seconds
Epoch 450 | loss 0.15894249639849536 | correct 50 | epoch time 0.10494089126586914 seconds | total time 68.24664044380188 seconds
Epoch 460 | loss 0.04398457020033284 | correct 50 | epoch time 0.10294151306152344 seconds | total time 69.2878749370575 seconds
Epoch 470 | loss 0.015353175805095542 | correct 50 | epoch time 0.1010599136352539 seconds | total time 70.32522630691528 seconds
Epoch 480 | loss 0.040803385998783526 | correct 50 | epoch time 0.09859013557434082 seconds | total time 71.35219144821167 seconds
Epoch 490 | loss 0.4740975494884174 | correct 50 | epoch time 0.10004997253417969 seconds | total time 72.38329887390137 seconds
```

## Split
```
Epoch 0 | loss 6.195230110115921 | correct 34 | epoch time 13.322509527206421 seconds | total time 13.322511672973633 seconds
Epoch 10 | loss 4.326292703220332 | correct 39 | epoch time 0.2250354290008545 seconds | total time 18.43011713027954 seconds
Epoch 20 | loss 4.703451280908129 | correct 43 | epoch time 0.10044026374816895 seconds | total time 19.557780981063843 seconds
Epoch 30 | loss 3.1170742761026196 | correct 42 | epoch time 0.10043764114379883 seconds | total time 20.589861154556274 seconds
Epoch 40 | loss 3.360183128420745 | correct 45 | epoch time 0.10095524787902832 seconds | total time 21.621669054031372 seconds
Epoch 50 | loss 3.0303328695986225 | correct 47 | epoch time 0.10156536102294922 seconds | total time 22.656344175338745 seconds
Epoch 60 | loss 2.676060875395309 | correct 47 | epoch time 0.09877276420593262 seconds | total time 23.712589979171753 seconds
Epoch 70 | loss 0.9913780969533798 | correct 47 | epoch time 0.10157155990600586 seconds | total time 24.742567777633667 seconds
Epoch 80 | loss 1.9497116642782135 | correct 50 | epoch time 0.09865450859069824 seconds | total time 25.768738985061646 seconds
Epoch 90 | loss 1.4110974618537095 | correct 48 | epoch time 0.10118842124938965 seconds | total time 26.804526805877686 seconds
Epoch 100 | loss 1.042622015635545 | correct 48 | epoch time 0.10127377510070801 seconds | total time 27.825562715530396 seconds
Epoch 110 | loss 1.7560628335407618 | correct 50 | epoch time 0.2212810516357422 seconds | total time 29.107764720916748 seconds
Epoch 120 | loss 1.9057648048504605 | correct 50 | epoch time 0.22223281860351562 seconds | total time 31.029073238372803 seconds
Epoch 130 | loss 1.0122721473414131 | correct 50 | epoch time 0.09961986541748047 seconds | total time 32.05000591278076 seconds
Epoch 140 | loss 1.5432796820286228 | correct 50 | epoch time 0.09884762763977051 seconds | total time 33.06395220756531 seconds
Epoch 150 | loss 1.1807960428329511 | correct 50 | epoch time 0.10508227348327637 seconds | total time 34.09935164451599 seconds
Epoch 160 | loss 0.8282264000967013 | correct 50 | epoch time 0.0986320972442627 seconds | total time 35.13401746749878 seconds
Epoch 170 | loss 0.9225716257823915 | correct 50 | epoch time 0.10213327407836914 seconds | total time 36.16698169708252 seconds
Epoch 180 | loss 0.22755885157157835 | correct 50 | epoch time 0.10187506675720215 seconds | total time 37.19090414047241 seconds
Epoch 190 | loss 0.6127171691257233 | correct 50 | epoch time 0.1010134220123291 seconds | total time 38.211732149124146 seconds
Epoch 200 | loss 1.639911614340638 | correct 50 | epoch time 0.10245919227600098 seconds | total time 39.243802309036255 seconds
Epoch 210 | loss 0.4850747473757441 | correct 50 | epoch time 0.10010933876037598 seconds | total time 40.26374530792236 seconds
Epoch 220 | loss 0.6087239777346554 | correct 50 | epoch time 0.21976184844970703 seconds | total time 41.56474852561951 seconds
Epoch 230 | loss 0.8143400894572437 | correct 50 | epoch time 0.24617648124694824 seconds | total time 43.49818563461304 seconds
Epoch 240 | loss 0.378697922832246 | correct 50 | epoch time 0.10530638694763184 seconds | total time 44.580453395843506 seconds
Epoch 250 | loss 0.4865605945497923 | correct 50 | epoch time 0.10069060325622559 seconds | total time 45.6104474067688 seconds
Epoch 260 | loss 0.5718410015942709 | correct 50 | epoch time 0.1801300048828125 seconds | total time 46.71123385429382 seconds
Epoch 270 | loss 0.47064198313522987 | correct 50 | epoch time 0.09819865226745605 seconds | total time 47.72871279716492 seconds
Epoch 280 | loss 0.4264264474919723 | correct 50 | epoch time 0.09904718399047852 seconds | total time 48.747530698776245 seconds
Epoch 290 | loss 0.38045379738398694 | correct 49 | epoch time 0.09922456741333008 seconds | total time 49.76236629486084 seconds
Epoch 300 | loss 0.7899254766091408 | correct 50 | epoch time 0.09802651405334473 seconds | total time 50.80693435668945 seconds
Epoch 310 | loss 0.8158286177977464 | correct 50 | epoch time 0.10072588920593262 seconds | total time 51.8236939907074 seconds
Epoch 320 | loss 0.08992794405329985 | correct 50 | epoch time 0.09964585304260254 seconds | total time 52.84117007255554 seconds
Epoch 330 | loss 0.6141186145482819 | correct 50 | epoch time 0.1691570281982422 seconds | total time 54.008782386779785 seconds
Epoch 340 | loss 0.3006703322316962 | correct 50 | epoch time 0.22541260719299316 seconds | total time 55.89894509315491 seconds
Epoch 350 | loss 0.4363255558379101 | correct 50 | epoch time 0.09857964515686035 seconds | total time 57.011053800582886 seconds
Epoch 360 | loss 0.5186313860859335 | correct 50 | epoch time 0.09969472885131836 seconds | total time 58.026556968688965 seconds
Epoch 370 | loss 0.8397649427374653 | correct 50 | epoch time 0.09848189353942871 seconds | total time 59.04698610305786 seconds
Epoch 380 | loss 0.7001835379146341 | correct 50 | epoch time 0.09872674942016602 seconds | total time 60.07257866859436 seconds
Epoch 390 | loss 0.45274050171225066 | correct 50 | epoch time 0.10829496383666992 seconds | total time 61.07403516769409 seconds
Epoch 400 | loss 0.4122746901530185 | correct 50 | epoch time 0.10454630851745605 seconds | total time 62.07711935043335 seconds
Epoch 410 | loss 0.3742271334252516 | correct 50 | epoch time 0.1051034927368164 seconds | total time 63.0798454284668 seconds
Epoch 420 | loss 0.36247521423057577 | correct 50 | epoch time 0.10931944847106934 seconds | total time 64.08583474159241 seconds
Epoch 430 | loss 0.10306971722064624 | correct 50 | epoch time 0.10198569297790527 seconds | total time 65.09077072143555 seconds
Epoch 440 | loss 0.04414178602431027 | correct 50 | epoch time 0.11801409721374512 seconds | total time 66.11028861999512 seconds
Epoch 450 | loss 0.3210190399236587 | correct 50 | epoch time 0.1981980800628662 seconds | total time 68.01803255081177 seconds
Epoch 460 | loss 0.15431429926222429 | correct 50 | epoch time 0.11239981651306152 seconds | total time 69.32813334465027 seconds
Epoch 470 | loss 0.3168235112059267 | correct 50 | epoch time 0.10765480995178223 seconds | total time 70.35067582130432 seconds
Epoch 480 | loss 0.36133603755249144 | correct 50 | epoch time 0.1370084285736084 seconds | total time 71.39489722251892 seconds
Epoch 490 | loss 0.16440025078522322 | correct 50 | epoch time 0.11135005950927734 seconds | total time 72.41293215751648 seconds
```

## XOR
```
Epoch 0 | loss 7.551652621109283 | correct 26 | epoch time 13.367004871368408 seconds | total time 13.367007970809937 seconds
Epoch 10 | loss 3.62769062752028 | correct 36 | epoch time 0.10032868385314941 seconds | total time 18.54718542098999 seconds
Epoch 20 | loss 5.9615639589679645 | correct 44 | epoch time 0.10083222389221191 seconds | total time 19.58026671409607 seconds
Epoch 30 | loss 4.556563448092184 | correct 46 | epoch time 0.10187625885009766 seconds | total time 20.60471820831299 seconds
Epoch 40 | loss 2.710835397597237 | correct 42 | epoch time 0.10004281997680664 seconds | total time 21.628762245178223 seconds
Epoch 50 | loss 2.5939644130636337 | correct 45 | epoch time 0.10133123397827148 seconds | total time 22.64624571800232 seconds
Epoch 60 | loss 3.7720380400245794 | correct 43 | epoch time 0.1010735034942627 seconds | total time 23.667778968811035 seconds
Epoch 70 | loss 3.390599454111227 | correct 45 | epoch time 0.09958910942077637 seconds | total time 24.693949699401855 seconds
Epoch 80 | loss 2.0142260661908447 | correct 46 | epoch time 0.10123443603515625 seconds | total time 25.73138928413391 seconds
Epoch 90 | loss 1.5908417885926744 | correct 45 | epoch time 0.10463976860046387 seconds | total time 26.766757488250732 seconds
Epoch 100 | loss 3.767945779595084 | correct 44 | epoch time 0.1011958122253418 seconds | total time 27.799360752105713 seconds
Epoch 110 | loss 2.7567976686066094 | correct 43 | epoch time 0.19696497917175293 seconds | total time 29.283279418945312 seconds
Epoch 120 | loss 3.034316451757854 | correct 46 | epoch time 0.1017599105834961 seconds | total time 31.017195224761963 seconds
Epoch 130 | loss 2.503166304376597 | correct 46 | epoch time 0.10253310203552246 seconds | total time 32.045183181762695 seconds
Epoch 140 | loss 2.5471383416294704 | correct 47 | epoch time 0.10067152976989746 seconds | total time 33.079397201538086 seconds
Epoch 150 | loss 1.8821185317319893 | correct 45 | epoch time 0.10018658638000488 seconds | total time 34.10653638839722 seconds
Epoch 160 | loss 2.9628699239555734 | correct 45 | epoch time 0.10109210014343262 seconds | total time 35.13300347328186 seconds
Epoch 170 | loss 3.07141074347994 | correct 47 | epoch time 0.10128235816955566 seconds | total time 36.15461206436157 seconds
Epoch 180 | loss 1.526968953700304 | correct 47 | epoch time 0.1002962589263916 seconds | total time 37.188032150268555 seconds
Epoch 190 | loss 4.431238566815322 | correct 46 | epoch time 0.09888362884521484 seconds | total time 38.217609882354736 seconds
Epoch 200 | loss 2.3699887909699044 | correct 46 | epoch time 0.0991973876953125 seconds | total time 39.24074959754944 seconds
Epoch 210 | loss 1.06377726311263 | correct 46 | epoch time 0.10042953491210938 seconds | total time 40.262460708618164 seconds
Epoch 220 | loss 2.9545038088993003 | correct 46 | epoch time 0.23546361923217773 seconds | total time 41.57454466819763 seconds
Epoch 230 | loss 2.0635212584801357 | correct 46 | epoch time 0.09973812103271484 seconds | total time 43.51033854484558 seconds
Epoch 240 | loss 0.8403458792957216 | correct 45 | epoch time 0.09920597076416016 seconds | total time 44.52248573303223 seconds
Epoch 250 | loss 1.3469231755338242 | correct 45 | epoch time 0.1002953052520752 seconds | total time 45.54119944572449 seconds
Epoch 260 | loss 1.6244298923930154 | correct 48 | epoch time 0.1738433837890625 seconds | total time 46.636017084121704 seconds
Epoch 270 | loss 0.8165652683943134 | correct 46 | epoch time 0.10015273094177246 seconds | total time 47.658291816711426 seconds
Epoch 280 | loss 2.5245534248721726 | correct 46 | epoch time 0.0981130599975586 seconds | total time 48.679383516311646 seconds
Epoch 290 | loss 1.7608267056939144 | correct 47 | epoch time 0.0980372428894043 seconds | total time 49.70152568817139 seconds
Epoch 300 | loss 2.3535786318801915 | correct 45 | epoch time 0.10197758674621582 seconds | total time 50.72525501251221 seconds
Epoch 310 | loss 2.216829406968672 | correct 48 | epoch time 0.10322260856628418 seconds | total time 51.750964403152466 seconds
Epoch 320 | loss 1.097476013364437 | correct 47 | epoch time 0.10073637962341309 seconds | total time 52.77143406867981 seconds
Epoch 330 | loss 2.3617948827807718 | correct 49 | epoch time 0.20967531204223633 seconds | total time 54.188156604766846 seconds
Epoch 340 | loss 1.2234437704667929 | correct 49 | epoch time 0.09956097602844238 seconds | total time 55.95328330993652 seconds
Epoch 350 | loss 1.4781674855449545 | correct 48 | epoch time 0.09964847564697266 seconds | total time 56.97505807876587 seconds
Epoch 360 | loss 1.2437108843774078 | correct 48 | epoch time 0.09988164901733398 seconds | total time 58.006086587905884 seconds
Epoch 370 | loss 1.1129938501244359 | correct 49 | epoch time 0.10220122337341309 seconds | total time 59.056376457214355 seconds
Epoch 380 | loss 3.2918504957375987 | correct 46 | epoch time 0.10099649429321289 seconds | total time 60.074995279312134 seconds
Epoch 390 | loss 1.5853429777146775 | correct 50 | epoch time 0.1033334732055664 seconds | total time 61.09262251853943 seconds
Epoch 400 | loss 2.310398563587031 | correct 46 | epoch time 0.1016690731048584 seconds | total time 62.1300253868103 seconds
Epoch 410 | loss 2.1642962732721123 | correct 50 | epoch time 0.09899592399597168 seconds | total time 63.16367173194885 seconds
Epoch 420 | loss 0.8151038416501468 | correct 49 | epoch time 0.09878420829772949 seconds | total time 64.18091440200806 seconds
Epoch 430 | loss 0.4889780748649018 | correct 48 | epoch time 0.10076665878295898 seconds | total time 65.19370126724243 seconds
Epoch 440 | loss 2.4436200443073033 | correct 48 | epoch time 0.2185986042022705 seconds | total time 66.58801698684692 seconds
Epoch 450 | loss 3.228075866521558 | correct 47 | epoch time 0.16608214378356934 seconds | total time 68.35391354560852 seconds
Epoch 460 | loss 0.5225786048516153 | correct 46 | epoch time 0.10023188591003418 seconds | total time 69.38517761230469 seconds
Epoch 470 | loss 0.5424742715706752 | correct 48 | epoch time 0.09944725036621094 seconds | total time 70.3991687297821 seconds
Epoch 480 | loss 1.260334353445442 | correct 48 | epoch time 0.09915590286254883 seconds | total time 71.42786526679993 seconds
Epoch 490 | loss 1.6447225326816848 | correct 46 | epoch time 0.10241055488586426 seconds | total time 72.47996258735657 seconds
```
