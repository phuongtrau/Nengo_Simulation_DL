▒╣
Ч═
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
╝
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.6.02v2.6.0-rc2-32-g919f693420e8Ю┼
ё
conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_60/kernel
}
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_60/bias
m
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes
:@*
dtype0
Ё
conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*!
shared_nameconv2d_61/kernel
~
$conv2d_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_61/kernel*'
_output_shapes
:@ђ*
dtype0
u
conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_61/bias
n
"conv2d_61/bias/Read/ReadVariableOpReadVariableOpconv2d_61/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_62/kernel

$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_62/bias
n
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_63/kernel

$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_63/bias
n
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_64/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_64/kernel

$conv2d_64/kernel/Read/ReadVariableOpReadVariableOpconv2d_64/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_64/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_64/bias
n
"conv2d_64/bias/Read/ReadVariableOpReadVariableOpconv2d_64/bias*
_output_shapes	
:ђ*
dtype0
є
conv2d_65/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*!
shared_nameconv2d_65/kernel

$conv2d_65/kernel/Read/ReadVariableOpReadVariableOpconv2d_65/kernel*(
_output_shapes
:ђђ*
dtype0
u
conv2d_65/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_nameconv2d_65/bias
n
"conv2d_65/bias/Read/ReadVariableOpReadVariableOpconv2d_65/bias*
_output_shapes	
:ђ*
dtype0
{
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ* 
shared_namedense_10/kernel
t
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes
:	ђ*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
њ
Adam/conv2d_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_60/kernel/m
І
+Adam/conv2d_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/m*&
_output_shapes
:@*
dtype0
ѓ
Adam/conv2d_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_60/bias/m
{
)Adam/conv2d_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/m*
_output_shapes
:@*
dtype0
Њ
Adam/conv2d_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*(
shared_nameAdam/conv2d_61/kernel/m
ї
+Adam/conv2d_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/m*'
_output_shapes
:@ђ*
dtype0
Ѓ
Adam/conv2d_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_61/bias/m
|
)Adam/conv2d_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_62/kernel/m
Ї
+Adam/conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_62/bias/m
|
)Adam/conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_63/kernel/m
Ї
+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_63/bias/m
|
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_64/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_64/kernel/m
Ї
+Adam/conv2d_64/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_64/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_64/bias/m
|
)Adam/conv2d_64/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/m*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_65/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_65/kernel/m
Ї
+Adam/conv2d_65/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/kernel/m*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_65/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_65/bias/m
|
)Adam/conv2d_65/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/bias/m*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_10/kernel/m
ѓ
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:*
dtype0
њ
Adam/conv2d_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_60/kernel/v
І
+Adam/conv2d_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/v*&
_output_shapes
:@*
dtype0
ѓ
Adam/conv2d_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_60/bias/v
{
)Adam/conv2d_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/v*
_output_shapes
:@*
dtype0
Њ
Adam/conv2d_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ђ*(
shared_nameAdam/conv2d_61/kernel/v
ї
+Adam/conv2d_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/v*'
_output_shapes
:@ђ*
dtype0
Ѓ
Adam/conv2d_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_61/bias/v
|
)Adam/conv2d_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_62/kernel/v
Ї
+Adam/conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_62/bias/v
|
)Adam/conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_63/kernel/v
Ї
+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_63/bias/v
|
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_64/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_64/kernel/v
Ї
+Adam/conv2d_64/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_64/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_64/bias/v
|
)Adam/conv2d_64/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_64/bias/v*
_output_shapes	
:ђ*
dtype0
ћ
Adam/conv2d_65/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђђ*(
shared_nameAdam/conv2d_65/kernel/v
Ї
+Adam/conv2d_65/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/kernel/v*(
_output_shapes
:ђђ*
dtype0
Ѓ
Adam/conv2d_65/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*&
shared_nameAdam/conv2d_65/bias/v
|
)Adam/conv2d_65/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_65/bias/v*
_output_shapes	
:ђ*
dtype0
Ѕ
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*'
shared_nameAdam/dense_10/kernel/v
ѓ
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes
:	ђ*
dtype0
ђ
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
щT
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┤T
valueфTBДT BаT
М
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
R
:regularization_losses
;	variables
<trainable_variables
=	keras_api
h

>kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
R
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
R
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
п
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemБmцmЦmд(mД)mе.mЕ/mф4mФ5mг>mГ?m«Lm»Mm░v▒v▓v│v┤(vх)vХ.vи/vИ4v╣5v║>v╗?v╝LvйMvЙ
 
f
0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
L12
M13
f
0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
L12
M13
Г
Wlayer_metrics
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
regularization_losses
	variables
trainable_variables
[metrics
 
\Z
VARIABLE_VALUEconv2d_60/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_60/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
\layer_metrics
]layer_regularization_losses
^non_trainable_variables

_layers
regularization_losses
	variables
trainable_variables
`metrics
 
 
 
Г
alayer_metrics
blayer_regularization_losses
cnon_trainable_variables

dlayers
regularization_losses
	variables
trainable_variables
emetrics
\Z
VARIABLE_VALUEconv2d_61/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_61/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
flayer_metrics
glayer_regularization_losses
hnon_trainable_variables

ilayers
 regularization_losses
!	variables
"trainable_variables
jmetrics
 
 
 
Г
klayer_metrics
llayer_regularization_losses
mnon_trainable_variables

nlayers
$regularization_losses
%	variables
&trainable_variables
ometrics
\Z
VARIABLE_VALUEconv2d_62/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_62/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
Г
player_metrics
qlayer_regularization_losses
rnon_trainable_variables

slayers
*regularization_losses
+	variables
,trainable_variables
tmetrics
\Z
VARIABLE_VALUEconv2d_63/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_63/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
Г
ulayer_metrics
vlayer_regularization_losses
wnon_trainable_variables

xlayers
0regularization_losses
1	variables
2trainable_variables
ymetrics
\Z
VARIABLE_VALUEconv2d_64/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_64/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
Г
zlayer_metrics
{layer_regularization_losses
|non_trainable_variables

}layers
6regularization_losses
7	variables
8trainable_variables
~metrics
 
 
 
▒
layer_metrics
 ђlayer_regularization_losses
Ђnon_trainable_variables
ѓlayers
:regularization_losses
;	variables
<trainable_variables
Ѓmetrics
\Z
VARIABLE_VALUEconv2d_65/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_65/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

>0
?1
▓
ёlayer_metrics
 Ёlayer_regularization_losses
єnon_trainable_variables
Єlayers
@regularization_losses
A	variables
Btrainable_variables
ѕmetrics
 
 
 
▓
Ѕlayer_metrics
 іlayer_regularization_losses
Іnon_trainable_variables
їlayers
Dregularization_losses
E	variables
Ftrainable_variables
Їmetrics
 
 
 
▓
јlayer_metrics
 Јlayer_regularization_losses
љnon_trainable_variables
Љlayers
Hregularization_losses
I	variables
Jtrainable_variables
њmetrics
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
▓
Њlayer_metrics
 ћlayer_regularization_losses
Ћnon_trainable_variables
ќlayers
Nregularization_losses
O	variables
Ptrainable_variables
Ќmetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
^
0
1
2
3
4
5
6
7
	8

9
10
11
12

ў0
Ў1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

џtotal

Џcount
ю	variables
Ю	keras_api
I

ъtotal

Ъcount
а
_fn_kwargs
А	variables
б	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

џ0
Џ1

ю	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ъ0
Ъ1

А	variables
}
VARIABLE_VALUEAdam/conv2d_60/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_60/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_61/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_61/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_62/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_62/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_63/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_64/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_64/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_65/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_65/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_60/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_60/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_61/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_61/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_62/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_62/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_63/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_63/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_64/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_64/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_65/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_65/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
І
serving_default_input_11Placeholder*/
_output_shapes
:         @ *
dtype0*$
shape:         @ 
┐
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11conv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_64/kernelconv2d_64/biasconv2d_65/kernelconv2d_65/biasdense_10/kerneldense_10/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *,
f'R%
#__inference_signature_wrapper_61263
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╗
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp$conv2d_61/kernel/Read/ReadVariableOp"conv2d_61/bias/Read/ReadVariableOp$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp$conv2d_64/kernel/Read/ReadVariableOp"conv2d_64/bias/Read/ReadVariableOp$conv2d_65/kernel/Read/ReadVariableOp"conv2d_65/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_60/kernel/m/Read/ReadVariableOp)Adam/conv2d_60/bias/m/Read/ReadVariableOp+Adam/conv2d_61/kernel/m/Read/ReadVariableOp)Adam/conv2d_61/bias/m/Read/ReadVariableOp+Adam/conv2d_62/kernel/m/Read/ReadVariableOp)Adam/conv2d_62/bias/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp+Adam/conv2d_64/kernel/m/Read/ReadVariableOp)Adam/conv2d_64/bias/m/Read/ReadVariableOp+Adam/conv2d_65/kernel/m/Read/ReadVariableOp)Adam/conv2d_65/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp+Adam/conv2d_60/kernel/v/Read/ReadVariableOp)Adam/conv2d_60/bias/v/Read/ReadVariableOp+Adam/conv2d_61/kernel/v/Read/ReadVariableOp)Adam/conv2d_61/bias/v/Read/ReadVariableOp+Adam/conv2d_62/kernel/v/Read/ReadVariableOp)Adam/conv2d_62/bias/v/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp+Adam/conv2d_64/kernel/v/Read/ReadVariableOp)Adam/conv2d_64/bias/v/Read/ReadVariableOp+Adam/conv2d_65/kernel/v/Read/ReadVariableOp)Adam/conv2d_65/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOpConst*@
Tin9
725	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *'
f"R 
__inference__traced_save_61855
║

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_60/kernelconv2d_60/biasconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasconv2d_63/kernelconv2d_63/biasconv2d_64/kernelconv2d_64/biasconv2d_65/kernelconv2d_65/biasdense_10/kerneldense_10/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_60/kernel/mAdam/conv2d_60/bias/mAdam/conv2d_61/kernel/mAdam/conv2d_61/bias/mAdam/conv2d_62/kernel/mAdam/conv2d_62/bias/mAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/mAdam/conv2d_64/kernel/mAdam/conv2d_64/bias/mAdam/conv2d_65/kernel/mAdam/conv2d_65/bias/mAdam/dense_10/kernel/mAdam/dense_10/bias/mAdam/conv2d_60/kernel/vAdam/conv2d_60/bias/vAdam/conv2d_61/kernel/vAdam/conv2d_61/bias/vAdam/conv2d_62/kernel/vAdam/conv2d_62/bias/vAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/vAdam/conv2d_64/kernel/vAdam/conv2d_64/bias/vAdam/conv2d_65/kernel/vAdam/conv2d_65/bias/vAdam/dense_10/kernel/vAdam/dense_10/bias/v*?
Tin8
624*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ **
f%R#
!__inference__traced_restore_62018¤у	
¤S
│
C__inference_model_10_layer_call_and_return_conditional_losses_61322

inputsB
(conv2d_60_conv2d_readvariableop_resource:@7
)conv2d_60_biasadd_readvariableop_resource:@C
(conv2d_61_conv2d_readvariableop_resource:@ђ8
)conv2d_61_biasadd_readvariableop_resource:	ђD
(conv2d_62_conv2d_readvariableop_resource:ђђ8
)conv2d_62_biasadd_readvariableop_resource:	ђD
(conv2d_63_conv2d_readvariableop_resource:ђђ8
)conv2d_63_biasadd_readvariableop_resource:	ђD
(conv2d_64_conv2d_readvariableop_resource:ђђ8
)conv2d_64_biasadd_readvariableop_resource:	ђD
(conv2d_65_conv2d_readvariableop_resource:ђђ8
)conv2d_65_biasadd_readvariableop_resource:	ђ:
'dense_10_matmul_readvariableop_resource:	ђ6
(dense_10_biasadd_readvariableop_resource:
identityѕб conv2d_60/BiasAdd/ReadVariableOpбconv2d_60/Conv2D/ReadVariableOpб conv2d_61/BiasAdd/ReadVariableOpбconv2d_61/Conv2D/ReadVariableOpб conv2d_62/BiasAdd/ReadVariableOpбconv2d_62/Conv2D/ReadVariableOpб conv2d_63/BiasAdd/ReadVariableOpбconv2d_63/Conv2D/ReadVariableOpб conv2d_64/BiasAdd/ReadVariableOpбconv2d_64/Conv2D/ReadVariableOpб conv2d_65/BiasAdd/ReadVariableOpбconv2d_65/Conv2D/ReadVariableOpбdense_10/BiasAdd/ReadVariableOpбdense_10/MatMul/ReadVariableOp│
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_60/Conv2D/ReadVariableOp┴
conv2d_60/Conv2DConv2Dinputs'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @*
paddingSAME*
strides
2
conv2d_60/Conv2Dф
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp░
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @2
conv2d_60/BiasAdd~
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:         @ @2
conv2d_60/Relu█
average_pooling2d_30/AvgPoolAvgPoolconv2d_60/Relu:activations:0*
T0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_30/AvgPool┤
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_61/Conv2D/ReadVariableOpр
conv2d_61/Conv2DConv2D%average_pooling2d_30/AvgPool:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ*
paddingSAME*
strides
2
conv2d_61/Conv2DФ
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp▒
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ2
conv2d_61/BiasAdd
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:          ђ2
conv2d_61/Relu▄
average_pooling2d_31/AvgPoolAvgPoolconv2d_61/Relu:activations:0*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_31/AvgPoolх
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_62/Conv2D/ReadVariableOpр
conv2d_62/Conv2DConv2D%average_pooling2d_31/AvgPool:output:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_62/Conv2DФ
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp▒
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_62/BiasAdd
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_62/Reluх
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_63/Conv2D/ReadVariableOpп
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_63/Conv2DФ
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp▒
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_63/BiasAdd
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_63/Reluх
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_64/Conv2D/ReadVariableOpп
conv2d_64/Conv2DConv2Dconv2d_63/Relu:activations:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_64/Conv2DФ
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_64/BiasAdd/ReadVariableOp▒
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_64/BiasAdd
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_64/Relu▄
average_pooling2d_32/AvgPoolAvgPoolconv2d_64/Relu:activations:0*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_32/AvgPoolх
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_65/Conv2D/ReadVariableOpр
conv2d_65/Conv2DConv2D%average_pooling2d_32/AvgPool:output:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_65/Conv2DФ
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_65/BiasAdd/ReadVariableOp▒
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_65/BiasAdd
conv2d_65/ReluReluconv2d_65/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_65/Relu╣
2global_average_pooling2d_10/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_10/Mean/reduction_indices┌
 global_average_pooling2d_10/MeanMeanconv2d_65/Relu:activations:0;global_average_pooling2d_10/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2"
 global_average_pooling2d_10/Meanu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_10/Constг
flatten_10/ReshapeReshape)global_average_pooling2d_10/Mean:output:0flatten_10/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_10/ReshapeЕ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_10/MatMul/ReadVariableOpБ
dense_10/MatMulMatMulflatten_10/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/MatMulД
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЦ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/BiasAddt
IdentityIdentitydense_10/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity»
NoOpNoOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
й
W
;__inference_global_average_pooling2d_10_layer_call_fn_61644

inputs
identityР
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_606912
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
д
А
)__inference_conv2d_64_layer_call_fn_61587

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_608032
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_62_layer_call_and_return_conditional_losses_61538

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_60668

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Е

ш
C__inference_dense_10_layer_call_and_return_conditional_losses_61670

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_62_layer_call_and_return_conditional_losses_60769

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
╦
k
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_61477

inputs
identityЏ
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @ @:W S
/
_output_shapes
:         @ @
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_60624

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦
k
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_60733

inputs
identityЏ
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @ @:W S
/
_output_shapes
:         @ @
 
_user_specified_nameinputs
К
Ю
(__inference_model_10_layer_call_fn_60895
input_11!
unknown:@
	unknown_0:@$
	unknown_1:@ђ
	unknown_2:	ђ%
	unknown_3:ђђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_608642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @ 
"
_user_specified_name
input_11
Т
r
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_60691

inputs
identityЂ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
¤
k
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_61597

inputs
identityю
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Е

ш
C__inference_dense_10_layer_call_and_return_conditional_losses_60857

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Х▄
▓ 
!__inference__traced_restore_62018
file_prefix;
!assignvariableop_conv2d_60_kernel:@/
!assignvariableop_1_conv2d_60_bias:@>
#assignvariableop_2_conv2d_61_kernel:@ђ0
!assignvariableop_3_conv2d_61_bias:	ђ?
#assignvariableop_4_conv2d_62_kernel:ђђ0
!assignvariableop_5_conv2d_62_bias:	ђ?
#assignvariableop_6_conv2d_63_kernel:ђђ0
!assignvariableop_7_conv2d_63_bias:	ђ?
#assignvariableop_8_conv2d_64_kernel:ђђ0
!assignvariableop_9_conv2d_64_bias:	ђ@
$assignvariableop_10_conv2d_65_kernel:ђђ1
"assignvariableop_11_conv2d_65_bias:	ђ6
#assignvariableop_12_dense_10_kernel:	ђ/
!assignvariableop_13_dense_10_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: E
+assignvariableop_23_adam_conv2d_60_kernel_m:@7
)assignvariableop_24_adam_conv2d_60_bias_m:@F
+assignvariableop_25_adam_conv2d_61_kernel_m:@ђ8
)assignvariableop_26_adam_conv2d_61_bias_m:	ђG
+assignvariableop_27_adam_conv2d_62_kernel_m:ђђ8
)assignvariableop_28_adam_conv2d_62_bias_m:	ђG
+assignvariableop_29_adam_conv2d_63_kernel_m:ђђ8
)assignvariableop_30_adam_conv2d_63_bias_m:	ђG
+assignvariableop_31_adam_conv2d_64_kernel_m:ђђ8
)assignvariableop_32_adam_conv2d_64_bias_m:	ђG
+assignvariableop_33_adam_conv2d_65_kernel_m:ђђ8
)assignvariableop_34_adam_conv2d_65_bias_m:	ђ=
*assignvariableop_35_adam_dense_10_kernel_m:	ђ6
(assignvariableop_36_adam_dense_10_bias_m:E
+assignvariableop_37_adam_conv2d_60_kernel_v:@7
)assignvariableop_38_adam_conv2d_60_bias_v:@F
+assignvariableop_39_adam_conv2d_61_kernel_v:@ђ8
)assignvariableop_40_adam_conv2d_61_bias_v:	ђG
+assignvariableop_41_adam_conv2d_62_kernel_v:ђђ8
)assignvariableop_42_adam_conv2d_62_bias_v:	ђG
+assignvariableop_43_adam_conv2d_63_kernel_v:ђђ8
)assignvariableop_44_adam_conv2d_63_bias_v:	ђG
+assignvariableop_45_adam_conv2d_64_kernel_v:ђђ8
)assignvariableop_46_adam_conv2d_64_bias_v:	ђG
+assignvariableop_47_adam_conv2d_65_kernel_v:ђђ8
)assignvariableop_48_adam_conv2d_65_bias_v:	ђ=
*assignvariableop_49_adam_dense_10_kernel_v:	ђ6
(assignvariableop_50_adam_dense_10_bias_v:
identity_52ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Щ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*є
valueЧBщ4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesШ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices▓
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Т
_output_shapesМ
л::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityа
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_60_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_60_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2е
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_61_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3д
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_61_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4е
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_62_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5д
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_62_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6е
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_63_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7д
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_63_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8е
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_64_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9д
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_64_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10г
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_65_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ф
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_65_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ф
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_10_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Е
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_10_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14Ц
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Д
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Д
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17д
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19А
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20А
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Б
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Б
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23│
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_60_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▒
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_60_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25│
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_61_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26▒
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_61_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27│
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_62_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▒
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_62_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29│
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_63_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_63_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31│
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_64_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32▒
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_64_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33│
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_65_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_65_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35▓
AssignVariableOp_35AssignVariableOp*assignvariableop_35_adam_dense_10_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36░
AssignVariableOp_36AssignVariableOp(assignvariableop_36_adam_dense_10_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37│
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_60_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_60_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39│
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_61_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40▒
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_61_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41│
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_62_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42▒
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_62_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43│
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_63_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44▒
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_63_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45│
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_64_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46▒
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_64_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47│
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_65_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48▒
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_65_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49▓
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_dense_10_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50░
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_dense_10_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp└	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51f
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_52е	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_52Identity_52:output:0*{
_input_shapesj
h: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ч
P
4__inference_average_pooling2d_31_layer_call_fn_61527

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_607562
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:          ђ:X T
0
_output_shapes
:          ђ
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_65_layer_call_and_return_conditional_losses_60826

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
щ
W
;__inference_global_average_pooling2d_10_layer_call_fn_61649

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_608372
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_60646

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
пj
Ё
__inference__traced_save_61855
file_prefix/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop/
+savev2_conv2d_61_kernel_read_readvariableop-
)savev2_conv2d_61_bias_read_readvariableop/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop/
+savev2_conv2d_64_kernel_read_readvariableop-
)savev2_conv2d_64_bias_read_readvariableop/
+savev2_conv2d_65_kernel_read_readvariableop-
)savev2_conv2d_65_bias_read_readvariableop.
*savev2_dense_10_kernel_read_readvariableop,
(savev2_dense_10_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_60_kernel_m_read_readvariableop4
0savev2_adam_conv2d_60_bias_m_read_readvariableop6
2savev2_adam_conv2d_61_kernel_m_read_readvariableop4
0savev2_adam_conv2d_61_bias_m_read_readvariableop6
2savev2_adam_conv2d_62_kernel_m_read_readvariableop4
0savev2_adam_conv2d_62_bias_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableop6
2savev2_adam_conv2d_64_kernel_m_read_readvariableop4
0savev2_adam_conv2d_64_bias_m_read_readvariableop6
2savev2_adam_conv2d_65_kernel_m_read_readvariableop4
0savev2_adam_conv2d_65_bias_m_read_readvariableop5
1savev2_adam_dense_10_kernel_m_read_readvariableop3
/savev2_adam_dense_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_60_kernel_v_read_readvariableop4
0savev2_adam_conv2d_60_bias_v_read_readvariableop6
2savev2_adam_conv2d_61_kernel_v_read_readvariableop4
0savev2_adam_conv2d_61_bias_v_read_readvariableop6
2savev2_adam_conv2d_62_kernel_v_read_readvariableop4
0savev2_adam_conv2d_62_bias_v_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableop6
2savev2_adam_conv2d_64_kernel_v_read_readvariableop4
0savev2_adam_conv2d_64_bias_v_read_readvariableop6
2savev2_adam_conv2d_65_kernel_v_read_readvariableop4
0savev2_adam_conv2d_65_bias_v_read_readvariableop5
1savev2_adam_dense_10_kernel_v_read_readvariableop3
/savev2_adam_dense_10_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЗ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*є
valueЧBщ4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices╗
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop+savev2_conv2d_61_kernel_read_readvariableop)savev2_conv2d_61_bias_read_readvariableop+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop+savev2_conv2d_64_kernel_read_readvariableop)savev2_conv2d_64_bias_read_readvariableop+savev2_conv2d_65_kernel_read_readvariableop)savev2_conv2d_65_bias_read_readvariableop*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_60_kernel_m_read_readvariableop0savev2_adam_conv2d_60_bias_m_read_readvariableop2savev2_adam_conv2d_61_kernel_m_read_readvariableop0savev2_adam_conv2d_61_bias_m_read_readvariableop2savev2_adam_conv2d_62_kernel_m_read_readvariableop0savev2_adam_conv2d_62_bias_m_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop2savev2_adam_conv2d_64_kernel_m_read_readvariableop0savev2_adam_conv2d_64_bias_m_read_readvariableop2savev2_adam_conv2d_65_kernel_m_read_readvariableop0savev2_adam_conv2d_65_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop2savev2_adam_conv2d_60_kernel_v_read_readvariableop0savev2_adam_conv2d_60_bias_v_read_readvariableop2savev2_adam_conv2d_61_kernel_v_read_readvariableop0savev2_adam_conv2d_61_bias_v_read_readvariableop2savev2_adam_conv2d_62_kernel_v_read_readvariableop0savev2_adam_conv2d_62_bias_v_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop2savev2_adam_conv2d_64_kernel_v_read_readvariableop0savev2_adam_conv2d_64_bias_v_read_readvariableop2savev2_adam_conv2d_65_kernel_v_read_readvariableop0savev2_adam_conv2d_65_bias_v_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*И
_input_shapesд
Б: :@:@:@ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:	ђ:: : : : : : : : : :@:@:@ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:	ђ::@:@:@ђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:ђђ:ђ:	ђ:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.	*
(
_output_shapes
:ђђ:!


_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@ђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:.*
(
_output_shapes
:ђђ:!

_output_shapes	
:ђ:. *
(
_output_shapes
:ђђ:!!

_output_shapes	
:ђ:."*
(
_output_shapes
:ђђ:!#

_output_shapes	
:ђ:%$!

_output_shapes
:	ђ: %

_output_shapes
::,&(
&
_output_shapes
:@: '

_output_shapes
:@:-()
'
_output_shapes
:@ђ:!)

_output_shapes	
:ђ:.**
(
_output_shapes
:ђђ:!+

_output_shapes	
:ђ:.,*
(
_output_shapes
:ђђ:!-

_output_shapes	
:ђ:..*
(
_output_shapes
:ђђ:!/

_output_shapes	
:ђ:.0*
(
_output_shapes
:ђђ:!1

_output_shapes	
:ђ:%2!

_output_shapes
:	ђ: 3

_output_shapes
::4

_output_shapes
: 
б
r
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_60837

inputs
identityЂ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_63_layer_call_and_return_conditional_losses_60786

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
э
P
4__inference_average_pooling2d_30_layer_call_fn_61487

inputs
identity┌
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_607332
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:          @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @ @:W S
/
_output_shapes
:         @ @
 
_user_specified_nameinputs
Ж
§
D__inference_conv2d_60_layer_call_and_return_conditional_losses_61458

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @ @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @ @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
┘
a
E__inference_flatten_10_layer_call_and_return_conditional_losses_60845

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
¤
k
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_61517

inputs
identityю
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:          ђ:X T
0
_output_shapes
:          ђ
 
_user_specified_nameinputs
д
А
)__inference_conv2d_65_layer_call_fn_61627

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_608262
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
╩9
У
C__inference_model_10_layer_call_and_return_conditional_losses_60864

inputs)
conv2d_60_60724:@
conv2d_60_60726:@*
conv2d_61_60747:@ђ
conv2d_61_60749:	ђ+
conv2d_62_60770:ђђ
conv2d_62_60772:	ђ+
conv2d_63_60787:ђђ
conv2d_63_60789:	ђ+
conv2d_64_60804:ђђ
conv2d_64_60806:	ђ+
conv2d_65_60827:ђђ
conv2d_65_60829:	ђ!
dense_10_60858:	ђ
dense_10_60860:
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб!conv2d_62/StatefulPartitionedCallб!conv2d_63/StatefulPartitionedCallб!conv2d_64/StatefulPartitionedCallб!conv2d_65/StatefulPartitionedCallб dense_10/StatefulPartitionedCallБ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_60_60724conv2d_60_60726*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_607232#
!conv2d_60/StatefulPartitionedCallе
$average_pooling2d_30/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_607332&
$average_pooling2d_30/PartitionedCall╦
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_30/PartitionedCall:output:0conv2d_61_60747conv2d_61_60749*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_607462#
!conv2d_61/StatefulPartitionedCallЕ
$average_pooling2d_31/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_607562&
$average_pooling2d_31/PartitionedCall╦
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_31/PartitionedCall:output:0conv2d_62_60770conv2d_62_60772*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_607692#
!conv2d_62/StatefulPartitionedCall╚
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_60787conv2d_63_60789*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_607862#
!conv2d_63/StatefulPartitionedCall╚
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_60804conv2d_64_60806*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_608032#
!conv2d_64/StatefulPartitionedCallЕ
$average_pooling2d_32/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_608132&
$average_pooling2d_32/PartitionedCall╦
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_32/PartitionedCall:output:0conv2d_65_60827conv2d_65_60829*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_608262#
!conv2d_65/StatefulPartitionedCallХ
+global_average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_608372-
+global_average_pooling2d_10/PartitionedCallЇ
flatten_10/PartitionedCallPartitionedCall4global_average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_608452
flatten_10/PartitionedCall│
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_10_60858dense_10_60860*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_608572"
 dense_10/StatefulPartitionedCallё
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╔
NoOpNoOp"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
Ъ
ъ
)__inference_conv2d_60_layer_call_fn_61467

inputs!
unknown:@
	unknown_0:@
identityѕбStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_607232
StatefulPartitionedCallЃ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @ @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
б
r
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_61639

inputs
identityЂ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_61592

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
С
P
4__inference_average_pooling2d_32_layer_call_fn_61602

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_606682
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Б
а
)__inference_conv2d_61_layer_call_fn_61507

inputs"
unknown:@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_607462
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:          ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
¤
k
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_60813

inputs
identityю
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ж
§
D__inference_conv2d_60_layer_call_and_return_conditional_losses_60723

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЋ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpБ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @*
paddingSAME*
strides
2
Conv2Dї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         @ @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @ @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
л9
Ж
C__inference_model_10_layer_call_and_return_conditional_losses_61222
input_11)
conv2d_60_61181:@
conv2d_60_61183:@*
conv2d_61_61187:@ђ
conv2d_61_61189:	ђ+
conv2d_62_61193:ђђ
conv2d_62_61195:	ђ+
conv2d_63_61198:ђђ
conv2d_63_61200:	ђ+
conv2d_64_61203:ђђ
conv2d_64_61205:	ђ+
conv2d_65_61209:ђђ
conv2d_65_61211:	ђ!
dense_10_61216:	ђ
dense_10_61218:
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб!conv2d_62/StatefulPartitionedCallб!conv2d_63/StatefulPartitionedCallб!conv2d_64/StatefulPartitionedCallб!conv2d_65/StatefulPartitionedCallб dense_10/StatefulPartitionedCallЦ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinput_11conv2d_60_61181conv2d_60_61183*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_607232#
!conv2d_60/StatefulPartitionedCallе
$average_pooling2d_30/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_607332&
$average_pooling2d_30/PartitionedCall╦
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_30/PartitionedCall:output:0conv2d_61_61187conv2d_61_61189*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_607462#
!conv2d_61/StatefulPartitionedCallЕ
$average_pooling2d_31/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_607562&
$average_pooling2d_31/PartitionedCall╦
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_31/PartitionedCall:output:0conv2d_62_61193conv2d_62_61195*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_607692#
!conv2d_62/StatefulPartitionedCall╚
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_61198conv2d_63_61200*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_607862#
!conv2d_63/StatefulPartitionedCall╚
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_61203conv2d_64_61205*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_608032#
!conv2d_64/StatefulPartitionedCallЕ
$average_pooling2d_32/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_608132&
$average_pooling2d_32/PartitionedCall╦
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_32/PartitionedCall:output:0conv2d_65_61209conv2d_65_61211*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_608262#
!conv2d_65/StatefulPartitionedCallХ
+global_average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_608372-
+global_average_pooling2d_10/PartitionedCallЇ
flatten_10/PartitionedCallPartitionedCall4global_average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_608452
flatten_10/PartitionedCall│
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_10_61216dense_10_61218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_608572"
 dense_10/StatefulPartitionedCallё
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╔
NoOpNoOp"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:Y U
/
_output_shapes
:         @ 
"
_user_specified_name
input_11
С
P
4__inference_average_pooling2d_31_layer_call_fn_61522

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_606462
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
ђb
ј
 __inference__wrapped_model_60615
input_11K
1model_10_conv2d_60_conv2d_readvariableop_resource:@@
2model_10_conv2d_60_biasadd_readvariableop_resource:@L
1model_10_conv2d_61_conv2d_readvariableop_resource:@ђA
2model_10_conv2d_61_biasadd_readvariableop_resource:	ђM
1model_10_conv2d_62_conv2d_readvariableop_resource:ђђA
2model_10_conv2d_62_biasadd_readvariableop_resource:	ђM
1model_10_conv2d_63_conv2d_readvariableop_resource:ђђA
2model_10_conv2d_63_biasadd_readvariableop_resource:	ђM
1model_10_conv2d_64_conv2d_readvariableop_resource:ђђA
2model_10_conv2d_64_biasadd_readvariableop_resource:	ђM
1model_10_conv2d_65_conv2d_readvariableop_resource:ђђA
2model_10_conv2d_65_biasadd_readvariableop_resource:	ђC
0model_10_dense_10_matmul_readvariableop_resource:	ђ?
1model_10_dense_10_biasadd_readvariableop_resource:
identityѕб)model_10/conv2d_60/BiasAdd/ReadVariableOpб(model_10/conv2d_60/Conv2D/ReadVariableOpб)model_10/conv2d_61/BiasAdd/ReadVariableOpб(model_10/conv2d_61/Conv2D/ReadVariableOpб)model_10/conv2d_62/BiasAdd/ReadVariableOpб(model_10/conv2d_62/Conv2D/ReadVariableOpб)model_10/conv2d_63/BiasAdd/ReadVariableOpб(model_10/conv2d_63/Conv2D/ReadVariableOpб)model_10/conv2d_64/BiasAdd/ReadVariableOpб(model_10/conv2d_64/Conv2D/ReadVariableOpб)model_10/conv2d_65/BiasAdd/ReadVariableOpб(model_10/conv2d_65/Conv2D/ReadVariableOpб(model_10/dense_10/BiasAdd/ReadVariableOpб'model_10/dense_10/MatMul/ReadVariableOp╬
(model_10/conv2d_60/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02*
(model_10/conv2d_60/Conv2D/ReadVariableOpя
model_10/conv2d_60/Conv2DConv2Dinput_110model_10/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @*
paddingSAME*
strides
2
model_10/conv2d_60/Conv2D┼
)model_10/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)model_10/conv2d_60/BiasAdd/ReadVariableOpн
model_10/conv2d_60/BiasAddBiasAdd"model_10/conv2d_60/Conv2D:output:01model_10/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @2
model_10/conv2d_60/BiasAddЎ
model_10/conv2d_60/ReluRelu#model_10/conv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:         @ @2
model_10/conv2d_60/ReluШ
%model_10/average_pooling2d_30/AvgPoolAvgPool%model_10/conv2d_60/Relu:activations:0*
T0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
2'
%model_10/average_pooling2d_30/AvgPool¤
(model_10/conv2d_61/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02*
(model_10/conv2d_61/Conv2D/ReadVariableOpЁ
model_10/conv2d_61/Conv2DConv2D.model_10/average_pooling2d_30/AvgPool:output:00model_10/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ*
paddingSAME*
strides
2
model_10/conv2d_61/Conv2Dк
)model_10/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)model_10/conv2d_61/BiasAdd/ReadVariableOpН
model_10/conv2d_61/BiasAddBiasAdd"model_10/conv2d_61/Conv2D:output:01model_10/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ2
model_10/conv2d_61/BiasAddџ
model_10/conv2d_61/ReluRelu#model_10/conv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:          ђ2
model_10/conv2d_61/Reluэ
%model_10/average_pooling2d_31/AvgPoolAvgPool%model_10/conv2d_61/Relu:activations:0*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%model_10/average_pooling2d_31/AvgPoolл
(model_10/conv2d_62/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02*
(model_10/conv2d_62/Conv2D/ReadVariableOpЁ
model_10/conv2d_62/Conv2DConv2D.model_10/average_pooling2d_31/AvgPool:output:00model_10/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
model_10/conv2d_62/Conv2Dк
)model_10/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)model_10/conv2d_62/BiasAdd/ReadVariableOpН
model_10/conv2d_62/BiasAddBiasAdd"model_10/conv2d_62/Conv2D:output:01model_10/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_62/BiasAddџ
model_10/conv2d_62/ReluRelu#model_10/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_62/Reluл
(model_10/conv2d_63/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02*
(model_10/conv2d_63/Conv2D/ReadVariableOpЧ
model_10/conv2d_63/Conv2DConv2D%model_10/conv2d_62/Relu:activations:00model_10/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
model_10/conv2d_63/Conv2Dк
)model_10/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)model_10/conv2d_63/BiasAdd/ReadVariableOpН
model_10/conv2d_63/BiasAddBiasAdd"model_10/conv2d_63/Conv2D:output:01model_10/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_63/BiasAddџ
model_10/conv2d_63/ReluRelu#model_10/conv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_63/Reluл
(model_10/conv2d_64/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_64_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02*
(model_10/conv2d_64/Conv2D/ReadVariableOpЧ
model_10/conv2d_64/Conv2DConv2D%model_10/conv2d_63/Relu:activations:00model_10/conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
model_10/conv2d_64/Conv2Dк
)model_10/conv2d_64/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_64_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)model_10/conv2d_64/BiasAdd/ReadVariableOpН
model_10/conv2d_64/BiasAddBiasAdd"model_10/conv2d_64/Conv2D:output:01model_10/conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_64/BiasAddџ
model_10/conv2d_64/ReluRelu#model_10/conv2d_64/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_64/Reluэ
%model_10/average_pooling2d_32/AvgPoolAvgPool%model_10/conv2d_64/Relu:activations:0*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2'
%model_10/average_pooling2d_32/AvgPoolл
(model_10/conv2d_65/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_65_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02*
(model_10/conv2d_65/Conv2D/ReadVariableOpЁ
model_10/conv2d_65/Conv2DConv2D.model_10/average_pooling2d_32/AvgPool:output:00model_10/conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
model_10/conv2d_65/Conv2Dк
)model_10/conv2d_65/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_65_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02+
)model_10/conv2d_65/BiasAdd/ReadVariableOpН
model_10/conv2d_65/BiasAddBiasAdd"model_10/conv2d_65/Conv2D:output:01model_10/conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_65/BiasAddџ
model_10/conv2d_65/ReluRelu#model_10/conv2d_65/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
model_10/conv2d_65/Relu╦
;model_10/global_average_pooling2d_10/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2=
;model_10/global_average_pooling2d_10/Mean/reduction_indices■
)model_10/global_average_pooling2d_10/MeanMean%model_10/conv2d_65/Relu:activations:0Dmodel_10/global_average_pooling2d_10/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2+
)model_10/global_average_pooling2d_10/MeanЄ
model_10/flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
model_10/flatten_10/Constл
model_10/flatten_10/ReshapeReshape2model_10/global_average_pooling2d_10/Mean:output:0"model_10/flatten_10/Const:output:0*
T0*(
_output_shapes
:         ђ2
model_10/flatten_10/Reshape─
'model_10/dense_10/MatMul/ReadVariableOpReadVariableOp0model_10_dense_10_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02)
'model_10/dense_10/MatMul/ReadVariableOpК
model_10/dense_10/MatMulMatMul$model_10/flatten_10/Reshape:output:0/model_10/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_10/dense_10/MatMul┬
(model_10/dense_10/BiasAdd/ReadVariableOpReadVariableOp1model_10_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_10/dense_10/BiasAdd/ReadVariableOp╔
model_10/dense_10/BiasAddBiasAdd"model_10/dense_10/MatMul:product:00model_10/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_10/dense_10/BiasAdd}
IdentityIdentity"model_10/dense_10/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

IdentityГ
NoOpNoOp*^model_10/conv2d_60/BiasAdd/ReadVariableOp)^model_10/conv2d_60/Conv2D/ReadVariableOp*^model_10/conv2d_61/BiasAdd/ReadVariableOp)^model_10/conv2d_61/Conv2D/ReadVariableOp*^model_10/conv2d_62/BiasAdd/ReadVariableOp)^model_10/conv2d_62/Conv2D/ReadVariableOp*^model_10/conv2d_63/BiasAdd/ReadVariableOp)^model_10/conv2d_63/Conv2D/ReadVariableOp*^model_10/conv2d_64/BiasAdd/ReadVariableOp)^model_10/conv2d_64/Conv2D/ReadVariableOp*^model_10/conv2d_65/BiasAdd/ReadVariableOp)^model_10/conv2d_65/Conv2D/ReadVariableOp)^model_10/dense_10/BiasAdd/ReadVariableOp(^model_10/dense_10/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 2V
)model_10/conv2d_60/BiasAdd/ReadVariableOp)model_10/conv2d_60/BiasAdd/ReadVariableOp2T
(model_10/conv2d_60/Conv2D/ReadVariableOp(model_10/conv2d_60/Conv2D/ReadVariableOp2V
)model_10/conv2d_61/BiasAdd/ReadVariableOp)model_10/conv2d_61/BiasAdd/ReadVariableOp2T
(model_10/conv2d_61/Conv2D/ReadVariableOp(model_10/conv2d_61/Conv2D/ReadVariableOp2V
)model_10/conv2d_62/BiasAdd/ReadVariableOp)model_10/conv2d_62/BiasAdd/ReadVariableOp2T
(model_10/conv2d_62/Conv2D/ReadVariableOp(model_10/conv2d_62/Conv2D/ReadVariableOp2V
)model_10/conv2d_63/BiasAdd/ReadVariableOp)model_10/conv2d_63/BiasAdd/ReadVariableOp2T
(model_10/conv2d_63/Conv2D/ReadVariableOp(model_10/conv2d_63/Conv2D/ReadVariableOp2V
)model_10/conv2d_64/BiasAdd/ReadVariableOp)model_10/conv2d_64/BiasAdd/ReadVariableOp2T
(model_10/conv2d_64/Conv2D/ReadVariableOp(model_10/conv2d_64/Conv2D/ReadVariableOp2V
)model_10/conv2d_65/BiasAdd/ReadVariableOp)model_10/conv2d_65/BiasAdd/ReadVariableOp2T
(model_10/conv2d_65/Conv2D/ReadVariableOp(model_10/conv2d_65/Conv2D/ReadVariableOp2T
(model_10/dense_10/BiasAdd/ReadVariableOp(model_10/dense_10/BiasAdd/ReadVariableOp2R
'model_10/dense_10/MatMul/ReadVariableOp'model_10/dense_10/MatMul/ReadVariableOp:Y U
/
_output_shapes
:         @ 
"
_user_specified_name
input_11
╩9
У
C__inference_model_10_layer_call_and_return_conditional_losses_61070

inputs)
conv2d_60_61029:@
conv2d_60_61031:@*
conv2d_61_61035:@ђ
conv2d_61_61037:	ђ+
conv2d_62_61041:ђђ
conv2d_62_61043:	ђ+
conv2d_63_61046:ђђ
conv2d_63_61048:	ђ+
conv2d_64_61051:ђђ
conv2d_64_61053:	ђ+
conv2d_65_61057:ђђ
conv2d_65_61059:	ђ!
dense_10_61064:	ђ
dense_10_61066:
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб!conv2d_62/StatefulPartitionedCallб!conv2d_63/StatefulPartitionedCallб!conv2d_64/StatefulPartitionedCallб!conv2d_65/StatefulPartitionedCallб dense_10/StatefulPartitionedCallБ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_60_61029conv2d_60_61031*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_607232#
!conv2d_60/StatefulPartitionedCallе
$average_pooling2d_30/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_607332&
$average_pooling2d_30/PartitionedCall╦
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_30/PartitionedCall:output:0conv2d_61_61035conv2d_61_61037*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_607462#
!conv2d_61/StatefulPartitionedCallЕ
$average_pooling2d_31/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_607562&
$average_pooling2d_31/PartitionedCall╦
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_31/PartitionedCall:output:0conv2d_62_61041conv2d_62_61043*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_607692#
!conv2d_62/StatefulPartitionedCall╚
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_61046conv2d_63_61048*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_607862#
!conv2d_63/StatefulPartitionedCall╚
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_61051conv2d_64_61053*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_608032#
!conv2d_64/StatefulPartitionedCallЕ
$average_pooling2d_32/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_608132&
$average_pooling2d_32/PartitionedCall╦
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_32/PartitionedCall:output:0conv2d_65_61057conv2d_65_61059*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_608262#
!conv2d_65/StatefulPartitionedCallХ
+global_average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_608372-
+global_average_pooling2d_10/PartitionedCallЇ
flatten_10/PartitionedCallPartitionedCall4global_average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_608452
flatten_10/PartitionedCall│
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_10_61064dense_10_61066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_608572"
 dense_10/StatefulPartitionedCallё
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╔
NoOpNoOp"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
ч
P
4__inference_average_pooling2d_32_layer_call_fn_61607

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_608132
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         ђ:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
┴
Џ
(__inference_model_10_layer_call_fn_61414

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@ђ
	unknown_2:	ђ%
	unknown_3:ђђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:
identityѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_608642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
┴
Џ
(__inference_model_10_layer_call_fn_61447

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@ђ
	unknown_2:	ђ%
	unknown_3:ђђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:
identityѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_610702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
¤S
│
C__inference_model_10_layer_call_and_return_conditional_losses_61381

inputsB
(conv2d_60_conv2d_readvariableop_resource:@7
)conv2d_60_biasadd_readvariableop_resource:@C
(conv2d_61_conv2d_readvariableop_resource:@ђ8
)conv2d_61_biasadd_readvariableop_resource:	ђD
(conv2d_62_conv2d_readvariableop_resource:ђђ8
)conv2d_62_biasadd_readvariableop_resource:	ђD
(conv2d_63_conv2d_readvariableop_resource:ђђ8
)conv2d_63_biasadd_readvariableop_resource:	ђD
(conv2d_64_conv2d_readvariableop_resource:ђђ8
)conv2d_64_biasadd_readvariableop_resource:	ђD
(conv2d_65_conv2d_readvariableop_resource:ђђ8
)conv2d_65_biasadd_readvariableop_resource:	ђ:
'dense_10_matmul_readvariableop_resource:	ђ6
(dense_10_biasadd_readvariableop_resource:
identityѕб conv2d_60/BiasAdd/ReadVariableOpбconv2d_60/Conv2D/ReadVariableOpб conv2d_61/BiasAdd/ReadVariableOpбconv2d_61/Conv2D/ReadVariableOpб conv2d_62/BiasAdd/ReadVariableOpбconv2d_62/Conv2D/ReadVariableOpб conv2d_63/BiasAdd/ReadVariableOpбconv2d_63/Conv2D/ReadVariableOpб conv2d_64/BiasAdd/ReadVariableOpбconv2d_64/Conv2D/ReadVariableOpб conv2d_65/BiasAdd/ReadVariableOpбconv2d_65/Conv2D/ReadVariableOpбdense_10/BiasAdd/ReadVariableOpбdense_10/MatMul/ReadVariableOp│
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_60/Conv2D/ReadVariableOp┴
conv2d_60/Conv2DConv2Dinputs'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @*
paddingSAME*
strides
2
conv2d_60/Conv2Dф
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_60/BiasAdd/ReadVariableOp░
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ @2
conv2d_60/BiasAdd~
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:         @ @2
conv2d_60/Relu█
average_pooling2d_30/AvgPoolAvgPoolconv2d_60/Relu:activations:0*
T0*/
_output_shapes
:          @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_30/AvgPool┤
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02!
conv2d_61/Conv2D/ReadVariableOpр
conv2d_61/Conv2DConv2D%average_pooling2d_30/AvgPool:output:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ*
paddingSAME*
strides
2
conv2d_61/Conv2DФ
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_61/BiasAdd/ReadVariableOp▒
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ2
conv2d_61/BiasAdd
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:          ђ2
conv2d_61/Relu▄
average_pooling2d_31/AvgPoolAvgPoolconv2d_61/Relu:activations:0*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_31/AvgPoolх
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_62/Conv2D/ReadVariableOpр
conv2d_62/Conv2DConv2D%average_pooling2d_31/AvgPool:output:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_62/Conv2DФ
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_62/BiasAdd/ReadVariableOp▒
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_62/BiasAdd
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_62/Reluх
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_63/Conv2D/ReadVariableOpп
conv2d_63/Conv2DConv2Dconv2d_62/Relu:activations:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_63/Conv2DФ
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_63/BiasAdd/ReadVariableOp▒
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_63/BiasAdd
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_63/Reluх
conv2d_64/Conv2D/ReadVariableOpReadVariableOp(conv2d_64_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_64/Conv2D/ReadVariableOpп
conv2d_64/Conv2DConv2Dconv2d_63/Relu:activations:0'conv2d_64/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_64/Conv2DФ
 conv2d_64/BiasAdd/ReadVariableOpReadVariableOp)conv2d_64_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_64/BiasAdd/ReadVariableOp▒
conv2d_64/BiasAddBiasAddconv2d_64/Conv2D:output:0(conv2d_64/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_64/BiasAdd
conv2d_64/ReluReluconv2d_64/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_64/Relu▄
average_pooling2d_32/AvgPoolAvgPoolconv2d_64/Relu:activations:0*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_32/AvgPoolх
conv2d_65/Conv2D/ReadVariableOpReadVariableOp(conv2d_65_conv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02!
conv2d_65/Conv2D/ReadVariableOpр
conv2d_65/Conv2DConv2D%average_pooling2d_32/AvgPool:output:0'conv2d_65/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
conv2d_65/Conv2DФ
 conv2d_65/BiasAdd/ReadVariableOpReadVariableOp)conv2d_65_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02"
 conv2d_65/BiasAdd/ReadVariableOp▒
conv2d_65/BiasAddBiasAddconv2d_65/Conv2D:output:0(conv2d_65/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2
conv2d_65/BiasAdd
conv2d_65/ReluReluconv2d_65/BiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
conv2d_65/Relu╣
2global_average_pooling2d_10/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      24
2global_average_pooling2d_10/Mean/reduction_indices┌
 global_average_pooling2d_10/MeanMeanconv2d_65/Relu:activations:0;global_average_pooling2d_10/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:         ђ2"
 global_average_pooling2d_10/Meanu
flatten_10/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
flatten_10/Constг
flatten_10/ReshapeReshape)global_average_pooling2d_10/Mean:output:0flatten_10/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten_10/ReshapeЕ
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype02 
dense_10/MatMul/ReadVariableOpБ
dense_10/MatMulMatMulflatten_10/Reshape:output:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/MatMulД
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpЦ
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/BiasAddt
IdentityIdentitydense_10/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity»
NoOpNoOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp!^conv2d_64/BiasAdd/ReadVariableOp ^conv2d_64/Conv2D/ReadVariableOp!^conv2d_65/BiasAdd/ReadVariableOp ^conv2d_65/Conv2D/ReadVariableOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2D
 conv2d_64/BiasAdd/ReadVariableOp conv2d_64/BiasAdd/ReadVariableOp2B
conv2d_64/Conv2D/ReadVariableOpconv2d_64/Conv2D/ReadVariableOp2D
 conv2d_65/BiasAdd/ReadVariableOp conv2d_65/BiasAdd/ReadVariableOp2B
conv2d_65/Conv2D/ReadVariableOpconv2d_65/Conv2D/ReadVariableOp2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @ 
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_63_layer_call_and_return_conditional_losses_61558

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
К
Ю
(__inference_model_10_layer_call_fn_61134
input_11!
unknown:@
	unknown_0:@$
	unknown_1:@ђ
	unknown_2:	ђ%
	unknown_3:ђђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_610702
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @ 
"
_user_specified_name
input_11
И
k
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_61472

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
э
ќ
(__inference_dense_10_layer_call_fn_61679

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_608572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
¤
k
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_60756

inputs
identityю
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:         ђ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:          ђ:X T
0
_output_shapes
:          ђ
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_64_layer_call_and_return_conditional_losses_60803

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
л9
Ж
C__inference_model_10_layer_call_and_return_conditional_losses_61178
input_11)
conv2d_60_61137:@
conv2d_60_61139:@*
conv2d_61_61143:@ђ
conv2d_61_61145:	ђ+
conv2d_62_61149:ђђ
conv2d_62_61151:	ђ+
conv2d_63_61154:ђђ
conv2d_63_61156:	ђ+
conv2d_64_61159:ђђ
conv2d_64_61161:	ђ+
conv2d_65_61165:ђђ
conv2d_65_61167:	ђ!
dense_10_61172:	ђ
dense_10_61174:
identityѕб!conv2d_60/StatefulPartitionedCallб!conv2d_61/StatefulPartitionedCallб!conv2d_62/StatefulPartitionedCallб!conv2d_63/StatefulPartitionedCallб!conv2d_64/StatefulPartitionedCallб!conv2d_65/StatefulPartitionedCallб dense_10/StatefulPartitionedCallЦ
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCallinput_11conv2d_60_61137conv2d_60_61139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_60_layer_call_and_return_conditional_losses_607232#
!conv2d_60/StatefulPartitionedCallе
$average_pooling2d_30/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_607332&
$average_pooling2d_30/PartitionedCall╦
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_30/PartitionedCall:output:0conv2d_61_61143conv2d_61_61145*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:          ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_61_layer_call_and_return_conditional_losses_607462#
!conv2d_61/StatefulPartitionedCallЕ
$average_pooling2d_31/PartitionedCallPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_607562&
$average_pooling2d_31/PartitionedCall╦
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_31/PartitionedCall:output:0conv2d_62_61149conv2d_62_61151*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_607692#
!conv2d_62/StatefulPartitionedCall╚
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0conv2d_63_61154conv2d_63_61156*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_607862#
!conv2d_63/StatefulPartitionedCall╚
!conv2d_64/StatefulPartitionedCallStatefulPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0conv2d_64_61159conv2d_64_61161*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_64_layer_call_and_return_conditional_losses_608032#
!conv2d_64/StatefulPartitionedCallЕ
$average_pooling2d_32/PartitionedCallPartitionedCall*conv2d_64/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_608132&
$average_pooling2d_32/PartitionedCall╦
!conv2d_65/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_32/PartitionedCall:output:0conv2d_65_61165conv2d_65_61167*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_65_layer_call_and_return_conditional_losses_608262#
!conv2d_65/StatefulPartitionedCallХ
+global_average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_65/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *_
fZRX
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_608372-
+global_average_pooling2d_10/PartitionedCallЇ
flatten_10/PartitionedCallPartitionedCall4global_average_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_608452
flatten_10/PartitionedCall│
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#flatten_10/PartitionedCall:output:0dense_10_61172dense_10_61174*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *L
fGRE
C__inference_dense_10_layer_call_and_return_conditional_losses_608572"
 dense_10/StatefulPartitionedCallё
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identity╔
NoOpNoOp"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall"^conv2d_64/StatefulPartitionedCall"^conv2d_65/StatefulPartitionedCall!^dense_10/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2F
!conv2d_64/StatefulPartitionedCall!conv2d_64/StatefulPartitionedCall2F
!conv2d_65/StatefulPartitionedCall!conv2d_65/StatefulPartitionedCall2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall:Y U
/
_output_shapes
:         @ 
"
_user_specified_name
input_11
д
А
)__inference_conv2d_63_layer_call_fn_61567

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_63_layer_call_and_return_conditional_losses_607862
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
д
А
)__inference_conv2d_62_layer_call_fn_61547

inputs#
unknown:ђђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         ђ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *M
fHRF
D__inference_conv2d_62_layer_call_and_return_conditional_losses_607692
StatefulPartitionedCallё
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
С
P
4__inference_average_pooling2d_30_layer_call_fn_61482

inputs
identityш
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *X
fSRQ
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_606242
PartitionedCallЈ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_65_layer_call_and_return_conditional_losses_61618

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ы
 
D__inference_conv2d_61_layer_call_and_return_conditional_losses_60746

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:          ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:          ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
┘
a
E__inference_flatten_10_layer_call_and_return_conditional_losses_61655

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
ђ
D__inference_conv2d_64_layer_call_and_return_conditional_losses_61578

inputs:
conv2d_readvariableop_resource:ђђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpЌ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:ђђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:         ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:         ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:         ђ
 
_user_specified_nameinputs
К
F
*__inference_flatten_10_layer_call_fn_61660

inputs
identity╔
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8ѓ *N
fIRG
E__inference_flatten_10_layer_call_and_return_conditional_losses_608452
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Т
r
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_61633

inputs
identityЂ
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ы
 
D__inference_conv2d_61_layer_call_and_return_conditional_losses_61498

inputs9
conv2d_readvariableop_resource:@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@ђ*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ*
paddingSAME*
strides
2
Conv2DЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype02
BiasAdd/ReadVariableOpЅ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:          ђ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:          ђ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:          ђ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          @
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_61512

inputs
identityХ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
2	
AvgPoolЄ
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4                                    2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ъ
ў
#__inference_signature_wrapper_61263
input_11!
unknown:@
	unknown_0:@$
	unknown_1:@ђ
	unknown_2:	ђ%
	unknown_3:ђђ
	unknown_4:	ђ%
	unknown_5:ђђ
	unknown_6:	ђ%
	unknown_7:ђђ
	unknown_8:	ђ%
	unknown_9:ђђ

unknown_10:	ђ

unknown_11:	ђ

unknown_12:
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinput_11unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8ѓ *)
f$R"
 __inference__wrapped_model_606152
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:         @ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:         @ 
"
_user_specified_name
input_11"еL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*х
serving_defaultА
E
input_119
serving_default_input_11:0         @ <
dense_100
StatefulPartitionedCall:0         tensorflow/serving/predict:дя
╚
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer-11
layer_with_weights-6
layer-12
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
┐_default_save_signature
+└&call_and_return_all_conditional_losses
┴__call__"
_tf_keras_network
"
_tf_keras_input_layer
й

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"
_tf_keras_layer
Д
regularization_losses
	variables
trainable_variables
	keras_api
+─&call_and_return_all_conditional_losses
┼__call__"
_tf_keras_layer
й

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+к&call_and_return_all_conditional_losses
К__call__"
_tf_keras_layer
Д
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+╚&call_and_return_all_conditional_losses
╔__call__"
_tf_keras_layer
й

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+╩&call_and_return_all_conditional_losses
╦__call__"
_tf_keras_layer
й

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+╠&call_and_return_all_conditional_losses
═__call__"
_tf_keras_layer
й

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+╬&call_and_return_all_conditional_losses
¤__call__"
_tf_keras_layer
Д
:regularization_losses
;	variables
<trainable_variables
=	keras_api
+л&call_and_return_all_conditional_losses
Л__call__"
_tf_keras_layer
й

>kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+м&call_and_return_all_conditional_losses
М__call__"
_tf_keras_layer
Д
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+н&call_and_return_all_conditional_losses
Н__call__"
_tf_keras_layer
Д
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
+о&call_and_return_all_conditional_losses
О__call__"
_tf_keras_layer
й

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+п&call_and_return_all_conditional_losses
┘__call__"
_tf_keras_layer
в
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemБmцmЦmд(mД)mе.mЕ/mф4mФ5mг>mГ?m«Lm»Mm░v▒v▓v│v┤(vх)vХ.vи/vИ4v╣5v║>v╗?v╝LvйMvЙ"
	optimizer
 "
trackable_list_wrapper
є
0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
L12
M13"
trackable_list_wrapper
є
0
1
2
3
(4
)5
.6
/7
48
59
>10
?11
L12
M13"
trackable_list_wrapper
╬
Wlayer_metrics
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
regularization_losses
	variables
trainable_variables
[metrics
┴__call__
┐_default_save_signature
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
-
┌serving_default"
signature_map
*:(@2conv2d_60/kernel
:@2conv2d_60/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
\layer_metrics
]layer_regularization_losses
^non_trainable_variables

_layers
regularization_losses
	variables
trainable_variables
`metrics
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
alayer_metrics
blayer_regularization_losses
cnon_trainable_variables

dlayers
regularization_losses
	variables
trainable_variables
emetrics
┼__call__
+─&call_and_return_all_conditional_losses
'─"call_and_return_conditional_losses"
_generic_user_object
+:)@ђ2conv2d_61/kernel
:ђ2conv2d_61/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
flayer_metrics
glayer_regularization_losses
hnon_trainable_variables

ilayers
 regularization_losses
!	variables
"trainable_variables
jmetrics
К__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
klayer_metrics
llayer_regularization_losses
mnon_trainable_variables

nlayers
$regularization_losses
%	variables
&trainable_variables
ometrics
╔__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
,:*ђђ2conv2d_62/kernel
:ђ2conv2d_62/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
░
player_metrics
qlayer_regularization_losses
rnon_trainable_variables

slayers
*regularization_losses
+	variables
,trainable_variables
tmetrics
╦__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
,:*ђђ2conv2d_63/kernel
:ђ2conv2d_63/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
░
ulayer_metrics
vlayer_regularization_losses
wnon_trainable_variables

xlayers
0regularization_losses
1	variables
2trainable_variables
ymetrics
═__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
,:*ђђ2conv2d_64/kernel
:ђ2conv2d_64/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
░
zlayer_metrics
{layer_regularization_losses
|non_trainable_variables

}layers
6regularization_losses
7	variables
8trainable_variables
~metrics
¤__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
┤
layer_metrics
 ђlayer_regularization_losses
Ђnon_trainable_variables
ѓlayers
:regularization_losses
;	variables
<trainable_variables
Ѓmetrics
Л__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
,:*ђђ2conv2d_65/kernel
:ђ2conv2d_65/bias
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
х
ёlayer_metrics
 Ёlayer_regularization_losses
єnon_trainable_variables
Єlayers
@regularization_losses
A	variables
Btrainable_variables
ѕmetrics
М__call__
+м&call_and_return_all_conditional_losses
'м"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
Ѕlayer_metrics
 іlayer_regularization_losses
Іnon_trainable_variables
їlayers
Dregularization_losses
E	variables
Ftrainable_variables
Їmetrics
Н__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
х
јlayer_metrics
 Јlayer_regularization_losses
љnon_trainable_variables
Љlayers
Hregularization_losses
I	variables
Jtrainable_variables
њmetrics
О__call__
+о&call_and_return_all_conditional_losses
'о"call_and_return_conditional_losses"
_generic_user_object
": 	ђ2dense_10/kernel
:2dense_10/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
х
Њlayer_metrics
 ћlayer_regularization_losses
Ћnon_trainable_variables
ќlayers
Nregularization_losses
O	variables
Ptrainable_variables
Ќmetrics
┘__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
	8

9
10
11
12"
trackable_list_wrapper
0
ў0
Ў1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

џtotal

Џcount
ю	variables
Ю	keras_api"
_tf_keras_metric
c

ъtotal

Ъcount
а
_fn_kwargs
А	variables
б	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
џ0
Џ1"
trackable_list_wrapper
.
ю	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ъ0
Ъ1"
trackable_list_wrapper
.
А	variables"
_generic_user_object
/:-@2Adam/conv2d_60/kernel/m
!:@2Adam/conv2d_60/bias/m
0:.@ђ2Adam/conv2d_61/kernel/m
": ђ2Adam/conv2d_61/bias/m
1:/ђђ2Adam/conv2d_62/kernel/m
": ђ2Adam/conv2d_62/bias/m
1:/ђђ2Adam/conv2d_63/kernel/m
": ђ2Adam/conv2d_63/bias/m
1:/ђђ2Adam/conv2d_64/kernel/m
": ђ2Adam/conv2d_64/bias/m
1:/ђђ2Adam/conv2d_65/kernel/m
": ђ2Adam/conv2d_65/bias/m
':%	ђ2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
/:-@2Adam/conv2d_60/kernel/v
!:@2Adam/conv2d_60/bias/v
0:.@ђ2Adam/conv2d_61/kernel/v
": ђ2Adam/conv2d_61/bias/v
1:/ђђ2Adam/conv2d_62/kernel/v
": ђ2Adam/conv2d_62/bias/v
1:/ђђ2Adam/conv2d_63/kernel/v
": ђ2Adam/conv2d_63/bias/v
1:/ђђ2Adam/conv2d_64/kernel/v
": ђ2Adam/conv2d_64/bias/v
1:/ђђ2Adam/conv2d_65/kernel/v
": ђ2Adam/conv2d_65/bias/v
':%	ђ2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
╠B╔
 __inference__wrapped_model_60615input_11"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
┌2О
C__inference_model_10_layer_call_and_return_conditional_losses_61322
C__inference_model_10_layer_call_and_return_conditional_losses_61381
C__inference_model_10_layer_call_and_return_conditional_losses_61178
C__inference_model_10_layer_call_and_return_conditional_losses_61222└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
(__inference_model_10_layer_call_fn_60895
(__inference_model_10_layer_call_fn_61414
(__inference_model_10_layer_call_fn_61447
(__inference_model_10_layer_call_fn_61134└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ь2в
D__inference_conv2d_60_layer_call_and_return_conditional_losses_61458б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv2d_60_layer_call_fn_61467б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╩2К
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_61472
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_61477б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
4__inference_average_pooling2d_30_layer_call_fn_61482
4__inference_average_pooling2d_30_layer_call_fn_61487б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_61_layer_call_and_return_conditional_losses_61498б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv2d_61_layer_call_fn_61507б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╩2К
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_61512
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_61517б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
4__inference_average_pooling2d_31_layer_call_fn_61522
4__inference_average_pooling2d_31_layer_call_fn_61527б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_62_layer_call_and_return_conditional_losses_61538б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv2d_62_layer_call_fn_61547б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_63_layer_call_and_return_conditional_losses_61558б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv2d_63_layer_call_fn_61567б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_64_layer_call_and_return_conditional_losses_61578б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv2d_64_layer_call_fn_61587б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╩2К
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_61592
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_61597б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
4__inference_average_pooling2d_32_layer_call_fn_61602
4__inference_average_pooling2d_32_layer_call_fn_61607б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv2d_65_layer_call_and_return_conditional_losses_61618б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv2d_65_layer_call_fn_61627б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
п2Н
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_61633
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_61639б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
б2Ъ
;__inference_global_average_pooling2d_10_layer_call_fn_61644
;__inference_global_average_pooling2d_10_layer_call_fn_61649б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
№2В
E__inference_flatten_10_layer_call_and_return_conditional_losses_61655б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
н2Л
*__inference_flatten_10_layer_call_fn_61660б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense_10_layer_call_and_return_conditional_losses_61670б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense_10_layer_call_fn_61679б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╦B╚
#__inference_signature_wrapper_61263input_11"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ц
 __inference__wrapped_model_60615ђ()./45>?LM9б6
/б,
*і'
input_11         @ 
ф "3ф0
.
dense_10"і
dense_10         Ы
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_61472ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ╗
O__inference_average_pooling2d_30_layer_call_and_return_conditional_losses_61477h7б4
-б*
(і%
inputs         @ @
ф "-б*
#і 
0          @
џ ╩
4__inference_average_pooling2d_30_layer_call_fn_61482ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Њ
4__inference_average_pooling2d_30_layer_call_fn_61487[7б4
-б*
(і%
inputs         @ @
ф " і          @Ы
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_61512ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ й
O__inference_average_pooling2d_31_layer_call_and_return_conditional_losses_61517j8б5
.б+
)і&
inputs          ђ
ф ".б+
$і!
0         ђ
џ ╩
4__inference_average_pooling2d_31_layer_call_fn_61522ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ћ
4__inference_average_pooling2d_31_layer_call_fn_61527]8б5
.б+
)і&
inputs          ђ
ф "!і         ђЫ
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_61592ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ й
O__inference_average_pooling2d_32_layer_call_and_return_conditional_losses_61597j8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ ╩
4__inference_average_pooling2d_32_layer_call_fn_61602ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Ћ
4__inference_average_pooling2d_32_layer_call_fn_61607]8б5
.б+
)і&
inputs         ђ
ф "!і         ђ┤
D__inference_conv2d_60_layer_call_and_return_conditional_losses_61458l7б4
-б*
(і%
inputs         @ 
ф "-б*
#і 
0         @ @
џ ї
)__inference_conv2d_60_layer_call_fn_61467_7б4
-б*
(і%
inputs         @ 
ф " і         @ @х
D__inference_conv2d_61_layer_call_and_return_conditional_losses_61498m7б4
-б*
(і%
inputs          @
ф ".б+
$і!
0          ђ
џ Ї
)__inference_conv2d_61_layer_call_fn_61507`7б4
-б*
(і%
inputs          @
ф "!і          ђХ
D__inference_conv2d_62_layer_call_and_return_conditional_losses_61538n()8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ ј
)__inference_conv2d_62_layer_call_fn_61547a()8б5
.б+
)і&
inputs         ђ
ф "!і         ђХ
D__inference_conv2d_63_layer_call_and_return_conditional_losses_61558n./8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ ј
)__inference_conv2d_63_layer_call_fn_61567a./8б5
.б+
)і&
inputs         ђ
ф "!і         ђХ
D__inference_conv2d_64_layer_call_and_return_conditional_losses_61578n458б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ ј
)__inference_conv2d_64_layer_call_fn_61587a458б5
.б+
)і&
inputs         ђ
ф "!і         ђХ
D__inference_conv2d_65_layer_call_and_return_conditional_losses_61618n>?8б5
.б+
)і&
inputs         ђ
ф ".б+
$і!
0         ђ
џ ј
)__inference_conv2d_65_layer_call_fn_61627a>?8б5
.б+
)і&
inputs         ђ
ф "!і         ђц
C__inference_dense_10_layer_call_and_return_conditional_losses_61670]LM0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ |
(__inference_dense_10_layer_call_fn_61679PLM0б-
&б#
!і
inputs         ђ
ф "і         Б
E__inference_flatten_10_layer_call_and_return_conditional_losses_61655Z0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ {
*__inference_flatten_10_layer_call_fn_61660M0б-
&б#
!і
inputs         ђ
ф "і         ђ▀
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_61633ёRбO
HбE
Cі@
inputs4                                    
ф ".б+
$і!
0                  
џ ╝
V__inference_global_average_pooling2d_10_layer_call_and_return_conditional_losses_61639b8б5
.б+
)і&
inputs         ђ
ф "&б#
і
0         ђ
џ Х
;__inference_global_average_pooling2d_10_layer_call_fn_61644wRбO
HбE
Cі@
inputs4                                    
ф "!і                  ћ
;__inference_global_average_pooling2d_10_layer_call_fn_61649U8б5
.б+
)і&
inputs         ђ
ф "і         ђ┴
C__inference_model_10_layer_call_and_return_conditional_losses_61178z()./45>?LMAб>
7б4
*і'
input_11         @ 
p 

 
ф "%б"
і
0         
џ ┴
C__inference_model_10_layer_call_and_return_conditional_losses_61222z()./45>?LMAб>
7б4
*і'
input_11         @ 
p

 
ф "%б"
і
0         
џ ┐
C__inference_model_10_layer_call_and_return_conditional_losses_61322x()./45>?LM?б<
5б2
(і%
inputs         @ 
p 

 
ф "%б"
і
0         
џ ┐
C__inference_model_10_layer_call_and_return_conditional_losses_61381x()./45>?LM?б<
5б2
(і%
inputs         @ 
p

 
ф "%б"
і
0         
џ Ў
(__inference_model_10_layer_call_fn_60895m()./45>?LMAб>
7б4
*і'
input_11         @ 
p 

 
ф "і         Ў
(__inference_model_10_layer_call_fn_61134m()./45>?LMAб>
7б4
*і'
input_11         @ 
p

 
ф "і         Ќ
(__inference_model_10_layer_call_fn_61414k()./45>?LM?б<
5б2
(і%
inputs         @ 
p 

 
ф "і         Ќ
(__inference_model_10_layer_call_fn_61447k()./45>?LM?б<
5б2
(і%
inputs         @ 
p

 
ф "і         ┤
#__inference_signature_wrapper_61263ї()./45>?LMEбB
б 
;ф8
6
input_11*і'
input_11         @ "3ф0
.
dense_10"і
dense_10         