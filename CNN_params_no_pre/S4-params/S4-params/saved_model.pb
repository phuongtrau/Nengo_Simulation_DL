ЮД
ќЭ
B
AssignVariableOp
resource
value"dtype"
dtypetype
М
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

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

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
delete_old_dirsbool(
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
О
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.02v2.6.0-rc2-32-g919f693420e8љР

conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
:@*
dtype0

conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_19/kernel
~
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*'
_output_shapes
:@*
dtype0
u
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_19/bias
n
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes	
:*
dtype0

conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_20/kernel

$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*(
_output_shapes
:*
dtype0
u
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_20/bias
n
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes	
:*
dtype0

conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_21/kernel

$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*(
_output_shapes
:*
dtype0
u
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_21/bias
n
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes	
:*
dtype0

conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_22/kernel

$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*(
_output_shapes
:*
dtype0
u
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_22/bias
n
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes	
:*
dtype0

conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_23/kernel

$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*(
_output_shapes
:*
dtype0
u
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_23/bias
n
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes	
:*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
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

Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_18/kernel/m

+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_18/bias/m
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_19/kernel/m

+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/m
|
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_20/kernel/m

+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_20/bias/m
|
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_21/kernel/m

+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_21/bias/m
|
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_22/kernel/m

+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_22/bias/m
|
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_23/kernel/m

+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_23/bias/m
|
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_18/kernel/v

+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_18/bias/v
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_19/kernel/v

+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_19/bias/v
|
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_20/kernel/v

+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_20/bias/v
|
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_21/kernel/v

+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_21/bias/v
|
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_22/kernel/v

+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_22/bias/v
|
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_23/kernel/v

+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_23/bias/v
|
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ѓT
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЎT
valueЄTBЁT BT
г
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
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
R
:regularization_losses
;trainable_variables
<	variables
=	keras_api
h

>kernel
?bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
R
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
и
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemЃmЄmЅmІ(mЇ)mЈ.mЉ/mЊ4mЋ5mЌ>m­?mЎLmЏMmАvБvВvГvД(vЕ)vЖ.vЗ/vИ4vЙ5vК>vЛ?vМLvНMvО
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
­
Wnon_trainable_variables
regularization_losses
Xlayer_regularization_losses
trainable_variables
Ymetrics

Zlayers
	variables
[layer_metrics
 
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
\non_trainable_variables
regularization_losses
]layer_regularization_losses
trainable_variables
^metrics

_layers
	variables
`layer_metrics
 
 
 
­
anon_trainable_variables
regularization_losses
blayer_regularization_losses
trainable_variables
cmetrics

dlayers
	variables
elayer_metrics
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
fnon_trainable_variables
 regularization_losses
glayer_regularization_losses
!trainable_variables
hmetrics

ilayers
"	variables
jlayer_metrics
 
 
 
­
knon_trainable_variables
$regularization_losses
llayer_regularization_losses
%trainable_variables
mmetrics

nlayers
&	variables
olayer_metrics
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
­
pnon_trainable_variables
*regularization_losses
qlayer_regularization_losses
+trainable_variables
rmetrics

slayers
,	variables
tlayer_metrics
\Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
­
unon_trainable_variables
0regularization_losses
vlayer_regularization_losses
1trainable_variables
wmetrics

xlayers
2	variables
ylayer_metrics
\Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
­
znon_trainable_variables
6regularization_losses
{layer_regularization_losses
7trainable_variables
|metrics

}layers
8	variables
~layer_metrics
 
 
 
Б
non_trainable_variables
:regularization_losses
 layer_regularization_losses
;trainable_variables
metrics
layers
<	variables
layer_metrics
\Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

>0
?1
В
non_trainable_variables
@regularization_losses
 layer_regularization_losses
Atrainable_variables
metrics
layers
B	variables
layer_metrics
 
 
 
В
non_trainable_variables
Dregularization_losses
 layer_regularization_losses
Etrainable_variables
metrics
layers
F	variables
layer_metrics
 
 
 
В
non_trainable_variables
Hregularization_losses
 layer_regularization_losses
Itrainable_variables
metrics
layers
J	variables
layer_metrics
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
В
non_trainable_variables
Nregularization_losses
 layer_regularization_losses
Otrainable_variables
metrics
layers
P	variables
layer_metrics
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

0
1
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
 
8

total

count
	variables
	keras_api
I

total

count
 
_fn_kwargs
Ё	variables
Ђ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

Ё	variables
}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_4Placeholder*/
_output_shapes
:џџџџџџџџџ@ *
dtype0*$
shape:џџџџџџџџџ@ 
М
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4conv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *,
f'R%
#__inference_signature_wrapper_32357
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Е
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOpConst*@
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
GPU2*0,1J 8 *'
f"R 
__inference__traced_save_32949
Д

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_3/kerneldense_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_18/kernel/mAdam/conv2d_18/bias/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/v*?
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
GPU2*0,1J 8 **
f%R#
!__inference__traced_restore_33112лу	
Я
k
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_32701

inputs
identity
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Б9
ф
B__inference_model_3_layer_call_and_return_conditional_losses_31958

inputs)
conv2d_18_31818:@
conv2d_18_31820:@*
conv2d_19_31841:@
conv2d_19_31843:	+
conv2d_20_31864:
conv2d_20_31866:	+
conv2d_21_31881:
conv2d_21_31883:	+
conv2d_22_31898:
conv2d_22_31900:	+
conv2d_23_31921:
conv2d_23_31923:	 
dense_3_31952:	
dense_3_31954:
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЃ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_18_31818conv2d_18_31820*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_318172#
!conv2d_18/StatefulPartitionedCallЅ
#average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_318272%
#average_pooling2d_9/PartitionedCallЪ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_9/PartitionedCall:output:0conv2d_19_31841conv2d_19_31843*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_318402#
!conv2d_19/StatefulPartitionedCallЉ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_318502&
$average_pooling2d_10/PartitionedCallЫ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_20_31864conv2d_20_31866*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_318632#
!conv2d_20/StatefulPartitionedCallШ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_31881conv2d_21_31883*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_318802#
!conv2d_21/StatefulPartitionedCallШ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_31898conv2d_22_31900*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_318972#
!conv2d_22/StatefulPartitionedCallЉ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_319072&
$average_pooling2d_11/PartitionedCallЫ
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_11/PartitionedCall:output:0conv2d_23_31921conv2d_23_31923*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_319202#
!conv2d_23/StatefulPartitionedCallГ
*global_average_pooling2d_3/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_319312,
*global_average_pooling2d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_319392
flatten_3/PartitionedCall­
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_31952dense_3_31954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_319512!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityШ
NoOpNoOp"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Д9
х
B__inference_model_3_layer_call_and_return_conditional_losses_32316
input_4)
conv2d_18_32275:@
conv2d_18_32277:@*
conv2d_19_32281:@
conv2d_19_32283:	+
conv2d_20_32287:
conv2d_20_32289:	+
conv2d_21_32292:
conv2d_21_32294:	+
conv2d_22_32297:
conv2d_22_32299:	+
conv2d_23_32303:
conv2d_23_32305:	 
dense_3_32310:	
dense_3_32312:
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЄ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_18_32275conv2d_18_32277*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_318172#
!conv2d_18/StatefulPartitionedCallЅ
#average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_318272%
#average_pooling2d_9/PartitionedCallЪ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_9/PartitionedCall:output:0conv2d_19_32281conv2d_19_32283*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_318402#
!conv2d_19/StatefulPartitionedCallЉ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_318502&
$average_pooling2d_10/PartitionedCallЫ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_20_32287conv2d_20_32289*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_318632#
!conv2d_20/StatefulPartitionedCallШ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_32292conv2d_21_32294*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_318802#
!conv2d_21/StatefulPartitionedCallШ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_32297conv2d_22_32299*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_318972#
!conv2d_22/StatefulPartitionedCallЉ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_319072&
$average_pooling2d_11/PartitionedCallЫ
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_11/PartitionedCall:output:0conv2d_23_32303conv2d_23_32305*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_319202#
!conv2d_23/StatefulPartitionedCallГ
*global_average_pooling2d_3/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_319312,
*global_average_pooling2d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_319392
flatten_3/PartitionedCall­
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_32310dense_3_32312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_319512!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityШ
NoOpNoOp"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_4
И
k
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_31762

inputs
identityЖ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І
Ё
)__inference_conv2d_20_layer_call_fn_32630

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_318632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
j
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_32576

inputs
identityЖ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
І
Ё
)__inference_conv2d_21_layer_call_fn_32650

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_318802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЇS
Ў
B__inference_model_3_layer_call_and_return_conditional_losses_32541

inputsB
(conv2d_18_conv2d_readvariableop_resource:@7
)conv2d_18_biasadd_readvariableop_resource:@C
(conv2d_19_conv2d_readvariableop_resource:@8
)conv2d_19_biasadd_readvariableop_resource:	D
(conv2d_20_conv2d_readvariableop_resource:8
)conv2d_20_biasadd_readvariableop_resource:	D
(conv2d_21_conv2d_readvariableop_resource:8
)conv2d_21_biasadd_readvariableop_resource:	D
(conv2d_22_conv2d_readvariableop_resource:8
)conv2d_22_biasadd_readvariableop_resource:	D
(conv2d_23_conv2d_readvariableop_resource:8
)conv2d_23_biasadd_readvariableop_resource:	9
&dense_3_matmul_readvariableop_resource:	5
'dense_3_biasadd_readvariableop_resource:
identityЂ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂ conv2d_19/BiasAdd/ReadVariableOpЂconv2d_19/Conv2D/ReadVariableOpЂ conv2d_20/BiasAdd/ReadVariableOpЂconv2d_20/Conv2D/ReadVariableOpЂ conv2d_21/BiasAdd/ReadVariableOpЂconv2d_21/Conv2D/ReadVariableOpЂ conv2d_22/BiasAdd/ReadVariableOpЂconv2d_22/Conv2D/ReadVariableOpЂ conv2d_23/BiasAdd/ReadVariableOpЂconv2d_23/Conv2D/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpГ
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_18/Conv2D/ReadVariableOpС
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @*
paddingSAME*
strides
2
conv2d_18/Conv2DЊ
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOpА
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
conv2d_18/Reluй
average_pooling2d_9/AvgPoolAvgPoolconv2d_18/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_9/AvgPoolД
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_19/Conv2D/ReadVariableOpр
conv2d_19/Conv2DConv2D$average_pooling2d_9/AvgPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d_19/Conv2DЋ
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOpБ
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
conv2d_19/Reluм
average_pooling2d_10/AvgPoolAvgPoolconv2d_19/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_10/AvgPoolЕ
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpс
conv2d_20/Conv2DConv2D%average_pooling2d_10/AvgPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_20/Conv2DЋ
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOpБ
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_20/BiasAdd
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_20/ReluЕ
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_21/Conv2D/ReadVariableOpи
conv2d_21/Conv2DConv2Dconv2d_20/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_21/Conv2DЋ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOpБ
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_21/BiasAdd
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_21/ReluЕ
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOpи
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_22/Conv2DЋ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOpБ
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_22/Reluм
average_pooling2d_11/AvgPoolAvgPoolconv2d_22/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_11/AvgPoolЕ
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_23/Conv2D/ReadVariableOpс
conv2d_23/Conv2DConv2D%average_pooling2d_11/AvgPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_23/Conv2DЋ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOpБ
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_23/ReluЗ
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_3/Mean/reduction_indicesз
global_average_pooling2d_3/MeanMeanconv2d_23/Relu:activations:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
global_average_pooling2d_3/Means
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_3/ConstЈ
flatten_3/ReshapeReshape(global_average_pooling2d_3/Mean:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_3/ReshapeІ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity­
NoOpNoOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ф
P
4__inference_average_pooling2d_10_layer_call_fn_32606

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_317402
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_32616

inputs
identityЖ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Л
V
:__inference_global_average_pooling2d_3_layer_call_fn_32726

inputs
identityс
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_317852
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
П

'__inference_model_3_layer_call_fn_32423

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_321642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ћ
P
4__inference_average_pooling2d_10_layer_call_fn_32611

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_318502
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ :X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Ј

є
B__inference_dense_3_layer_call_and_return_conditional_losses_31951

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Т

'__inference_model_3_layer_call_fn_31989
input_4!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_319582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_4
Б9
ф
B__inference_model_3_layer_call_and_return_conditional_losses_32164

inputs)
conv2d_18_32123:@
conv2d_18_32125:@*
conv2d_19_32129:@
conv2d_19_32131:	+
conv2d_20_32135:
conv2d_20_32137:	+
conv2d_21_32140:
conv2d_21_32142:	+
conv2d_22_32145:
conv2d_22_32147:	+
conv2d_23_32151:
conv2d_23_32153:	 
dense_3_32158:	
dense_3_32160:
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЃ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_18_32123conv2d_18_32125*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_318172#
!conv2d_18/StatefulPartitionedCallЅ
#average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_318272%
#average_pooling2d_9/PartitionedCallЪ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_9/PartitionedCall:output:0conv2d_19_32129conv2d_19_32131*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_318402#
!conv2d_19/StatefulPartitionedCallЉ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_318502&
$average_pooling2d_10/PartitionedCallЫ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_20_32135conv2d_20_32137*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_318632#
!conv2d_20/StatefulPartitionedCallШ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_32140conv2d_21_32142*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_318802#
!conv2d_21/StatefulPartitionedCallШ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_32145conv2d_22_32147*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_318972#
!conv2d_22/StatefulPartitionedCallЉ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_319072&
$average_pooling2d_11/PartitionedCallЫ
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_11/PartitionedCall:output:0conv2d_23_32151conv2d_23_32153*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_319202#
!conv2d_23/StatefulPartitionedCallГ
*global_average_pooling2d_3/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_319312,
*global_average_pooling2d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_319392
flatten_3/PartitionedCall­
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_32158dense_3_32160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_319512!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityШ
NoOpNoOp"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Ѓ
 
)__inference_conv2d_19_layer_call_fn_32590

inputs"
unknown:@
	unknown_0:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_318402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ @
 
_user_specified_nameinputs
ѕ

'__inference_dense_3_layer_call_fn_32763

inputs
unknown:	
	unknown_0:
identityЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_319512
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П

'__inference_model_3_layer_call_fn_32390

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_319582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ъ
§
D__inference_conv2d_18_layer_call_and_return_conditional_losses_32561

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@ @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
Х
E
)__inference_flatten_3_layer_call_fn_32748

inputs
identityШ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_319392
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і

D__inference_conv2d_22_layer_call_and_return_conditional_losses_31897

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
џ
D__inference_conv2d_19_layer_call_and_return_conditional_losses_31840

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ @
 
_user_specified_nameinputs
і

D__inference_conv2d_22_layer_call_and_return_conditional_losses_32681

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђ
џ
D__inference_conv2d_19_layer_call_and_return_conditional_losses_32601

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ @
 
_user_specified_nameinputs
І
Ё
)__inference_conv2d_22_layer_call_fn_32670

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_318972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ф
P
4__inference_average_pooling2d_11_layer_call_fn_32686

inputs
identityѕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_317622
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј

є
B__inference_dense_3_layer_call_and_return_conditional_losses_32773

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і

D__inference_conv2d_20_layer_call_and_return_conditional_losses_31863

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ьj
џ
__inference__traced_save_32949
file_prefix/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop6
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
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
Const_1
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
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameє
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueќBљ4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names№
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesЕ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
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
_input_shapesІ
Ѓ: :@:@:@::::::::::	:: : : : : : : : : :@:@:@::::::::::	::@:@:@::::::::::	:: 2(
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
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::%!

_output_shapes
:	: 
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
:@:!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::. *
(
_output_shapes
::!!

_output_shapes	
::."*
(
_output_shapes
::!#

_output_shapes	
::%$!

_output_shapes
:	: %
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
:@:!)

_output_shapes	
::.**
(
_output_shapes
::!+

_output_shapes	
::.,*
(
_output_shapes
::!-

_output_shapes	
::..*
(
_output_shapes
::!/

_output_shapes	
::.0*
(
_output_shapes
::!1

_output_shapes	
::%2!

_output_shapes
:	: 3

_output_shapes
::4

_output_shapes
: 
х
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_32737

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ъ
j
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_32581

inputs
identity
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@ @:W S
/
_output_shapes
:џџџџџџџџџ@ @
 
_user_specified_nameinputs
і

D__inference_conv2d_23_layer_call_and_return_conditional_losses_31920

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ъ
j
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_31827

inputs
identity
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:џџџџџџџџџ @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@ @:W S
/
_output_shapes
:џџџџџџџџџ@ @
 
_user_specified_nameinputs
і

D__inference_conv2d_21_layer_call_and_return_conditional_losses_32661

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_31931

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і

D__inference_conv2d_21_layer_call_and_return_conditional_losses_31880

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_31939

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и
`
D__inference_flatten_3_layer_call_and_return_conditional_losses_32754

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_32696

inputs
identityЖ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Я
k
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_32621

inputs
identity
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ :X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Д9
х
B__inference_model_3_layer_call_and_return_conditional_losses_32272
input_4)
conv2d_18_32231:@
conv2d_18_32233:@*
conv2d_19_32237:@
conv2d_19_32239:	+
conv2d_20_32243:
conv2d_20_32245:	+
conv2d_21_32248:
conv2d_21_32250:	+
conv2d_22_32253:
conv2d_22_32255:	+
conv2d_23_32259:
conv2d_23_32261:	 
dense_3_32266:	
dense_3_32268:
identityЂ!conv2d_18/StatefulPartitionedCallЂ!conv2d_19/StatefulPartitionedCallЂ!conv2d_20/StatefulPartitionedCallЂ!conv2d_21/StatefulPartitionedCallЂ!conv2d_22/StatefulPartitionedCallЂ!conv2d_23/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЄ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_18_32231conv2d_18_32233*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_318172#
!conv2d_18/StatefulPartitionedCallЅ
#average_pooling2d_9/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_318272%
#average_pooling2d_9/PartitionedCallЪ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_9/PartitionedCall:output:0conv2d_19_32237conv2d_19_32239*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_19_layer_call_and_return_conditional_losses_318402#
!conv2d_19/StatefulPartitionedCallЉ
$average_pooling2d_10/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_318502&
$average_pooling2d_10/PartitionedCallЫ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_10/PartitionedCall:output:0conv2d_20_32243conv2d_20_32245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_20_layer_call_and_return_conditional_losses_318632#
!conv2d_20/StatefulPartitionedCallШ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0conv2d_21_32248conv2d_21_32250*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_21_layer_call_and_return_conditional_losses_318802#
!conv2d_21/StatefulPartitionedCallШ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_32253conv2d_22_32255*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_22_layer_call_and_return_conditional_losses_318972#
!conv2d_22/StatefulPartitionedCallЉ
$average_pooling2d_11/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_319072&
$average_pooling2d_11/PartitionedCallЫ
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_11/PartitionedCall:output:0conv2d_23_32259conv2d_23_32261*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_319202#
!conv2d_23/StatefulPartitionedCallГ
*global_average_pooling2d_3/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_319312,
*global_average_pooling2d_3/PartitionedCall
flatten_3/PartitionedCallPartitionedCall3global_average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_flatten_3_layer_call_and_return_conditional_losses_319392
flatten_3/PartitionedCall­
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_3_32266dense_3_32268*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_319512!
dense_3/StatefulPartitionedCall
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

IdentityШ
NoOpNoOp"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_4
Т

'__inference_model_3_layer_call_fn_32228
input_4!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_model_3_layer_call_and_return_conditional_losses_321642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_4
`
э
 __inference__wrapped_model_31709
input_4J
0model_3_conv2d_18_conv2d_readvariableop_resource:@?
1model_3_conv2d_18_biasadd_readvariableop_resource:@K
0model_3_conv2d_19_conv2d_readvariableop_resource:@@
1model_3_conv2d_19_biasadd_readvariableop_resource:	L
0model_3_conv2d_20_conv2d_readvariableop_resource:@
1model_3_conv2d_20_biasadd_readvariableop_resource:	L
0model_3_conv2d_21_conv2d_readvariableop_resource:@
1model_3_conv2d_21_biasadd_readvariableop_resource:	L
0model_3_conv2d_22_conv2d_readvariableop_resource:@
1model_3_conv2d_22_biasadd_readvariableop_resource:	L
0model_3_conv2d_23_conv2d_readvariableop_resource:@
1model_3_conv2d_23_biasadd_readvariableop_resource:	A
.model_3_dense_3_matmul_readvariableop_resource:	=
/model_3_dense_3_biasadd_readvariableop_resource:
identityЂ(model_3/conv2d_18/BiasAdd/ReadVariableOpЂ'model_3/conv2d_18/Conv2D/ReadVariableOpЂ(model_3/conv2d_19/BiasAdd/ReadVariableOpЂ'model_3/conv2d_19/Conv2D/ReadVariableOpЂ(model_3/conv2d_20/BiasAdd/ReadVariableOpЂ'model_3/conv2d_20/Conv2D/ReadVariableOpЂ(model_3/conv2d_21/BiasAdd/ReadVariableOpЂ'model_3/conv2d_21/Conv2D/ReadVariableOpЂ(model_3/conv2d_22/BiasAdd/ReadVariableOpЂ'model_3/conv2d_22/Conv2D/ReadVariableOpЂ(model_3/conv2d_23/BiasAdd/ReadVariableOpЂ'model_3/conv2d_23/Conv2D/ReadVariableOpЂ&model_3/dense_3/BiasAdd/ReadVariableOpЂ%model_3/dense_3/MatMul/ReadVariableOpЫ
'model_3/conv2d_18/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'model_3/conv2d_18/Conv2D/ReadVariableOpк
model_3/conv2d_18/Conv2DConv2Dinput_4/model_3/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @*
paddingSAME*
strides
2
model_3/conv2d_18/Conv2DТ
(model_3/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_3/conv2d_18/BiasAdd/ReadVariableOpа
model_3/conv2d_18/BiasAddBiasAdd!model_3/conv2d_18/Conv2D:output:00model_3/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
model_3/conv2d_18/BiasAdd
model_3/conv2d_18/ReluRelu"model_3/conv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
model_3/conv2d_18/Reluё
#model_3/average_pooling2d_9/AvgPoolAvgPool$model_3/conv2d_18/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides
2%
#model_3/average_pooling2d_9/AvgPoolЬ
'model_3/conv2d_19/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02)
'model_3/conv2d_19/Conv2D/ReadVariableOp
model_3/conv2d_19/Conv2DConv2D,model_3/average_pooling2d_9/AvgPool:output:0/model_3/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
model_3/conv2d_19/Conv2DУ
(model_3/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_19/BiasAdd/ReadVariableOpб
model_3/conv2d_19/BiasAddBiasAdd!model_3/conv2d_19/Conv2D:output:00model_3/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
model_3/conv2d_19/BiasAdd
model_3/conv2d_19/ReluRelu"model_3/conv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
model_3/conv2d_19/Reluє
$model_3/average_pooling2d_10/AvgPoolAvgPool$model_3/conv2d_19/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2&
$model_3/average_pooling2d_10/AvgPoolЭ
'model_3/conv2d_20/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_20/Conv2D/ReadVariableOp
model_3/conv2d_20/Conv2DConv2D-model_3/average_pooling2d_10/AvgPool:output:0/model_3/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
model_3/conv2d_20/Conv2DУ
(model_3/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_20/BiasAdd/ReadVariableOpб
model_3/conv2d_20/BiasAddBiasAdd!model_3/conv2d_20/Conv2D:output:00model_3/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_20/BiasAdd
model_3/conv2d_20/ReluRelu"model_3/conv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_20/ReluЭ
'model_3/conv2d_21/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_21/Conv2D/ReadVariableOpј
model_3/conv2d_21/Conv2DConv2D$model_3/conv2d_20/Relu:activations:0/model_3/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
model_3/conv2d_21/Conv2DУ
(model_3/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_21/BiasAdd/ReadVariableOpб
model_3/conv2d_21/BiasAddBiasAdd!model_3/conv2d_21/Conv2D:output:00model_3/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_21/BiasAdd
model_3/conv2d_21/ReluRelu"model_3/conv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_21/ReluЭ
'model_3/conv2d_22/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_22/Conv2D/ReadVariableOpј
model_3/conv2d_22/Conv2DConv2D$model_3/conv2d_21/Relu:activations:0/model_3/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
model_3/conv2d_22/Conv2DУ
(model_3/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_22/BiasAdd/ReadVariableOpб
model_3/conv2d_22/BiasAddBiasAdd!model_3/conv2d_22/Conv2D:output:00model_3/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_22/BiasAdd
model_3/conv2d_22/ReluRelu"model_3/conv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_22/Reluє
$model_3/average_pooling2d_11/AvgPoolAvgPool$model_3/conv2d_22/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2&
$model_3/average_pooling2d_11/AvgPoolЭ
'model_3/conv2d_23/Conv2D/ReadVariableOpReadVariableOp0model_3_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02)
'model_3/conv2d_23/Conv2D/ReadVariableOp
model_3/conv2d_23/Conv2DConv2D-model_3/average_pooling2d_11/AvgPool:output:0/model_3/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
model_3/conv2d_23/Conv2DУ
(model_3/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp1model_3_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02*
(model_3/conv2d_23/BiasAdd/ReadVariableOpб
model_3/conv2d_23/BiasAddBiasAdd!model_3/conv2d_23/Conv2D:output:00model_3/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_23/BiasAdd
model_3/conv2d_23/ReluRelu"model_3/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
model_3/conv2d_23/ReluЧ
9model_3/global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_3/global_average_pooling2d_3/Mean/reduction_indicesї
'model_3/global_average_pooling2d_3/MeanMean$model_3/conv2d_23/Relu:activations:0Bmodel_3/global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2)
'model_3/global_average_pooling2d_3/Mean
model_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
model_3/flatten_3/ConstШ
model_3/flatten_3/ReshapeReshape0model_3/global_average_pooling2d_3/Mean:output:0 model_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
model_3/flatten_3/ReshapeО
%model_3/dense_3/MatMul/ReadVariableOpReadVariableOp.model_3_dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02'
%model_3/dense_3/MatMul/ReadVariableOpП
model_3/dense_3/MatMulMatMul"model_3/flatten_3/Reshape:output:0-model_3/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/dense_3/MatMulМ
&model_3/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_3_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_3/dense_3/BiasAdd/ReadVariableOpС
model_3/dense_3/BiasAddBiasAdd model_3/dense_3/MatMul:product:0.model_3/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_3/dense_3/BiasAdd{
IdentityIdentity model_3/dense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp)^model_3/conv2d_18/BiasAdd/ReadVariableOp(^model_3/conv2d_18/Conv2D/ReadVariableOp)^model_3/conv2d_19/BiasAdd/ReadVariableOp(^model_3/conv2d_19/Conv2D/ReadVariableOp)^model_3/conv2d_20/BiasAdd/ReadVariableOp(^model_3/conv2d_20/Conv2D/ReadVariableOp)^model_3/conv2d_21/BiasAdd/ReadVariableOp(^model_3/conv2d_21/Conv2D/ReadVariableOp)^model_3/conv2d_22/BiasAdd/ReadVariableOp(^model_3/conv2d_22/Conv2D/ReadVariableOp)^model_3/conv2d_23/BiasAdd/ReadVariableOp(^model_3/conv2d_23/Conv2D/ReadVariableOp'^model_3/dense_3/BiasAdd/ReadVariableOp&^model_3/dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 2T
(model_3/conv2d_18/BiasAdd/ReadVariableOp(model_3/conv2d_18/BiasAdd/ReadVariableOp2R
'model_3/conv2d_18/Conv2D/ReadVariableOp'model_3/conv2d_18/Conv2D/ReadVariableOp2T
(model_3/conv2d_19/BiasAdd/ReadVariableOp(model_3/conv2d_19/BiasAdd/ReadVariableOp2R
'model_3/conv2d_19/Conv2D/ReadVariableOp'model_3/conv2d_19/Conv2D/ReadVariableOp2T
(model_3/conv2d_20/BiasAdd/ReadVariableOp(model_3/conv2d_20/BiasAdd/ReadVariableOp2R
'model_3/conv2d_20/Conv2D/ReadVariableOp'model_3/conv2d_20/Conv2D/ReadVariableOp2T
(model_3/conv2d_21/BiasAdd/ReadVariableOp(model_3/conv2d_21/BiasAdd/ReadVariableOp2R
'model_3/conv2d_21/Conv2D/ReadVariableOp'model_3/conv2d_21/Conv2D/ReadVariableOp2T
(model_3/conv2d_22/BiasAdd/ReadVariableOp(model_3/conv2d_22/BiasAdd/ReadVariableOp2R
'model_3/conv2d_22/Conv2D/ReadVariableOp'model_3/conv2d_22/Conv2D/ReadVariableOp2T
(model_3/conv2d_23/BiasAdd/ReadVariableOp(model_3/conv2d_23/BiasAdd/ReadVariableOp2R
'model_3/conv2d_23/Conv2D/ReadVariableOp'model_3/conv2d_23/Conv2D/ReadVariableOp2P
&model_3/dense_3/BiasAdd/ReadVariableOp&model_3/dense_3/BiasAdd/ReadVariableOp2N
%model_3/dense_3/MatMul/ReadVariableOp%model_3/dense_3/MatMul/ReadVariableOp:X T
/
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_4
І
Ё
)__inference_conv2d_23_layer_call_fn_32710

inputs#
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_23_layer_call_and_return_conditional_losses_319202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
х
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_31785

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ї
V
:__inference_global_average_pooling2d_3_layer_call_fn_32731

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *^
fYRW
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_319312
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
і

D__inference_conv2d_20_layer_call_and_return_conditional_losses_32641

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
k
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_31740

inputs
identityЖ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
і

D__inference_conv2d_23_layer_call_and_return_conditional_losses_32721

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ё
q
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_32743

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЇS
Ў
B__inference_model_3_layer_call_and_return_conditional_losses_32482

inputsB
(conv2d_18_conv2d_readvariableop_resource:@7
)conv2d_18_biasadd_readvariableop_resource:@C
(conv2d_19_conv2d_readvariableop_resource:@8
)conv2d_19_biasadd_readvariableop_resource:	D
(conv2d_20_conv2d_readvariableop_resource:8
)conv2d_20_biasadd_readvariableop_resource:	D
(conv2d_21_conv2d_readvariableop_resource:8
)conv2d_21_biasadd_readvariableop_resource:	D
(conv2d_22_conv2d_readvariableop_resource:8
)conv2d_22_biasadd_readvariableop_resource:	D
(conv2d_23_conv2d_readvariableop_resource:8
)conv2d_23_biasadd_readvariableop_resource:	9
&dense_3_matmul_readvariableop_resource:	5
'dense_3_biasadd_readvariableop_resource:
identityЂ conv2d_18/BiasAdd/ReadVariableOpЂconv2d_18/Conv2D/ReadVariableOpЂ conv2d_19/BiasAdd/ReadVariableOpЂconv2d_19/Conv2D/ReadVariableOpЂ conv2d_20/BiasAdd/ReadVariableOpЂconv2d_20/Conv2D/ReadVariableOpЂ conv2d_21/BiasAdd/ReadVariableOpЂconv2d_21/Conv2D/ReadVariableOpЂ conv2d_22/BiasAdd/ReadVariableOpЂconv2d_22/Conv2D/ReadVariableOpЂ conv2d_23/BiasAdd/ReadVariableOpЂconv2d_23/Conv2D/ReadVariableOpЂdense_3/BiasAdd/ReadVariableOpЂdense_3/MatMul/ReadVariableOpГ
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_18/Conv2D/ReadVariableOpС
conv2d_18/Conv2DConv2Dinputs'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @*
paddingSAME*
strides
2
conv2d_18/Conv2DЊ
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_18/BiasAdd/ReadVariableOpА
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
conv2d_18/BiasAdd~
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
conv2d_18/Reluй
average_pooling2d_9/AvgPoolAvgPoolconv2d_18/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_9/AvgPoolД
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02!
conv2d_19/Conv2D/ReadVariableOpр
conv2d_19/Conv2DConv2D$average_pooling2d_9/AvgPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ *
paddingSAME*
strides
2
conv2d_19/Conv2DЋ
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_19/BiasAdd/ReadVariableOpБ
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
conv2d_19/BiasAdd
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ 2
conv2d_19/Reluм
average_pooling2d_10/AvgPoolAvgPoolconv2d_19/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_10/AvgPoolЕ
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_20/Conv2D/ReadVariableOpс
conv2d_20/Conv2DConv2D%average_pooling2d_10/AvgPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_20/Conv2DЋ
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_20/BiasAdd/ReadVariableOpБ
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_20/BiasAdd
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_20/ReluЕ
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_21/Conv2D/ReadVariableOpи
conv2d_21/Conv2DConv2Dconv2d_20/Relu:activations:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_21/Conv2DЋ
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_21/BiasAdd/ReadVariableOpБ
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_21/BiasAdd
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_21/ReluЕ
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_22/Conv2D/ReadVariableOpи
conv2d_22/Conv2DConv2Dconv2d_21/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_22/Conv2DЋ
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOpБ
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_22/BiasAdd
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_22/Reluм
average_pooling2d_11/AvgPoolAvgPoolconv2d_22/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2
average_pooling2d_11/AvgPoolЕ
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02!
conv2d_23/Conv2D/ReadVariableOpс
conv2d_23/Conv2DConv2D%average_pooling2d_11/AvgPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ*
paddingSAME*
strides
2
conv2d_23/Conv2DЋ
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOpБ
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_23/BiasAdd
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2
conv2d_23/ReluЗ
1global_average_pooling2d_3/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_3/Mean/reduction_indicesз
global_average_pooling2d_3/MeanMeanconv2d_23/Relu:activations:0:global_average_pooling2d_3/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
global_average_pooling2d_3/Means
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   2
flatten_3/ConstЈ
flatten_3/ReshapeReshape(global_average_pooling2d_3/Mean:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
flatten_3/ReshapeІ
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_3/MatMul/ReadVariableOp
dense_3/MatMulMatMulflatten_3/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/MatMulЄ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOpЁ
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2
dense_3/BiasAdds
IdentityIdentitydense_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identity­
NoOpNoOp!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs


)__inference_conv2d_18_layer_call_fn_32550

inputs!
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *M
fHRF
D__inference_conv2d_18_layer_call_and_return_conditional_losses_318172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@ @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
т
O
3__inference_average_pooling2d_9_layer_call_fn_32566

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_317182
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ѕ
O
3__inference_average_pooling2d_9_layer_call_fn_32571

inputs
identityй
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *W
fRRP
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_318272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:џџџџџџџџџ @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@ @:W S
/
_output_shapes
:џџџџџџџџџ@ @
 
_user_specified_nameinputs
ъ
§
D__inference_conv2d_18_layer_call_and_return_conditional_losses_31817

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOpЃ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@ @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:џџџџџџџџџ@ @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:џџџџџџџџџ@ 
 
_user_specified_nameinputs
ћ
P
4__inference_average_pooling2d_11_layer_call_fn_32691

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *X
fSRQ
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_319072
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


#__inference_signature_wrapper_32357
input_4!
unknown:@
	unknown_0:@$
	unknown_1:@
	unknown_2:	%
	unknown_3:
	unknown_4:	%
	unknown_5:
	unknown_6:	%
	unknown_7:
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:џџџџџџџџџ*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8 *)
f$R"
 __inference__wrapped_model_317092
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџ@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ@ 
!
_user_specified_name	input_4
Њм
Ќ 
!__inference__traced_restore_33112
file_prefix;
!assignvariableop_conv2d_18_kernel:@/
!assignvariableop_1_conv2d_18_bias:@>
#assignvariableop_2_conv2d_19_kernel:@0
!assignvariableop_3_conv2d_19_bias:	?
#assignvariableop_4_conv2d_20_kernel:0
!assignvariableop_5_conv2d_20_bias:	?
#assignvariableop_6_conv2d_21_kernel:0
!assignvariableop_7_conv2d_21_bias:	?
#assignvariableop_8_conv2d_22_kernel:0
!assignvariableop_9_conv2d_22_bias:	@
$assignvariableop_10_conv2d_23_kernel:1
"assignvariableop_11_conv2d_23_bias:	5
"assignvariableop_12_dense_3_kernel:	.
 assignvariableop_13_dense_3_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: E
+assignvariableop_23_adam_conv2d_18_kernel_m:@7
)assignvariableop_24_adam_conv2d_18_bias_m:@F
+assignvariableop_25_adam_conv2d_19_kernel_m:@8
)assignvariableop_26_adam_conv2d_19_bias_m:	G
+assignvariableop_27_adam_conv2d_20_kernel_m:8
)assignvariableop_28_adam_conv2d_20_bias_m:	G
+assignvariableop_29_adam_conv2d_21_kernel_m:8
)assignvariableop_30_adam_conv2d_21_bias_m:	G
+assignvariableop_31_adam_conv2d_22_kernel_m:8
)assignvariableop_32_adam_conv2d_22_bias_m:	G
+assignvariableop_33_adam_conv2d_23_kernel_m:8
)assignvariableop_34_adam_conv2d_23_bias_m:	<
)assignvariableop_35_adam_dense_3_kernel_m:	5
'assignvariableop_36_adam_dense_3_bias_m:E
+assignvariableop_37_adam_conv2d_18_kernel_v:@7
)assignvariableop_38_adam_conv2d_18_bias_v:@F
+assignvariableop_39_adam_conv2d_19_kernel_v:@8
)assignvariableop_40_adam_conv2d_19_bias_v:	G
+assignvariableop_41_adam_conv2d_20_kernel_v:8
)assignvariableop_42_adam_conv2d_20_bias_v:	G
+assignvariableop_43_adam_conv2d_21_kernel_v:8
)assignvariableop_44_adam_conv2d_21_bias_v:	G
+assignvariableop_45_adam_conv2d_22_kernel_v:8
)assignvariableop_46_adam_conv2d_22_bias_v:	G
+assignvariableop_47_adam_conv2d_23_kernel_v:8
)assignvariableop_48_adam_conv2d_23_bias_v:	<
)assignvariableop_49_adam_dense_3_kernel_v:	5
'assignvariableop_50_adam_dense_3_bias_v:
identity_52ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9њ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*
valueќBљ4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesі
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesВ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ц
_output_shapesг
а::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ј
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3І
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_19_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ј
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_20_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5І
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_20_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ј
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_21_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7І
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_21_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ј
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_22_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9І
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_22_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ќ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_23_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Њ
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_23_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Њ
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ј
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14Ѕ
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ї
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ї
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17І
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Ў
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ё
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ё
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ѓ
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ѓ
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Г
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_18_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_18_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Г
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_19_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Б
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_19_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Г
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_20_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Б
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_20_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Г
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_21_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Б
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_21_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Г
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_22_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Б
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_22_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Г
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_23_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Б
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_23_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Б
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_3_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36Џ
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_3_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Г
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_18_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Б
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_18_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Г
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_19_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Б
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_19_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Г
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_20_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Б
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_20_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43Г
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_21_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Б
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_21_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Г
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_22_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Б
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_22_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47Г
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_23_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48Б
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_23_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Б
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_3_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Џ
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_3_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpР	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51f
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_52Ј	
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
Я
k
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_31850

inputs
identity
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ :X T
0
_output_shapes
:џџџџџџџџџ 
 
_user_specified_nameinputs
Я
k
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_31907

inputs
identity
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:џџџџџџџџџ:X T
0
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
j
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_31718

inputs
identityЖ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ЈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*В
serving_default
C
input_48
serving_default_input_4:0џџџџџџџџџ@ ;
dense_30
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:ън
Ш
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
trainable_variables
	variables
	keras_api

signatures
П__call__
Р_default_save_signature
+С&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
Н

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
regularization_losses
trainable_variables
	variables
	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
$regularization_losses
%trainable_variables
&	variables
'	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

(kernel
)bias
*regularization_losses
+trainable_variables
,	variables
-	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

.kernel
/bias
0regularization_losses
1trainable_variables
2	variables
3	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

4kernel
5bias
6regularization_losses
7trainable_variables
8	variables
9	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
:regularization_losses
;trainable_variables
<	variables
=	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

>kernel
?bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
в__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
Н

Lkernel
Mbias
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
и__call__
+й&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratemЃmЄmЅmІ(mЇ)mЈ.mЉ/mЊ4mЋ5mЌ>m­?mЎLmЏMmАvБvВvГvД(vЕ)vЖ.vЗ/vИ4vЙ5vК>vЛ?vМLvНMvО"
	optimizer
 "
trackable_list_wrapper

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

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
Ю
Wnon_trainable_variables
regularization_losses
Xlayer_regularization_losses
trainable_variables
Ymetrics

Zlayers
	variables
[layer_metrics
П__call__
Р_default_save_signature
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
-
кserving_default"
signature_map
*:(@2conv2d_18/kernel
:@2conv2d_18/bias
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
А
\non_trainable_variables
regularization_losses
]layer_regularization_losses
trainable_variables
^metrics

_layers
	variables
`layer_metrics
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
anon_trainable_variables
regularization_losses
blayer_regularization_losses
trainable_variables
cmetrics

dlayers
	variables
elayer_metrics
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
+:)@2conv2d_19/kernel
:2conv2d_19/bias
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
А
fnon_trainable_variables
 regularization_losses
glayer_regularization_losses
!trainable_variables
hmetrics

ilayers
"	variables
jlayer_metrics
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
knon_trainable_variables
$regularization_losses
llayer_regularization_losses
%trainable_variables
mmetrics

nlayers
&	variables
olayer_metrics
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_20/kernel
:2conv2d_20/bias
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
А
pnon_trainable_variables
*regularization_losses
qlayer_regularization_losses
+trainable_variables
rmetrics

slayers
,	variables
tlayer_metrics
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_21/kernel
:2conv2d_21/bias
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
А
unon_trainable_variables
0regularization_losses
vlayer_regularization_losses
1trainable_variables
wmetrics

xlayers
2	variables
ylayer_metrics
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_22/kernel
:2conv2d_22/bias
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
А
znon_trainable_variables
6regularization_losses
{layer_regularization_losses
7trainable_variables
|metrics

}layers
8	variables
~layer_metrics
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Д
non_trainable_variables
:regularization_losses
 layer_regularization_losses
;trainable_variables
metrics
layers
<	variables
layer_metrics
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
,:*2conv2d_23/kernel
:2conv2d_23/bias
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
Е
non_trainable_variables
@regularization_losses
 layer_regularization_losses
Atrainable_variables
metrics
layers
B	variables
layer_metrics
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
Dregularization_losses
 layer_regularization_losses
Etrainable_variables
metrics
layers
F	variables
layer_metrics
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
non_trainable_variables
Hregularization_losses
 layer_regularization_losses
Itrainable_variables
metrics
layers
J	variables
layer_metrics
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_3/kernel
:2dense_3/bias
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
Е
non_trainable_variables
Nregularization_losses
 layer_regularization_losses
Otrainable_variables
metrics
layers
P	variables
layer_metrics
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
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
 "
trackable_dict_wrapper
R

total

count
	variables
	keras_api"
_tf_keras_metric
c

total

count
 
_fn_kwargs
Ё	variables
Ђ	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
Ё	variables"
_generic_user_object
/:-@2Adam/conv2d_18/kernel/m
!:@2Adam/conv2d_18/bias/m
0:.@2Adam/conv2d_19/kernel/m
": 2Adam/conv2d_19/bias/m
1:/2Adam/conv2d_20/kernel/m
": 2Adam/conv2d_20/bias/m
1:/2Adam/conv2d_21/kernel/m
": 2Adam/conv2d_21/bias/m
1:/2Adam/conv2d_22/kernel/m
": 2Adam/conv2d_22/bias/m
1:/2Adam/conv2d_23/kernel/m
": 2Adam/conv2d_23/bias/m
&:$	2Adam/dense_3/kernel/m
:2Adam/dense_3/bias/m
/:-@2Adam/conv2d_18/kernel/v
!:@2Adam/conv2d_18/bias/v
0:.@2Adam/conv2d_19/kernel/v
": 2Adam/conv2d_19/bias/v
1:/2Adam/conv2d_20/kernel/v
": 2Adam/conv2d_20/bias/v
1:/2Adam/conv2d_21/kernel/v
": 2Adam/conv2d_21/bias/v
1:/2Adam/conv2d_22/kernel/v
": 2Adam/conv2d_22/bias/v
1:/2Adam/conv2d_23/kernel/v
": 2Adam/conv2d_23/bias/v
&:$	2Adam/dense_3/kernel/v
:2Adam/dense_3/bias/v
ъ2ч
'__inference_model_3_layer_call_fn_31989
'__inference_model_3_layer_call_fn_32390
'__inference_model_3_layer_call_fn_32423
'__inference_model_3_layer_call_fn_32228Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ЫBШ
 __inference__wrapped_model_31709input_4"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
B__inference_model_3_layer_call_and_return_conditional_losses_32482
B__inference_model_3_layer_call_and_return_conditional_losses_32541
B__inference_model_3_layer_call_and_return_conditional_losses_32272
B__inference_model_3_layer_call_and_return_conditional_losses_32316Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
г2а
)__inference_conv2d_18_layer_call_fn_32550Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_18_layer_call_and_return_conditional_losses_32561Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
3__inference_average_pooling2d_9_layer_call_fn_32566
3__inference_average_pooling2d_9_layer_call_fn_32571Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ш2Х
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_32576
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_32581Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv2d_19_layer_call_fn_32590Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_19_layer_call_and_return_conditional_losses_32601Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
4__inference_average_pooling2d_10_layer_call_fn_32606
4__inference_average_pooling2d_10_layer_call_fn_32611Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ъ2Ч
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_32616
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_32621Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv2d_20_layer_call_fn_32630Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_20_layer_call_and_return_conditional_losses_32641Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv2d_21_layer_call_fn_32650Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_21_layer_call_and_return_conditional_losses_32661Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv2d_22_layer_call_fn_32670Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_22_layer_call_and_return_conditional_losses_32681Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
4__inference_average_pooling2d_11_layer_call_fn_32686
4__inference_average_pooling2d_11_layer_call_fn_32691Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ъ2Ч
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_32696
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_32701Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_conv2d_23_layer_call_fn_32710Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_23_layer_call_and_return_conditional_losses_32721Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 2
:__inference_global_average_pooling2d_3_layer_call_fn_32726
:__inference_global_average_pooling2d_3_layer_call_fn_32731Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_32737
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_32743Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
г2а
)__inference_flatten_3_layer_call_fn_32748Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_flatten_3_layer_call_and_return_conditional_losses_32754Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_3_layer_call_fn_32763Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ь2щ
B__inference_dense_3_layer_call_and_return_conditional_losses_32773Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЪBЧ
#__inference_signature_wrapper_32357input_4"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Ё
 __inference__wrapped_model_31709}()./45>?LM8Ђ5
.Ђ+
)&
input_4џџџџџџџџџ@ 
Њ "1Њ.
,
dense_3!
dense_3џџџџџџџџџђ
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_32616RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
O__inference_average_pooling2d_10_layer_call_and_return_conditional_losses_32621j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ 
Њ ".Ђ+
$!
0џџџџџџџџџ
 Ъ
4__inference_average_pooling2d_10_layer_call_fn_32606RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
4__inference_average_pooling2d_10_layer_call_fn_32611]8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ 
Њ "!џџџџџџџџџђ
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_32696RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Н
O__inference_average_pooling2d_11_layer_call_and_return_conditional_losses_32701j8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 Ъ
4__inference_average_pooling2d_11_layer_call_fn_32686RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
4__inference_average_pooling2d_11_layer_call_fn_32691]8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџё
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_32576RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 К
N__inference_average_pooling2d_9_layer_call_and_return_conditional_losses_32581h7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@ @
Њ "-Ђ*
# 
0џџџџџџџџџ @
 Щ
3__inference_average_pooling2d_9_layer_call_fn_32566RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
3__inference_average_pooling2d_9_layer_call_fn_32571[7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@ @
Њ " џџџџџџџџџ @Д
D__inference_conv2d_18_layer_call_and_return_conditional_losses_32561l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@ 
Њ "-Ђ*
# 
0џџџџџџџџџ@ @
 
)__inference_conv2d_18_layer_call_fn_32550_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ@ 
Њ " џџџџџџџџџ@ @Е
D__inference_conv2d_19_layer_call_and_return_conditional_losses_32601m7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ @
Њ ".Ђ+
$!
0џџџџџџџџџ 
 
)__inference_conv2d_19_layer_call_fn_32590`7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ @
Њ "!џџџџџџџџџ Ж
D__inference_conv2d_20_layer_call_and_return_conditional_losses_32641n()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
)__inference_conv2d_20_layer_call_fn_32630a()8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЖ
D__inference_conv2d_21_layer_call_and_return_conditional_losses_32661n./8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
)__inference_conv2d_21_layer_call_fn_32650a./8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЖ
D__inference_conv2d_22_layer_call_and_return_conditional_losses_32681n458Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
)__inference_conv2d_22_layer_call_fn_32670a458Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЖ
D__inference_conv2d_23_layer_call_and_return_conditional_losses_32721n>?8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџ
 
)__inference_conv2d_23_layer_call_fn_32710a>?8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "!џџџџџџџџџЃ
B__inference_dense_3_layer_call_and_return_conditional_losses_32773]LM0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 {
'__inference_dense_3_layer_call_fn_32763PLM0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЂ
D__inference_flatten_3_layer_call_and_return_conditional_losses_32754Z0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 z
)__inference_flatten_3_layer_call_fn_32748M0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџо
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_32737RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Л
U__inference_global_average_pooling2d_3_layer_call_and_return_conditional_losses_32743b8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 Е
:__inference_global_average_pooling2d_3_layer_call_fn_32726wRЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "!џџџџџџџџџџџџџџџџџџ
:__inference_global_average_pooling2d_3_layer_call_fn_32731U8Ђ5
.Ђ+
)&
inputsџџџџџџџџџ
Њ "џџџџџџџџџП
B__inference_model_3_layer_call_and_return_conditional_losses_32272y()./45>?LM@Ђ=
6Ђ3
)&
input_4џџџџџџџџџ@ 
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 П
B__inference_model_3_layer_call_and_return_conditional_losses_32316y()./45>?LM@Ђ=
6Ђ3
)&
input_4џџџџџџџџџ@ 
p

 
Њ "%Ђ"

0џџџџџџџџџ
 О
B__inference_model_3_layer_call_and_return_conditional_losses_32482x()./45>?LM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@ 
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 О
B__inference_model_3_layer_call_and_return_conditional_losses_32541x()./45>?LM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@ 
p

 
Њ "%Ђ"

0џџџџџџџџџ
 
'__inference_model_3_layer_call_fn_31989l()./45>?LM@Ђ=
6Ђ3
)&
input_4џџџџџџџџџ@ 
p 

 
Њ "џџџџџџџџџ
'__inference_model_3_layer_call_fn_32228l()./45>?LM@Ђ=
6Ђ3
)&
input_4џџџџџџџџџ@ 
p

 
Њ "џџџџџџџџџ
'__inference_model_3_layer_call_fn_32390k()./45>?LM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@ 
p 

 
Њ "џџџџџџџџџ
'__inference_model_3_layer_call_fn_32423k()./45>?LM?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ@ 
p

 
Њ "џџџџџџџџџА
#__inference_signature_wrapper_32357()./45>?LMCЂ@
Ђ 
9Њ6
4
input_4)&
input_4џџџџџџџџџ@ "1Њ.
,
dense_3!
dense_3џџџџџџџџџ