��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
�
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
�
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
�
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
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.02v2.6.0-rc2-32-g919f693420e8��
�
conv2d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_48/kernel
}
$conv2d_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_48/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_48/bias
m
"conv2d_48/bias/Read/ReadVariableOpReadVariableOpconv2d_48/bias*
_output_shapes
:@*
dtype0
�
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*!
shared_nameconv2d_49/kernel
~
$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*'
_output_shapes
:@�*
dtype0
u
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_49/bias
n
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes	
:�*
dtype0
�
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_50/kernel

$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_50/bias
n
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
_output_shapes	
:�*
dtype0
�
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_51/kernel

$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_51/bias
n
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes	
:�*
dtype0
�
conv2d_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_52/kernel

$conv2d_52/kernel/Read/ReadVariableOpReadVariableOpconv2d_52/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_52/bias
n
"conv2d_52/bias/Read/ReadVariableOpReadVariableOpconv2d_52/bias*
_output_shapes	
:�*
dtype0
�
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_53/kernel

$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_53/bias
n
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes	
:�*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	�*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
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
�
Adam/conv2d_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_48/kernel/m
�
+Adam/conv2d_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/m*&
_output_shapes
:@*
dtype0
�
Adam/conv2d_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_48/bias/m
{
)Adam/conv2d_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*(
shared_nameAdam/conv2d_49/kernel/m
�
+Adam/conv2d_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/conv2d_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_49/bias/m
|
)Adam/conv2d_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_50/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_50/kernel/m
�
+Adam/conv2d_50/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_50/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_50/bias/m
|
)Adam/conv2d_50/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_51/kernel/m
�
+Adam/conv2d_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_51/bias/m
|
)Adam/conv2d_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_52/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_52/kernel/m
�
+Adam/conv2d_52/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_52/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_52/bias/m
|
)Adam/conv2d_52/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_53/kernel/m
�
+Adam/conv2d_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_53/bias/m
|
)Adam/conv2d_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/dense_8/kernel/m
�
)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	�*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_48/kernel/v
�
+Adam/conv2d_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/kernel/v*&
_output_shapes
:@*
dtype0
�
Adam/conv2d_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_48/bias/v
{
)Adam/conv2d_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_48/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*(
shared_nameAdam/conv2d_49/kernel/v
�
+Adam/conv2d_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/conv2d_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_49/bias/v
|
)Adam/conv2d_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_50/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_50/kernel/v
�
+Adam/conv2d_50/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_50/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_50/bias/v
|
)Adam/conv2d_50/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_50/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_51/kernel/v
�
+Adam/conv2d_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_51/bias/v
|
)Adam/conv2d_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_52/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_52/kernel/v
�
+Adam/conv2d_52/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_52/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_52/bias/v
|
)Adam/conv2d_52/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_52/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_53/kernel/v
�
+Adam/conv2d_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_53/bias/v
|
)Adam/conv2d_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/dense_8/kernel/v
�
)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	�*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�T
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�T
value�TB�T B�T
�
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
R
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
R
:	variables
;trainable_variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
R
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
�
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem�m�m�m�(m�)m�.m�/m�4m�5m�>m�?m�Lm�Mm�v�v�v�v�(v�)v�.v�/v�4v�5v�>v�?v�Lv�Mv�
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
 
�
	variables
Wlayer_metrics
Xlayer_regularization_losses
trainable_variables

Ylayers
Znon_trainable_variables
[metrics
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_48/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_48/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
	variables
\layer_metrics
]layer_regularization_losses
trainable_variables
regularization_losses

^layers
_non_trainable_variables
`metrics
 
 
 
�
	variables
alayer_metrics
blayer_regularization_losses
trainable_variables
regularization_losses

clayers
dnon_trainable_variables
emetrics
\Z
VARIABLE_VALUEconv2d_49/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_49/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
 	variables
flayer_metrics
glayer_regularization_losses
!trainable_variables
"regularization_losses

hlayers
inon_trainable_variables
jmetrics
 
 
 
�
$	variables
klayer_metrics
llayer_regularization_losses
%trainable_variables
&regularization_losses

mlayers
nnon_trainable_variables
ometrics
\Z
VARIABLE_VALUEconv2d_50/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_50/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
�
*	variables
player_metrics
qlayer_regularization_losses
+trainable_variables
,regularization_losses

rlayers
snon_trainable_variables
tmetrics
\Z
VARIABLE_VALUEconv2d_51/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_51/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
�
0	variables
ulayer_metrics
vlayer_regularization_losses
1trainable_variables
2regularization_losses

wlayers
xnon_trainable_variables
ymetrics
\Z
VARIABLE_VALUEconv2d_52/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_52/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
�
6	variables
zlayer_metrics
{layer_regularization_losses
7trainable_variables
8regularization_losses

|layers
}non_trainable_variables
~metrics
 
 
 
�
:	variables
layer_metrics
 �layer_regularization_losses
;trainable_variables
<regularization_losses
�layers
�non_trainable_variables
�metrics
\Z
VARIABLE_VALUEconv2d_53/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_53/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
�
@	variables
�layer_metrics
 �layer_regularization_losses
Atrainable_variables
Bregularization_losses
�layers
�non_trainable_variables
�metrics
 
 
 
�
D	variables
�layer_metrics
 �layer_regularization_losses
Etrainable_variables
Fregularization_losses
�layers
�non_trainable_variables
�metrics
 
 
 
�
H	variables
�layer_metrics
 �layer_regularization_losses
Itrainable_variables
Jregularization_losses
�layers
�non_trainable_variables
�metrics
ZX
VARIABLE_VALUEdense_8/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_8/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
�
N	variables
�layer_metrics
 �layer_regularization_losses
Otrainable_variables
Pregularization_losses
�layers
�non_trainable_variables
�metrics
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

�0
�1
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

�total

�count
�	variables
�	keras_api
I

�total

�count
�
_fn_kwargs
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

�0
�1

�	variables
}
VARIABLE_VALUEAdam/conv2d_48/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_48/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_49/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_49/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_50/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_50/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_51/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_51/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_52/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_52/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_53/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_53/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_48/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_48/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_49/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_49/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_50/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_50/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_51/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_51/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_52/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_52/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_53/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_53/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_9Placeholder*/
_output_shapes
:���������@ *
dtype0*$
shape:���������@ 
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_9conv2d_48/kernelconv2d_48/biasconv2d_49/kernelconv2d_49/biasconv2d_50/kernelconv2d_50/biasconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasdense_8/kerneldense_8/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *,
f'R%
#__inference_signature_wrapper_73739
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_48/kernel/Read/ReadVariableOp"conv2d_48/bias/Read/ReadVariableOp$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOp$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp$conv2d_52/kernel/Read/ReadVariableOp"conv2d_52/bias/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_48/kernel/m/Read/ReadVariableOp)Adam/conv2d_48/bias/m/Read/ReadVariableOp+Adam/conv2d_49/kernel/m/Read/ReadVariableOp)Adam/conv2d_49/bias/m/Read/ReadVariableOp+Adam/conv2d_50/kernel/m/Read/ReadVariableOp)Adam/conv2d_50/bias/m/Read/ReadVariableOp+Adam/conv2d_51/kernel/m/Read/ReadVariableOp)Adam/conv2d_51/bias/m/Read/ReadVariableOp+Adam/conv2d_52/kernel/m/Read/ReadVariableOp)Adam/conv2d_52/bias/m/Read/ReadVariableOp+Adam/conv2d_53/kernel/m/Read/ReadVariableOp)Adam/conv2d_53/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp+Adam/conv2d_48/kernel/v/Read/ReadVariableOp)Adam/conv2d_48/bias/v/Read/ReadVariableOp+Adam/conv2d_49/kernel/v/Read/ReadVariableOp)Adam/conv2d_49/bias/v/Read/ReadVariableOp+Adam/conv2d_50/kernel/v/Read/ReadVariableOp)Adam/conv2d_50/bias/v/Read/ReadVariableOp+Adam/conv2d_51/kernel/v/Read/ReadVariableOp)Adam/conv2d_51/bias/v/Read/ReadVariableOp+Adam/conv2d_52/kernel/v/Read/ReadVariableOp)Adam/conv2d_52/bias/v/Read/ReadVariableOp+Adam/conv2d_53/kernel/v/Read/ReadVariableOp)Adam/conv2d_53/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOpConst*@
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
GPU2*0,1J 8� *'
f"R 
__inference__traced_save_74331
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_48/kernelconv2d_48/biasconv2d_49/kernelconv2d_49/biasconv2d_50/kernelconv2d_50/biasconv2d_51/kernelconv2d_51/biasconv2d_52/kernelconv2d_52/biasconv2d_53/kernelconv2d_53/biasdense_8/kerneldense_8/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_48/kernel/mAdam/conv2d_48/bias/mAdam/conv2d_49/kernel/mAdam/conv2d_49/bias/mAdam/conv2d_50/kernel/mAdam/conv2d_50/bias/mAdam/conv2d_51/kernel/mAdam/conv2d_51/bias/mAdam/conv2d_52/kernel/mAdam/conv2d_52/bias/mAdam/conv2d_53/kernel/mAdam/conv2d_53/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/mAdam/conv2d_48/kernel/vAdam/conv2d_48/bias/vAdam/conv2d_49/kernel/vAdam/conv2d_49/bias/vAdam/conv2d_50/kernel/vAdam/conv2d_50/bias/vAdam/conv2d_51/kernel/vAdam/conv2d_51/bias/vAdam/conv2d_52/kernel/vAdam/conv2d_52/bias/vAdam/conv2d_53/kernel/vAdam/conv2d_53/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/v*?
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
GPU2*0,1J 8� **
f%R#
!__inference__traced_restore_74494��	
�
�
'__inference_dense_8_layer_call_fn_74145

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_733332
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_48_layer_call_fn_73932

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_731992
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@ @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_73167

inputs
identity�
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73209

inputs
identity�
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:��������� @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@ @:W S
/
_output_shapes
:���������@ @
 
_user_specified_nameinputs
�
P
4__inference_average_pooling2d_26_layer_call_fn_74068

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_731442
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_74003

inputs
identity�
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:��������� �:X T
0
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
�
)__inference_conv2d_51_layer_call_fn_74032

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_732622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_51_layer_call_and_return_conditional_losses_73262

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_73998

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
)__inference_conv2d_49_layer_call_fn_73972

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:��������� �*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_732222
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:��������� �2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
D__inference_conv2d_49_layer_call_and_return_conditional_losses_73983

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:��������� �2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
P
4__inference_average_pooling2d_25_layer_call_fn_73988

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_731222
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
B__inference_dense_8_layer_call_and_return_conditional_losses_74155

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�9
�
B__inference_model_8_layer_call_and_return_conditional_losses_73654
input_9)
conv2d_48_73613:@
conv2d_48_73615:@*
conv2d_49_73619:@�
conv2d_49_73621:	�+
conv2d_50_73625:��
conv2d_50_73627:	�+
conv2d_51_73630:��
conv2d_51_73632:	�+
conv2d_52_73635:��
conv2d_52_73637:	�+
conv2d_53_73641:��
conv2d_53_73643:	� 
dense_8_73648:	�
dense_8_73650:
identity��!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_48_73613conv2d_48_73615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_731992#
!conv2d_48/StatefulPartitionedCall�
$average_pooling2d_24/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_732092&
$average_pooling2d_24/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_24/PartitionedCall:output:0conv2d_49_73619conv2d_49_73621*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:��������� �*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_732222#
!conv2d_49/StatefulPartitionedCall�
$average_pooling2d_25/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_732322&
$average_pooling2d_25/PartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_25/PartitionedCall:output:0conv2d_50_73625conv2d_50_73627*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_732452#
!conv2d_50/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0conv2d_51_73630conv2d_51_73632*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_732622#
!conv2d_51/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_73635conv2d_52_73637*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_732792#
!conv2d_52/StatefulPartitionedCall�
$average_pooling2d_26/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_732892&
$average_pooling2d_26/PartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_26/PartitionedCall:output:0conv2d_53_73641conv2d_53_73643*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_733022#
!conv2d_53/StatefulPartitionedCall�
*global_average_pooling2d_8/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *^
fYRW
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_733132,
*global_average_pooling2d_8/PartitionedCall�
flatten_8/PartitionedCallPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_733212
flatten_8/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_73648dense_8_73650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_733332!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_9
�
�
D__inference_conv2d_52_layer_call_and_return_conditional_losses_74063

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_74125

inputs
identity�
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_73289

inputs
identity�
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
4__inference_average_pooling2d_24_layer_call_fn_73953

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_732092
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:��������� @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@ @:W S
/
_output_shapes
:���������@ @
 
_user_specified_nameinputs
�
�
)__inference_conv2d_50_layer_call_fn_74012

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_732452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�S
�
B__inference_model_8_layer_call_and_return_conditional_losses_73864

inputsB
(conv2d_48_conv2d_readvariableop_resource:@7
)conv2d_48_biasadd_readvariableop_resource:@C
(conv2d_49_conv2d_readvariableop_resource:@�8
)conv2d_49_biasadd_readvariableop_resource:	�D
(conv2d_50_conv2d_readvariableop_resource:��8
)conv2d_50_biasadd_readvariableop_resource:	�D
(conv2d_51_conv2d_readvariableop_resource:��8
)conv2d_51_biasadd_readvariableop_resource:	�D
(conv2d_52_conv2d_readvariableop_resource:��8
)conv2d_52_biasadd_readvariableop_resource:	�D
(conv2d_53_conv2d_readvariableop_resource:��8
)conv2d_53_biasadd_readvariableop_resource:	�9
&dense_8_matmul_readvariableop_resource:	�5
'dense_8_biasadd_readvariableop_resource:
identity�� conv2d_48/BiasAdd/ReadVariableOp�conv2d_48/Conv2D/ReadVariableOp� conv2d_49/BiasAdd/ReadVariableOp�conv2d_49/Conv2D/ReadVariableOp� conv2d_50/BiasAdd/ReadVariableOp�conv2d_50/Conv2D/ReadVariableOp� conv2d_51/BiasAdd/ReadVariableOp�conv2d_51/Conv2D/ReadVariableOp� conv2d_52/BiasAdd/ReadVariableOp�conv2d_52/Conv2D/ReadVariableOp� conv2d_53/BiasAdd/ReadVariableOp�conv2d_53/Conv2D/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_48/Conv2D/ReadVariableOp�
conv2d_48/Conv2DConv2Dinputs'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
conv2d_48/Conv2D�
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp�
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2
conv2d_48/BiasAdd~
conv2d_48/ReluReluconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
conv2d_48/Relu�
average_pooling2d_24/AvgPoolAvgPoolconv2d_48/Relu:activations:0*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_24/AvgPool�
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_49/Conv2D/ReadVariableOp�
conv2d_49/Conv2DConv2D%average_pooling2d_24/AvgPool:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
conv2d_49/Conv2D�
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp�
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
conv2d_49/Relu�
average_pooling2d_25/AvgPoolAvgPoolconv2d_49/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_25/AvgPool�
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_50/Conv2D/ReadVariableOp�
conv2d_50/Conv2DConv2D%average_pooling2d_25/AvgPool:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_50/Conv2D�
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp�
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_50/BiasAdd
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_50/Relu�
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_51/Conv2D/ReadVariableOp�
conv2d_51/Conv2DConv2Dconv2d_50/Relu:activations:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_51/Conv2D�
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp�
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_51/BiasAdd
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_51/Relu�
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_52/Conv2D/ReadVariableOp�
conv2d_52/Conv2DConv2Dconv2d_51/Relu:activations:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_52/Conv2D�
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp�
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_52/BiasAdd
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_52/Relu�
average_pooling2d_26/AvgPoolAvgPoolconv2d_52/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_26/AvgPool�
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_53/Conv2D/ReadVariableOp�
conv2d_53/Conv2DConv2D%average_pooling2d_26/AvgPool:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_53/Conv2D�
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp�
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_53/Relu�
1global_average_pooling2d_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_8/Mean/reduction_indices�
global_average_pooling2d_8/MeanMeanconv2d_53/Relu:activations:0:global_average_pooling2d_8/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2!
global_average_pooling2d_8/Means
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_8/Const�
flatten_8/ReshapeReshape(global_average_pooling2d_8/Mean:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_8/Reshape�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
�
)__inference_conv2d_53_layer_call_fn_74092

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_733022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_53_layer_call_and_return_conditional_losses_74103

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_52_layer_call_and_return_conditional_losses_73279

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_73144

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_73321

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
4__inference_average_pooling2d_25_layer_call_fn_73993

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_732322
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:��������� �:X T
0
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
�
D__inference_conv2d_51_layer_call_and_return_conditional_losses_74043

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_50_layer_call_and_return_conditional_losses_74023

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_74078

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_74083

inputs
identity�
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_8_layer_call_fn_73371
input_9!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_733402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_9
�`
�
 __inference__wrapped_model_73091
input_9J
0model_8_conv2d_48_conv2d_readvariableop_resource:@?
1model_8_conv2d_48_biasadd_readvariableop_resource:@K
0model_8_conv2d_49_conv2d_readvariableop_resource:@�@
1model_8_conv2d_49_biasadd_readvariableop_resource:	�L
0model_8_conv2d_50_conv2d_readvariableop_resource:��@
1model_8_conv2d_50_biasadd_readvariableop_resource:	�L
0model_8_conv2d_51_conv2d_readvariableop_resource:��@
1model_8_conv2d_51_biasadd_readvariableop_resource:	�L
0model_8_conv2d_52_conv2d_readvariableop_resource:��@
1model_8_conv2d_52_biasadd_readvariableop_resource:	�L
0model_8_conv2d_53_conv2d_readvariableop_resource:��@
1model_8_conv2d_53_biasadd_readvariableop_resource:	�A
.model_8_dense_8_matmul_readvariableop_resource:	�=
/model_8_dense_8_biasadd_readvariableop_resource:
identity��(model_8/conv2d_48/BiasAdd/ReadVariableOp�'model_8/conv2d_48/Conv2D/ReadVariableOp�(model_8/conv2d_49/BiasAdd/ReadVariableOp�'model_8/conv2d_49/Conv2D/ReadVariableOp�(model_8/conv2d_50/BiasAdd/ReadVariableOp�'model_8/conv2d_50/Conv2D/ReadVariableOp�(model_8/conv2d_51/BiasAdd/ReadVariableOp�'model_8/conv2d_51/Conv2D/ReadVariableOp�(model_8/conv2d_52/BiasAdd/ReadVariableOp�'model_8/conv2d_52/Conv2D/ReadVariableOp�(model_8/conv2d_53/BiasAdd/ReadVariableOp�'model_8/conv2d_53/Conv2D/ReadVariableOp�&model_8/dense_8/BiasAdd/ReadVariableOp�%model_8/dense_8/MatMul/ReadVariableOp�
'model_8/conv2d_48/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02)
'model_8/conv2d_48/Conv2D/ReadVariableOp�
model_8/conv2d_48/Conv2DConv2Dinput_9/model_8/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
model_8/conv2d_48/Conv2D�
(model_8/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model_8/conv2d_48/BiasAdd/ReadVariableOp�
model_8/conv2d_48/BiasAddBiasAdd!model_8/conv2d_48/Conv2D:output:00model_8/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2
model_8/conv2d_48/BiasAdd�
model_8/conv2d_48/ReluRelu"model_8/conv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
model_8/conv2d_48/Relu�
$model_8/average_pooling2d_24/AvgPoolAvgPool$model_8/conv2d_48/Relu:activations:0*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2&
$model_8/average_pooling2d_24/AvgPool�
'model_8/conv2d_49/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02)
'model_8/conv2d_49/Conv2D/ReadVariableOp�
model_8/conv2d_49/Conv2DConv2D-model_8/average_pooling2d_24/AvgPool:output:0/model_8/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
model_8/conv2d_49/Conv2D�
(model_8/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_8/conv2d_49/BiasAdd/ReadVariableOp�
model_8/conv2d_49/BiasAddBiasAdd!model_8/conv2d_49/Conv2D:output:00model_8/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2
model_8/conv2d_49/BiasAdd�
model_8/conv2d_49/ReluRelu"model_8/conv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
model_8/conv2d_49/Relu�
$model_8/average_pooling2d_25/AvgPoolAvgPool$model_8/conv2d_49/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2&
$model_8/average_pooling2d_25/AvgPool�
'model_8/conv2d_50/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02)
'model_8/conv2d_50/Conv2D/ReadVariableOp�
model_8/conv2d_50/Conv2DConv2D-model_8/average_pooling2d_25/AvgPool:output:0/model_8/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_8/conv2d_50/Conv2D�
(model_8/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_8/conv2d_50/BiasAdd/ReadVariableOp�
model_8/conv2d_50/BiasAddBiasAdd!model_8/conv2d_50/Conv2D:output:00model_8/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_50/BiasAdd�
model_8/conv2d_50/ReluRelu"model_8/conv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_50/Relu�
'model_8/conv2d_51/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_51_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02)
'model_8/conv2d_51/Conv2D/ReadVariableOp�
model_8/conv2d_51/Conv2DConv2D$model_8/conv2d_50/Relu:activations:0/model_8/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_8/conv2d_51/Conv2D�
(model_8/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_51_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_8/conv2d_51/BiasAdd/ReadVariableOp�
model_8/conv2d_51/BiasAddBiasAdd!model_8/conv2d_51/Conv2D:output:00model_8/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_51/BiasAdd�
model_8/conv2d_51/ReluRelu"model_8/conv2d_51/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_51/Relu�
'model_8/conv2d_52/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_52_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02)
'model_8/conv2d_52/Conv2D/ReadVariableOp�
model_8/conv2d_52/Conv2DConv2D$model_8/conv2d_51/Relu:activations:0/model_8/conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_8/conv2d_52/Conv2D�
(model_8/conv2d_52/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_8/conv2d_52/BiasAdd/ReadVariableOp�
model_8/conv2d_52/BiasAddBiasAdd!model_8/conv2d_52/Conv2D:output:00model_8/conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_52/BiasAdd�
model_8/conv2d_52/ReluRelu"model_8/conv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_52/Relu�
$model_8/average_pooling2d_26/AvgPoolAvgPool$model_8/conv2d_52/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2&
$model_8/average_pooling2d_26/AvgPool�
'model_8/conv2d_53/Conv2D/ReadVariableOpReadVariableOp0model_8_conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02)
'model_8/conv2d_53/Conv2D/ReadVariableOp�
model_8/conv2d_53/Conv2DConv2D-model_8/average_pooling2d_26/AvgPool:output:0/model_8/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_8/conv2d_53/Conv2D�
(model_8/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp1model_8_conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_8/conv2d_53/BiasAdd/ReadVariableOp�
model_8/conv2d_53/BiasAddBiasAdd!model_8/conv2d_53/Conv2D:output:00model_8/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_53/BiasAdd�
model_8/conv2d_53/ReluRelu"model_8/conv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_8/conv2d_53/Relu�
9model_8/global_average_pooling2d_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_8/global_average_pooling2d_8/Mean/reduction_indices�
'model_8/global_average_pooling2d_8/MeanMean$model_8/conv2d_53/Relu:activations:0Bmodel_8/global_average_pooling2d_8/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2)
'model_8/global_average_pooling2d_8/Mean�
model_8/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
model_8/flatten_8/Const�
model_8/flatten_8/ReshapeReshape0model_8/global_average_pooling2d_8/Mean:output:0 model_8/flatten_8/Const:output:0*
T0*(
_output_shapes
:����������2
model_8/flatten_8/Reshape�
%model_8/dense_8/MatMul/ReadVariableOpReadVariableOp.model_8_dense_8_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%model_8/dense_8/MatMul/ReadVariableOp�
model_8/dense_8/MatMulMatMul"model_8/flatten_8/Reshape:output:0-model_8/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_8/dense_8/MatMul�
&model_8/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_8_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_8/dense_8/BiasAdd/ReadVariableOp�
model_8/dense_8/BiasAddBiasAdd model_8/dense_8/MatMul:product:0.model_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_8/dense_8/BiasAdd{
IdentityIdentity model_8/dense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^model_8/conv2d_48/BiasAdd/ReadVariableOp(^model_8/conv2d_48/Conv2D/ReadVariableOp)^model_8/conv2d_49/BiasAdd/ReadVariableOp(^model_8/conv2d_49/Conv2D/ReadVariableOp)^model_8/conv2d_50/BiasAdd/ReadVariableOp(^model_8/conv2d_50/Conv2D/ReadVariableOp)^model_8/conv2d_51/BiasAdd/ReadVariableOp(^model_8/conv2d_51/Conv2D/ReadVariableOp)^model_8/conv2d_52/BiasAdd/ReadVariableOp(^model_8/conv2d_52/Conv2D/ReadVariableOp)^model_8/conv2d_53/BiasAdd/ReadVariableOp(^model_8/conv2d_53/Conv2D/ReadVariableOp'^model_8/dense_8/BiasAdd/ReadVariableOp&^model_8/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2T
(model_8/conv2d_48/BiasAdd/ReadVariableOp(model_8/conv2d_48/BiasAdd/ReadVariableOp2R
'model_8/conv2d_48/Conv2D/ReadVariableOp'model_8/conv2d_48/Conv2D/ReadVariableOp2T
(model_8/conv2d_49/BiasAdd/ReadVariableOp(model_8/conv2d_49/BiasAdd/ReadVariableOp2R
'model_8/conv2d_49/Conv2D/ReadVariableOp'model_8/conv2d_49/Conv2D/ReadVariableOp2T
(model_8/conv2d_50/BiasAdd/ReadVariableOp(model_8/conv2d_50/BiasAdd/ReadVariableOp2R
'model_8/conv2d_50/Conv2D/ReadVariableOp'model_8/conv2d_50/Conv2D/ReadVariableOp2T
(model_8/conv2d_51/BiasAdd/ReadVariableOp(model_8/conv2d_51/BiasAdd/ReadVariableOp2R
'model_8/conv2d_51/Conv2D/ReadVariableOp'model_8/conv2d_51/Conv2D/ReadVariableOp2T
(model_8/conv2d_52/BiasAdd/ReadVariableOp(model_8/conv2d_52/BiasAdd/ReadVariableOp2R
'model_8/conv2d_52/Conv2D/ReadVariableOp'model_8/conv2d_52/Conv2D/ReadVariableOp2T
(model_8/conv2d_53/BiasAdd/ReadVariableOp(model_8/conv2d_53/BiasAdd/ReadVariableOp2R
'model_8/conv2d_53/Conv2D/ReadVariableOp'model_8/conv2d_53/Conv2D/ReadVariableOp2P
&model_8/dense_8/BiasAdd/ReadVariableOp&model_8/dense_8/BiasAdd/ReadVariableOp2N
%model_8/dense_8/MatMul/ReadVariableOp%model_8/dense_8/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_9
�
�
)__inference_conv2d_52_layer_call_fn_74052

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_732792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_73313

inputs
identity�
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesp
MeanMeaninputsMean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2
Meanb
IdentityIdentityMean:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
P
4__inference_average_pooling2d_26_layer_call_fn_74073

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_732892
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_8_layer_call_fn_73772

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
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
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_733402
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
�
'__inference_model_8_layer_call_fn_73610
input_9!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_735462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_9
�
�
D__inference_conv2d_53_layer_call_and_return_conditional_losses_73302

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73100

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
B__inference_dense_8_layer_call_and_return_conditional_losses_73333

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�9
�
B__inference_model_8_layer_call_and_return_conditional_losses_73340

inputs)
conv2d_48_73200:@
conv2d_48_73202:@*
conv2d_49_73223:@�
conv2d_49_73225:	�+
conv2d_50_73246:��
conv2d_50_73248:	�+
conv2d_51_73263:��
conv2d_51_73265:	�+
conv2d_52_73280:��
conv2d_52_73282:	�+
conv2d_53_73303:��
conv2d_53_73305:	� 
dense_8_73334:	�
dense_8_73336:
identity��!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_48_73200conv2d_48_73202*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_731992#
!conv2d_48/StatefulPartitionedCall�
$average_pooling2d_24/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_732092&
$average_pooling2d_24/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_24/PartitionedCall:output:0conv2d_49_73223conv2d_49_73225*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:��������� �*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_732222#
!conv2d_49/StatefulPartitionedCall�
$average_pooling2d_25/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_732322&
$average_pooling2d_25/PartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_25/PartitionedCall:output:0conv2d_50_73246conv2d_50_73248*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_732452#
!conv2d_50/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0conv2d_51_73263conv2d_51_73265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_732622#
!conv2d_51/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_73280conv2d_52_73282*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_732792#
!conv2d_52/StatefulPartitionedCall�
$average_pooling2d_26/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_732892&
$average_pooling2d_26/PartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_26/PartitionedCall:output:0conv2d_53_73303conv2d_53_73305*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_733022#
!conv2d_53/StatefulPartitionedCall�
*global_average_pooling2d_8/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *^
fYRW
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_733132,
*global_average_pooling2d_8/PartitionedCall�
flatten_8/PartitionedCallPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_733212
flatten_8/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_73334dense_8_73336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_733332!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_49_layer_call_and_return_conditional_losses_73222

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:��������� �2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:��������� @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:��������� @
 
_user_specified_nameinputs
�
�
D__inference_conv2d_50_layer_call_and_return_conditional_losses_73245

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������2
Reluv
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_73232

inputs
identity�
AvgPoolAvgPoolinputs*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2	
AvgPoolm
IdentityIdentityAvgPool:output:0*
T0*0
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:��������� �:X T
0
_output_shapes
:��������� �
 
_user_specified_nameinputs
�
q
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_74119

inputs
identity�
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:������������������2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
E
)__inference_flatten_8_layer_call_fn_74130

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_733212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
� 
!__inference__traced_restore_74494
file_prefix;
!assignvariableop_conv2d_48_kernel:@/
!assignvariableop_1_conv2d_48_bias:@>
#assignvariableop_2_conv2d_49_kernel:@�0
!assignvariableop_3_conv2d_49_bias:	�?
#assignvariableop_4_conv2d_50_kernel:��0
!assignvariableop_5_conv2d_50_bias:	�?
#assignvariableop_6_conv2d_51_kernel:��0
!assignvariableop_7_conv2d_51_bias:	�?
#assignvariableop_8_conv2d_52_kernel:��0
!assignvariableop_9_conv2d_52_bias:	�@
$assignvariableop_10_conv2d_53_kernel:��1
"assignvariableop_11_conv2d_53_bias:	�5
"assignvariableop_12_dense_8_kernel:	�.
 assignvariableop_13_dense_8_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: E
+assignvariableop_23_adam_conv2d_48_kernel_m:@7
)assignvariableop_24_adam_conv2d_48_bias_m:@F
+assignvariableop_25_adam_conv2d_49_kernel_m:@�8
)assignvariableop_26_adam_conv2d_49_bias_m:	�G
+assignvariableop_27_adam_conv2d_50_kernel_m:��8
)assignvariableop_28_adam_conv2d_50_bias_m:	�G
+assignvariableop_29_adam_conv2d_51_kernel_m:��8
)assignvariableop_30_adam_conv2d_51_bias_m:	�G
+assignvariableop_31_adam_conv2d_52_kernel_m:��8
)assignvariableop_32_adam_conv2d_52_bias_m:	�G
+assignvariableop_33_adam_conv2d_53_kernel_m:��8
)assignvariableop_34_adam_conv2d_53_bias_m:	�<
)assignvariableop_35_adam_dense_8_kernel_m:	�5
'assignvariableop_36_adam_dense_8_bias_m:E
+assignvariableop_37_adam_conv2d_48_kernel_v:@7
)assignvariableop_38_adam_conv2d_48_bias_v:@F
+assignvariableop_39_adam_conv2d_49_kernel_v:@�8
)assignvariableop_40_adam_conv2d_49_bias_v:	�G
+assignvariableop_41_adam_conv2d_50_kernel_v:��8
)assignvariableop_42_adam_conv2d_50_bias_v:	�G
+assignvariableop_43_adam_conv2d_51_kernel_v:��8
)assignvariableop_44_adam_conv2d_51_bias_v:	�G
+assignvariableop_45_adam_conv2d_52_kernel_v:��8
)assignvariableop_46_adam_conv2d_52_bias_v:	�G
+assignvariableop_47_adam_conv2d_53_kernel_v:��8
)assignvariableop_48_adam_conv2d_53_bias_v:	�<
)assignvariableop_49_adam_dense_8_kernel_v:	�5
'assignvariableop_50_adam_dense_8_bias_v:
identity_52��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::*B
dtypes8
624	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_48_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_48_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_49_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_49_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_50_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_50_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_51_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_51_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_52_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_52_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_53_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_53_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_8_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_8_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOpassignvariableop_15_adam_beta_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_beta_2Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_decayIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_learning_rateIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_totalIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_countIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_total_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_count_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_48_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_48_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_49_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_49_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_50_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_50_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_51_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_51_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_52_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_52_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_53_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_53_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_8_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_8_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_48_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_48_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_conv2d_49_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_conv2d_49_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_50_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_50_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_51_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_51_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_52_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_52_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_53_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_53_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_8_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_8_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_509
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�	
Identity_51Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_51f
Identity_52IdentityIdentity_51:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_52�	
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
�
V
:__inference_global_average_pooling2d_8_layer_call_fn_74108

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *^
fYRW
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_731672
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
V
:__inference_global_average_pooling2d_8_layer_call_fn_74113

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *^
fYRW
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_733132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73958

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
'__inference_model_8_layer_call_fn_73805

inputs!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
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
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_model_8_layer_call_and_return_conditional_losses_735462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�9
�
B__inference_model_8_layer_call_and_return_conditional_losses_73546

inputs)
conv2d_48_73505:@
conv2d_48_73507:@*
conv2d_49_73511:@�
conv2d_49_73513:	�+
conv2d_50_73517:��
conv2d_50_73519:	�+
conv2d_51_73522:��
conv2d_51_73524:	�+
conv2d_52_73527:��
conv2d_52_73529:	�+
conv2d_53_73533:��
conv2d_53_73535:	� 
dense_8_73540:	�
dense_8_73542:
identity��!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_48_73505conv2d_48_73507*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_731992#
!conv2d_48/StatefulPartitionedCall�
$average_pooling2d_24/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_732092&
$average_pooling2d_24/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_24/PartitionedCall:output:0conv2d_49_73511conv2d_49_73513*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:��������� �*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_732222#
!conv2d_49/StatefulPartitionedCall�
$average_pooling2d_25/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_732322&
$average_pooling2d_25/PartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_25/PartitionedCall:output:0conv2d_50_73517conv2d_50_73519*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_732452#
!conv2d_50/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0conv2d_51_73522conv2d_51_73524*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_732622#
!conv2d_51/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_73527conv2d_52_73529*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_732792#
!conv2d_52/StatefulPartitionedCall�
$average_pooling2d_26/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_732892&
$average_pooling2d_26/PartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_26/PartitionedCall:output:0conv2d_53_73533conv2d_53_73535*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_733022#
!conv2d_53/StatefulPartitionedCall�
*global_average_pooling2d_8/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *^
fYRW
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_733132,
*global_average_pooling2d_8/PartitionedCall�
flatten_8/PartitionedCallPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_733212
flatten_8/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_73540dense_8_73542*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_733332!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
k
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_73122

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
D__inference_conv2d_48_layer_call_and_return_conditional_losses_73943

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@ @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
`
D__inference_flatten_8_layer_call_and_return_conditional_losses_74136

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_73739
input_9!
unknown:@
	unknown_0:@$
	unknown_1:@�
	unknown_2:	�%
	unknown_3:��
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:���������*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2*0,1J 8� *)
f$R"
 __inference__wrapped_model_730912
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_9
�
P
4__inference_average_pooling2d_24_layer_call_fn_73948

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_731002
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�j
�
__inference__traced_save_74331
file_prefix/
+savev2_conv2d_48_kernel_read_readvariableop-
)savev2_conv2d_48_bias_read_readvariableop/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop/
+savev2_conv2d_52_kernel_read_readvariableop-
)savev2_conv2d_52_bias_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_48_kernel_m_read_readvariableop4
0savev2_adam_conv2d_48_bias_m_read_readvariableop6
2savev2_adam_conv2d_49_kernel_m_read_readvariableop4
0savev2_adam_conv2d_49_bias_m_read_readvariableop6
2savev2_adam_conv2d_50_kernel_m_read_readvariableop4
0savev2_adam_conv2d_50_bias_m_read_readvariableop6
2savev2_adam_conv2d_51_kernel_m_read_readvariableop4
0savev2_adam_conv2d_51_bias_m_read_readvariableop6
2savev2_adam_conv2d_52_kernel_m_read_readvariableop4
0savev2_adam_conv2d_52_bias_m_read_readvariableop6
2savev2_adam_conv2d_53_kernel_m_read_readvariableop4
0savev2_adam_conv2d_53_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableop6
2savev2_adam_conv2d_48_kernel_v_read_readvariableop4
0savev2_adam_conv2d_48_bias_v_read_readvariableop6
2savev2_adam_conv2d_49_kernel_v_read_readvariableop4
0savev2_adam_conv2d_49_bias_v_read_readvariableop6
2savev2_adam_conv2d_50_kernel_v_read_readvariableop4
0savev2_adam_conv2d_50_bias_v_read_readvariableop6
2savev2_adam_conv2d_51_kernel_v_read_readvariableop4
0savev2_adam_conv2d_51_bias_v_read_readvariableop6
2savev2_adam_conv2d_52_kernel_v_read_readvariableop4
0savev2_adam_conv2d_52_bias_v_read_readvariableop6
2savev2_adam_conv2d_53_kernel_v_read_readvariableop4
0savev2_adam_conv2d_53_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*�
value�B�4B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:4*
dtype0*{
valuerBp4B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_48_kernel_read_readvariableop)savev2_conv2d_48_bias_read_readvariableop+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableop+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop+savev2_conv2d_52_kernel_read_readvariableop)savev2_conv2d_52_bias_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_48_kernel_m_read_readvariableop0savev2_adam_conv2d_48_bias_m_read_readvariableop2savev2_adam_conv2d_49_kernel_m_read_readvariableop0savev2_adam_conv2d_49_bias_m_read_readvariableop2savev2_adam_conv2d_50_kernel_m_read_readvariableop0savev2_adam_conv2d_50_bias_m_read_readvariableop2savev2_adam_conv2d_51_kernel_m_read_readvariableop0savev2_adam_conv2d_51_bias_m_read_readvariableop2savev2_adam_conv2d_52_kernel_m_read_readvariableop0savev2_adam_conv2d_52_bias_m_read_readvariableop2savev2_adam_conv2d_53_kernel_m_read_readvariableop0savev2_adam_conv2d_53_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop2savev2_adam_conv2d_48_kernel_v_read_readvariableop0savev2_adam_conv2d_48_bias_v_read_readvariableop2savev2_adam_conv2d_49_kernel_v_read_readvariableop0savev2_adam_conv2d_49_bias_v_read_readvariableop2savev2_adam_conv2d_50_kernel_v_read_readvariableop0savev2_adam_conv2d_50_bias_v_read_readvariableop2savev2_adam_conv2d_51_kernel_v_read_readvariableop0savev2_adam_conv2d_51_bias_v_read_readvariableop2savev2_adam_conv2d_52_kernel_v_read_readvariableop0savev2_adam_conv2d_52_bias_v_read_readvariableop2savev2_adam_conv2d_53_kernel_v_read_readvariableop0savev2_adam_conv2d_53_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *B
dtypes8
624	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:@:@�:�:��:�:��:�:��:�:��:�:	�:: : : : : : : : : :@:@:@�:�:��:�:��:�:��:�:��:�:	�::@:@:@�:�:��:�:��:�:��:�:��:�:	�:: 2(
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
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.	*
(
_output_shapes
:��:!


_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:%!

_output_shapes
:	�: 
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
:@�:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:. *
(
_output_shapes
:��:!!

_output_shapes	
:�:."*
(
_output_shapes
:��:!#

_output_shapes	
:�:%$!

_output_shapes
:	�: %
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
:@�:!)

_output_shapes	
:�:.**
(
_output_shapes
:��:!+

_output_shapes	
:�:.,*
(
_output_shapes
:��:!-

_output_shapes	
:�:..*
(
_output_shapes
:��:!/

_output_shapes	
:�:.0*
(
_output_shapes
:��:!1

_output_shapes	
:�:%2!

_output_shapes
:	�: 3

_output_shapes
::4

_output_shapes
: 
�
k
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73963

inputs
identity�
AvgPoolAvgPoolinputs*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2	
AvgPooll
IdentityIdentityAvgPool:output:0*
T0*/
_output_shapes
:��������� @2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@ @:W S
/
_output_shapes
:���������@ @
 
_user_specified_nameinputs
�S
�
B__inference_model_8_layer_call_and_return_conditional_losses_73923

inputsB
(conv2d_48_conv2d_readvariableop_resource:@7
)conv2d_48_biasadd_readvariableop_resource:@C
(conv2d_49_conv2d_readvariableop_resource:@�8
)conv2d_49_biasadd_readvariableop_resource:	�D
(conv2d_50_conv2d_readvariableop_resource:��8
)conv2d_50_biasadd_readvariableop_resource:	�D
(conv2d_51_conv2d_readvariableop_resource:��8
)conv2d_51_biasadd_readvariableop_resource:	�D
(conv2d_52_conv2d_readvariableop_resource:��8
)conv2d_52_biasadd_readvariableop_resource:	�D
(conv2d_53_conv2d_readvariableop_resource:��8
)conv2d_53_biasadd_readvariableop_resource:	�9
&dense_8_matmul_readvariableop_resource:	�5
'dense_8_biasadd_readvariableop_resource:
identity�� conv2d_48/BiasAdd/ReadVariableOp�conv2d_48/Conv2D/ReadVariableOp� conv2d_49/BiasAdd/ReadVariableOp�conv2d_49/Conv2D/ReadVariableOp� conv2d_50/BiasAdd/ReadVariableOp�conv2d_50/Conv2D/ReadVariableOp� conv2d_51/BiasAdd/ReadVariableOp�conv2d_51/Conv2D/ReadVariableOp� conv2d_52/BiasAdd/ReadVariableOp�conv2d_52/Conv2D/ReadVariableOp� conv2d_53/BiasAdd/ReadVariableOp�conv2d_53/Conv2D/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_48/Conv2D/ReadVariableOp�
conv2d_48/Conv2DConv2Dinputs'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
conv2d_48/Conv2D�
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp�
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2
conv2d_48/BiasAdd~
conv2d_48/ReluReluconv2d_48/BiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
conv2d_48/Relu�
average_pooling2d_24/AvgPoolAvgPoolconv2d_48/Relu:activations:0*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_24/AvgPool�
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02!
conv2d_49/Conv2D/ReadVariableOp�
conv2d_49/Conv2DConv2D%average_pooling2d_24/AvgPool:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
conv2d_49/Conv2D�
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp�
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2
conv2d_49/BiasAdd
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
conv2d_49/Relu�
average_pooling2d_25/AvgPoolAvgPoolconv2d_49/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_25/AvgPool�
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_50/Conv2D/ReadVariableOp�
conv2d_50/Conv2DConv2D%average_pooling2d_25/AvgPool:output:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_50/Conv2D�
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp�
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_50/BiasAdd
conv2d_50/ReluReluconv2d_50/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_50/Relu�
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_51/Conv2D/ReadVariableOp�
conv2d_51/Conv2DConv2Dconv2d_50/Relu:activations:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_51/Conv2D�
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp�
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_51/BiasAdd
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_51/Relu�
conv2d_52/Conv2D/ReadVariableOpReadVariableOp(conv2d_52_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_52/Conv2D/ReadVariableOp�
conv2d_52/Conv2DConv2Dconv2d_51/Relu:activations:0'conv2d_52/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_52/Conv2D�
 conv2d_52/BiasAdd/ReadVariableOpReadVariableOp)conv2d_52_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_52/BiasAdd/ReadVariableOp�
conv2d_52/BiasAddBiasAddconv2d_52/Conv2D:output:0(conv2d_52/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_52/BiasAdd
conv2d_52/ReluReluconv2d_52/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_52/Relu�
average_pooling2d_26/AvgPoolAvgPoolconv2d_52/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_26/AvgPool�
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_53/Conv2D/ReadVariableOp�
conv2d_53/Conv2DConv2D%average_pooling2d_26/AvgPool:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_53/Conv2D�
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_53/BiasAdd/ReadVariableOp�
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_53/BiasAdd
conv2d_53/ReluReluconv2d_53/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_53/Relu�
1global_average_pooling2d_8/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_8/Mean/reduction_indices�
global_average_pooling2d_8/MeanMeanconv2d_53/Relu:activations:0:global_average_pooling2d_8/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2!
global_average_pooling2d_8/Means
flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_8/Const�
flatten_8/ReshapeReshape(global_average_pooling2d_8/Mean:output:0flatten_8/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_8/Reshape�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_8/MatMul/ReadVariableOp�
dense_8/MatMulMatMulflatten_8/Reshape:output:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/MatMul�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_8/BiasAdds
IdentityIdentitydense_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_52/BiasAdd/ReadVariableOp ^conv2d_52/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_52/BiasAdd/ReadVariableOp conv2d_52/BiasAdd/ReadVariableOp2B
conv2d_52/Conv2D/ReadVariableOpconv2d_52/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
�
D__inference_conv2d_48_layer_call_and_return_conditional_losses_73199

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@ @2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�9
�
B__inference_model_8_layer_call_and_return_conditional_losses_73698
input_9)
conv2d_48_73657:@
conv2d_48_73659:@*
conv2d_49_73663:@�
conv2d_49_73665:	�+
conv2d_50_73669:��
conv2d_50_73671:	�+
conv2d_51_73674:��
conv2d_51_73676:	�+
conv2d_52_73679:��
conv2d_52_73681:	�+
conv2d_53_73685:��
conv2d_53_73687:	� 
dense_8_73692:	�
dense_8_73694:
identity��!conv2d_48/StatefulPartitionedCall�!conv2d_49/StatefulPartitionedCall�!conv2d_50/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�!conv2d_52/StatefulPartitionedCall�!conv2d_53/StatefulPartitionedCall�dense_8/StatefulPartitionedCall�
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCallinput_9conv2d_48_73657conv2d_48_73659*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@ @*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_48_layer_call_and_return_conditional_losses_731992#
!conv2d_48/StatefulPartitionedCall�
$average_pooling2d_24/PartitionedCallPartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:��������� @* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_732092&
$average_pooling2d_24/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_24/PartitionedCall:output:0conv2d_49_73663conv2d_49_73665*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:��������� �*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_732222#
!conv2d_49/StatefulPartitionedCall�
$average_pooling2d_25/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_732322&
$average_pooling2d_25/PartitionedCall�
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_25/PartitionedCall:output:0conv2d_50_73669conv2d_50_73671*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_50_layer_call_and_return_conditional_losses_732452#
!conv2d_50/StatefulPartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_50/StatefulPartitionedCall:output:0conv2d_51_73674conv2d_51_73676*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_732622#
!conv2d_51/StatefulPartitionedCall�
!conv2d_52/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0conv2d_52_73679conv2d_52_73681*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_52_layer_call_and_return_conditional_losses_732792#
!conv2d_52/StatefulPartitionedCall�
$average_pooling2d_26/PartitionedCallPartitionedCall*conv2d_52/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *X
fSRQ
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_732892&
$average_pooling2d_26/PartitionedCall�
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_26/PartitionedCall:output:0conv2d_53_73685conv2d_53_73687*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_733022#
!conv2d_53/StatefulPartitionedCall�
*global_average_pooling2d_8/PartitionedCallPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *^
fYRW
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_733132,
*global_average_pooling2d_8/PartitionedCall�
flatten_8/PartitionedCallPartitionedCall3global_average_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8� *M
fHRF
D__inference_flatten_8_layer_call_and_return_conditional_losses_733212
flatten_8/PartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCall"flatten_8/PartitionedCall:output:0dense_8_73692dense_8_73694*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8� *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_733332!
dense_8/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_52/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_52/StatefulPartitionedCall!conv2d_52/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_9"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_98
serving_default_input_9:0���������@ ;
dense_80
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
:	variables
;trainable_variables
<regularization_losses
=	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

>kernel
?bias
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem�m�m�m�(m�)m�.m�/m�4m�5m�>m�?m�Lm�Mm�v�v�v�v�(v�)v�.v�/v�4v�5v�>v�?v�Lv�Mv�"
	optimizer
�
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
�
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
 "
trackable_list_wrapper
�
	variables
Wlayer_metrics
Xlayer_regularization_losses
trainable_variables

Ylayers
Znon_trainable_variables
[metrics
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
*:(@2conv2d_48/kernel
:@2conv2d_48/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
\layer_metrics
]layer_regularization_losses
trainable_variables
regularization_losses

^layers
_non_trainable_variables
`metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	variables
alayer_metrics
blayer_regularization_losses
trainable_variables
regularization_losses

clayers
dnon_trainable_variables
emetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)@�2conv2d_49/kernel
:�2conv2d_49/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
 	variables
flayer_metrics
glayer_regularization_losses
!trainable_variables
"regularization_losses

hlayers
inon_trainable_variables
jmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
$	variables
klayer_metrics
llayer_regularization_losses
%trainable_variables
&regularization_losses

mlayers
nnon_trainable_variables
ometrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_50/kernel
:�2conv2d_50/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
*	variables
player_metrics
qlayer_regularization_losses
+trainable_variables
,regularization_losses

rlayers
snon_trainable_variables
tmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_51/kernel
:�2conv2d_51/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
0	variables
ulayer_metrics
vlayer_regularization_losses
1trainable_variables
2regularization_losses

wlayers
xnon_trainable_variables
ymetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_52/kernel
:�2conv2d_52/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
�
6	variables
zlayer_metrics
{layer_regularization_losses
7trainable_variables
8regularization_losses

|layers
}non_trainable_variables
~metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
:	variables
layer_metrics
 �layer_regularization_losses
;trainable_variables
<regularization_losses
�layers
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_53/kernel
:�2conv2d_53/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@	variables
�layer_metrics
 �layer_regularization_losses
Atrainable_variables
Bregularization_losses
�layers
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
D	variables
�layer_metrics
 �layer_regularization_losses
Etrainable_variables
Fregularization_losses
�layers
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
H	variables
�layer_metrics
 �layer_regularization_losses
Itrainable_variables
Jregularization_losses
�layers
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_8/kernel
:2dense_8/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
N	variables
�layer_metrics
 �layer_regularization_losses
Otrainable_variables
Pregularization_losses
�layers
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
trackable_list_wrapper
0
�0
�1"
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

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
/:-@2Adam/conv2d_48/kernel/m
!:@2Adam/conv2d_48/bias/m
0:.@�2Adam/conv2d_49/kernel/m
": �2Adam/conv2d_49/bias/m
1:/��2Adam/conv2d_50/kernel/m
": �2Adam/conv2d_50/bias/m
1:/��2Adam/conv2d_51/kernel/m
": �2Adam/conv2d_51/bias/m
1:/��2Adam/conv2d_52/kernel/m
": �2Adam/conv2d_52/bias/m
1:/��2Adam/conv2d_53/kernel/m
": �2Adam/conv2d_53/bias/m
&:$	�2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
/:-@2Adam/conv2d_48/kernel/v
!:@2Adam/conv2d_48/bias/v
0:.@�2Adam/conv2d_49/kernel/v
": �2Adam/conv2d_49/bias/v
1:/��2Adam/conv2d_50/kernel/v
": �2Adam/conv2d_50/bias/v
1:/��2Adam/conv2d_51/kernel/v
": �2Adam/conv2d_51/bias/v
1:/��2Adam/conv2d_52/kernel/v
": �2Adam/conv2d_52/bias/v
1:/��2Adam/conv2d_53/kernel/v
": �2Adam/conv2d_53/bias/v
&:$	�2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
�2�
'__inference_model_8_layer_call_fn_73371
'__inference_model_8_layer_call_fn_73772
'__inference_model_8_layer_call_fn_73805
'__inference_model_8_layer_call_fn_73610�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
 __inference__wrapped_model_73091input_9"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_model_8_layer_call_and_return_conditional_losses_73864
B__inference_model_8_layer_call_and_return_conditional_losses_73923
B__inference_model_8_layer_call_and_return_conditional_losses_73654
B__inference_model_8_layer_call_and_return_conditional_losses_73698�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
)__inference_conv2d_48_layer_call_fn_73932�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_48_layer_call_and_return_conditional_losses_73943�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_average_pooling2d_24_layer_call_fn_73948
4__inference_average_pooling2d_24_layer_call_fn_73953�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73958
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73963�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_49_layer_call_fn_73972�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_49_layer_call_and_return_conditional_losses_73983�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_average_pooling2d_25_layer_call_fn_73988
4__inference_average_pooling2d_25_layer_call_fn_73993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_73998
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_74003�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_50_layer_call_fn_74012�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_50_layer_call_and_return_conditional_losses_74023�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_51_layer_call_fn_74032�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_51_layer_call_and_return_conditional_losses_74043�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_52_layer_call_fn_74052�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_52_layer_call_and_return_conditional_losses_74063�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
4__inference_average_pooling2d_26_layer_call_fn_74068
4__inference_average_pooling2d_26_layer_call_fn_74073�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_74078
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_74083�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_conv2d_53_layer_call_fn_74092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_conv2d_53_layer_call_and_return_conditional_losses_74103�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
:__inference_global_average_pooling2d_8_layer_call_fn_74108
:__inference_global_average_pooling2d_8_layer_call_fn_74113�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_74119
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_74125�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_flatten_8_layer_call_fn_74130�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_flatten_8_layer_call_and_return_conditional_losses_74136�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_8_layer_call_fn_74145�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_8_layer_call_and_return_conditional_losses_74155�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_73739input_9"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
 __inference__wrapped_model_73091}()./45>?LM8�5
.�+
)�&
input_9���������@ 
� "1�.
,
dense_8!�
dense_8����������
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73958�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
O__inference_average_pooling2d_24_layer_call_and_return_conditional_losses_73963h7�4
-�*
(�%
inputs���������@ @
� "-�*
#� 
0��������� @
� �
4__inference_average_pooling2d_24_layer_call_fn_73948�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
4__inference_average_pooling2d_24_layer_call_fn_73953[7�4
-�*
(�%
inputs���������@ @
� " ���������� @�
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_73998�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
O__inference_average_pooling2d_25_layer_call_and_return_conditional_losses_74003j8�5
.�+
)�&
inputs��������� �
� ".�+
$�!
0����������
� �
4__inference_average_pooling2d_25_layer_call_fn_73988�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
4__inference_average_pooling2d_25_layer_call_fn_73993]8�5
.�+
)�&
inputs��������� �
� "!������������
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_74078�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
O__inference_average_pooling2d_26_layer_call_and_return_conditional_losses_74083j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
4__inference_average_pooling2d_26_layer_call_fn_74068�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
4__inference_average_pooling2d_26_layer_call_fn_74073]8�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_48_layer_call_and_return_conditional_losses_73943l7�4
-�*
(�%
inputs���������@ 
� "-�*
#� 
0���������@ @
� �
)__inference_conv2d_48_layer_call_fn_73932_7�4
-�*
(�%
inputs���������@ 
� " ����������@ @�
D__inference_conv2d_49_layer_call_and_return_conditional_losses_73983m7�4
-�*
(�%
inputs��������� @
� ".�+
$�!
0��������� �
� �
)__inference_conv2d_49_layer_call_fn_73972`7�4
-�*
(�%
inputs��������� @
� "!���������� ��
D__inference_conv2d_50_layer_call_and_return_conditional_losses_74023n()8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_50_layer_call_fn_74012a()8�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_51_layer_call_and_return_conditional_losses_74043n./8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_51_layer_call_fn_74032a./8�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_52_layer_call_and_return_conditional_losses_74063n458�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_52_layer_call_fn_74052a458�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_53_layer_call_and_return_conditional_losses_74103n>?8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_53_layer_call_fn_74092a>?8�5
.�+
)�&
inputs����������
� "!������������
B__inference_dense_8_layer_call_and_return_conditional_losses_74155]LM0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_dense_8_layer_call_fn_74145PLM0�-
&�#
!�
inputs����������
� "�����������
D__inference_flatten_8_layer_call_and_return_conditional_losses_74136Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� z
)__inference_flatten_8_layer_call_fn_74130M0�-
&�#
!�
inputs����������
� "������������
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_74119�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
U__inference_global_average_pooling2d_8_layer_call_and_return_conditional_losses_74125b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
:__inference_global_average_pooling2d_8_layer_call_fn_74108wR�O
H�E
C�@
inputs4������������������������������������
� "!��������������������
:__inference_global_average_pooling2d_8_layer_call_fn_74113U8�5
.�+
)�&
inputs����������
� "������������
B__inference_model_8_layer_call_and_return_conditional_losses_73654y()./45>?LM@�=
6�3
)�&
input_9���������@ 
p 

 
� "%�"
�
0���������
� �
B__inference_model_8_layer_call_and_return_conditional_losses_73698y()./45>?LM@�=
6�3
)�&
input_9���������@ 
p

 
� "%�"
�
0���������
� �
B__inference_model_8_layer_call_and_return_conditional_losses_73864x()./45>?LM?�<
5�2
(�%
inputs���������@ 
p 

 
� "%�"
�
0���������
� �
B__inference_model_8_layer_call_and_return_conditional_losses_73923x()./45>?LM?�<
5�2
(�%
inputs���������@ 
p

 
� "%�"
�
0���������
� �
'__inference_model_8_layer_call_fn_73371l()./45>?LM@�=
6�3
)�&
input_9���������@ 
p 

 
� "�����������
'__inference_model_8_layer_call_fn_73610l()./45>?LM@�=
6�3
)�&
input_9���������@ 
p

 
� "�����������
'__inference_model_8_layer_call_fn_73772k()./45>?LM?�<
5�2
(�%
inputs���������@ 
p 

 
� "�����������
'__inference_model_8_layer_call_fn_73805k()./45>?LM?�<
5�2
(�%
inputs���������@ 
p

 
� "�����������
#__inference_signature_wrapper_73739�()./45>?LMC�@
� 
9�6
4
input_9)�&
input_9���������@ "1�.
,
dense_8!�
dense_8���������