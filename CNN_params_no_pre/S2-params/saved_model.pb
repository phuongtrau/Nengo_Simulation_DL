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
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
�
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�* 
shared_nameconv2d_7/kernel
|
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*'
_output_shapes
:@�*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:�*
dtype0
�
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_8/kernel
}
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_8/bias
l
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes	
:�*
dtype0
�
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��* 
shared_nameconv2d_9/kernel
}
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*(
_output_shapes
:��*
dtype0
s
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_9/bias
l
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes	
:�*
dtype0
�
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_10/kernel

$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_10/bias
n
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes	
:�*
dtype0
�
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*!
shared_nameconv2d_11/kernel

$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*(
_output_shapes
:��*
dtype0
u
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nameconv2d_11/bias
n
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes	
:�*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	�*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
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
Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/m
�
*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:@*
dtype0
�
Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:@*
dtype0
�
Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/conv2d_7/kernel/m
�
*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*'
_output_shapes
:@�*
dtype0
�
Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2d_7/bias/m
z
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/conv2d_8/kernel/m
�
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2d_8/bias/m
z
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/conv2d_9/kernel/m
�
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2d_9/bias/m
z
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_10/kernel/m
�
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_10/bias/m
|
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_11/kernel/m
�
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_11/bias/m
|
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/dense_1/kernel/m
�
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	�*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_6/kernel/v
�
*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:@*
dtype0
�
Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:@*
dtype0
�
Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*'
shared_nameAdam/conv2d_7/kernel/v
�
*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*'
_output_shapes
:@�*
dtype0
�
Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2d_7/bias/v
z
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/conv2d_8/kernel/v
�
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2d_8/bias/v
z
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*'
shared_nameAdam/conv2d_9/kernel/v
�
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/conv2d_9/bias/v
z
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_10/kernel/v
�
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_10/bias/v
|
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*(
shared_nameAdam/conv2d_11/kernel/v
�
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*(
_output_shapes
:��*
dtype0
�
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*&
shared_nameAdam/conv2d_11/bias/v
|
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*&
shared_nameAdam/dense_1/kernel/v
�
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	�*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
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
�
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem�m�m�m�(m�)m�.m�/m�4m�5m�>m�?m�Lm�Mm�v�v�v�v�(v�)v�.v�/v�4v�5v�>v�?v�Lv�Mv�
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
�
Wlayer_metrics
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
regularization_losses
	variables
trainable_variables
[metrics
 
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
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
�
alayer_metrics
blayer_regularization_losses
cnon_trainable_variables

dlayers
regularization_losses
	variables
trainable_variables
emetrics
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
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
�
klayer_metrics
llayer_regularization_losses
mnon_trainable_variables

nlayers
$regularization_losses
%	variables
&trainable_variables
ometrics
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
�
player_metrics
qlayer_regularization_losses
rnon_trainable_variables

slayers
*regularization_losses
+	variables
,trainable_variables
tmetrics
[Y
VARIABLE_VALUEconv2d_9/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_9/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
�
ulayer_metrics
vlayer_regularization_losses
wnon_trainable_variables

xlayers
0regularization_losses
1	variables
2trainable_variables
ymetrics
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
�
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
�
layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
:regularization_losses
;	variables
<trainable_variables
�metrics
\Z
VARIABLE_VALUEconv2d_11/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_11/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

>0
?1

>0
?1
�
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
@regularization_losses
A	variables
Btrainable_variables
�metrics
 
 
 
�
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Dregularization_losses
E	variables
Ftrainable_variables
�metrics
 
 
 
�
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Hregularization_losses
I	variables
Jtrainable_variables
�metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
�
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Nregularization_losses
O	variables
Ptrainable_variables
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
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_10/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_6/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_6/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_7/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_7/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_8/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_8/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_9/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_9/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_10/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_10/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_11/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_11/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_2Placeholder*/
_output_shapes
:���������@ *
dtype0*$
shape:���������@ 
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2conv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_1/kerneldense_1/bias*
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
#__inference_signature_wrapper_10163
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*@
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
__inference__traced_save_10755
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*?
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
!__inference__traced_restore_10918��	
�
i
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_9633

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
�
i
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_9524

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
)__inference_conv2d_10_layer_call_fn_10487

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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_97032
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
�
j
N__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_10497

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
�
B__inference_conv2d_8_layer_call_and_return_conditional_losses_9669

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
�

�
B__inference_dense_1_layer_call_and_return_conditional_losses_10570

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
�_
�
__inference__wrapped_model_9515
input_2I
/model_1_conv2d_6_conv2d_readvariableop_resource:@>
0model_1_conv2d_6_biasadd_readvariableop_resource:@J
/model_1_conv2d_7_conv2d_readvariableop_resource:@�?
0model_1_conv2d_7_biasadd_readvariableop_resource:	�K
/model_1_conv2d_8_conv2d_readvariableop_resource:��?
0model_1_conv2d_8_biasadd_readvariableop_resource:	�K
/model_1_conv2d_9_conv2d_readvariableop_resource:��?
0model_1_conv2d_9_biasadd_readvariableop_resource:	�L
0model_1_conv2d_10_conv2d_readvariableop_resource:��@
1model_1_conv2d_10_biasadd_readvariableop_resource:	�L
0model_1_conv2d_11_conv2d_readvariableop_resource:��@
1model_1_conv2d_11_biasadd_readvariableop_resource:	�A
.model_1_dense_1_matmul_readvariableop_resource:	�=
/model_1_dense_1_biasadd_readvariableop_resource:
identity��(model_1/conv2d_10/BiasAdd/ReadVariableOp�'model_1/conv2d_10/Conv2D/ReadVariableOp�(model_1/conv2d_11/BiasAdd/ReadVariableOp�'model_1/conv2d_11/Conv2D/ReadVariableOp�'model_1/conv2d_6/BiasAdd/ReadVariableOp�&model_1/conv2d_6/Conv2D/ReadVariableOp�'model_1/conv2d_7/BiasAdd/ReadVariableOp�&model_1/conv2d_7/Conv2D/ReadVariableOp�'model_1/conv2d_8/BiasAdd/ReadVariableOp�&model_1/conv2d_8/Conv2D/ReadVariableOp�'model_1/conv2d_9/BiasAdd/ReadVariableOp�&model_1/conv2d_9/Conv2D/ReadVariableOp�&model_1/dense_1/BiasAdd/ReadVariableOp�%model_1/dense_1/MatMul/ReadVariableOp�
&model_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02(
&model_1/conv2d_6/Conv2D/ReadVariableOp�
model_1/conv2d_6/Conv2DConv2Dinput_2.model_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
model_1/conv2d_6/Conv2D�
'model_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'model_1/conv2d_6/BiasAdd/ReadVariableOp�
model_1/conv2d_6/BiasAddBiasAdd model_1/conv2d_6/Conv2D:output:0/model_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2
model_1/conv2d_6/BiasAdd�
model_1/conv2d_6/ReluRelu!model_1/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
model_1/conv2d_6/Relu�
#model_1/average_pooling2d_3/AvgPoolAvgPool#model_1/conv2d_6/Relu:activations:0*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2%
#model_1/average_pooling2d_3/AvgPool�
&model_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02(
&model_1/conv2d_7/Conv2D/ReadVariableOp�
model_1/conv2d_7/Conv2DConv2D,model_1/average_pooling2d_3/AvgPool:output:0.model_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
model_1/conv2d_7/Conv2D�
'model_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_1/conv2d_7/BiasAdd/ReadVariableOp�
model_1/conv2d_7/BiasAddBiasAdd model_1/conv2d_7/Conv2D:output:0/model_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2
model_1/conv2d_7/BiasAdd�
model_1/conv2d_7/ReluRelu!model_1/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
model_1/conv2d_7/Relu�
#model_1/average_pooling2d_4/AvgPoolAvgPool#model_1/conv2d_7/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2%
#model_1/average_pooling2d_4/AvgPool�
&model_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&model_1/conv2d_8/Conv2D/ReadVariableOp�
model_1/conv2d_8/Conv2DConv2D,model_1/average_pooling2d_4/AvgPool:output:0.model_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_1/conv2d_8/Conv2D�
'model_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_1/conv2d_8/BiasAdd/ReadVariableOp�
model_1/conv2d_8/BiasAddBiasAdd model_1/conv2d_8/Conv2D:output:0/model_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_8/BiasAdd�
model_1/conv2d_8/ReluRelu!model_1/conv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_8/Relu�
&model_1/conv2d_9/Conv2D/ReadVariableOpReadVariableOp/model_1_conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02(
&model_1/conv2d_9/Conv2D/ReadVariableOp�
model_1/conv2d_9/Conv2DConv2D#model_1/conv2d_8/Relu:activations:0.model_1/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_1/conv2d_9/Conv2D�
'model_1/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp0model_1_conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02)
'model_1/conv2d_9/BiasAdd/ReadVariableOp�
model_1/conv2d_9/BiasAddBiasAdd model_1/conv2d_9/Conv2D:output:0/model_1/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_9/BiasAdd�
model_1/conv2d_9/ReluRelu!model_1/conv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_9/Relu�
'model_1/conv2d_10/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02)
'model_1/conv2d_10/Conv2D/ReadVariableOp�
model_1/conv2d_10/Conv2DConv2D#model_1/conv2d_9/Relu:activations:0/model_1/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_1/conv2d_10/Conv2D�
(model_1/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_1/conv2d_10/BiasAdd/ReadVariableOp�
model_1/conv2d_10/BiasAddBiasAdd!model_1/conv2d_10/Conv2D:output:00model_1/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_10/BiasAdd�
model_1/conv2d_10/ReluRelu"model_1/conv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_10/Relu�
#model_1/average_pooling2d_5/AvgPoolAvgPool$model_1/conv2d_10/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2%
#model_1/average_pooling2d_5/AvgPool�
'model_1/conv2d_11/Conv2D/ReadVariableOpReadVariableOp0model_1_conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02)
'model_1/conv2d_11/Conv2D/ReadVariableOp�
model_1/conv2d_11/Conv2DConv2D,model_1/average_pooling2d_5/AvgPool:output:0/model_1/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
model_1/conv2d_11/Conv2D�
(model_1/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp1model_1_conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02*
(model_1/conv2d_11/BiasAdd/ReadVariableOp�
model_1/conv2d_11/BiasAddBiasAdd!model_1/conv2d_11/Conv2D:output:00model_1/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_11/BiasAdd�
model_1/conv2d_11/ReluRelu"model_1/conv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
model_1/conv2d_11/Relu�
9model_1/global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2;
9model_1/global_average_pooling2d_1/Mean/reduction_indices�
'model_1/global_average_pooling2d_1/MeanMean$model_1/conv2d_11/Relu:activations:0Bmodel_1/global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2)
'model_1/global_average_pooling2d_1/Mean�
model_1/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
model_1/flatten_1/Const�
model_1/flatten_1/ReshapeReshape0model_1/global_average_pooling2d_1/Mean:output:0 model_1/flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
model_1/flatten_1/Reshape�
%model_1/dense_1/MatMul/ReadVariableOpReadVariableOp.model_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02'
%model_1/dense_1/MatMul/ReadVariableOp�
model_1/dense_1/MatMulMatMul"model_1/flatten_1/Reshape:output:0-model_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/dense_1/MatMul�
&model_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_1/BiasAdd/ReadVariableOp�
model_1/dense_1/BiasAddBiasAdd model_1/dense_1/MatMul:product:0.model_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/dense_1/BiasAdd{
IdentityIdentity model_1/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp)^model_1/conv2d_10/BiasAdd/ReadVariableOp(^model_1/conv2d_10/Conv2D/ReadVariableOp)^model_1/conv2d_11/BiasAdd/ReadVariableOp(^model_1/conv2d_11/Conv2D/ReadVariableOp(^model_1/conv2d_6/BiasAdd/ReadVariableOp'^model_1/conv2d_6/Conv2D/ReadVariableOp(^model_1/conv2d_7/BiasAdd/ReadVariableOp'^model_1/conv2d_7/Conv2D/ReadVariableOp(^model_1/conv2d_8/BiasAdd/ReadVariableOp'^model_1/conv2d_8/Conv2D/ReadVariableOp(^model_1/conv2d_9/BiasAdd/ReadVariableOp'^model_1/conv2d_9/Conv2D/ReadVariableOp'^model_1/dense_1/BiasAdd/ReadVariableOp&^model_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2T
(model_1/conv2d_10/BiasAdd/ReadVariableOp(model_1/conv2d_10/BiasAdd/ReadVariableOp2R
'model_1/conv2d_10/Conv2D/ReadVariableOp'model_1/conv2d_10/Conv2D/ReadVariableOp2T
(model_1/conv2d_11/BiasAdd/ReadVariableOp(model_1/conv2d_11/BiasAdd/ReadVariableOp2R
'model_1/conv2d_11/Conv2D/ReadVariableOp'model_1/conv2d_11/Conv2D/ReadVariableOp2R
'model_1/conv2d_6/BiasAdd/ReadVariableOp'model_1/conv2d_6/BiasAdd/ReadVariableOp2P
&model_1/conv2d_6/Conv2D/ReadVariableOp&model_1/conv2d_6/Conv2D/ReadVariableOp2R
'model_1/conv2d_7/BiasAdd/ReadVariableOp'model_1/conv2d_7/BiasAdd/ReadVariableOp2P
&model_1/conv2d_7/Conv2D/ReadVariableOp&model_1/conv2d_7/Conv2D/ReadVariableOp2R
'model_1/conv2d_8/BiasAdd/ReadVariableOp'model_1/conv2d_8/BiasAdd/ReadVariableOp2P
&model_1/conv2d_8/Conv2D/ReadVariableOp&model_1/conv2d_8/Conv2D/ReadVariableOp2R
'model_1/conv2d_9/BiasAdd/ReadVariableOp'model_1/conv2d_9/BiasAdd/ReadVariableOp2P
&model_1/conv2d_9/Conv2D/ReadVariableOp&model_1/conv2d_9/Conv2D/ReadVariableOp2P
&model_1/dense_1/BiasAdd/ReadVariableOp&model_1/dense_1/BiasAdd/ReadVariableOp2N
%model_1/dense_1/MatMul/ReadVariableOp%model_1/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_2
�
V
:__inference_global_average_pooling2d_1_layer_call_fn_10544

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
GPU2*0,1J 8� *]
fXRV
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_95912
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
�R
�
B__inference_model_1_layer_call_and_return_conditional_losses_10281

inputsA
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@B
'conv2d_7_conv2d_readvariableop_resource:@�7
(conv2d_7_biasadd_readvariableop_resource:	�C
'conv2d_8_conv2d_readvariableop_resource:��7
(conv2d_8_biasadd_readvariableop_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��7
(conv2d_9_biasadd_readvariableop_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�5
'dense_1_biasadd_readvariableop_resource:
identity�� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
conv2d_6/Conv2D�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
conv2d_6/Relu�
average_pooling2d_3/AvgPoolAvgPoolconv2d_6/Relu:activations:0*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_3/AvgPool�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2DConv2D$average_pooling2d_3/AvgPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
conv2d_7/Conv2D�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2
conv2d_7/BiasAdd|
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
conv2d_7/Relu�
average_pooling2d_4/AvgPoolAvgPoolconv2d_7/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_4/AvgPool�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2DConv2D$average_pooling2d_4/AvgPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_8/Conv2D�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_8/Relu�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_9/BiasAdd|
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_9/Relu�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_10/Relu�
average_pooling2d_5/AvgPoolAvgPoolconv2d_10/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_5/AvgPool�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D$average_pooling2d_5/AvgPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_11/Relu�
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indices�
global_average_pooling2d_1/MeanMeanconv2d_11/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2!
global_average_pooling2d_1/Means
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_1/Const�
flatten_1/ReshapeReshape(global_average_pooling2d_1/Mean:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
�
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10358

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
�8
�
A__inference_model_1_layer_call_and_return_conditional_losses_9970

inputs'
conv2d_6_9929:@
conv2d_6_9931:@(
conv2d_7_9935:@�
conv2d_7_9937:	�)
conv2d_8_9941:��
conv2d_8_9943:	�)
conv2d_9_9946:��
conv2d_9_9948:	�*
conv2d_10_9951:��
conv2d_10_9953:	�*
conv2d_11_9957:��
conv2d_11_9959:	�
dense_1_9964:	�
dense_1_9966:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_9929conv2d_6_9931*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_96232"
 conv2d_6/StatefulPartitionedCall�
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_96332%
#average_pooling2d_3/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0conv2d_7_9935conv2d_7_9937*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_96462"
 conv2d_7/StatefulPartitionedCall�
#average_pooling2d_4/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_96562%
#average_pooling2d_4/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_4/PartitionedCall:output:0conv2d_8_9941conv2d_8_9943*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_96692"
 conv2d_8/StatefulPartitionedCall�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_9946conv2d_9_9948*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_96862"
 conv2d_9/StatefulPartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_9951conv2d_10_9953*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_97032#
!conv2d_10/StatefulPartitionedCall�
#average_pooling2d_5/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_97132%
#average_pooling2d_5/PartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_5/PartitionedCall:output:0conv2d_11_9957conv2d_11_9959*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_97262#
!conv2d_11/StatefulPartitionedCall�
*global_average_pooling2d_1/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *]
fXRV
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_97372,
*global_average_pooling2d_1/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
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
GPU2*0,1J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_97452
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_9964dense_1_9966*
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
GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_97572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
j
N__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_10412

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
�
V
:__inference_global_average_pooling2d_1_layer_call_fn_10549

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
GPU2*0,1J 8� *]
fXRV
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_97372
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
��
� 
!__inference__traced_restore_10918
file_prefix:
 assignvariableop_conv2d_6_kernel:@.
 assignvariableop_1_conv2d_6_bias:@=
"assignvariableop_2_conv2d_7_kernel:@�/
 assignvariableop_3_conv2d_7_bias:	�>
"assignvariableop_4_conv2d_8_kernel:��/
 assignvariableop_5_conv2d_8_bias:	�>
"assignvariableop_6_conv2d_9_kernel:��/
 assignvariableop_7_conv2d_9_bias:	�?
#assignvariableop_8_conv2d_10_kernel:��0
!assignvariableop_9_conv2d_10_bias:	�@
$assignvariableop_10_conv2d_11_kernel:��1
"assignvariableop_11_conv2d_11_bias:	�5
"assignvariableop_12_dense_1_kernel:	�.
 assignvariableop_13_dense_1_bias:'
assignvariableop_14_adam_iter:	 )
assignvariableop_15_adam_beta_1: )
assignvariableop_16_adam_beta_2: (
assignvariableop_17_adam_decay: 0
&assignvariableop_18_adam_learning_rate: #
assignvariableop_19_total: #
assignvariableop_20_count: %
assignvariableop_21_total_1: %
assignvariableop_22_count_1: D
*assignvariableop_23_adam_conv2d_6_kernel_m:@6
(assignvariableop_24_adam_conv2d_6_bias_m:@E
*assignvariableop_25_adam_conv2d_7_kernel_m:@�7
(assignvariableop_26_adam_conv2d_7_bias_m:	�F
*assignvariableop_27_adam_conv2d_8_kernel_m:��7
(assignvariableop_28_adam_conv2d_8_bias_m:	�F
*assignvariableop_29_adam_conv2d_9_kernel_m:��7
(assignvariableop_30_adam_conv2d_9_bias_m:	�G
+assignvariableop_31_adam_conv2d_10_kernel_m:��8
)assignvariableop_32_adam_conv2d_10_bias_m:	�G
+assignvariableop_33_adam_conv2d_11_kernel_m:��8
)assignvariableop_34_adam_conv2d_11_bias_m:	�<
)assignvariableop_35_adam_dense_1_kernel_m:	�5
'assignvariableop_36_adam_dense_1_bias_m:D
*assignvariableop_37_adam_conv2d_6_kernel_v:@6
(assignvariableop_38_adam_conv2d_6_bias_v:@E
*assignvariableop_39_adam_conv2d_7_kernel_v:@�7
(assignvariableop_40_adam_conv2d_7_bias_v:	�F
*assignvariableop_41_adam_conv2d_8_kernel_v:��7
(assignvariableop_42_adam_conv2d_8_bias_v:	�F
*assignvariableop_43_adam_conv2d_9_kernel_v:��7
(assignvariableop_44_adam_conv2d_9_bias_v:	�G
+assignvariableop_45_adam_conv2d_10_kernel_v:��8
)assignvariableop_46_adam_conv2d_10_bias_v:	�G
+assignvariableop_47_adam_conv2d_11_kernel_v:��8
)assignvariableop_48_adam_conv2d_11_bias_v:	�<
)assignvariableop_49_adam_dense_1_kernel_v:	�5
'assignvariableop_50_adam_dense_1_bias_v:
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
AssignVariableOpAssignVariableOp assignvariableop_conv2d_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_6_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_7_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_7_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_8_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_8_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_9_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_9_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_10_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_10_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_11_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_11_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_1_biasIdentity_13:output:0"/device:CPU:0*
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
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_6_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_6_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_7_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_7_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_conv2d_8_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_8_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_9_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_9_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_10_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_10_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_11_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_11_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_6_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_6_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_7_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_7_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_8_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_8_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_9_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_9_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_10_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_10_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_11_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_11_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_adam_dense_1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_adam_dense_1_bias_vIdentity_50:output:0"/device:CPU:0*
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
�
O
3__inference_average_pooling2d_4_layer_call_fn_10427

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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_96562
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
�
j
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_10377

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
�
�
#__inference_signature_wrapper_10163
input_2!
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
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0,1J 8� *(
f#R!
__inference__wrapped_model_95152
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
_user_specified_name	input_2
�
�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10398

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
�
q
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_10539

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
_
C__inference_flatten_1_layer_call_and_return_conditional_losses_9745

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
�
D__inference_conv2d_10_layer_call_and_return_conditional_losses_10478

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
E
)__inference_flatten_1_layer_call_fn_10560

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
GPU2*0,1J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_97452
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
�
�
'__inference_model_1_layer_call_fn_10347

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
GPU2*0,1J 8� *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_99702
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
�
O
3__inference_average_pooling2d_3_layer_call_fn_10387

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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_96332
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
(__inference_conv2d_8_layer_call_fn_10447

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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_96692
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
�
j
N__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_10417

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
�j
�
__inference__traced_save_10755
file_prefix.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
�
i
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_9568

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
�R
�
B__inference_model_1_layer_call_and_return_conditional_losses_10222

inputsA
'conv2d_6_conv2d_readvariableop_resource:@6
(conv2d_6_biasadd_readvariableop_resource:@B
'conv2d_7_conv2d_readvariableop_resource:@�7
(conv2d_7_biasadd_readvariableop_resource:	�C
'conv2d_8_conv2d_readvariableop_resource:��7
(conv2d_8_biasadd_readvariableop_resource:	�C
'conv2d_9_conv2d_readvariableop_resource:��7
(conv2d_9_biasadd_readvariableop_resource:	�D
(conv2d_10_conv2d_readvariableop_resource:��8
)conv2d_10_biasadd_readvariableop_resource:	�D
(conv2d_11_conv2d_readvariableop_resource:��8
)conv2d_11_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�5
'dense_1_biasadd_readvariableop_resource:
identity�� conv2d_10/BiasAdd/ReadVariableOp�conv2d_10/Conv2D/ReadVariableOp� conv2d_11/BiasAdd/ReadVariableOp�conv2d_11/Conv2D/ReadVariableOp�conv2d_6/BiasAdd/ReadVariableOp�conv2d_6/Conv2D/ReadVariableOp�conv2d_7/BiasAdd/ReadVariableOp�conv2d_7/Conv2D/ReadVariableOp�conv2d_8/BiasAdd/ReadVariableOp�conv2d_8/Conv2D/ReadVariableOp�conv2d_9/BiasAdd/ReadVariableOp�conv2d_9/Conv2D/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp�
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @*
paddingSAME*
strides
2
conv2d_6/Conv2D�
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp�
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@ @2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:���������@ @2
conv2d_6/Relu�
average_pooling2d_3/AvgPoolAvgPoolconv2d_6/Relu:activations:0*
T0*/
_output_shapes
:��������� @*
ksize
*
paddingVALID*
strides
2
average_pooling2d_3/AvgPool�
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype02 
conv2d_7/Conv2D/ReadVariableOp�
conv2d_7/Conv2DConv2D$average_pooling2d_3/AvgPool:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �*
paddingSAME*
strides
2
conv2d_7/Conv2D�
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp�
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:��������� �2
conv2d_7/BiasAdd|
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:��������� �2
conv2d_7/Relu�
average_pooling2d_4/AvgPoolAvgPoolconv2d_7/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_4/AvgPool�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2DConv2D$average_pooling2d_4/AvgPool:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_8/Conv2D�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_8/BiasAdd|
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_8/Relu�
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02 
conv2d_9/Conv2D/ReadVariableOp�
conv2d_9/Conv2DConv2Dconv2d_8/Relu:activations:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_9/Conv2D�
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp�
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_9/BiasAdd|
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_9/Relu�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2Dconv2d_9/Relu:activations:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_10/BiasAdd
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_10/Relu�
average_pooling2d_5/AvgPoolAvgPoolconv2d_10/Relu:activations:0*
T0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
2
average_pooling2d_5/AvgPool�
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype02!
conv2d_11/Conv2D/ReadVariableOp�
conv2d_11/Conv2DConv2D$average_pooling2d_5/AvgPool:output:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
2
conv2d_11/Conv2D�
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp�
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������2
conv2d_11/BiasAdd
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*0
_output_shapes
:����������2
conv2d_11/Relu�
1global_average_pooling2d_1/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      23
1global_average_pooling2d_1/Mean/reduction_indices�
global_average_pooling2d_1/MeanMeanconv2d_11/Relu:activations:0:global_average_pooling2d_1/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:����������2!
global_average_pooling2d_1/Means
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   2
flatten_1/Const�
flatten_1/ReshapeReshape(global_average_pooling2d_1/Mean:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:����������2
flatten_1/Reshape�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_1/BiasAdds
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�
O
3__inference_average_pooling2d_4_layer_call_fn_10422

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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_95462
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_10458

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
�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10438

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
j
N__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_10492

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
(__inference_conv2d_6_layer_call_fn_10367

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_96232
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
O
3__inference_average_pooling2d_5_layer_call_fn_10507

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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_97132
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
�8
�
B__inference_model_1_layer_call_and_return_conditional_losses_10078
input_2(
conv2d_6_10037:@
conv2d_6_10039:@)
conv2d_7_10043:@�
conv2d_7_10045:	�*
conv2d_8_10049:��
conv2d_8_10051:	�*
conv2d_9_10054:��
conv2d_9_10056:	�+
conv2d_10_10059:��
conv2d_10_10061:	�+
conv2d_11_10065:��
conv2d_11_10067:	� 
dense_1_10072:	�
dense_1_10074:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_6_10037conv2d_6_10039*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_96232"
 conv2d_6/StatefulPartitionedCall�
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_96332%
#average_pooling2d_3/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0conv2d_7_10043conv2d_7_10045*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_96462"
 conv2d_7/StatefulPartitionedCall�
#average_pooling2d_4/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_96562%
#average_pooling2d_4/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_4/PartitionedCall:output:0conv2d_8_10049conv2d_8_10051*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_96692"
 conv2d_8/StatefulPartitionedCall�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_10054conv2d_9_10056*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_96862"
 conv2d_9/StatefulPartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_10059conv2d_10_10061*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_97032#
!conv2d_10/StatefulPartitionedCall�
#average_pooling2d_5/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_97132%
#average_pooling2d_5/PartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_5/PartitionedCall:output:0conv2d_11_10065conv2d_11_10067*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_97262#
!conv2d_11/StatefulPartitionedCall�
*global_average_pooling2d_1/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *]
fXRV
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_97372,
*global_average_pooling2d_1/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
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
GPU2*0,1J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_97452
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_10072dense_1_10074*
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
GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_97572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_2
�
q
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_10533

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
i
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_9656

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
�
�
B__inference_conv2d_9_layer_call_and_return_conditional_losses_9686

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
i
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_9546

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
C__inference_conv2d_10_layer_call_and_return_conditional_losses_9703

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
�

�
A__inference_dense_1_layer_call_and_return_conditional_losses_9757

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
�
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_10555

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
�
B__inference_conv2d_6_layer_call_and_return_conditional_losses_9623

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
�
�
C__inference_conv2d_11_layer_call_and_return_conditional_losses_9726

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
�
�
'__inference_dense_1_layer_call_fn_10579

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
GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_97572
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
�
�
'__inference_model_1_layer_call_fn_10314

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
GPU2*0,1J 8� *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_97642
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
�
O
3__inference_average_pooling2d_5_layer_call_fn_10502

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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_95682
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
i
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_9713

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
�
�
(__inference_conv2d_7_layer_call_fn_10407

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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_96462
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
�8
�
A__inference_model_1_layer_call_and_return_conditional_losses_9764

inputs'
conv2d_6_9624:@
conv2d_6_9626:@(
conv2d_7_9647:@�
conv2d_7_9649:	�)
conv2d_8_9670:��
conv2d_8_9672:	�)
conv2d_9_9687:��
conv2d_9_9689:	�*
conv2d_10_9704:��
conv2d_10_9706:	�*
conv2d_11_9727:��
conv2d_11_9729:	�
dense_1_9758:	�
dense_1_9760:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_6_9624conv2d_6_9626*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_96232"
 conv2d_6/StatefulPartitionedCall�
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_96332%
#average_pooling2d_3/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0conv2d_7_9647conv2d_7_9649*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_96462"
 conv2d_7/StatefulPartitionedCall�
#average_pooling2d_4/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_96562%
#average_pooling2d_4/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_4/PartitionedCall:output:0conv2d_8_9670conv2d_8_9672*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_96692"
 conv2d_8/StatefulPartitionedCall�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_9687conv2d_9_9689*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_96862"
 conv2d_9/StatefulPartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_9704conv2d_10_9706*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_97032#
!conv2d_10/StatefulPartitionedCall�
#average_pooling2d_5/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_97132%
#average_pooling2d_5/PartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_5/PartitionedCall:output:0conv2d_11_9727conv2d_11_9729*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_97262#
!conv2d_11/StatefulPartitionedCall�
*global_average_pooling2d_1/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *]
fXRV
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_97372,
*global_average_pooling2d_1/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
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
GPU2*0,1J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_97452
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_9758dense_1_9760*
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
GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_97572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������@ 
 
_user_specified_nameinputs
�8
�
B__inference_model_1_layer_call_and_return_conditional_losses_10122
input_2(
conv2d_6_10081:@
conv2d_6_10083:@)
conv2d_7_10087:@�
conv2d_7_10089:	�*
conv2d_8_10093:��
conv2d_8_10095:	�*
conv2d_9_10098:��
conv2d_9_10100:	�+
conv2d_10_10103:��
conv2d_10_10105:	�+
conv2d_11_10109:��
conv2d_11_10111:	� 
dense_1_10116:	�
dense_1_10118:
identity��!conv2d_10/StatefulPartitionedCall�!conv2d_11/StatefulPartitionedCall� conv2d_6/StatefulPartitionedCall� conv2d_7/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall� conv2d_9/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_6_10081conv2d_6_10083*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_96232"
 conv2d_6/StatefulPartitionedCall�
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_96332%
#average_pooling2d_3/PartitionedCall�
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0conv2d_7_10087conv2d_7_10089*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_96462"
 conv2d_7/StatefulPartitionedCall�
#average_pooling2d_4/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_96562%
#average_pooling2d_4/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_4/PartitionedCall:output:0conv2d_8_10093conv2d_8_10095*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_8_layer_call_and_return_conditional_losses_96692"
 conv2d_8/StatefulPartitionedCall�
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0conv2d_9_10098conv2d_9_10100*
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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_96862"
 conv2d_9/StatefulPartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0conv2d_10_10103conv2d_10_10105*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_10_layer_call_and_return_conditional_losses_97032#
!conv2d_10/StatefulPartitionedCall�
#average_pooling2d_5/PartitionedCallPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_97132%
#average_pooling2d_5/PartitionedCall�
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_5/PartitionedCall:output:0conv2d_11_10109conv2d_11_10111*
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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_97262#
!conv2d_11/StatefulPartitionedCall�
*global_average_pooling2d_1/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8� *]
fXRV
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_97372,
*global_average_pooling2d_1/PartitionedCall�
flatten_1/PartitionedCallPartitionedCall3global_average_pooling2d_1/PartitionedCall:output:0*
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
GPU2*0,1J 8� *L
fGRE
C__inference_flatten_1_layer_call_and_return_conditional_losses_97452
flatten_1/PartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_10116dense_1_10118*
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
GPU2*0,1J 8� *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_97572!
dense_1/StatefulPartitionedCall�
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:���������@ : : : : : : : : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������@ 
!
_user_specified_name	input_2
�
�
(__inference_conv2d_9_layer_call_fn_10467

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
GPU2*0,1J 8� *K
fFRD
B__inference_conv2d_9_layer_call_and_return_conditional_losses_96862
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
�
p
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_9737

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
j
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_10372

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
�
O
3__inference_average_pooling2d_3_layer_call_fn_10382

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
GPU2*0,1J 8� *V
fQRO
M__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_95242
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
B__inference_conv2d_7_layer_call_and_return_conditional_losses_9646

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
�
p
T__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_9591

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
�
�
'__inference_model_1_layer_call_fn_10034
input_2!
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
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0,1J 8� *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_99702
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
_user_specified_name	input_2
�
�
&__inference_model_1_layer_call_fn_9795
input_2!
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
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0,1J 8� *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_97642
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
_user_specified_name	input_2
�
�
D__inference_conv2d_11_layer_call_and_return_conditional_losses_10518

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
�
�
)__inference_conv2d_11_layer_call_fn_10527

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
GPU2*0,1J 8� *L
fGRE
C__inference_conv2d_11_layer_call_and_return_conditional_losses_97262
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
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_28
serving_default_input_2:0���������@ ;
dense_10
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
�_default_save_signature
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
regularization_losses
	variables
trainable_variables
	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
:regularization_losses
;	variables
<trainable_variables
=	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

>kernel
?bias
@regularization_losses
A	variables
Btrainable_variables
C	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+�&call_and_return_all_conditional_losses
�__call__"
_tf_keras_layer
�
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem�m�m�m�(m�)m�.m�/m�4m�5m�>m�?m�Lm�Mm�v�v�v�v�(v�)v�.v�/v�4v�5v�>v�?v�Lv�Mv�"
	optimizer
 "
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
�
Wlayer_metrics
Xnon_trainable_variables
Ylayer_regularization_losses

Zlayers
regularization_losses
	variables
trainable_variables
[metrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
):'@2conv2d_6/kernel
:@2conv2d_6/bias
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
�
\layer_metrics
]layer_regularization_losses
^non_trainable_variables

_layers
regularization_losses
	variables
trainable_variables
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
alayer_metrics
blayer_regularization_losses
cnon_trainable_variables

dlayers
regularization_losses
	variables
trainable_variables
emetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@�2conv2d_7/kernel
:�2conv2d_7/bias
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
�
flayer_metrics
glayer_regularization_losses
hnon_trainable_variables

ilayers
 regularization_losses
!	variables
"trainable_variables
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
klayer_metrics
llayer_regularization_losses
mnon_trainable_variables

nlayers
$regularization_losses
%	variables
&trainable_variables
ometrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2conv2d_8/kernel
:�2conv2d_8/bias
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
�
player_metrics
qlayer_regularization_losses
rnon_trainable_variables

slayers
*regularization_losses
+	variables
,trainable_variables
tmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)��2conv2d_9/kernel
:�2conv2d_9/bias
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
�
ulayer_metrics
vlayer_regularization_losses
wnon_trainable_variables

xlayers
0regularization_losses
1	variables
2trainable_variables
ymetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_10/kernel
:�2conv2d_10/bias
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
�
zlayer_metrics
{layer_regularization_losses
|non_trainable_variables

}layers
6regularization_losses
7	variables
8trainable_variables
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
layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
:regularization_losses
;	variables
<trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*��2conv2d_11/kernel
:�2conv2d_11/bias
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
�
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
@regularization_losses
A	variables
Btrainable_variables
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
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Dregularization_losses
E	variables
Ftrainable_variables
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
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Hregularization_losses
I	variables
Jtrainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
!:	�2dense_1/kernel
:2dense_1/bias
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
�
�layer_metrics
 �layer_regularization_losses
�non_trainable_variables
�layers
Nregularization_losses
O	variables
Ptrainable_variables
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
.:,@2Adam/conv2d_6/kernel/m
 :@2Adam/conv2d_6/bias/m
/:-@�2Adam/conv2d_7/kernel/m
!:�2Adam/conv2d_7/bias/m
0:.��2Adam/conv2d_8/kernel/m
!:�2Adam/conv2d_8/bias/m
0:.��2Adam/conv2d_9/kernel/m
!:�2Adam/conv2d_9/bias/m
1:/��2Adam/conv2d_10/kernel/m
": �2Adam/conv2d_10/bias/m
1:/��2Adam/conv2d_11/kernel/m
": �2Adam/conv2d_11/bias/m
&:$	�2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
.:,@2Adam/conv2d_6/kernel/v
 :@2Adam/conv2d_6/bias/v
/:-@�2Adam/conv2d_7/kernel/v
!:�2Adam/conv2d_7/bias/v
0:.��2Adam/conv2d_8/kernel/v
!:�2Adam/conv2d_8/bias/v
0:.��2Adam/conv2d_9/kernel/v
!:�2Adam/conv2d_9/bias/v
1:/��2Adam/conv2d_10/kernel/v
": �2Adam/conv2d_10/bias/v
1:/��2Adam/conv2d_11/kernel/v
": �2Adam/conv2d_11/bias/v
&:$	�2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
�B�
__inference__wrapped_model_9515input_2"�
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
B__inference_model_1_layer_call_and_return_conditional_losses_10222
B__inference_model_1_layer_call_and_return_conditional_losses_10281
B__inference_model_1_layer_call_and_return_conditional_losses_10078
B__inference_model_1_layer_call_and_return_conditional_losses_10122�
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
�2�
&__inference_model_1_layer_call_fn_9795
'__inference_model_1_layer_call_fn_10314
'__inference_model_1_layer_call_fn_10347
'__inference_model_1_layer_call_fn_10034�
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
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10358�
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
(__inference_conv2d_6_layer_call_fn_10367�
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
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_10372
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_10377�
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
3__inference_average_pooling2d_3_layer_call_fn_10382
3__inference_average_pooling2d_3_layer_call_fn_10387�
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
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10398�
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
(__inference_conv2d_7_layer_call_fn_10407�
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
N__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_10412
N__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_10417�
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
3__inference_average_pooling2d_4_layer_call_fn_10422
3__inference_average_pooling2d_4_layer_call_fn_10427�
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10438�
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
(__inference_conv2d_8_layer_call_fn_10447�
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
C__inference_conv2d_9_layer_call_and_return_conditional_losses_10458�
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
(__inference_conv2d_9_layer_call_fn_10467�
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
D__inference_conv2d_10_layer_call_and_return_conditional_losses_10478�
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
)__inference_conv2d_10_layer_call_fn_10487�
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
N__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_10492
N__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_10497�
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
3__inference_average_pooling2d_5_layer_call_fn_10502
3__inference_average_pooling2d_5_layer_call_fn_10507�
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
D__inference_conv2d_11_layer_call_and_return_conditional_losses_10518�
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
)__inference_conv2d_11_layer_call_fn_10527�
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
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_10533
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_10539�
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
:__inference_global_average_pooling2d_1_layer_call_fn_10544
:__inference_global_average_pooling2d_1_layer_call_fn_10549�
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
D__inference_flatten_1_layer_call_and_return_conditional_losses_10555�
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
)__inference_flatten_1_layer_call_fn_10560�
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
B__inference_dense_1_layer_call_and_return_conditional_losses_10570�
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
'__inference_dense_1_layer_call_fn_10579�
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
#__inference_signature_wrapper_10163input_2"�
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
__inference__wrapped_model_9515}()./45>?LM8�5
.�+
)�&
input_2���������@ 
� "1�.
,
dense_1!�
dense_1����������
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_10372�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_10377h7�4
-�*
(�%
inputs���������@ @
� "-�*
#� 
0��������� @
� �
3__inference_average_pooling2d_3_layer_call_fn_10382�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
3__inference_average_pooling2d_3_layer_call_fn_10387[7�4
-�*
(�%
inputs���������@ @
� " ���������� @�
N__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_10412�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
N__inference_average_pooling2d_4_layer_call_and_return_conditional_losses_10417j8�5
.�+
)�&
inputs��������� �
� ".�+
$�!
0����������
� �
3__inference_average_pooling2d_4_layer_call_fn_10422�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
3__inference_average_pooling2d_4_layer_call_fn_10427]8�5
.�+
)�&
inputs��������� �
� "!������������
N__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_10492�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
N__inference_average_pooling2d_5_layer_call_and_return_conditional_losses_10497j8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
3__inference_average_pooling2d_5_layer_call_fn_10502�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
3__inference_average_pooling2d_5_layer_call_fn_10507]8�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_10_layer_call_and_return_conditional_losses_10478n458�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_10_layer_call_fn_10487a458�5
.�+
)�&
inputs����������
� "!������������
D__inference_conv2d_11_layer_call_and_return_conditional_losses_10518n>?8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
)__inference_conv2d_11_layer_call_fn_10527a>?8�5
.�+
)�&
inputs����������
� "!������������
C__inference_conv2d_6_layer_call_and_return_conditional_losses_10358l7�4
-�*
(�%
inputs���������@ 
� "-�*
#� 
0���������@ @
� �
(__inference_conv2d_6_layer_call_fn_10367_7�4
-�*
(�%
inputs���������@ 
� " ����������@ @�
C__inference_conv2d_7_layer_call_and_return_conditional_losses_10398m7�4
-�*
(�%
inputs��������� @
� ".�+
$�!
0��������� �
� �
(__inference_conv2d_7_layer_call_fn_10407`7�4
-�*
(�%
inputs��������� @
� "!���������� ��
C__inference_conv2d_8_layer_call_and_return_conditional_losses_10438n()8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
(__inference_conv2d_8_layer_call_fn_10447a()8�5
.�+
)�&
inputs����������
� "!������������
C__inference_conv2d_9_layer_call_and_return_conditional_losses_10458n./8�5
.�+
)�&
inputs����������
� ".�+
$�!
0����������
� �
(__inference_conv2d_9_layer_call_fn_10467a./8�5
.�+
)�&
inputs����������
� "!������������
B__inference_dense_1_layer_call_and_return_conditional_losses_10570]LM0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� {
'__inference_dense_1_layer_call_fn_10579PLM0�-
&�#
!�
inputs����������
� "�����������
D__inference_flatten_1_layer_call_and_return_conditional_losses_10555Z0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� z
)__inference_flatten_1_layer_call_fn_10560M0�-
&�#
!�
inputs����������
� "������������
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_10533�R�O
H�E
C�@
inputs4������������������������������������
� ".�+
$�!
0������������������
� �
U__inference_global_average_pooling2d_1_layer_call_and_return_conditional_losses_10539b8�5
.�+
)�&
inputs����������
� "&�#
�
0����������
� �
:__inference_global_average_pooling2d_1_layer_call_fn_10544wR�O
H�E
C�@
inputs4������������������������������������
� "!��������������������
:__inference_global_average_pooling2d_1_layer_call_fn_10549U8�5
.�+
)�&
inputs����������
� "������������
B__inference_model_1_layer_call_and_return_conditional_losses_10078y()./45>?LM@�=
6�3
)�&
input_2���������@ 
p 

 
� "%�"
�
0���������
� �
B__inference_model_1_layer_call_and_return_conditional_losses_10122y()./45>?LM@�=
6�3
)�&
input_2���������@ 
p

 
� "%�"
�
0���������
� �
B__inference_model_1_layer_call_and_return_conditional_losses_10222x()./45>?LM?�<
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
B__inference_model_1_layer_call_and_return_conditional_losses_10281x()./45>?LM?�<
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
'__inference_model_1_layer_call_fn_10034l()./45>?LM@�=
6�3
)�&
input_2���������@ 
p

 
� "�����������
'__inference_model_1_layer_call_fn_10314k()./45>?LM?�<
5�2
(�%
inputs���������@ 
p 

 
� "�����������
'__inference_model_1_layer_call_fn_10347k()./45>?LM?�<
5�2
(�%
inputs���������@ 
p

 
� "�����������
&__inference_model_1_layer_call_fn_9795l()./45>?LM@�=
6�3
)�&
input_2���������@ 
p 

 
� "�����������
#__inference_signature_wrapper_10163�()./45>?LMC�@
� 
9�6
4
input_2)�&
input_2���������@ "1�.
,
dense_1!�
dense_1���������