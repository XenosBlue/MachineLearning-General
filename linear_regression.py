import numpy
import pandas



x=pandas.read_csv("car_price.csv",header=0,usecols=[1,3,4,5,6,7,8,9,10,12],converters={i: str for i in range(13)})
#print(x)

####_____________________________________________________DATA_PREPROCESSING

####_________________________MAX_POWER
temp=[]
zeros=0
for each in x['max_power']:
  if type(each)==str:
    number=each.split()
    if len(number)>1:
      temp.append(float(number[0]))
    else:
      temp.append(0)
      zeros+=1
  else:
    temp.append(0)
    zeros=1
avg=numpy.sum(temp)/(len(temp)-zeros)
maxx=numpy.max(temp)
minn=numpy.min(temp)
for i in range(len(temp)):
  if temp[i]==0:
    temp[i]=(avg-minn)/(maxx-minn)
  else:
    temp[i]=(temp[i]-minn)/(maxx-minn)
x['max_power']=temp


####___________________________mileage
temp=[]
zeros=0
for each in x['mileage']:
  if type(each)==str:
    number=each.split()
    if len(number)>1:
      temp.append(float(number[0]))
    else:
      temp.append(0)
      zeros+=1
  else:
    temp.append(0)
    zeros=1
avg=numpy.sum(temp)/(len(temp)-zeros)
maxx=numpy.max(temp)
minn=numpy.min(temp)
for i in range(len(temp)):
  if temp[i]==0:
    temp[i]=(avg-minn)/(maxx-minn)
  else:
    temp[i]=(temp[i]-minn)/(maxx-minn)
x['mileage']=temp


####________________________engine
temp=[]
zeros=0
for each in x['engine']:
  if type(each)==str:
    number=each.split()
    if len(number)>1:
      temp.append(float(number[0]))
    else:
      temp.append(0)
      zeros+=1
  else:
    temp.append(0)
    zeros=1
avg=numpy.sum(temp)/(len(temp)-zeros)
maxx=numpy.max(temp)
minn=numpy.min(temp)
for i in range(len(temp)):
  if temp[i]==0:
    temp[i]=(avg-minn)/(maxx-minn)
  else:
    temp[i]=(temp[i]-minn)/(maxx-minn)
x['engine']=temp


####________________________YEAR
temp=[]
zeros=0
for each in x['year']:
  if len(str(each))!=0:
      temp.append(float(each))
  else:
    temp.append(0)
    zeros=1
avg=numpy.sum(temp)/(len(temp)-zeros)
maxx=numpy.max(temp)
minn=numpy.min(temp)
for i in range(len(temp)):
  if temp[i]==0:
    temp[i]=(avg-minn)/(maxx-minn)
  else:
    temp[i]=(temp[i]-minn)/(maxx-minn)
x['year']=temp

###_________________________km_driven
temp=[]
zeros=0
for each in x['km_driven']:
  if len(str(each))!=0:
      temp.append(float(each))
  else:
    temp.append(0)
    zeros=1
avg=numpy.sum(temp)/(len(temp)-zeros)
maxx=numpy.max(temp)
minn=numpy.min(temp)
for i in range(len(temp)):
  if temp[i]==0:
    temp[i]=(avg-minn)/(maxx-minn)
  else:
    temp[i]=(temp[i]-minn)/(maxx-minn)
x['km_driven']=temp


##_________________________seats
temp=[]
zeros=0
for each in x['seats']:
  if len(str(each))!=0:
      temp.append(float(each))
  else:
    temp.append(0)
    zeros=1
avg=numpy.sum(temp)/(len(temp)-zeros)
maxx=numpy.max(temp)
minn=numpy.min(temp)
for i in range(len(temp)):
  if temp[i]==0:
    temp[i]=(avg-minn)/(maxx-minn)
  else:
    temp[i]=(temp[i]-minn)/(maxx-minn)
x['seats']=temp
#x

####_____________________________________________fuel
types=numpy.unique(x['fuel'])
keys=numpy.linspace(0,1,len(types))
print(types)
print(keys)
temp=[]
for each in x['fuel']:
  for i,every in enumerate(types):
    if every == each:
      temp.append(keys[i])
x['fuel']=temp


####_____________________________________________seller_type
types=numpy.unique(x['seller_type'])
print(types)
keys=numpy.linspace(0,1,len(types))
print(keys)
temp=[]
for each in x['seller_type']:
  for i,every in enumerate(types):
    if every == each:
      temp.append(keys[i])
x['seller_type']=temp



####_____________________________________________transmission
types=numpy.unique(x['transmission'])
print(types)
keys=numpy.linspace(0,1,len(types))
print(keys)
temp=[]
for each in x['transmission']:
  for i,every in enumerate(types):
    if every == each:
      temp.append(keys[i])
x['transmission']=temp

###________________________________________________owner
types=numpy.unique(x['owner'])
print(types)
keys=numpy.linspace(0,1,len(types))
print(keys)
temp=[]
for each in x['owner']:
  for i,every in enumerate(types):
    if every == each:
      temp.append(keys[i])
x['owner']=temp
###__________________________________________________END


###_______________________________________________SLPITTING_SAMPLES_AND_ADDING_BIAS_WEIGHTS

l=len(x)
print(l)
x=x.to_numpy()

x = numpy.hstack([numpy.ones((l, 1)),x])
print(numpy.shape(x))
print(x)

y=pandas.read_csv("car_price.csv",usecols=[2])
y=y.to_numpy()
maxx=numpy.max(y)
minn=numpy.min(y)
y=(y-minn)/(maxx-minn)
samples=len(y)
print(numpy.shape(y))

y_train=y[0:samples-128]
y_test=y[samples-128:]
x_train=x[0:samples-128]
x_test=x[samples-128:]

#print(numpy.shape(x_test))
###_______________________________________________________END


###________________________________________________LINEAR_REGRESSION
def loss(x,y,w):
    l = len(y)
    h = x.dot(w)
    error = (h-y)**2
    return numpy.sum(error)/(2*l)


def gradient_descent(x,y,w):
    m = len(y)
    for i in range(10000):
        h = x.dot(w)
        #print(numpy.shape(h))
        error = ((h-y).transpose()).dot(x)
        #print(numpy.shape(error))
        w -= 0.00001*(1/m)*(error.transpose())
        print("Loss : ",loss(x,y,w))
    return w

  

  
weights= numpy.zeros([x_train.shape[1],1])
print(numpy.shape(weights))
weights= gradient_descent(x_train,y_train,weights)
###______________________________________________________END


###___________________________________________________PREDICT
print("Prediction Loss: "loss(x_test,y_test,weights))



