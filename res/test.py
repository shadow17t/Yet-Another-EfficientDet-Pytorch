x = "20210510"
# initializing length list
cus_lens = [4, 2, 2]
 
res = []
strt = 0
for size in cus_lens:
     
    # slicing for particular length
    res.append(x[strt : strt + size])
    strt += size
dat=res[0]+"\\"+res[1]+"\\"+res[2]
# printing result 
print("Strings after splitting : " + dat) 